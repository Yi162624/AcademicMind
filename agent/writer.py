# 写作者 Agent（Writer）
# 作用：把研究员整理好的结构化证据组织成一份面向用户的 Markdown 报告（正常可读的学术报告）。
#       关键：Writer 只负责"组织已有证据"，不做二次研究——证据来自研究员（findings/claims/sources），
#       正文引用论文处标注（来源：论文标题链接），供验证员核对引用真假；不再向用户暴露内部证据编号 E1/E2。
# 位置：agent/writer.py，被 main.py 的 writer_node 调用
# 产出：Report（outline + markdown 正文）
#
# 配图说明：本期研究员还没接入图片搜集，先出纯文字 Markdown；
#           等 researcher 接入图片后，再考虑把图嵌进报告（Markdown 支持图片）。

from core.llm_client import chat
from core.logger import get_logger
from core.schemas import FlowState, MODE_PAPER, Report

log = get_logger("writer")

# 调研模式：写领域调研 Markdown 报告
_SURVEY_SYSTEM = "你是学术调研报告撰写助手。根据研究任务和已整理好的证据，写一篇结构完整的 Markdown 报告。只输出 Markdown 代码，不要加任何解释或 Markdown 代码块。"
# 论文模式：写论文对比综述 Markdown 报告
_PAPER_SYSTEM = "你是论文对比分析报告撰写助手。根据对比维度和已整理好的证据，写一篇论文对比综述的 Markdown 报告。只输出 Markdown 代码，不要加任何解释或 Markdown 代码块。"


def run_writer(flow: FlowState) -> Report:
    """写作者入口：把证据组织成 Markdown 报告。重写时 flow.verifier_feedback 里带验证员意见"""
    md = _generate_markdown(flow)

    # 配图预留：等 researcher 接入图片搜集后，这里对每章调 visual_retrieval.retrieve() 拿图嵌进报告。
    # 本期图片没搜集，retrieve 也必然返回空，所以先纯文字输出，不影响主流程跑通。
    return Report(outline=flow.outline, markdown=md)


def _generate_markdown(flow: FlowState) -> str:
    """调 LLM 生成 Markdown 报告正文，按模式选不同的写法和结构标题"""
    outline = flow.outline

    if flow.mode == MODE_PAPER:
        title = flow.question or "论文对比分析报告"
        # 论文模式按"对比维度"组织章节（每章一个维度，章内对比各论文）
        structure = (outline.analysis_dimensions if outline else []) or ["研究方法", "实验设计", "核心结论"]
        system = _PAPER_SYSTEM
    else:
        title = flow.question or "领域调研报告"
        structure = (outline.sections if outline else []) or ["研究背景", "核心方法", "关键结论"]
        system = _SURVEY_SYSTEM

    structure_lines = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(structure))
    prompt = f"""报告标题：{title}

报告章节（严格按此顺序写）：
{structure_lines}

研究任务与证据（研究员已整理好的结构化证据，是"唯一事实来源"）：
{_serialize_research_context(flow)}

写作要求：
1. 输出规范 Markdown 报告：章节用 ## 标题、正文用普通段落、列表用 - 或 1. 编号
2. 严格按上面给定的章节顺序组织，每章用 ## 标题
3. 正文引用论文时，在句末标注「（来源：[论文标题](链接)）」，给出可点开的出处；禁止使用 E1/E2 等内部证据编号，不要把编号暴露给读者
4. 引用论文链接必须取自上面证据里的来源链接，禁止编造或拼凑链接
5. 内容只能基于上面的证据，禁止凭空编造数据、结论或引用不存在的论文
6. 结尾附一个「## 参考文献」章节，用有序列表列出所有来源论文的标题和链接"""

    # 验证员退回重写时，把修改意见塞进 prompt，让 LLM 针对性改
    if flow.verifier_feedback.strip():
        prompt += f"\n\n验证员提出的修改意见（必须逐条落实）：\n{flow.verifier_feedback}"

    try:
        resp = chat(system=system, user=prompt, max_tokens=12288)
        return _clean_markdown(resp.text)
    except Exception as e:
        log.warning("写作者生成报告失败（%s），返回降级报告", type(e).__name__)
        return _fallback_markdown(flow)


def _serialize_research_context(flow: FlowState) -> str:
    """把研究任务单 + 编号证据拼成一段文本，让 LLM 知道每条证据属于哪个任务、回答什么问题、该放哪章"""
    evidences = flow.evidences
    if not evidences:
        return "（未搜集到证据）"
    tasks = flow.outline.research_tasks if flow.outline else []
    task_map = {t.id: t for t in tasks}   # 建"任务编号 → 任务"映射，方便给证据标它回答的问题

    blocks = []
    for ev in evidences:
        task = task_map.get(ev.task_id)
        header = f"【{ev.evidence_id}】任务 {ev.task_id}"
        if task:
            target = "、".join(task.target_sections) if task.target_sections else "未指定"
            header += f"（要回答：{task.question}；服务章节：{target}）"
        blocks.append(header)
        if ev.findings:
            blocks.append("  综合结论：")
            blocks.extend(f"    - {f}" for f in ev.findings)
        if ev.claims:
            blocks.append("  可追溯论断：")
            blocks.extend(f"    - {c.claim}（来源 {c.source_id}）" for c in ev.claims)
        if ev.sources:
            blocks.append("  来源论文：")
            blocks.extend(f"    - {s.id}《{s.title}》{s.link}" for s in ev.sources)
    return "\n".join(blocks)


def _clean_markdown(text: str) -> str:
    """去掉 LLM 输出里可能包裹 Markdown 的代码块围栏，只留干净的内容"""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")        # 找到 ``` 这一行的结尾
        if first_newline != -1:
            text = text[first_newline + 1:]    # 去掉开头的 ```markdown 或 ```
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]          # 去掉结尾的 ```
    return text.strip()


def _fallback_markdown(flow: FlowState) -> str:
    """降级报告：LLM 生成失败时返回一段简单 Markdown，保证前端有东西可展示"""
    title = flow.question or "报告"
    return f"""# {title}

> 报告生成失败，请稍后重试或人工复核证据素材。"""
