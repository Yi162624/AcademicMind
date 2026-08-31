# 规划师 Agent（Planner）
# 作用：把用户的研究问题（或论文列表）拆解成可执行的研究大纲，
#       是整个调研流程的"总设计师"，下游研究员按它分工、写作者按它排版
# 位置：agent/planner.py，被 main.py 的 planner_node 调用
# 产出：Outline（调研模式=子主题+章节+图片需求；论文模式=分析维度+论文分组）

from core.json_utils import parse_json
from core.llm_client import chat
from core.logger import get_logger
from core.schemas import MODE_PAPER, FlowState, Outline, ResearchTask

log = get_logger("planner")

# ── 调研模式 prompt ──
# 让 LLM 把问题拆成 3 个子主题 + 报告章节 + 每章图片需求
_SURVEY_SYSTEM = """
你是学术调研规划师。
你的任务是将研究问题拆解为可并行执行的调研方向，并规划最终报告结构。
只输出 JSON，不要加任何解释或 Markdown 代码块。
"""

_SURVEY_USER = """研究问题：{question}

请输出如下 JSON（严格按字段）：
{{
  "subtopics": ["子主题1", "子主题2", "子主题3"],
  "sections": ["章节标题1", "章节标题2", "..."],
  "image_requirements": ["章节1需要的图片类型", "章节2需要的图片类型", "..."],
  "research_tasks": [
    {{
      "question": "研究员要回答的具体研究问题",
      "purpose": "为什么研究这个问题",
      "required_evidence": ["需要什么类型的证据1", "需要什么类型的证据2"],
      "target_sections": ["这些证据服务报告哪几章"]
    }}
  ]
}}

要求：
1. subtopics 必须恰好 3 个，三个子主题应从不同角度共同回答研究问题，彼此尽量不重叠，并且每个子主题都能够独立进行资料搜集。
2. sections 是最终报告的章节结构，必须包含引言和结论。章节应围绕研究问题形成清晰的逻辑链条，不要简单复制 subtopics。
3. image_requirements 必须与 sections 一一对应。只有图片能够明显帮助理解该章节时才指定图片；不需要图片的章节填写空字符串 ""。
4. research_tasks 是本次拆解的核心，也是研究员真正的"工作依据"，务必认真设计。每个 research_task 都要明确告诉研究员：搜什么（question）、为什么搜（purpose）、要什么证据（required_evidence）、证据用在报告哪里（target_sections）。
5. research_tasks 数量建议 3 个、与 subtopics 对应，但要比 subtopics 更具体——question 必须是"要回答的具体研究问题"，而不是泛泛的关键词。
6. 不要臆造具体图片、数据或资料来源。你只负责规划研究方向、报告结构和研究任务，不负责实际研究。
"""

# ── 论文模式 prompt ──
# 让 LLM 确定对比分析维度 + 论文分组
_PAPER_SYSTEM = """
你是论文对比分析规划师。
你的任务是根据论文集合确定真正具有研究价值的对比维度，并设计合理的论文分组。
只输出 JSON，不要加任何解释或 Markdown 代码块。
"""

_PAPER_USER = """论文列表：
{papers}

请输出如下 JSON（严格按字段）：
{{
  "analysis_dimensions": ["对比维度1", "对比维度2", "..."],
  "paper_assignments": [["论文标题A", "论文标题B"], ["论文标题C"]],
  "research_tasks": [
    {{
      "question": "围绕某个对比维度，要回答的具体研究问题",
      "purpose": "为什么比较这个维度",
      "required_evidence": ["需要从论文中提取什么证据"],
      "target_sections": ["这个维度对应报告哪一章"]
    }}
  ]
}}

要求：
1. analysis_dimensions 必须根据给定论文集合动态确定，体现这些论文之间真正值得比较的关键差异。
2. 对比维度可以涉及研究目标、核心方法、模型架构、训练策略、数据、实验设置、性能、局限、适用场景等，但必须根据具体论文集合选择，不要机械套用固定模板。
3. 不要简单使用"方法、实验、结果、结论"等过于宽泛的维度。维度应具体到能够进行有意义的横向比较，例如"预训练目标差异""模型架构选择""训练数据与规模""不同方法的性能表现"等。
4. analysis_dimensions 控制在 5–8 个最重要的维度，避免重复、过细或高度重叠。
5. paper_assignments 将所有论文分成若干个适合并行研究的组。每篇论文必须且只能被分到一个组。
6. 分组应具有研究意义，优先依据研究方向、方法路线、模型范式或技术演进关系进行分组，不要为了平均分配而随机分组。
7. 如果论文数量较少或论文高度相关，可以使用较少的组；如果存在明显不同的研究路线，应优先形成主题一致的分组。
8. research_tasks 与 analysis_dimensions 对应：每个分析维度至少一个研究任务，question 要具体到"这些论文在该维度上分别有什么做法、结果如何、差异在哪"。
9. 不要臆造论文内容。你只负责确定"比较什么"、"哪些论文放在一起研究"和"围绕什么研究问题组织证据"，不负责分析论文本身。
"""

def run_planner(flow: FlowState) -> Outline:
    """规划师入口：按任务模式拆大纲。被用户退回重拆时，flow.verifier_feedback 里带意见"""
    if flow.mode == MODE_PAPER:
        return _plan_paper(flow)   # 论文模式走另一套拆法
    return _plan_survey(flow)      # 默认调研模式


def _plan_survey(flow: FlowState) -> Outline:
    """调研模式：把问题拆成研究任务单+子主题+章节+图片需求。LLM 失败时用兜底大纲保证流程不断"""
    prompt = _SURVEY_USER.format(question=flow.question)
    # 用户退回重拆时，把修改意见塞进 prompt，让 LLM 照着改
    if flow.verifier_feedback.strip():
        prompt += f"\n\n用户对上一版大纲的修改意见：{flow.verifier_feedback}\n请按意见调整后重新输出。"

    try:
        resp = chat(system=_SURVEY_SYSTEM, user=prompt, max_tokens=8192)
        data = parse_json(resp.text)      # 把 LLM 的"脏 JSON"洗成 dict
        if data:
            sections = _as_str_list(data.get("sections"))
            # 研究任务单是核心，编号由系统分配（T1/T2...），不信 LLM
            tasks = _normalize_tasks(data.get("research_tasks"))
            return Outline(
                mode=flow.mode,               # 保持调研模式
                question=flow.question,       # 保持用户问题
                subtopics=_as_str_list(data.get("subtopics"))[:3],       # 最多取 3 个，别让研究员分太散
                sections=sections,               # 保持章节结构
                image_requirements=_align_image_reqs(data.get("image_requirements"), sections),  # 保持图片需求
                research_tasks=tasks,            # 研究任务单（研究员真正的工作依据）
            )
    except Exception as e:
        log.warning("规划师拆解失败（%s），用兜底大纲", type(e).__name__)

    return _fallback_survey_outline(flow)        # 保持调研模式


def _plan_paper(flow: FlowState) -> Outline:
    """论文模式：确定分析维度+论文分组+研究任务单。LLM 失败时退化成每篇一组"""
    # 把论文列表拼成带编号的文本，喂给 LLM
    lines = "\n".join(
        f"{i + 1}. {p.title}" + (f"（{p.link}）" if p.link else "")
        for i, p in enumerate(flow.papers)
    )
    prompt = _PAPER_USER.format(papers=lines)      # 保持论文列表
    if flow.verifier_feedback.strip():
        prompt += f"\n\n用户对上一版大纲的修改意见：{flow.verifier_feedback}\n请按意见调整后重新输出。"

    try:
        resp = chat(system=_PAPER_SYSTEM, user=prompt, max_tokens=8192)
        data = parse_json(resp.text)
        if data:
            return Outline(
                mode=flow.mode,               # 保持论文模式
                question=flow.question,       # 保持用户问题
                analysis_dimensions=_as_str_list(data.get("analysis_dimensions")),  # 保持分析维度
                paper_assignments=_normalize_assignments(data.get("paper_assignments"), flow.papers),  # 保持论文分组
                research_tasks=_normalize_tasks(data.get("research_tasks")),  # 研究任务单（系统分配编号）
            )
    except Exception as e:
        log.warning("规划师（论文模式）拆解失败（%s），用兜底分组", type(e).__name__)

    return _fallback_paper_outline(flow)        # 保持论文模式


def _as_str_list(value) -> list[str]:
    """把 LLM 可能返回的 list/str/None 统一转成字符串列表，顺手去掉空项和首尾空格"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _normalize_tasks(raw) -> list[ResearchTask]:
    """把 LLM 返回的研究任务列表转成 ResearchTask 对象，并系统分配 T1/T2/T3 编号。
    编号不信 LLM，按"有效任务"顺序重新分配；question 为空的任务直接跳过"""
    if not isinstance(raw, list):
        return []
    tasks: list[ResearchTask] = []
    seq = 0   # 有效任务的计数器，保证编号连续（T1/T2/T3...）
    for t in raw:
        if not isinstance(t, dict):
            continue
        q = str(t.get("question") or "").strip()
        if not q:
            continue   # 没有问题的任务没意义，跳过
        seq += 1
        tasks.append(ResearchTask(
            id=f"T{seq}",                          # 系统分配编号，不信 LLM
            question=q,
            purpose=str(t.get("purpose") or "").strip(),
            required_evidence=_as_str_list(t.get("required_evidence")),
            target_sections=_as_str_list(t.get("target_sections")),
        ))
    return tasks


def _align_image_reqs(reqs, sections: list[str]) -> list[str]:
    """把图片需求列表对齐到章节数：少了补空串，多了截断，保证和 sections 一一对应"""
    reqs = _as_str_list(reqs)
    n = len(sections)
    if len(reqs) >= n:
        return reqs[:n]
    return reqs + [""] * (n - len(reqs))


def _normalize_assignments(assignments, papers) -> list[list[str]]:
    """把 LLM 给的分组整理成合法形式：确保每篇论文都被分到、且不重复。
    LLM 可能漏分组或分重了，这里兜底补全，避免研究员漏查某篇论文"""
    titles = [p.title for p in papers]
    groups: list[list[str]] = []
    seen: set[str] = set()

    if isinstance(assignments, list):
        for g in assignments:
            if isinstance(g, list):
                names = [str(x).strip() for x in g if str(x).strip()]
            elif isinstance(g, str):
                names = [g.strip()] if g.strip() else []
            else:
                names = []
            if names:
                # 已经分过的论文不重复收进新组
                groups.append([n for n in names if n not in seen])
                seen.update(names)

    # 没被分到的论文，每篇单独补一组
    for t in titles:
        if t not in seen:
            groups.append([t])
            seen.add(t)

    return groups or [[t] for t in titles]


def _fallback_survey_outline(flow: FlowState) -> Outline:
    """调研模式兜底大纲：LLM 失败时用最朴素的拆分，保证流程不中断"""
    q = flow.question.strip()
    return Outline(
        mode=flow.mode,               # 保持调研模式
        question=flow.question,       # 保持用户问题
        subtopics=[q] if q else ["研究问题"],
        sections=["研究背景", "核心方法", "关键结论"],
        image_requirements=["", "", ""],
        research_tasks=[ResearchTask(   # 兜底任务单：一个任务回答整个问题，研究员有活干
            id="T1",
            question=q or "研究问题",
            purpose="理解研究问题的背景、方法与结论",
            required_evidence=["相关论文", "方法", "结论"],
            target_sections=["研究背景", "核心方法", "关键结论"],
        )],
    )


def _fallback_paper_outline(flow: FlowState) -> Outline:
    """论文模式兜底：每篇论文一组，用通用分析维度，每个维度一个研究任务"""
    dims = ["研究方法", "实验设计", "核心结论", "局限性"]
    return Outline(
        mode=flow.mode,               # 保持论文模式
        question=flow.question,       # 保持用户问题
        analysis_dimensions=dims,
        paper_assignments=[[p.title] for p in flow.papers],
        research_tasks=[ResearchTask(   # 每个维度一个任务，研究员按维度整理证据
            id=f"T{i + 1}",
            question=f"这些论文在「{d}」上分别有什么做法和结果？",
            purpose=f"横向对比{d}",
            required_evidence=[f"各论文在{d}上的具体做法/数据"],
            target_sections=[d],
        ) for i, d in enumerate(dims)],
    )
