# 行动建议 Agent（Advisor）
# 作用：基于最终报告给出可执行的研究建议，是流程的"收尾顾问"。
#       调研模式给 3 个研究方向 + 5 篇核心论文 + 行动清单；论文模式只给论文 + 行动（不给方向）
# 位置：agent/advisor.py，被 main.py 的 advisor_node 调用
# 产出：Suggestion（directions + papers + actions）

from core.json_utils import parse_json
from core.llm_client import chat
from core.logger import get_logger
from core.schemas import FlowState, MODE_PAPER, Paper, Suggestion

log = get_logger("advisor")

# 调研模式：给方向 + 论文 + 行动
_SURVEY_SYSTEM = "你是科研方向规划顾问。基于调研报告给出可执行的研究建议，只输出 JSON，不要加任何解释或 Markdown 代码块。"
# 论文模式：只给论文 + 行动，不给方向（文档规定分析模式 directions 留空）
_PAPER_SYSTEM = "你是论文分析顾问。基于对比分析报告给出下一步建议，只输出 JSON，不要加任何解释或 Markdown 代码块。"


def run_advisor(flow: FlowState) -> Suggestion:
    """建议 Agent 入口：按模式生成建议。失败时返回空建议，不卡主流程"""
    report = flow.report
    markdown = report.markdown if report else "（报告缺失）"

    if flow.mode == MODE_PAPER:
        return _advise_paper(flow, markdown)
    return _advise_survey(flow, markdown)


def _serialize_context(flow: FlowState) -> str:
    """把研究任务 + 证据概况拼成文本，让 Advisor 知道哪些任务研究充分、哪些证据不足、哪些是空白"""
    tasks = flow.outline.research_tasks if flow.outline else []
    evidences = flow.evidences
    lines = []
    for t in tasks:
        related = [ev for ev in evidences if ev.task_id == t.id]
        if related:
            total_claims = sum(len(ev.claims) for ev in related)
            total_sources = sum(len(ev.sources) for ev in related)
            lines.append(f"- {t.id}「{t.question}」：{len(related)} 组证据、{total_claims} 条论断、{total_sources} 个来源")
        else:
            lines.append(f"- {t.id}「{t.question}」：无证据（可能是研究空白）")
    return "\n".join(lines) if lines else "（无研究任务）"


def _advise_survey(flow: FlowState, markdown: str) -> Suggestion:
    """调研模式：生成 3 个方向 + 5 篇论文 + 行动清单（参考研究任务和证据概况，不只吃报告正文）"""
    prompt = f"""调研问题：{flow.question}

研究任务与证据概况（据此判断哪些任务研究充分、哪些证据不足、哪些是研究空白）：
{_serialize_context(flow)}

调研报告正文（Markdown）：
{markdown}

请输出如下 JSON（严格按字段）：
{{
  "directions": ["研究方向1", "研究方向2", "研究方向3"],
  "papers": [{{"title": "论文标题", "reason": "一句话推荐理由", "link": "论文链接"}}],
  "actions": ["下一步行动1", "下一步行动2", "下一步行动3"]
}}
要求：directions 给 3 个值得深入研究的方向（优先针对证据不足的研究空白）；papers 给 5 篇最值得精读的核心论文（优先从报告引用的来源论文里挑，链接必须真实）；actions 给 3-5 条具体可执行的下一步行动。"""

    try:
        resp = chat(system=_SURVEY_SYSTEM, user=prompt, max_tokens=8192)
        data = parse_json(resp.text)
        if data:
            return Suggestion(
                directions=_as_str_list(data.get("directions"))[:3],
                papers=_normalize_papers(data.get("papers"))[:5],
                actions=_as_str_list(data.get("actions")),
            )
    except Exception as e:
        log.warning("建议 Agent（调研模式）生成失败（%s），返回空建议", type(e).__name__)

    return Suggestion()


def _advise_paper(flow: FlowState, markdown: str) -> Suggestion:
    """论文模式：生成论文 + 行动清单，directions 按文档规定留空（参考研究任务和证据概况）"""
    prompt = f"""研究任务与证据概况（据此判断哪些对比维度证据充足、哪些不足）：
{_serialize_context(flow)}

论文对比分析报告正文（Markdown）：
{markdown}

请输出如下 JSON（严格按字段）：
{{
  "papers": [{{"title": "论文标题", "reason": "一句话推荐理由", "link": "论文链接"}}],
  "actions": ["下一步行动1", "下一步行动2", "下一步行动3"]
}}
要求：papers 给最值得精读的论文（从分析论文里挑，可为空数组）；actions 给 3-5 条具体可执行的下一步行动。"""

    try:
        resp = chat(system=_PAPER_SYSTEM, user=prompt, max_tokens=8192)
        data = parse_json(resp.text)
        if data:
            return Suggestion(
                directions=[],   # 论文模式不给研究方向，留空
                papers=_normalize_papers(data.get("papers"))[:5],
                actions=_as_str_list(data.get("actions")),
            )
    except Exception as e:
        log.warning("建议 Agent（论文模式）生成失败（%s），返回空建议", type(e).__name__)

    return Suggestion()


def _as_str_list(value) -> list[str]:
    """把 LLM 可能返回的 list/str/None 统一转成字符串列表，去掉空项和首尾空格"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _normalize_papers(raw) -> list[Paper]:
    """把 LLM 返回的论文列表转成 Paper 对象列表，过滤没标题的脏数据"""
    if not isinstance(raw, list):
        return []
    out: list[Paper] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        title = str(p.get("title") or "").strip()
        if not title:
            continue   # 没标题的论文没法给用户，跳过
        out.append(Paper(
            title=title,
            reason=str(p.get("reason") or "").strip(),
            link=str(p.get("link") or "").strip(),
        ))
    return out
