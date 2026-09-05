# 验证员 Agent（Verifier）
# 作用：检查写作者产出的报告质量——引用是否真实、事实是否有证据支撑、有没有编造。
#       不过关就退回写作者重写（最多 1 次，之后交用户拍板）。
# 位置：agent/verifier.py，被 main.py 的 verify_report_node 调用
# 产出：VerifierResult（passed 是否通过 + issues 问题清单 + feedback 修改意见）
#
# 注意：大纲已改由用户确认，不再用 AI 检查，所以这里只实现 check_report，没有 check_outline。

from core.json_utils import parse_json
from core.llm_client import chat
from core.logger import get_logger
from core.schemas import Evidence, FlowState, Issue, VerifierResult

log = get_logger("verifier")

# 检查报告质量的角色设定，逼 LLM 只输出结构化 JSON
_SYSTEM = "你是学术报告审核员。检查报告的引用真实性、事实准确性和忠实度，只输出 JSON，不要加任何解释或 Markdown 代码块。"


def check_report(flow: FlowState) -> VerifierResult:
    """验证员入口：检查报告质量，返回是否通过 + 问题清单 + 修改意见。
    检查失败（LLM 解析不出 JSON）时按"放行"处理，不卡死主流程"""
    report = flow.report
    if report is None:
        # 报告都没有，谈不上检查，直接判过，让流程尽早暴露问题
        return VerifierResult(passed=True)

    prompt = _build_prompt(flow, report.markdown)
    try:
        resp = chat(system=_SYSTEM, user=prompt, max_tokens=8192)
        data = parse_json(resp.text)
        if data is None:
            log.warning("验证员 JSON 解析失败，按放行处理")
            return VerifierResult(passed=True)
        issues = _normalize_issues(data.get("issues"))
        feedback = str(data.get("feedback") or "").strip()
        # 只要有 error 级问题就判不过，退回重写；只有 warning 也算过（只提示不改）
        passed = not any(i.level == "error" for i in issues)
        log.info("验证完成：通过=%s，问题 %d 条", passed, len(issues))
        return VerifierResult(passed=passed, issues=issues, feedback=feedback)
    except Exception as e:
        log.warning("验证员检查失败（%s），按放行处理", type(e).__name__)
        return VerifierResult(passed=True)


def _build_prompt(flow: FlowState, md: str) -> str:
    """拼验证用的 prompt：把报告正文 + 证据链一起给 LLM，让它核对"正文引用的论文标题"是否真实、论断有没有证据支撑"""
    sections = (flow.outline.sections if flow.outline else []) or ["未给大纲"]
    return f"""报告章节大纲：{sections}

报告正文（Markdown，引用论文处标注（来源：[论文标题](链接)））：
{md}

证据链（研究员整理的结构化证据，是"唯一事实来源"；E 编号=证据包，S 编号=来源论文）：
{_serialize_evidences(flow)}

请逐项检查并只输出如下 JSON：
{{
  "issues": [
    {{"level": "error 或 warning", "check_type": "fact/citation/fidelity", "message": "问题描述", "location": "哪个章节或哪条引用"}}
  ],
  "feedback": "综合修改意见，没有问题就写空字符串"
}}

检查维度：
1. fact（事实）：报告的事实/结论能否在证据链里找到对应论断（Claim）支撑，有没有凭空编造
2. citation（引用）：报告正文标注的论文标题是否真实存在于证据链的来源论文列表；论断的 source_id 是否对应真实来源论文；标题引用与正文内容是否对得上
3. fidelity（忠实度）：报告是否忠实于证据原文，有没有夸大、曲解、扩大结论

判定标准：error = 必须改（引用不存在的论文、编造数据、张冠李戴、严重曲解）；warning = 建议改（表述不严谨）。"""


def _serialize_evidences(flow: FlowState) -> str:
    """把证据链拼成清单（带 E/S 编号），供验证员核对报告引用真假、并沿 Claim → Source 追溯"""
    if not flow.evidences:
        return "（无证据）"
    blocks = []
    for ev in flow.evidences:
        blocks.append(f"【{ev.evidence_id}】任务 {ev.task_id}")
        if ev.findings:
            blocks.append("  结论：")
            blocks.extend(f"    - {f}" for f in ev.findings)
        if ev.claims:
            blocks.append("  论断（含出处）：")
            for c in ev.claims:
                src = _find_source(ev, c.source_id)
                blocks.append(f"    - {c.claim} [来源 {c.source_id}《{src.title if src else '?'}》]")
                if c.evidence_text:
                    blocks.append(f"      原文证据：{c.evidence_text}")
        if ev.sources:
            blocks.append("  来源论文：")
            blocks.extend(f"    - {s.id}《{s.title}》{s.link}" for s in ev.sources)
    return "\n".join(blocks)


def _find_source(ev: Evidence, source_id: str):
    """在证据的来源列表里按编号找来源论文，找不到返回 None"""
    for s in ev.sources:
        if s.id == source_id:
            return s
    return None


def _normalize_issues(raw) -> list[Issue]:
    """把 LLM 返回的问题列表转成 Issue 对象列表，顺便过滤脏数据"""
    if not isinstance(raw, list):
        return []
    out: list[Issue] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        level = str(it.get("level") or "warning").strip().lower()
        # 只认 error/warning 两档，其他都归 warning，避免脏数据漏判
        if level != "error":
            level = "warning"
        message = str(it.get("message") or "").strip()
        if not message:
            continue   # 没内容的问题不算问题
        out.append(Issue(
            level=level,
            check_type=str(it.get("check_type") or "fact").strip(),
            message=message,
            location=str(it.get("location") or "").strip(),
        ))
    return out
