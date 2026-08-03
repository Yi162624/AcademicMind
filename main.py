# main.py 系统入口
# 作用：启动任务（run_task）和恢复被暂停的任务（resume_task）
#       内部用 LangGraph 把 5 个 Agent 串起来跑
# 分工：编排逻辑在本文件，Agent 干活在 agent/（搭档实现）
# 流程：查论文相关性→规划师→用户确认大纲→研究员→写报告→验证报告（AI）→给建议

import argparse
import time
import uuid
from datetime import datetime
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from core.logger import get_logger
from core.schemas import (
    FinalResult,
    FlowState,
    MODE_PAPER,
    MODE_SURVEY,
    PaperSource,
    TaskRecord,
)

log = get_logger("main")  # 本模块日志器，排错看 main 的日志

# 编译好的状态机实例缓存：懒加载（agent 没实现时，import main 不报错，跑任务时才报）
_GRAPH = None

# thread_id → 任务开始时间，跨"挂起/恢复"算总耗时用
_START_TIMES: dict[str, float] = {}


class GraphState(TypedDict):
    """LangGraph 状态机的"盒子"：flow 是整个任务的"黑板"，route 是节点间的路由标记"""
    flow: FlowState   # 全流程数据（schemas.py 里定好的结构）
    route: str        # 路标：几个分支节点写 pass/retry/end，条件边读它决定往哪走


def _issue_to_dict(issue) -> dict:
    """把 Issue 转成普通字典：interrupt() 挂起的值要好序列化，不然前端接不住"""
    return {
        "level": issue.level,            # 错误等级：error/warning/info
        "check_type": issue.check_type,  # 校验类型：outline/report
        "message": issue.message,        # 错误信息
        "location": issue.location,      # 错误位置：文件名:行号
        }


def _outline_to_dict(outline) -> dict:
    """把 Outline 转成普通字典：大纲要展示给用户看，转成好序列化的格式"""
    return {
        "subtopics": outline.subtopics,                # 子主题（调研模式用）
        "sections": outline.sections,                  # 章节标题
        "image_requirements": outline.image_requirements,     # 每章需要什么图
        "analysis_dimensions": outline.analysis_dimensions,   # 分析维度（论文模式用）
        "paper_assignments": outline.paper_assignments,       # 论文分组（论文模式用）
    }


def _build_graph():
    """构建并编译 LangGraph 状态机（agent 模块没实现时给出明确提示而不是裸报错）"""
    # agent 是搭档负责的模块，还没写好时 import 会失败，这里转成好懂的中文提示
    try:
        from agent.advisor import run_advisor
        from agent.planner import run_planner
        from agent.researcher import run_researcher
        from agent.verifier import check_report
        from agent.writer import run_writer
    except ImportError as e:
        raise RuntimeError(
            f"agent 模块还没实现（搭档负责），状态机起不来：{e}。"
            f"请先让搭档按 main.py 头部注释的接口契约实现 agent/ 里的 5 个文件"
        ) from e

    # ═══ 节点函数：每个节点只做三件事：从黑板取数 → 调对应 Agent → 写回黑板 ═══

    def check_relevance_node(state: GraphState) -> dict:
        """论文相关性检查节点：paper模式≥2篇才查；相关直接过，低相关挂起问用户"""
        flow = state["flow"]
        # 调研模式或单篇论文不用查相关性，直接放行
        if flow.mode != MODE_PAPER or len(flow.papers) < 2:
            return {"flow": flow, "route": "pass"}
        # memory 模块是本人负责的，还没写好时给明确提示
        try:
            from memory.paper_relevance import check_paper_relevance
        except ImportError as e:
            raise RuntimeError(
                f"memory/paper_relevance.py 还没实现（本人负责），相关性检查起不来：{e}"
            ) from e
        flow.papers_relevant, flow.relevance_note = check_paper_relevance(flow.papers)
        if flow.papers_relevant:
            return {"flow": flow, "route": "pass"}   # 论文相关，直接进规划师
        # 低相关 → 挂起，告诉用户哪些论文差异大，让用户决定
        decision = interrupt({
            "stage": "relevance",                    # 前端据此弹"相关性提示"界面
            "note": flow.relevance_note,             # 哪些论文相关性低
        })
        if decision.get("choice") == "continue":
            # 用户坚持继续 → 记警告"对比仅供参考"，进规划师
            flow.warning_flags.append(f"论文相关性较低（用户选择继续）：{flow.relevance_note}，对比仅供参考")
            return {"flow": flow, "route": "pass"}
        # 用户选择取消 → 直接结束，不带报告
        flow.final = FinalResult(
            mode=flow.mode,
            question=flow.question,
            warning_flags=["已按用户要求取消：论文相关性低，建议分开分析"],
        )
        return {"flow": flow, "route": "end"}

    def planner_node(state: GraphState) -> dict:
        """规划师节点：拆解问题出大纲；被退回重拆时把用户的意见喂给它"""
        flow = state["flow"]
        flow.outline = run_planner(flow)   # 出大纲（重拆时 flow.verifier_feedback 里有用户上次的意见）
        flow.verifier_feedback = ""        # 意见用完就清掉，防止下次重拆时重复注入
        return {"flow": flow, "route": "user_check_outline"}

    def user_check_outline_node(state: GraphState) -> dict:
        """大纲人工确认节点：大纲一生成就挂起给用户看，用户说了算（大纲不再交给 AI 判）"""
        flow = state["flow"]
        # 挂起：把大纲内容打包给前端展示，等用户决定用不用
        decision = interrupt({
            "stage": "outline",                            # 前端据此弹"大纲确认"界面
            "outline": _outline_to_dict(flow.outline),     # 大纲内容（子主题/章节/图片需求）
            "question": flow.question,                     # 用户原问题，前端展示上下文用
        })
        if decision.get("choice") == "accept":
            # 用户认可当前大纲 → 直接用它去搜集资料
            return {"flow": flow, "route": "pass"}
        # 用户给了修改意见 → 写进 verifier_feedback，退回规划师按意见修改（改完会再挂起确认）
        flow.verifier_feedback = decision.get("feedback", "")
        if not flow.verifier_feedback.strip():
            # 选了重写但没写字 → 等同于接受当前大纲，别让规划师白跑一趟
            return {"flow": flow, "route": "pass"}
        return {"flow": flow, "route": "retry"}

    def researcher_node(state: GraphState) -> dict:
        """研究员节点：按大纲并行搜集图文证据（并行在 agent 内部做，返回合并好的完整列表）"""
        flow = state["flow"]
        flow.evidences = run_researcher(flow)   # list[Evidence]：文本证据 + 图片素材
        return {"flow": flow, "route": "writer"}

    def writer_node(state: GraphState) -> dict:
        """写作者节点：整合证据生成图文 HTML 报告；被退回重写时带上验证员的意见"""
        flow = state["flow"]
        flow.report = run_writer(flow)          # 生成报告（重写时 flow.verifier_feedback 里有修改意见）
        flow.verifier_feedback = ""             # 意见用完清掉，防止下次重复注入
        return {"flow": flow, "route": "verify_report"}

    def verify_report_node(state: GraphState) -> dict:
        """报告校验节点：不过→退回写作者按意见修改（最多1次）；次数满了→挂起等用户拍板"""
        flow = state["flow"]
        result = check_report(flow)             # 检查报告（事实/引用/图文一致）
        if result.passed:
            return {"flow": flow, "route": "pass"}   # 报告过了，去生成建议
        if flow.report_retry_count >= 1:
            decision = interrupt({
                "stage": "report",                       # 挂在哪一步，前端展示用
                "issues": [_issue_to_dict(i) for i in result.issues],
                "feedback": result.feedback,
            })
            if decision.get("choice") == "accept":
                flow.warning_flags.append(f"报告未完全通过（已人工接受）：{result.feedback}")
                return {"flow": flow, "route": "pass"}
            # 用户给了修改意见 → 写进 verifier_feedback，退回写作者按意见修改
            flow.verifier_feedback = decision.get("feedback", "")
            if not flow.verifier_feedback.strip():
                # 选了修改但没写字 → 等同于接受当前报告，别让写作者白跑一趟
                flow.warning_flags.append(f"报告未完全通过（已人工接受）：{result.feedback}")
                return {"flow": flow, "route": "pass"}
            # 不重置计数器：改完最多再自动试 1 次，还不过就再次挂起问用户
            return {"flow": flow, "route": "retry"}
        flow.report_retry_count += 1
        flow.verifier_feedback = result.feedback
        return {"flow": flow, "route": "retry"}

    def advisor_node(state: GraphState) -> dict:
        """行动建议节点：基于报告生成研究方向 + 核心论文 + 行动清单"""
        flow = state["flow"]
        flow.suggestion = run_advisor(flow)     # Suggestion（方向/论文/行动）
        return {"flow": flow, "route": "finalize"}

    def finalize_node(state: GraphState) -> dict:
        """收尾节点：把报告+建议+警告标签打包成前端唯一认的 FinalResult"""
        flow = state["flow"]
        flow.final = FinalResult(
            mode=flow.mode,
            question=flow.question,
            report=flow.report,
            suggestion=flow.suggestion,
            warning_flags=flow.warning_flags,
        )
        return {"flow": flow, "route": "end"}

    # ═══ 搭图：节点 + 连线 + 条件边 ═══
    g = StateGraph(GraphState)
    g.add_node("check_relevance", check_relevance_node)
    g.add_node("planner", planner_node)
    g.add_node("user_check_outline", user_check_outline_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("verify_report", verify_report_node)
    g.add_node("advisor", advisor_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "check_relevance")
    # 相关性检查结果决定下一步：pass 去规划师，end 直接结束（用户取消）
    g.add_conditional_edges("check_relevance", lambda s: s["route"],
                            {"pass": "planner", "end": END})
    g.add_edge("planner", "user_check_outline")
    # 用户确认大纲结果决定下一步：pass 去搜集资料，retry 退回规划师重拆
    g.add_conditional_edges("user_check_outline", lambda s: s["route"],
                            {"pass": "researcher", "retry": "planner"})
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "verify_report")
    # 报告校验结果决定下一步：pass 去生成建议，retry 退回写作者重写
    g.add_conditional_edges("verify_report", lambda s: s["route"],
                            {"pass": "advisor", "retry": "writer"})
    g.add_edge("advisor", "finalize")
    g.add_edge("finalize", END)

    # MemorySaver 是内存版检查点：interrupt() 挂起后能按 thread_id 恢复现场。
    # 注意：进程重启状态就没了，Streamlit 单进程跑没问题；以后要持久化换 SqliteSaver。
    return g.compile(checkpointer=MemorySaver())


def _get_graph():
    """拿编译好的状态机（懒加载：第一次调用才构建，agent 没实现时也允许 import main）"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def _run_deep_read(paper: PaperSource, config: dict) -> tuple[dict, FinalResult]:
    """单篇论文：直接走精读 skill（不经 5 Agent）。skill 是本人负责的模块，还没实现时给明确提示"""
    try:
        from skills.paper_deep_read.skill import run_deep_read as skill_run
    except ImportError as e:
        raise NotImplementedError(
            f"单篇精读 skill（skills/paper_deep_read）还没实现：{e}。先传 ≥2 篇论文走多 Agent 流程"
        ) from e
    deep = skill_run(paper)                        # 八段式精读报告
    flow = FlowState(mode=MODE_PAPER, question=paper.title, papers=[paper])
    flow.final = FinalResult(mode=MODE_PAPER, question=paper.title, deep_read=deep)
    _log_task(flow, 0.0)                           # 精读耗时由 skill 自己报，这里先记 0
    return config, flow.final


def _log_task(flow: FlowState, duration_sec: float) -> None:
    """记一次任务日志（文档 4.2）：耗时/模式/问题写进 SQLite task_log 表。
    非关键路径：memory 模块没实现或写库失败都不影响主流程，只记个 warning"""
    try:
        from memory.sqlite_store import save_task_log   # memory 是本人负责的模块
        save_task_log(TaskRecord(
            task_id=uuid.uuid4().hex,
            mode=flow.mode,
            question=flow.question,
            duration_sec=round(duration_sec, 2),
            created_at=datetime.now().isoformat(timespec="seconds"),
            # token_usage/cost 默认 0：等 agent 把 token 用量汇到 flow 里再填
        ))
    except Exception as e:
        log.warning("任务日志写入失败（不影响主流程）：%s", e)


def run_task(mode: str = MODE_SURVEY, question: str = "", papers: list[PaperSource] | None = None,
             thread_id: str | None = None) -> tuple[dict, dict | FinalResult]:
    """启动一次任务（前端唯一入口）。
    返回 (config, 结果)：
      - 结果是个 dict 且带 "stage" 字段 → 被 HUMAN_IN_LOOP 挂起了，config 留给 resume_task 用，前端弹按钮
      - 结果是 FinalResult → 任务完成，前端直接展示
    """
    papers = papers or []
    thread_id = thread_id or uuid.uuid4().hex   # 每次任务一个会话号，挂起恢复都靠它
    config = {"configurable": {"thread_id": thread_id}}

    if mode == MODE_PAPER:
        if not papers:
            raise ValueError("论文分析模式至少要给 1 篇论文")
        if len(papers) == 1:
            return _run_deep_read(papers[0], config)   # 单篇 → 精读 skill，不经多 Agent
    elif not question.strip():
        raise ValueError("调研模式需要输入研究问题")

    flow = FlowState(mode=mode, question=question, papers=papers)
    _START_TIMES[thread_id] = time.time()               # 记开始时间，跨挂起算总耗时
    result = _get_graph().invoke({"flow": flow, "route": "check_relevance"}, config)

    interrupts = result.get("__interrupt__")            # 被挂起时 LangGraph 会塞这个字段
    if interrupts:
        return config, interrupts[0].value              # 挂起 → 把问题清单丢给前端

    _log_task(result["flow"], time.time() - _START_TIMES.pop(thread_id, time.time()))
    return config, result["flow"].final


def resume_task(thread_id: str, choice: str, feedback: str = "") -> tuple[dict, dict | FinalResult]:
    """恢复被挂起的任务：让流程从挂起点接着跑，并把用户决定喂回去。
    choice：accept=接受当前版本继续 / revise=给修改意见（AI 按意见改，不是强制全部重写）
            / continue=相关性低仍继续生成 / cancel=相关性低时取消
    feedback：choice 选 revise 时填的修改意见
    返回格式和 run_task 一样：dict+stage=又挂起（相关性提示/大纲确认/报告复核），FinalResult=跑完了
    """
    config = {"configurable": {"thread_id": thread_id}}
    # Command(resume=...) 会把值原样塞回 interrupt() 的返回值，节点拿到后决定走向
    result = _get_graph().invoke(Command(resume={"choice": choice, "feedback": feedback}), config)

    interrupts = result.get("__interrupt__")            # 被挂起时 LangGraph 会塞这个字段
    if interrupts:
        return config, interrupts[0].value              # 挂起 → 把问题清单丢给前端
    _log_task(result["flow"], time.time() - _START_TIMES.pop(thread_id, time.time()))    # 记任务耗时
    return config, result["flow"].final       # 任务完成，前端直接展示


def main():
    """命令行自测入口：开发期没前端时用来验证状态机跑得通。
    正式使用由 Streamlit 前端调 run_task / resume_task，这个函数只是开发辅助"""
    parser = argparse.ArgumentParser(description="AcademicMind 命令行自测")      # 定义命令行参数解析器
    parser.add_argument("--mode", default=MODE_SURVEY, choices=[MODE_SURVEY, MODE_PAPER],
                        help="survey=调研 / paper=论文分析")
    parser.add_argument("--question", default="大模型在医疗影像的应用", help="调研问题（survey 用）")
    parser.add_argument("--papers", nargs="+", default=[], help="论文链接列表（paper 用）")
    args = parser.parse_args()          # 解析命令行参数

    papers = [PaperSource(title=p.split("/")[-1], link=p) for p in args.papers]       # 把论文链接转成 PaperSource
    config, out = run_task(mode=args.mode, question=args.question, papers=papers)
    if isinstance(out, dict) and "stage" in out:
        # 被挂起：命令行没法弹按钮，演示到挂起这一步就停
        if out["stage"] == "outline":
            # 大纲确认环节：把规划师生成的大纲打出来看
            log.info("规划师生成大纲（等待用户确认）：%s", out["outline"])
        elif out["stage"] == "relevance":
            # 论文相关性提示：打印哪些论文差异大
            log.info("论文相关性较低（等待用户决定）：%s", out["note"])
        else:
            log.info("报告校验挂起（%s），前端会弹按钮；命令行演示到此为止", out["stage"])
            for issue in out.get("issues", []):
                log.info("  未通过项：%s", issue["message"])
        return
    log.info("任务完成：mode=%s question=%s warning=%s", out.mode, out.question, out.warning_flags)


if __name__ == "__main__":
    main()
