# 前端入口（Streamlit · 对话式交互）
# 作用：以"用户和 AI 对话"的形式驱动后端多 Agent 流程。
#       用户用聊天框提需求，AI 以消息形式返回大纲确认/相关性提示/报告复核/最终结果。
# 运行：streamlit run frontend/app.py
# 关键设计：
#   - 后端 run_task/resume_task 是阻塞的（搜论文/调 API 要几十秒），放子线程跑，避免卡死 UI
#   - 子线程不能直接写 session_state（会丢 ScriptRunContext），改成写共享队列 mailbox，
#     主线程每次 rerun 时从队列取结果再写状态 —— 这是"输入后刷新无结果"的修复点
#   - 消息历史存 session_state.messages，rerun 后重新渲染，对话不丢

import os
import queue
import sys
import threading
import uuid

import streamlit as st
import streamlit.components.v1 as components

# 把项目根目录加到模块搜索路径：app.py 在 frontend/ 下，不 import 后端就找不到 main/core/agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.schemas import FinalResult, MODE_PAPER, MODE_SURVEY, PaperSource  # noqa: E402
from main import resume_task, run_task  # noqa: E402
from memory.sqlite_store import get_user_config, save_user_config  # noqa: E402

# ─────────────────────────────────────────────
# 页面基础配置
# ─────────────────────────────────────────────
st.set_page_config(page_title="AcademicMind · 研究助手", page_icon="🧠", layout="wide")

# 子线程 → 主线程 的结果信箱：后端跑完把 (kind, payload) 塞进来，主线程轮询取走
# kind 有 "pending"（被挂起）/"result"（完成）/"error"（报错）/"done"（线程结束标记）
_MAILBOX = queue.Queue()


# ─────────────────────────────────────────────
# 状态初始化与队列消费
# ─────────────────────────────────────────────
def _init_state():
    """初始化 session_state：跨 rerun 保存对话与任务上下文"""
    defaults = {
        "mode": get_user_config().preferred_mode,  # 模式偏好：从 SQLite 恢复上次用的模式
        "messages": [],        # 对话历史 [{role, kind, ...}]，rerun 后重新渲染
        "thread_id": None,     # 当前任务的会话号：挂起恢复靠它定位状态机现场
        "pending": None,       # 当前被挂起的 HIL 信息（dict，含 stage 字段）
        "running": False,      # 任务是否在跑：决定是否显示"思考中"
        "papers": [],          # 论文模式下当前收集到的论文（跨消息累积）
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _drain_mailbox():
    """主线程每次 rerun 时把子线程塞进队列的结果取出来写状态。
    这是修复核心：子线程写 session_state 会丢上下文，必须由主线程来写"""
    got = False
    while True:
        try:
            kind, payload = _MAILBOX.get_nowait()
        except queue.Empty:
            break
        got = True
        if kind == "pending":
            st.session_state.pending = payload      # 被挂起：渲染 HIL 确认界面
            st.session_state.messages.append({"role": "assistant", "kind": "pending", "data": payload})
        elif kind == "result":
            st.session_state.pending = None
            st.session_state.messages.append({"role": "assistant", "kind": "result", "data": payload})
        elif kind == "error":
            st.session_state.pending = None
            st.session_state.messages.append({"role": "assistant", "kind": "error", "data": payload})
        elif kind == "done":
            st.session_state.running = False        # 线程跑完了，关掉"思考中"
    return got


# ─────────────────────────────────────────────
# 后端调用（子线程里跑，结果写队列）
# ─────────────────────────────────────────────
def _start_task(mode: str, question: str, papers: list[PaperSource]):
    """启动新任务：开会话号、记录用户消息、子线程跑后端"""
    st.session_state.thread_id = uuid.uuid4().hex  # 新任务换新的会话号
    st.session_state.pending = None
    st.session_state.running = True
    tid = st.session_state.thread_id

    def _run():
        try:
            _, out = run_task(mode=mode, question=question, papers=papers, thread_id=tid)
            if isinstance(out, dict) and "stage" in out:
                _MAILBOX.put(("pending", out))   # 被挂起 → 通知主线程渲染确认界面
            else:
                _MAILBOX.put(("result", out))    # 跑完 → 通知主线程渲染结果
        except Exception as e:                    # 后端报错（key没配/网络挂了）也通知主线程
            _MAILBOX.put(("error", f"{type(e).__name__}：{e}"))
        finally:
            _MAILBOX.put(("done", None))          # 不管成功失败，标记线程结束

    threading.Thread(target=_run, daemon=True).start()


def _resume(choice: str, feedback: str = ""):
    """恢复被挂起的任务：把用户决定喂回后端，流程从挂起点接着跑"""
    st.session_state.running = True
    st.session_state.pending = None   # 当前这个确认已经处理，清掉避免重复渲染
    tid = st.session_state.thread_id

    def _run():
        try:
            _, out = resume_task(thread_id=tid, choice=choice, feedback=feedback)
            if isinstance(out, dict) and "stage" in out:
                _MAILBOX.put(("pending", out))   # 又挂起了（比如大纲改完再确认）
            else:
                _MAILBOX.put(("result", out))    # 跑完了
        except Exception as e:
            _MAILBOX.put(("error", f"{type(e).__name__}：{e}"))
        finally:
            _MAILBOX.put(("done", None))

    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────
# 消息内容渲染：把后端返回的各种结果渲染成对话里的"AI 消息"
# ─────────────────────────────────────────────
def _render_suggestion(sug):
    """渲染行动建议：方向（调研模式）+ 核心论文 + 行动清单"""
    if not sug:
        return
    if sug.directions:
        st.markdown("**💡 值得深入研究的方向**")
        for i, d in enumerate(sug.directions, 1):
            st.markdown(f"{i}. {d}")
    if sug.papers:
        st.markdown("**📚 推荐精读的核心论文**")
        for p in sug.papers:
            line = f"- **{p.title}**"
            if p.reason:
                line += f" — {p.reason}"
            if p.link:
                line += f"（[链接]({p.link})）"
            st.markdown(line)
    if sug.actions:
        st.markdown("**✅ 下一步行动**")
        for a in sug.actions:
            st.markdown(f"- {a}")


def _render_result(res: FinalResult):
    """把最终结果渲染成 AI 消息：警告 + 报告(HTML) / 建议 / 精读报告(Markdown)"""
    for w in res.warning_flags:
        st.warning(f"⚠️ {w}")
    if res.report:  # 调研/对比报告：Writer 生成的是完整 HTML，用 components 还原排版
        st.markdown("### 📄 研究报告")
        components.html(res.report.html, height=700, scrolling=True)
        st.download_button("⬇️ 下载报告 HTML", res.report.html,
                           file_name="report.html", mime="text/html")
    if res.suggestion:
        _render_suggestion(res.suggestion)
    if res.deep_read:  # 单篇精读报告：主产物是 Markdown
        dr = res.deep_read
        st.markdown("### 📖 论文精读报告")
        info = dr.paper_info
        meta = " · ".join(x for x in [info.title, info.authors, info.year] if x)
        if meta:
            st.caption(meta)
        if dr.one_line_summary:
            st.info(f"**一句话看懂**：{dr.one_line_summary}")
        st.markdown(dr.full_report)
        st.download_button("⬇️ 下载精读报告", dr.full_report,
                           file_name="deep_read.md", mime="text/markdown")


def _render_pending(p: dict, interactive: bool, key_prefix: str):
    """把挂起的 HIL 环节渲染成 AI 消息 + 确认按钮。
    interactive=True 时才渲染按钮（历史消息里只读，避免按钮状态混乱）"""
    stage = p.get("stage")

    if stage == "relevance":  # 论文相关性低
        st.warning("⚠️ 检测到论文之间相关性较低")
        st.markdown(p.get("note", ""))
        if interactive:
            c1, c2 = st.columns(2)
            if c1.button("继续生成（仅供参考）", key=f"{key_prefix}_rel_go", use_container_width=True):
                _resume("continue")
            if c2.button("取消，分开分析", key=f"{key_prefix}_rel_no", use_container_width=True):
                _resume("cancel")

    elif stage == "outline":  # 大纲确认：必经环节
        st.markdown("### 📋 请确认研究大纲")
        o = p.get("outline", {})
        if o.get("subtopics"):
            st.markdown("**子主题**：" + "、".join(o["subtopics"]))
        if o.get("sections"):
            st.markdown("**报告章节**：" + " → ".join(o["sections"]))
        if o.get("analysis_dimensions"):
            st.markdown("**分析维度**：" + "、".join(o["analysis_dimensions"]))
        tasks = o.get("research_tasks") or []
        if tasks:
            with st.expander(f"研究任务单（{len(tasks)} 项，研究员的工作依据）"):
                for t in tasks:
                    st.markdown(f"**{t.get('id')}：{t.get('question')}**")
                    if t.get("purpose"):
                        st.markdown(f"- 目的：{t['purpose']}")
                    if t.get("required_evidence"):
                        st.markdown(f"- 需要证据：{'、'.join(t['required_evidence'])}")
                    if t.get("target_sections"):
                        st.markdown(f"- 服务章节：{'、'.join(t['target_sections'])}")
        if interactive:
            if st.button("✅ 接受并开始研究", key=f"{key_prefix}_ol_ok", type="primary"):
                _resume("accept")
            fb = st.text_input("或输入修改意见", key=f"{key_prefix}_ol_fb",
                               placeholder="如：增加一个关于 XX 的子主题")
            if st.button("↩️ 按意见调整大纲", key=f"{key_prefix}_ol_re"):
                _resume("revise", fb)

    elif stage == "report":  # 报告复核：校验没通过
        st.warning("⚠️ 报告未完全通过质量检查，请人工复核")
        for issue in p.get("issues", []):
            st.markdown(f"- [{issue.get('level')}] {issue.get('message')}"
                        + (f"（{issue['location']}）" if issue.get("location") else ""))
        if p.get("feedback"):
            st.info(f"验证员意见：{p['feedback']}")
        if interactive:
            if st.button("✅ 接受当前版本并继续", key=f"{key_prefix}_rp_ok", type="primary"):
                _resume("accept")
            fb2 = st.text_input("或输入修改意见", key=f"{key_prefix}_rp_fb",
                                placeholder="如：第2章引用不准确")
            if st.button("↩️ 按意见修改报告", key=f"{key_prefix}_rp_re"):
                _resume("revise", fb2)


def _render_message(msg: dict, interactive: bool, key_prefix: str):
    """按消息类型分发渲染：用户文本 / AI 的各类返回"""
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        kind = msg.get("kind")
        if kind == "text":
            st.markdown(msg.get("data", ""))
        elif kind == "error":
            st.error(f"任务出错：{msg.get('data')}\n\n请检查 .env 的 API Key 是否配置、网络是否正常。")
        elif kind == "result":
            _render_result(msg.get("data"))
        elif kind == "pending":
            _render_pending(msg.get("data"), interactive, key_prefix)


# ─────────────────────────────────────────────
# 输入解析：把用户聊天内容解析成任务参数
# ─────────────────────────────────────────────
_URL_PREFIX = ("http://", "https://", "arxiv.org", "www.")


def _looks_like_paper(text: str) -> bool:
    """判断一句话是不是在"给论文"：含链接或像 PDF 路径"""
    t = text.strip()
    return t.startswith(_URL_PREFIX) or t.lower().endswith(".pdf")


def _save_uploaded(files) -> list[PaperSource]:
    """把侧边栏上传的 PDF 存到缓存目录，转成 PaperSource（后端按本地 PDF 识别）"""
    from core.config import CACHE_DIR
    up_dir = os.path.join(CACHE_DIR, "uploads")
    os.makedirs(up_dir, exist_ok=True)
    out = []
    for f in files or []:
        path = os.path.join(up_dir, f.name)
        with open(path, "wb") as fp:
            fp.write(f.getbuffer())
        out.append(PaperSource(title=os.path.splitext(f.name)[0], link=path))
    return out


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    """页面主入口：侧边栏（模式+上传）+ 主区（对话流 + 聊天输入框）"""
    _init_state()
    # 每次 rerun 先把子线程跑完的结果收进来（修复"刷新后没结果"的关键一步）
    _drain_mailbox()

    # ── 侧边栏：模式切换 + PDF 上传
    with st.sidebar:
        st.title("🧠 AcademicMind")
        cur = st.session_state.mode
        opts = {"🔍 领域调研": MODE_SURVEY, "📄 论文分析": MODE_PAPER}
        picked = st.radio("模式", list(opts.keys()),
                          index=0 if cur == MODE_SURVEY else 1)
        new_mode = opts[picked]
        if new_mode != cur:  # 切模式：持久化偏好 + 清空当前任务状态，避免串数据
            st.session_state.mode = new_mode
            save_user_config(new_mode)
            st.session_state.papers = []
            st.session_state.pending = None
            st.rerun()

        if st.session_state.mode == MODE_PAPER:
            st.divider()
            uploaded = st.file_uploader("上传 PDF（可多选）", type=["pdf"],
                                        accept_multiple_files=True)
            if uploaded and st.button("📥 加入论文列表"):
                new_papers = _save_uploaded(uploaded)
                st.session_state.papers.extend(new_papers)
                for p in new_papers:
                    st.session_state.messages.append(
                        {"role": "user", "kind": "text", "data": f"📎 上传论文：{p.title}"})
                st.rerun()
        if st.session_state.papers:
            st.caption(f"已收集论文 {len(st.session_state.papers)} 篇")

    st.title("🧠 AcademicMind · 多智能体深度研究助手")
    st.caption("像聊天一样做研究：输入研究问题或论文链接，AI 自动拆解、调研、写报告，关键步骤会来问你。")

    # ── 渲染对话历史：只有"最新一条 pending"是可交互的，历史 pending 只读
    last_pending_idx = max((i for i, m in enumerate(st.session_state.messages)
                            if m.get("kind") == "pending"), default=-1)
    for i, msg in enumerate(st.session_state.messages):
        _render_message(msg, interactive=(i == last_pending_idx), key_prefix=f"m{i}")

    # ── 运行中：显示 AI 正在思考（折叠状态，不刷屏）
    if st.session_state.running:
        with st.chat_message("assistant"):
            st.status("⏳ AI 正在研究…（多 Agent 并行工作中，可能要几十秒到几分钟）",
                      expanded=True, state="running")
        # 每 4 秒自动刷新，把子线程跑完的结果拉进对话（不用手动刷新页面）
        st.markdown('<meta http-equiv="refresh" content="4">', unsafe_allow_html=True)

    # ── 聊天输入框：用户在这里提需求（禁用条件：正在跑 / 有待确认的挂起）
    disabled = st.session_state.running or st.session_state.pending is not None
    prompt = st.chat_input(
        "输入研究问题，或粘贴论文链接…" if not disabled else "请先完成上面的确认，或等当前任务跑完",
        disabled=disabled)
    if prompt:
        mode = st.session_state.mode
        if mode == MODE_PAPER:
            # 论文模式：输入是链接就收集进论文列表，不是链接就忽略提示
            if _looks_like_paper(prompt):
                p = PaperSource(title=prompt.strip().split("/")[-1] or prompt.strip(), link=prompt.strip())
                st.session_state.papers.append(p)
                st.session_state.messages.append(
                    {"role": "user", "kind": "text", "data": f"📎 添加论文：{p.title}"})
                # 收集到论文后，自动开始分析（1 篇走精读，多篇走对比）
                st.session_state.messages.append(
                    {"role": "assistant", "kind": "text",
                     "data": f"已收到 {len(st.session_state.papers)} 篇论文，开始分析…"})
                _start_task(mode, "", list(st.session_state.papers))
                st.session_state.papers = []  # 开跑后清空，避免下次重复带
            else:
                st.session_state.messages.append(
                    {"role": "user", "kind": "text", "data": prompt})
                st.session_state.messages.append(
                    {"role": "assistant", "kind": "text",
                     "data": "论文模式下，请粘贴论文链接（http…/arxiv.org/…）或在左侧上传 PDF。"})
        else:
            # 调研模式：输入就是研究问题，直接开跑
            st.session_state.messages.append({"role": "user", "kind": "text", "data": prompt})
            _start_task(mode, prompt, [])
        st.rerun()


if __name__ == "__main__":
    main()
