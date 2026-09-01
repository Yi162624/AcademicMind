# 后端 API 服务（FastAPI）
# 作用：把 main.py 的 run_task / resume_task 包装成 HTTP 接口，给 React 前端调用。
#       本身不含任何业务逻辑（拆题/搜集/写报告/校验全在 main.py + agent/ 里），只是"传话筒"。
# 运行：uvicorn frontend.server:app --reload --port 8000
# 接口：
#   POST /api/task/start   启动任务（调研问题 或 论文列表）→ 返回挂起(interrupt)或最终结果(result)
#   POST /api/task/resume  恢复被挂起的任务（用户确认/给意见）→ 同上
#   GET  /api/config       读模式偏好（SQLite）
#   POST /api/config       存模式偏好（SQLite）
#   POST /api/upload       上传 PDF，存到缓存目录，返回本地路径
# 关键点：run_task/resume_task 是同步阻塞的（搜论文/调 API 要几十秒），
#         必须用 run_in_executor 扔到线程池跑，否则卡住整个事件循环，前端别的请求全排队。

import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 把项目根目录加到模块搜索路径：server.py 在 frontend/ 下，不 import 后端就找不到 main/core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import CACHE_DIR  # noqa: E402
from core.schemas import FinalResult, MODE_PAPER, MODE_SURVEY, PaperSource  # noqa: E402
from core.llm_client import chat  # noqa: E402
from core.json_utils import parse_json  # noqa: E402
from main import resume_task, run_task  # noqa: E402
from memory.sqlite_store import get_user_config, save_user_config  # noqa: E402

app = FastAPI(title="AcademicMind API")

# 允许 React 开发服务器（5173）跨域调本服务（8000）；生产打包后同源就无所谓了
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 线程池：同步阻塞的后端调用都扔这里跑，别堵事件循环
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────
# 请求 / 响应模型
# ─────────────────────────────────────────────
class StartReq(BaseModel):
    """启动任务请求：mode=调研/论文；question=调研问题；papers=论文链接列表"""
    mode: str = MODE_SURVEY
    question: str = ""
    papers: list[dict] = []          # [{title, link}]
    thread_id: str | None = None


class ResumeReq(BaseModel):
    """恢复任务请求：thread_id 定位挂起现场；choice=accept/revise/continue/cancel；feedback=修改意见"""
    thread_id: str
    choice: str
    feedback: str = ""


class ConfigReq(BaseModel):
    """存模式偏好请求"""
    mode: str


class ClassifyReq(BaseModel):
    """意图分类请求：用户输入的一句话"""
    text: str


# ─────────────────────────────────────────────
# 把后端返回统一成前端好处理的结构
# ─────────────────────────────────────────────
def _dataclass_to_dict(obj: Any) -> Any:
    """递归把 dataclass 转成 dict：FinalResult/Report/Suggestion 等都是 dataclass，JSON 序列化前要先转"""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    return obj


def _pack(out: Any, thread_id: str) -> dict:
    """把 run_task/resume_task 的返回打包成统一格式：
    - dict+stage → {"type":"interrupt", ...}  被挂起，前端弹确认
    - FinalResult → {"type":"result", ...}    跑完，前端展示报告
    thread_id 回传给前端，后续 resume 靠它定位现场"""
    if isinstance(out, dict) and "stage" in out:
        return {"type": "interrupt", "thread_id": thread_id, "stage": out.get("stage"), "data": out}
    return {"type": "result", "thread_id": thread_id, "data": _dataclass_to_dict(out)}


# ─────────────────────────────────────────────
# 接口
# ─────────────────────────────────────────────
@app.post("/api/task/start")
async def start_task(req: StartReq):
    """启动任务：在线程池里跑同步阻塞的 run_task，避免卡事件循环"""
    import asyncio
    tid = req.thread_id or uuid.uuid4().hex
    papers = [PaperSource(title=p.get("title", ""), link=p.get("link", "")) for p in req.papers]

    def _call():
        return run_task(mode=req.mode, question=req.question, papers=papers, thread_id=tid)

    loop = asyncio.get_event_loop()
    _, out = await loop.run_in_executor(_EXECUTOR, _call)
    return _pack(out, tid)


@app.post("/api/task/resume")
async def resume(req: ResumeReq):
    """恢复任务：把用户决定喂回 resume_task，流程从挂起点接着跑"""
    import asyncio

    def _call():
        return resume_task(thread_id=req.thread_id, choice=req.choice, feedback=req.feedback)

    loop = asyncio.get_event_loop()
    _, out = await loop.run_in_executor(_EXECUTOR, _call)
    return _pack(out, req.thread_id)


@app.get("/api/config")
async def read_config():
    """读模式偏好：前端进页面时恢复上次用的模式"""
    cfg = get_user_config()
    return {"mode": cfg.preferred_mode}


@app.post("/api/config")
async def write_config(req: ConfigReq):
    """存模式偏好：前端切模式时调用"""
    save_user_config(req.mode)
    return {"ok": True, "mode": req.mode}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF：存到缓存目录 uploads/，返回本地路径（后端按本地 PDF 识别）"""
    up_dir = os.path.join(CACHE_DIR, "uploads")
    os.makedirs(up_dir, exist_ok=True)
    path = os.path.join(up_dir, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    title = os.path.splitext(file.filename)[0]
    return {"title": title, "link": path}


@app.get("/api/health")
async def health():
    """健康检查：前端判断后端起没起"""
    return {"ok": True}


# ─────────────────────────────────────────────
# 意图分类：决定用户输入走哪条路
# ─────────────────────────────────────────────
# 三类：
#   simple  —— 简单问题/闲聊（你好、谢谢、什么是X这种能一句话答的）→ AI 直接答
#   research —— 专业学术调研/论文分析类问题 → 走 5 Agent 流程
#   other   —— 复杂但不是学术研究的问题（怎么写代码/怎么修电脑/情感问题…）→ 礼貌拒答并说明原因
_CLASSIFY_SYSTEM = """你是 AcademicMind 学术研究助手的意图路由器。
你的任务是判断用户输入属于哪一类，只输出 JSON，不要任何解释。

分类规则：
1. simple：问候/寒暄/感谢，或能用一两句话直接回答的常识性小问题（如"你好""在吗""什么是深度学习"）。
2. research：学术调研/文献综述/论文分析类问题，或用户给出了论文链接、论文相关请求（如"大模型在医疗影像的应用""帮我分析这篇论文")。用户输入看起来是研究问题（领域性问题、研究方向、学术话题探讨）时归此类。
3. other：复杂但非学术研究的问题——如编程求助、生活建议、情感咨询、写文案、翻译长文、技术排查等。

输出格式：{"type": "simple" | "research" | "other", "reason": "一句话说明判断理由"}"""

_CLASSIFY_USER = """用户输入：{text}
请分类并只输出 JSON。"""


def _classify(text: str) -> dict:
    """判断用户输入类别（simple/research/other）。
    分类失败时保守降级为 research（走正经流程），绝不把专业问题误判成闲聊"""
    if not text.strip():
        return {"type": "simple", "reason": "空输入"}
    try:
        resp = chat(system=_CLASSIFY_SYSTEM, user=_CLASSIFY_USER.format(text=text[:500]), max_tokens=256)
        data = parse_json(resp.text)
        t = (data or {}).get("type", "")
        if t in ("simple", "research", "other"):
            return {"type": t, "reason": (data or {}).get("reason", "")}
    except Exception:
        pass
    return {"type": "research", "reason": "分类失败，按专业问题处理"}


@app.post("/api/classify")
async def classify(req: ClassifyReq):
    """意图分类接口：前端发消息前先问一次，决定走"直接答/调研/拒答"这条路"""
    import asyncio

    def _call():
        return _classify(req.text)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, _call)


@app.post("/api/chat/simple")
async def chat_simple(req: ClassifyReq):
    """简单问题直答接口：只调一次 DeepSeek，让 AI 直接回答，不走 5 Agent"""
    import asyncio

    def _call():
        resp = chat(
            system="你是 AcademicMind 的助手。用户问了一个简单问题/闲聊，请用中文简洁友好地回答（2-4 句话）。",
            user=req.text,
            max_tokens=512,
        )
        return {"answer": resp.text.strip()}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, _call)
