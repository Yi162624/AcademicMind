# 视觉工作记忆模块
# 功能：Researcher 搜到图后，把图片"登记入库"——下载图片算 pHash 去重、
#       调 Qwen-VL 生成描述、BGE 转向量、最后存进 SQLite image 表
#       供 Writer Agent 生成图文报告时检索使用
# 位置：multimodal/visual_working_memory.py，被 agent/researcher.py 调用

import io

import httpx
import imagehash
from PIL import Image

from core.config import QWEN_VL
from core.logger import get_logger
from core.schemas import ImageItem
from memory import sqlite_store
from memory.embedding import to_embedding_json
from multimodal.image_understanding import describe_image

log = get_logger("visual_working_memory")  # 本模块日志器，排错时看图片入库情况用


def add_image(url: str, source: str, subtopic: str = "") -> ImageItem | None:
    """把一张图片登记进视觉工作记忆（Researcher 调用）。
    流程：下载 → pHash 去重（重复就丢）→ Qwen-VL 理解生成描述 → BGE 转向量 → 存 SQLite。
    任何一步失败都返回 None，不抛错——研究员搜图失败不能拖垮整个子主题任务"""
    log.info("图片入库开始：%s（子主题：%s）", url, subtopic)

    # ① 下载图片：pHash 要在本地算，先把图片拉下来（只下载字节，不落盘）
    image_bytes = _download_image(url)
    if image_bytes is None:
        log.warning("图片下载失败，跳过入库：%s", url)
        return None  # 下载都失败，这张图没法用，直接放弃

    # ② pHash 去重：同图不同 URL 也能拦住（汉明距离≤5 判重复）
    phash = _compute_phash(image_bytes)
    if phash and sqlite_store.image_exists_by_phash(phash):
        log.info("图片已存在（pHash 去重命中），跳过入库：%s", url)
        return None  # 库里已有几乎一模一样的图，不重复存

    # ③ 图片理解：调 Qwen-VL 生成结构化描述（内部已做失败兜底，返回降级描述）
    description = describe_image(url)

    # ④ 向量化：描述 → BGE 向量 JSON 字符串（失败返回空串，图照存，只是没有向量可检索）
    embedding = to_embedding_json(description)

    # ⑤ 组装 ImageItem 入库：image_id 用 URL 的短哈希，稳定唯一
    item = ImageItem(
        url=url,                      # 图片原始 URL，用于检索时直接用
        description=description,      # 图片描述，用于生成向量
        source=source,                # 图片来源，用于检索时筛选
        subtopic=subtopic,            # 图片所属主题，用于检索时筛选
        phash=phash,                  # 图片感知哈希，用于去重和检索
        embedding=embedding,          # 图片向量，用于检索时计算相似度
        image_id=_make_image_id(url),     # 图片唯一 ID，基于 URL 的短哈希，稳定唯一
       )
    sqlite_store.save_image(item)  # 存进 SQLite image 表（同 id 会覆盖，这里 id 唯一不会撞）
    log.info("图片入库完成：%s", url)
    return item


def _download_image(url: str) -> bytes | None:
    """下载图片内容（返回字节流），失败返回 None。
    超时设置参考 Qwen-VL 的请求超时，防止坏链/慢速服务器卡住研究员"""
    if not url.startswith(("http://", "https://")):         # 只支持网络图片（本地文件由调用方处理）
        log.warning("不是 http(s) 链接，无法下载：%s", url)
        return None  # 只支持网络图片（本地文件由调用方处理）
    try:
        with httpx.Client(timeout=QWEN_VL.timeout) as client:     # 超时设置参考 Qwen-VL 的请求超时，防止坏链/慢速服务器卡住研究员
            resp = client.get(url, follow_redirects=True)  # 跟随重定向，有些图床会跳转
            resp.raise_for_status()      # 抛出 HTTPError 异常，处理失败情况
            if not resp.content:
                log.warning("图片内容为空：%s", url)
                return None
            return resp.content      # 返回图片字节流，不落盘
    except Exception as e:
        log.warning("图片下载异常（%s）：%s", type(e).__name__, url)
        return None  # 下载失败统一返回 None，由调用方决定放弃


def _compute_phash(image_bytes: bytes) -> str:
    """算图片的感知哈希（pHash），返回 64 位 hex 字符串。
    imagehash.phash 的 hex 格式和 sqlite_store._hamming_distance 的解析方式匹配（16进制转int异或）"""
    try:
        img = Image.open(io.BytesIO(image_bytes))  # 字节流转 Pillow 图像对象
        return str(imagehash.phash(img))           # phash 转 hex 字符串（16个十六进制字符=64位）
    except Exception as e:
        log.warning("pHash 计算失败（%s），跳过去重直接入库", type(e).__name__)
        return ""  # 算不出哈希就不去重，图照常入库（靠 URL 或人工兜底）


def _make_image_id(url: str) -> str:
    """给图片生成稳定唯一编号：URL 的 md5 前 16 位（同 URL 永远同 id，跨任务复用）"""
    import hashlib
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
