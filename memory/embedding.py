# 记忆模块：BGE 文本向量化（公共工具）
# 作用：把一段文本转成 BGE-small 向量，给"论文相关性检查"和"图文检索排序"共用
#       全项目只有这一处加载 BGE 模型，别处都调 to_embedding()，避免重复加载浪费内存
# 位置：memory/embedding.py，被 paper_relevance.py 和 multimodal/visual_retrieval.py 调用

import json
import threading

from core.logger import get_logger

log = get_logger("embedding")  # 本模块日志器，模型加载/向量化失败时排错用

# BGE 小模型：中英文都认，CPU 就能跑，约 100MB（首次使用自动下载）
_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None                   # 全局缓存模型，整个进程只加载一次
_model_lock = threading.Lock()  # 防止多线程同时去加载模型（双检查锁）


def _get_model():
    """拿 BGE 模型（懒加载：第一次调用才加载，之后复用同一个，别重复下模型）"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # 延迟 import：没装 sentence-transformers 时，只影响向量化功能，不拖累别的模块
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(_MODEL_NAME)
    return _model


def to_embedding(text: str) -> list[float]:
    """把一段文本转成 BGE 向量（归一化后的单位向量，点积即余弦相似度）。
    模型没装或向量化失败时返回空列表，调用方自己判空兜底"""
    if not text:
        return []  # 空文本不浪费模型调用，直接返回空向量
    try:
        model = _get_model()
        # normalize_embeddings=True：返回单位向量，后面算相似度直接点积就行，不用再除模长
        vec = model.encode([text], normalize_embeddings=True)
        return vec[0].tolist()
    except Exception as e:
        log.warning("文本向量化失败，返回空向量兜底：%s", e)
        return []


def to_embedding_json(text: str) -> str:
    """把文本转向量并序列化成 JSON 字符串（存进 SQLite image 表的 image_embedding 列用）。
    失败时返回空串，调用方判空跳过即可"""
    vec = to_embedding(text)
    return json.dumps(vec) if vec else ""


def parse_embedding(json_str: str) -> list[float]:
    """把 SQLite 里存的 JSON 字符串解析回向量 list[float]（图文检索排序时用）。
    空串或脏数据返回空列表，不让排序流程崩掉"""
    if not json_str:
        return []
    try:
        vec = json.loads(json_str)
        return vec if isinstance(vec, list) else []
    except (json.JSONDecodeError, TypeError):
        log.warning("向量 JSON 解析失败，跳过该图：数据可能已损坏")
        return []
