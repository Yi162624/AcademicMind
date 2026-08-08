# 记忆模块：Milvus Lite 向量存储
# 作用：管三个向量集合（文档 2.1 定的名字）：
#       memory_survey  调研历史（主题+报告摘要向量）
#       memory_paper   论文分析历史（主题+分析结论向量）
#       image_memory   图片记忆（图片描述向量，去重+复用）
# 说明：本文件只管"存向量、查向量"，向量本身由调用方生成（BGE 模型在 paper_relevance 里）
# 位置：memory/milvus_lite_store.py，被 Researcher（图片去重）、前端（历史检索）调用

import os
import threading

from pymilvus import MilvusClient

from core.config import DATA_DIR
from core.logger import get_logger
from core.schemas import ImageItem, PaperRecord, SurveyRecord

log = get_logger("milvus_lite_store")  # 本模块日志器

# Milvus Lite 数据文件：data/milvus.db（嵌入式向量库，不用单独启动服务）
_DB_PATH = os.path.join(DATA_DIR, "milvus.db")

# BGE-small 系列向量统一是 512 维，三个集合都用这个维度
_VECTOR_DIM = 512

# 三个集合的名字（和文档 2.1 保持一致）
COL_SURVEY = "memory_survey"
COL_PAPER = "memory_paper"
COL_IMAGE = "image_memory"

_client = None                 # 全局客户端缓存，整个进程只连一次
_client_lock = threading.Lock()  # 防止多线程同时建客户端


def _get_client() -> MilvusClient:
    """拿 Milvus Lite 客户端（懒加载 + 顺手把三个集合建好）"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                os.makedirs(DATA_DIR, exist_ok=True)  # data/ 目录不存在就先建
                c = MilvusClient(_DB_PATH)
                _ensure_collections(c)
                _client = c
    return _client


def _ensure_collections(client: MilvusClient) -> None:
    """保证三个集合存在：没有就建，重复调用不报错（集合定义要改时得迁移数据）"""
    specs = [
        (COL_SURVEY, "record_id"),
        (COL_PAPER, "record_id"),
        (COL_IMAGE, "image_id"),
    ]
    for name, pk in specs:
        if not client.has_collection(name):
            client.create_collection(
                collection_name=name,
                dimension=_VECTOR_DIM,
                primary_field_name=pk,      # 主键字段：字符串 ID
                id_type="string",
                max_length=64,              # 字符串主键的长度上限
                vector_field_name="embedding",
                metric_type="COSINE",       # 相似度算法用余弦
                enable_dynamic_field=True,  # 集合里其他字段（topic 等）都能动态存，不用全列出来
            )


# ═══════════ 调研历史（memory_survey）═══════════

def save_survey_record(rec: SurveyRecord) -> None:
    """存一条调研历史（向量+摘要）；同一 record_id 重复存会覆盖（更新用）"""
    client = _get_client()
    client.upsert(COL_SURVEY, data=[{
        "record_id": rec.record_id,
        "topic": rec.topic,
        "report_summary": rec.report_summary,
        "created_at": rec.created_at,
        "embedding": rec.embedding,
    }])


def search_survey(query_embedding: list[float], top_k: int = 5) -> list[SurveyRecord]:
    """按向量搜历史调研记录，返回最像的 top_k 条（库里没数据时返回空列表）"""
    client = _get_client()
    hits = client.search(
        collection_name=COL_SURVEY,
        data=[query_embedding],
        limit=top_k,
        output_fields=["record_id", "topic", "report_summary", "created_at"],
        search_params={"metric_type": "COSINE"},
    )[0]
    return [
        SurveyRecord(
            record_id=h.get("entity", {}).get("record_id", h.get("id", "")),
            topic=h.get("entity", {}).get("topic", ""),
            report_summary=h.get("entity", {}).get("report_summary", ""),
            created_at=h.get("entity", {}).get("created_at", ""),
        )
        for h in hits
    ]


# ═══════════ 论文分析历史（memory_paper）═══════════

def save_paper_record(rec: PaperRecord) -> None:
    """存一条论文分析历史（向量+分析结论）；同一 record_id 重复存会覆盖"""
    client = _get_client()
    client.upsert(COL_PAPER, data=[{
        "record_id": rec.record_id,
        "topic": rec.topic,
        "analysis_conclusion": rec.analysis_conclusion,
        "paper_titles": rec.paper_titles,
        "created_at": rec.created_at,
        "embedding": rec.embedding,
    }])


def search_paper(query_embedding: list[float], top_k: int = 5) -> list[PaperRecord]:
    """按向量搜历史论文分析记录，返回最像的 top_k 条（库里没数据时返回空列表）"""
    client = _get_client()
    hits = client.search(
        collection_name=COL_PAPER,
        data=[query_embedding],
        limit=top_k,
        output_fields=["record_id", "topic", "analysis_conclusion", "paper_titles", "created_at"],
        search_params={"metric_type": "COSINE"},
    )[0]
    return [
        PaperRecord(
            record_id=h.get("entity", {}).get("record_id", h.get("id", "")),
            topic=h.get("entity", {}).get("topic", ""),
            analysis_conclusion=h.get("entity", {}).get("analysis_conclusion", ""),
            paper_titles=h.get("entity", {}).get("paper_titles", []),
            created_at=h.get("entity", {}).get("created_at", ""),
        )
        for h in hits
    ]


# ═══════════ 图片记忆（image_memory）═══════════

def save_image(item: ImageItem) -> None:
    """存一张图片记忆（图片+描述+向量）；同一 image_id 重复存会覆盖（去重更新常用）"""
    client = _get_client()
    client.upsert(COL_IMAGE, data=[{
        "image_id": item.image_id or item.url,  # 没给编号就用图片地址当主键
        "url": item.url,
        "description": item.description,
        "source": item.source,
        "subtopic": item.subtopic,
        "embedding": item.embedding,
    }])


def search_image(query_embedding: list[float], top_k: int = 5,
                 threshold: float = 0.85) -> list[tuple[ImageItem, float]]:
    """搜相似图片：只返回相似度超过 threshold 的（图片+相似度），做图片去重/复用用"""
    client = _get_client()
    hits = client.search(
        collection_name=COL_IMAGE,
        data=[query_embedding],
        limit=top_k,
        output_fields=["image_id", "url", "description", "source", "subtopic"],
        search_params={"metric_type": "COSINE"},
    )[0]
    out = []
    for h in hits:
        sim = float(h.get("distance", 0.0))  # COSINE 下 distance 就是相似度
        if sim < threshold:
            continue  # 不够像，跳过
        e = h.get("entity", {})
        out.append((
            ImageItem(
                url=e.get("url", ""),
                description=e.get("description", ""),
                source=e.get("source", ""),
                subtopic=e.get("subtopic", ""),
                image_id=e.get("image_id", h.get("id", "")),
            ),
            sim,
        ))
    return out


def image_exists(image_id: str) -> bool:
    """按 image_id 查图片在不在库里（去重前先查这个，比向量检索更准更省）"""
    client = _get_client()
    res = client.query(collection_name=COL_IMAGE, filter=f'image_id == "{image_id}"')
    return len(res) > 0


def delete_image(image_id: str) -> None:
    """按 image_id 删一张图片记忆（换图/清理不用了的图片）"""
    client = _get_client()
    client.delete(collection_name=COL_IMAGE, filter=f'image_id == "{image_id}"')
