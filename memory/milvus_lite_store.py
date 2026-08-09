# 记忆模块：Milvus Lite 向量存储
# 作用：管两个向量集合（文档 2.1 定的名字）：
#       memory_survey  调研历史（主题+报告摘要向量）
#       memory_paper   论文分析历史（主题+分析结论向量）
# 说明：本文件只管"存向量、查向量"，向量本身由调用方生成（BGE 模型在 paper_relevance 里）
#       图片记忆已迁到 SQLite（pHash 去重不依赖向量库），见 sqlite_store.py
# 位置：memory/milvus_lite_store.py，被前端（历史检索）调用

import os
import threading

from pymilvus import MilvusClient

from core.config import DATA_DIR
from core.logger import get_logger
from core.schemas import PaperRecord, SurveyRecord

log = get_logger("milvus_lite_store")  # 本模块日志器

# Milvus Lite 数据文件：data/milvus.db（嵌入式向量库，不用单独启动服务）
_DB_PATH = os.path.join(DATA_DIR, "milvus.db")

# BGE-small 系列向量统一是 512 维，两个集合都用这个维度
_VECTOR_DIM = 512

# 两个集合的名字（和文档 2.1 保持一致；图片记忆已迁到 SQLite）
COL_SURVEY = "memory_survey"
COL_PAPER = "memory_paper"

_client = None                 # 全局客户端缓存，整个进程只连一次
_client_lock = threading.Lock()  # 防止多线程同时建客户端


def _get_client() -> MilvusClient:
    """拿 Milvus Lite 客户端（懒加载 + 顺手把两个集合建好）"""
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
    """保证两个集合存在：没有就建，重复调用不报错（集合定义要改时得迁移数据）"""
    # 两个集合的清单：(集合名, 主键字段名) —— 后面循环按这个清单逐个建
    specs = [
        (COL_SURVEY, "record_id"),   # 调研历史集合，主键用记录 ID
        (COL_PAPER, "record_id"),    # 论文分析历史集合，主键用记录 ID
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
    client.upsert(COL_SURVEY, data=[{  # upsert = 存在则更新，不存在则插入
        "record_id": rec.record_id,       # 主键：这次调研的记录 ID
        "topic": rec.topic,                # 调研主题（生成向量的文本来源）
        "report_summary": rec.report_summary,  # 报告摘要（给人看的 payload）
        "created_at": rec.created_at,      # 创建时间
        "embedding": rec.embedding,        # 主题向量（检索用的）
    }])


def search_survey(query_embedding: list[float], top_k: int = 5) -> list[SurveyRecord]:
    """按向量搜历史调研记录，返回最像的 top_k 条（库里没数据时返回空列表）"""
    client = _get_client()
    result = client.search(               # 在 memory_survey 里做向量检索
        collection_name=COL_SURVEY,        # 集合名：memory_survey
        data=[query_embedding],            # 查询向量（用列表包了一层，Milvus 支持批量查）
        limit=top_k,                       # 最多返回几条
        output_fields=["record_id", "topic", "report_summary", "created_at"],  # 命中后要带回的字段
        search_params={"metric_type": "COSINE"},  # 用余弦相似度排序
    )
    if not result:                        # 集合为空时部分版本返回空列表，直接给空结果，别崩
        return []
    hits = result[0]                      # 只传了1个查询向量，取第0个查询的结果（剥掉外层）
    return [
        SurveyRecord(
            # Milvus 返回结构：{"id":主键, "distance":相似度, "entity":{附加字段}}
            # entity 优先取，没有就退到主键 id，再没有就空串（带兜底，永不崩）
            record_id=h.get("entity", {}).get("record_id", h.get("id", "")),
            topic=h.get("entity", {}).get("topic", ""),
            report_summary=h.get("entity", {}).get("report_summary", ""),
            created_at=h.get("entity", {}).get("created_at", ""),
        )
        for h in hits  # 逐条命中结果翻译成 SurveyRecord
    ]


# ═══════════ 论文分析历史（memory_paper）═══════════

def save_paper_record(rec: PaperRecord) -> None:
    """存一条论文分析历史（向量+分析结论）；同一 record_id 重复存会覆盖"""
    client = _get_client()
    client.upsert(COL_PAPER, data=[{           # upsert = 存在则更新，不存在则插入
        "record_id": rec.record_id,            # 主键：这次分析的记录 ID
        "topic": rec.topic,                    # 论文主题（生成向量的文本来源）
        "analysis_conclusion": rec.analysis_conclusion,  # 分析结论（给人看的 payload）
        "paper_titles": rec.paper_titles,      # 涉及哪些论文
        "created_at": rec.created_at,          # 创建时间
        "embedding": rec.embedding,            # 主题向量（检索用的）
    }])


def search_paper(query_embedding: list[float], top_k: int = 5) -> list[PaperRecord]:
    """按向量搜历史论文分析记录，返回最像的 top_k 条（库里没数据时返回空列表）"""
    client = _get_client()
    result = client.search(               # 在 memory_paper 里做向量检索
        collection_name=COL_PAPER,         # 集合名：memory_paper
        data=[query_embedding],            # 查询向量（用列表包了一层，Milvus 支持批量查）
        limit=top_k,                       # 最多返回几条
        output_fields=["record_id", "topic", "analysis_conclusion", "paper_titles", "created_at"],  # 命中后要带回的字段
        search_params={"metric_type": "COSINE"},  # 用余弦相似度排序
    )
    if not result:                        # 集合为空时部分版本返回空列表，直接给空结果，别崩
        return []
    hits = result[0]                      # 只传了1个查询向量，取第0个查询的结果（剥掉外层）
    return [
        PaperRecord(
            # Milvus 返回结构：{"id":主键, "distance":相似度, "entity":{附加字段}}
            # entity 优先取，没有就退到主键 id，再没有就空串/空列表（带兜底，永不崩）
            record_id=h.get("entity", {}).get("record_id", h.get("id", "")),
            topic=h.get("entity", {}).get("topic", ""),
            analysis_conclusion=h.get("entity", {}).get("analysis_conclusion", ""),
            paper_titles=h.get("entity", {}).get("paper_titles", []),
            created_at=h.get("entity", {}).get("created_at", ""),
        )
        for h in hits  # 逐条命中结果翻译成 PaperRecord
    ]
