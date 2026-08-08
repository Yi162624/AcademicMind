# 记忆模块：论文相关性检查
# 作用：main.py 在论文分析模式（≥2篇）跑规划师之前，先判断这批论文是不是"同一主题"。
#       差异太大就停下来问用户：继续硬生成（对比仅供参考）还是取消分开分析
# 原理：把每篇论文的标题+摘要用 BGE 小模型转成向量，两两算余弦相似度，
#       平均相似度低于阈值就判"低相关"，并把差异大的论文对写进说明文字
# 位置：memory/paper_relevance.py，被 main.py 的 check_relevance_node 调用

import threading

from core.logger import get_logger
from core.schemas import PaperSource

log = get_logger("paper_relevance")  # 本模块日志器

# BGE 小模型：中英文都认，CPU 就能跑，约 100MB（首次使用自动下载）
_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 相似度阈值：论文两两平均相似度低于它就判"低相关"
_RELEVANT_THRESHOLD = 0.6

_model = None                 # 全局缓存模型，整个进程只加载一次
_model_lock = threading.Lock()  # 防止多线程同时去加载模型（双检查锁）


def _get_model():
    """拿 BGE 模型（懒加载：第一次调用才加载，之后复用同一个，别重复下模型）"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # 延迟 import：没装 sentence-transformers 时，只影响相关性检查，不拖累别的模块
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _paper_text(p: PaperSource) -> str:
    """拼一篇论文的向量文本：标题为主，有摘要/链接再补上（信息越多向量越准）"""
    parts = [p.title]
    if p.abstract:
        parts.append(p.abstract)
    if p.link:
        parts.append(p.link)
    return " ".join(parts)


def check_paper_relevance(papers: list[PaperSource]) -> tuple[bool, str]:
    """判断论文之间是否相关（main.py 调用）。
    返回 (是否相关, 说明文字)：相关 → 说明为空串；不相关 → 说明里列出差异大的论文对"""
    n = len(papers)
    if n < 2:
        return True, ""  # 单篇论文不用查，直接放行

    # 模型加载/向量化失败时不硬卡流程：按"相关"放行，只记个 warning
    try:
        model = _get_model()
        vecs = model.encode([_paper_text(p) for p in papers], normalize_embeddings=True)
    except Exception as e:
        log.warning("论文相关性检查失败，按相关放行（模型没装或没网络？）：%s", e)
        return True, ""

    # 两两算相似度：normalize_embeddings=True 后每行是单位向量，点积就是余弦相似度
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float((vecs[i] * vecs[j]).sum())
            pairs.append((sim, i, j))

    avg_sim = sum(p[0] for p in pairs) / len(pairs)
    if avg_sim >= _RELEVANT_THRESHOLD:
        return True, ""  # 平均相似度够高 → 主题接近，正常走流程

    # 低相关：把低于阈值的论文对挑出来，按相似度从低到高写进说明，提示用户做决定
    low_pairs = sorted([p for p in pairs if p[0] < _RELEVANT_THRESHOLD], key=lambda x: x[0])
    note_parts = []
    for sim, i, j in low_pairs[:5]:
        note_parts.append(f"『{papers[i].title}』与『{papers[j].title}』相似度 {sim:.2f}")
    note = "；".join(note_parts) + "。建议分开分析，或继续生成对比报告（仅供参考）。"
    return False, note
