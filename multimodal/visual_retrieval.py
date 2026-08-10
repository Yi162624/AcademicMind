# VisualRAG 图文检索模块
# 作用：Writer 给某章配图时，从 SQLite 图片库里挑出"真正适合这章"的图
#       三步走：SQL 召回 → BGE 描述向量排序 → Qwen-VL 图文匹配精排
# 位置：multimodal/visual_retrieval.py，被 agent/writer.py 调用

from core.logger import get_logger
from core.schemas import ImageItem
from memory import sqlite_store
from memory.embedding import parse_embedding, to_embedding
from multimodal.image_understanding import match_image

log = get_logger("visual_retrieval")  # 本模块日志器，排错时看检索调用情况用

# 相似度阈值：BGE 余弦相似度低于它直接淘汰，不进 VLM 精排（省 API 调用）。
# 取值参考：项目内论文相关性检查用 0.6（见 paper_relevance.py），社区 BGE 常用 0.5~0.6；
# 这里因为后面还有 VLM 精排兜底，取宽松值 0.5，只拦"明显不相关"的图，不误杀边缘图
_SIM_THRESHOLD = 0.5


def retrieve(requirement: str, subtopic: str = "", top_k: int = 5) -> list[ImageItem]:
    """给某章节配图：三步取图，返回精排通过的图片素材。
    找不到合适的图就返回空列表（调用方拿到空列表就纯文字写，不配图）。

    参数：
        requirement: 章节图片需求（如"展示 Transformer 自注意力机制的架构图"）
        subtopic:    该章属于哪个子主题（按子主题粗筛，不传就全库搜）
        top_k:       BGE 排序后取前几张进精排（默认 5，太大费 VLM 调用）
    """
    log.info("开始配图检索：子主题=%s，需求=%s", subtopic, requirement[:50])

    # ① 召回：按子主题从 SQLite 粗筛（这步只走 SQL，不碰模型）
    candidates = sqlite_store.list_images(subtopic=subtopic)
    if not candidates:
        log.info("召回为空，该章不配图：子主题=%s", subtopic)
        return []  # 召回就空了，直接不配图

    # ② BGE 排序：按"图片描述"和"章节需求"的语义相似度取 Top-K
    ranked = _bge_rank(requirement, candidates, top_k)
    if not ranked:
        log.info("BGE 排序后无候选，该章不配图：子主题=%s", subtopic)
        return []  # 向量化全失败等极端情况，不配图

    # ③ VLM 精排：对 Top-K 逐一让 Qwen-VL 真看一眼图，筛掉"长得像但语义不符"的
    passed: list[ImageItem] = []
    for img in ranked:
        result = match_image(requirement, img.url)  # match_image 内部已做异常兜底（放行+标记）
        if result.matched:
            passed.append(img)
            log.debug("精排通过：%s（理由：%s）", img.url, result.reason)
        else:
            log.debug("精排拦掉：%s（理由：%s）", img.url, result.reason)

    if not passed:
        log.info("精排全不通过，该章不配图：子主题=%s", subtopic)
    else:
        log.info("配图检索完成：子主题=%s，通过 %d 张", subtopic, len(passed))
    return passed


def _bge_rank(requirement: str, candidates: list[ImageItem], top_k: int) -> list[ImageItem]:
    """BGE 向量排序：把章节需求和每张图的描述都转向量，算余弦相似度，取 Top-K。
    向量化失败时降级——直接返回原候选前 top_k 张，不让检索流程断掉"""
    req_vec = to_embedding(requirement)  # 章节需求转向量
    if not req_vec:
        log.warning("章节需求向量化失败，跳过 BGE 排序，直接拿召回前 %d 张进精排", top_k)
        return candidates[:top_k]

    # 给每张候选图算相似度：描述向量从 SQLite 的 JSON 字符串解析回来
    # 两道过滤语义不同：没向量=技术故障(降级放行)；相似度低=业务判定(淘汰)
    scored: list[tuple[float, ImageItem]] = []
    has_vector = False                     # 有没有图带有效向量，区分上面两种情况用
    for img in candidates:
        img_vec = parse_embedding(img.embedding)  # 解析 JSON 向量字符串
        if not img_vec:
            continue  # 这张图没向量（入库时 BGE 挂了），跳过不参与排序
        has_vector = True                  # 至少有一张图有向量，后面判断用
        sim = _cosine(req_vec, img_vec)            # 算余弦相似度
        if sim < _SIM_THRESHOLD:
            continue  # 相似度太低直接淘汰，别浪费 VLM 精排的 API 调用
        scored.append((sim, img))

    if not scored:
        if has_vector:
            # 有向量但全部低于阈值 → 按设计淘汰：该章不配图，纯文字输出
            log.info("所有候选图相似度均低于阈值 %.2f，该章不配图", _SIM_THRESHOLD)
            return []
        # 全都没向量（BGE 故障）→ 降级放行前 top_k，让 VLM 精排兜底
        log.warning("所有候选图都没有有效向量，跳过 BGE 排序")
        return candidates[:top_k]

    # 按相似度降序排，取 Top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in scored[:top_k]]


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """算两个向量的余弦相似度。
    BGE 返回的是归一化向量，理论上点积就是余弦相似度；
    这里仍做一次模长除法兜底，防止调用方传进来未归一化的向量"""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0  # 维度不一致或空向量，当完全不相关
    # 点积 + 模长，算标准余弦相似度
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 零向量当完全不相关，避免除零
    return dot / (norm_a * norm_b)
