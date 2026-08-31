# 研究员 Agent（Researcher）
# 作用：按规划师的研究任务单（ResearchTask）搜集并提炼结构化证据。
#       不是简单"搜论文返回摘要"，而是 Search（搜候选）→ Filter+Extract（LLM 提炼能回答任务的证据）。
#       产出的 Evidence 带 findings（综合结论）+ claims（可追溯论断）+ sources（来源论文），
#       让下游 Writer 按证据写、Verifier 沿证据链核对，而不是各自重新研究一遍。
# 位置：agent/researcher.py，被 main.py 的 researcher_node 调用
# 产出：list[Evidence]（每个研究任务一个 Evidence，证据编号 E1/E2 和来源编号 S1/S2 由系统统一分配）
#
# 图片来源：arXiv + Semantic Scholar（免费、无需 key）。
# 图片素材搜集暂未接入（图片搜索 API 还没定），Evidence.images 字段已预留，后续补。

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
import time

import httpx

from core.json_utils import parse_json
from core.llm_client import chat
from core.logger import get_logger
from core.schemas import Claim, Evidence, FlowState, MODE_PAPER, ResearchTask, Source

log = get_logger("researcher")

# 每个任务从单个来源搜多少篇，合并去重后大概 8-10 篇，够提炼证据用
_ARXIV_NUM = 5
_S2_NUM = 5

# 研究员提炼证据的角色设定，逼 LLM 只输出结构化 JSON
_SYSTEM = "你是学术研究员。从给定论文摘要中提炼能回答研究问题的证据，只输出 JSON，不要加任何解释或 Markdown 代码块。"


def run_researcher(flow: FlowState) -> list[Evidence]:
    """研究员入口：按研究任务单并行搜集并提炼证据，合并后统一分配 E/S 编号"""
    tasks = _get_tasks(flow)                       # 从大纲拿研究任务单
    evidences = _run_tasks_parallel(tasks, flow)   # 每个任务并行搜 + 提炼
    return _assign_ids(evidences)                  # 统一分配证据编号 E1/E2 和来源编号 S1/S2


def _get_tasks(flow: FlowState) -> list[ResearchTask]:
    """从大纲里拿研究任务单；没有任务单就按模式兜底造，保证研究员有活干"""
    tasks = (flow.outline.research_tasks if flow.outline else [])
    if tasks:
        return tasks
    # 兜底：大纲没给任务单时，调研用子主题、论文用分组临时造任务
    if flow.mode == MODE_PAPER:
        groups = (flow.outline.paper_assignments if flow.outline else []) or [[p.title] for p in flow.papers]
        return [ResearchTask(id=f"T{i + 1}", question=f"对比分析：{'、'.join(g)}", purpose="横向对比这些论文") for i, g in enumerate(groups)]
    subtopics = (flow.outline.subtopics if flow.outline else []) or [flow.question]
    return [ResearchTask(id=f"T{i + 1}", question=st, purpose="理解该子主题") for i, st in enumerate(subtopics)]


def _run_tasks_parallel(tasks: list[ResearchTask], flow: FlowState) -> list[Evidence]:
    """并行跑多个研究员（上限 3 并发），单个任务失败给空证据占位，不拖垮整体"""
    results: dict[int, Evidence] = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 3)) as ex:
        futures = {ex.submit(_research_task, t, flow): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                # 某个任务彻底失败，给个空证据占位，别让整个研究员崩
                log.warning("任务「%s」搜集失败（%s），返回空证据", tasks[idx].id, type(e).__name__)
                results[idx] = Evidence(task_id=tasks[idx].id)
    return [results[i] for i in range(len(tasks))]


def _research_task(task: ResearchTask, flow: FlowState) -> Evidence:
    """围绕一个研究任务搜集并提炼证据。调研模式搜 arXiv+S2，论文模式从用户给的论文提炼。
    # 图片素材预留：等确定图片搜索 API 后，在这里搜图并调 visual_working_memory.add_image() 入库"""
    if flow.mode == MODE_PAPER:
        papers = _papers_for_task(flow)   # 论文模式：候选是用户给的论文（摘要缺失的补搜）
    else:
        papers = _search_task(task)       # 调研模式：先把问题精炼成短搜索词再搜
    if not papers:
        return Evidence(task_id=task.id)   # 没搜到任何候选，给空证据
    return _extract_evidence(task, papers)


def _search_task(task: ResearchTask) -> list[dict]:
    """调研模式搜索：先把研究问题精炼成 1-2 条短搜索词（LLM），再逐条去 arXiv+S2 搜。
    用短关键词而非整句中文问句，arXiv 不会报错、S2 命中率也更高；合并去重后返回候选"""
    queries = _refine_queries(task)                     # 精炼搜索词（失败降级为原始问题）
    papers: list[dict] = []
    for q in queries:
        papers += _search_arxiv(q, _ARXIV_NUM)
        papers += _search_semantic_scholar(q, _S2_NUM)
    return _dedup_papers(papers)


def _refine_queries(task: ResearchTask) -> list[str]:
    """把研究问题精炼成 1-2 条短搜索词（LLM 一步搞定，无需额外思考）。
    学术搜索接口不认整句中文长问题：arXiv 超长 query 直接 HTTPError、S2 命中率差。
    改写成短英文关键词后两个源都搜得动。
    注意：deepseek-v4-pro 是思考型模型，输出不稳定（偶发不按 JSON 输出），
    失败时降级用本地关键词提取，绝不用原始整句——那会让搜索全部落空"""
    prompt = f"""研究问题：{task.question}
研究目的：{task.purpose}

请把上面的研究问题改写成 1-2 条学术搜索引擎能用的关键词查询：
- 每条必须短（不超过 40 个字符）、关键词型，不要疑问句
- 如果是中文问题，翻译成英文关键词（arXiv/Semantic Scholar 对英文支持更好）
- 只输出如下 JSON：{{"queries": ["关键词查询1", "关键词查询2"]}}"""

    try:
        resp = chat(
            system="你是搜索词优化专家。把研究问题改写成适合学术搜索引擎的短关键词查询，只输出 JSON，不要任何解释或 Markdown 代码块。",
            user=prompt,
            # 注意：deepseek-v4-pro 是思考型模型，会先消耗大量 token 思考再输出，
            # max_tokens 给太小（如 512）思考没完就被截断、content 为空，所以给足空间
            max_tokens=4096,
        )
        data = parse_json(resp.text)
        queries = _as_str_list(data.get("queries") if data else None)[:2]
        queries = [q.strip()[:60] for q in queries if q.strip()]   # 再压一遍长度，防 LLM 偷懒输出长句
        if queries:
            log.info("任务「%s」搜索词精炼为：%s", task.id, queries)
            return queries
    except Exception as e:
        log.warning("任务「%s」搜索词精炼失败（%s），用本地关键词提取", task.id, type(e).__name__)

    # LLM 没精炼出干净词时，本地兜底：去标点、去疑问词，压成一条短中文关键词
    # （比整句问话强：S2 能命中、arXiv 也少报错；英文关键词只能等 LLM 正常时才有）
    return _local_keywords(task.question)


def _local_keywords(question: str) -> list[str]:
    """本地兜底提取搜索关键词（纯 Python，不调 LLM）：
    去掉疑问词/标点/口语连接词，压缩成一条干净的中文关键词搜索词"""
    for w in ("如何", "为什么", "有哪些", "哪些", "什么", "怎么", "怎样", "是什么",
              "主要", "分别", "之间", "多大", "多少", "是否", "吗", "呢", "与", "和", "的"):
        question = question.replace(w, " ")
    text = re.sub(r"[，。？、；：！,.!?;:（）()\"'…\-]", " ", question)   # 标点统一压成空格
    words = [w for w in text.split() if w.strip()]                       # 分词去空
    return [" ".join(words)[:60]] if words else [question[:60]]


def _papers_for_task(flow: FlowState) -> list[dict]:
    """论文模式：把用户给的论文转成候选清单，摘要缺失的按标题去搜"""
    out = []
    for p in flow.papers:
        if p.abstract:
            out.append({"title": p.title, "abstract": p.abstract, "link": p.link})
            continue
        found = _search_arxiv(p.title, 1) or _search_semantic_scholar(p.title, 1)
        if found:
            out.append(found[0])
        else:
            out.append({"title": p.title, "abstract": "", "link": p.link})   # 搜不到用标题兜底
    return out


def _extract_evidence(task: ResearchTask, papers: list[dict]) -> Evidence:
    """调 LLM 从候选论文里提炼 findings 和 claims，并整理 sources。
    这是研究员和"搜索工具"的本质区别：搜索只拿摘要，研究员要提炼出能回答 task 的证据。
    sources/claims 的 id 暂用标题占位，合并阶段由 _assign_ids 统一换成 S 编号"""
    paper_lines = "\n".join(
        f"[{i + 1}] 《{p['title']}》\n    {p.get('abstract', '')}"
        for i, p in enumerate(papers)
    )
    prompt = f"""研究任务（围绕它提炼证据）：
- 要回答的问题：{task.question}
- 研究目的：{task.purpose}
- 需要证据：{task.required_evidence}

候选论文（标题 + 摘要）：
{paper_lines}

请只输出如下 JSON：
{{
  "findings": ["综合结论1", "综合结论2"],
  "claims": [
    {{"claim": "有出处的具体论断", "source_title": "支撑该论断的论文标题", "evidence_text": "摘要里支撑该论断的原文"}}
  ]
}}

要求：
1. findings 是"回答研究问题的综合结论"，用自己的话概括，不要照抄摘要。
2. claims 是"有明确出处的事实/论断"，source_title 必须和上面候选论文里的标题一字不差；evidence_text 是摘要里支撑该论断的原文片段。
3. 只保留和本任务真正相关的论文，不相关的忽略；不要编造不存在的论文、数据或结论。
4. 找不到相关证据就输出空数组：{{"findings": [], "claims": []}}。
"""

    try:
        resp = chat(system=_SYSTEM, user=prompt, max_tokens=8192)
        data = parse_json(resp.text)
        if not data:
            return Evidence(task_id=task.id)
        by_title = {p["title"]: p for p in papers}
        claims: list[Claim] = []
        used_titles: set[str] = set()
        for c in _as_dict_list(data.get("claims")):
            title = str(c.get("source_title") or "").strip()
            if not title or title not in by_title:
                continue   # 引用到不存在的论文，直接丢弃，防止编造来源
            used_titles.add(title)
            claims.append(Claim(
                claim=str(c.get("claim") or "").strip(),
                source_id=title,   # 先用标题占位，_assign_ids 阶段换成 S 编号
                evidence_text=str(c.get("evidence_text") or "").strip(),
            ))
        sources = [Source(
            id=title,   # 先用标题占位，_assign_ids 阶段换成 S 编号
            title=title,
            link=by_title[title].get("link", ""),
            abstract=by_title[title].get("abstract", ""),
        ) for title in used_titles]
        log.info("任务「%s」提炼出 %d 条结论、%d 条论断、%d 个来源", task.id, len(_as_str_list(data.get("findings"))), len(claims), len(sources))
        return Evidence(task_id=task.id, findings=_as_str_list(data.get("findings")), claims=claims, sources=sources)
    except Exception as e:
        log.warning("任务「%s」提炼证据失败（%s），返回空证据", task.id, type(e).__name__)
        return Evidence(task_id=task.id)


def _assign_ids(evidences: list[Evidence]) -> list[Evidence]:
    """合并后统一分配证据编号（E1/E2...）和来源编号（S1/S2...）。
    编号由系统分配、跨任务去重（同一篇论文只给一个 S 号），Writer/Verifier 只能引用、不能自造"""
    title_to_sid: dict[str, str] = {}
    sid_seq = 0
    for i, ev in enumerate(evidences):
        ev.evidence_id = f"E{i + 1}"   # 证据编号按顺序分配
        for src in ev.sources:
            if src.title not in title_to_sid:
                sid_seq += 1
                title_to_sid[src.title] = f"S{sid_seq}"   # 首次出现的论文给新编号
            src.id = title_to_sid[src.title]
        for c in ev.claims:
            c.source_id = title_to_sid.get(c.source_id, c.source_id)   # 标题占位换成 S 编号
    return evidences


def _as_str_list(value) -> list[str]:
    """把 LLM 可能返回的 list/str/None 统一转成字符串列表，去掉空项和首尾空格"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _as_dict_list(value) -> list[dict]:
    """把 LLM 返回的列表里的 dict 项过滤出来，丢弃脏数据"""
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


# ═══════════════════════════════════════
# 底层搜索工具（保留原实现，是研究员提炼证据的"原料来源"）
# ═══════════════════════════════════════
def _search_arxiv(query: str, max_results: int, retries: int = 2) -> list[dict]:
    """用 arxiv 库搜论文，返回 [{title, abstract, link}]。
    失败时指数退避重试（arxiv 库内部会重试，这里再兜一层），仍失败才返回空列表，不拖垮主流程"""
    try:
        import arxiv  # 函数内 import：arxiv 没装时只影响搜索，不影响别的模块加载
    except ImportError:
        log.warning("arxiv 库未安装，跳过 arXiv 搜索")
        return []

    for attempt in range(retries + 1):
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,   # 按相关度排，最相关的排前面
            )
            out = []
            for r in client.results(search):
                out.append({
                    "title": (r.title or "").strip(),
                    "abstract": (r.summary or "").strip().replace("\n", " "),   # 摘要里的换行压成空格，避免污染报告
                    "link": r.entry_id,                                          # entry_id 就是 arXiv 的 abs 链接
                })
            return out
        except Exception as e:
            if attempt < retries:
                # 退避重试：等 2^attempt 秒 + 随机抖动，错峰别撞限流
                wait = 2 ** attempt + random.uniform(0, 1)
                log.warning("arXiv 搜索失败（%s），%.1f秒后重试（第%d次）：%s", type(e).__name__, wait, attempt + 1, query)
                time.sleep(wait)
                continue
            log.warning("arXiv 搜索失败（%s）：%s", type(e).__name__, query)
            return []


def _search_semantic_scholar(query: str, max_results: int, retries: int = 3) -> list[dict]:
    """调 Semantic Scholar API 搜论文，返回 [{title, abstract, link}]。免费接口，无需 key。
    429 限流时指数退避重试（最多 retries 次），仍失败才返回空列表，不拖垮主流程"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,abstract,url",
        "limit": max_results,
    }
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    # 免费接口限流（请求太多）：等 2^attempt 秒 + 随机抖动再试，错峰避开限流池
                    if attempt < retries:
                        wait = 2 ** attempt + random.uniform(0, 1)
                        log.warning("Semantic Scholar 限流(429)，%.1f秒后重试（第%d次）：%s", wait, attempt + 1, query)
                        time.sleep(wait)
                        continue
                    log.warning("Semantic Scholar 限流(429)重试%d次仍失败：%s", retries, query)
                    return []
                resp.raise_for_status()
                data = resp.json()
                break
        except Exception as e:
            if attempt < retries:
                # 其他网络/服务端错误也退避重试，临时故障（5xx/断网）能扛过去
                wait = 2 ** attempt + random.uniform(0, 1)
                log.warning("Semantic Scholar 搜索失败（%s），%.1f秒后重试（第%d次）：%s", type(e).__name__, wait, attempt + 1, query)
                time.sleep(wait)
                continue
            log.warning("Semantic Scholar 搜索失败（%s）：%s", type(e).__name__, query)
            return []

    out = []
    for p in data.get("data", []):
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "abstract": (p.get("abstract") or "").strip().replace("\n", " "),
            "link": p.get("url") or "",
        })
    return out


def _dedup_papers(papers: list[dict]) -> list[dict]:
    """按标题去重：arXiv 和 Semantic Scholar 可能返回同一篇论文，只留第一次出现的"""
    seen: set[str] = set()
    out: list[dict] = []
    for p in papers:
        key = (p.get("title") or "").strip().lower()   # 标题转小写当去重键，防大小写差异
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out
