# 单篇论文精读 skill v3
# v3 架构：Paper → Stage0 结构地图 → Stage1 推理知识库 → Stage2 教学规划 → Stage3 动态分章 → Stage4 定点修正 → 报告
# 位置：skills/paper_deep_read/skill.py，被 main.py 的 _run_deep_read() 调用
# 流程：解析输入 → Docling 解析 PDF → Qwen-VL 理解图表 → Stage0 结构地图 → Stage1 推理知识库
#       → Stage2 教学规划 → Stage3 并行分章 → Stage4 定点修正 → 存入精读历史
# 目标：面向没读过论文的小白，还原"研究推理链"（为什么研究/为什么失败/作者想通了什么/为什么这样设计/实验如何证明）

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from core.config import CACHE_DIR, RESEARCH_SOURCE
from core.json_utils import parse_json
from core.llm_client import chat, vision
from core.logger import get_logger
from core.schemas import DeepReadReport, PaperInfo, PaperSource
from memory.sqlite_store import make_paper_key, save_paper_analysis, PaperAnalysis
from skills.paper_deep_read.prompts import (
    STAGE1_REASONING_PROMPT,
    STAGE_REPORT,
    TRUNCATE_PROMPT,
    DESCRIBE_FIGURE_PROMPT,
)

log = get_logger("paper_deep_read")


# ============================================================
# 公共入口
# ============================================================

def run_deep_read(paper: PaperSource) -> DeepReadReport:
    """单篇论文精读主入口 v3：结构地图 → 推理知识库 → 教学规划 → 动态分章 → 定点修正。
    main.py 的 _run_deep_read() 直接调这个函数，不经过 5 Agent 流程。

    参数：
        paper: 用户输入的论文（标题 + 链接/路径 + 可选摘要）
    返回：
        DeepReadReport: 精读报告（主产物 full_report 是完整动态叙事）
    """
    log.info("单篇精读开始：%s（链接：%s）", paper.title, paper.link)

    # 第一步：把论文链接/PDF 转成本地 PDF 文件路径
    pdf_path, arxiv_id = resolve_pdf(paper)
    if pdf_path is None:
        log.error("论文解析失败，无法获取 PDF：%s", paper.link)
        return _fallback_report(paper, "无法从链接获取论文全文")

    # 第二步：Docling 解析 PDF → 提取文本 + 图表图片
    full_text, figure_paths = parse_text(pdf_path)
    if not full_text.strip():
        log.error("Docling 解析 PDF 失败，文本为空：%s", pdf_path)
        return _fallback_report(paper, "PDF 解析后未能提取到有效文本")

    # 第三步：Qwen-VL 理解图表（有图才调，没图跳过）
    figure_notes: list[str] = []
    if figure_paths:
        log.info("开始理解图表，共 %d 张", len(figure_paths))
        figure_notes = _understand_figures(figure_paths)
    else:
        log.info("论文无图表，跳过视觉理解")

    # 第四步：跳过 Stage0 结构地图（非关键路径，直接用空地图）
    structure_map = {}

    # 第五步：Stage1 — 论文推理知识库（核心：还原作者推理链）
    stage1 = _stage1_reasoning_extract(full_text, figure_notes, structure_map)
    if stage1 is None:
        log.error("Stage1 推理知识库构建失败：%s", paper.link)
        return _fallback_report(paper, "LLM 提取论文推理链失败")

    # 链接/标题以用户输入为准（LLM 可能从正文抓成 GitHub 等别的链接）
    identity = stage1.setdefault("paper_identity", {})
    if paper.link:
        identity["link"] = paper.link
    if not identity.get("title"):
        identity["title"] = paper.title

    # 第六步：一次性生成整篇精读报告（参考 Paper Explainer 设计：
    # 直接拿论文全文 + Stage1 推理知识库，让 LLM 连贯地写完，不分章拼接）
    full_report_text = _generate_full_report(full_text, figure_notes, stage1)
    if not full_report_text:
        log.error("报告生成失败：%s", paper.link)
        return _fallback_report(paper, "LLM 生成精读报告失败")

    # 第七步：组装 DeepReadReport（full_report 用 LLM 生成的完整文本，其余字段从 Stage1 提取）
    report = _build_report(stage1, full_report_text)

    # 第八步：存精读历史（非关键路径，失败不影响主流程）
    _save_to_history(paper, arxiv_id, report)

    log.info("单篇精读完成：%s", paper.title)
    return report


# ============================================================
# 确定性操作：输入解析 + PDF 下载 + Docling 解析（纯工程，不需要 LLM 推理）
# ============================================================

def is_arxiv_url(link: str) -> str:
    """判断链接是不是 arXiv 链接，是的话返回 arXiv ID，不是返回空字符串"""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-]+)", link)
    if not m:
        return ""   # 不是 arXiv 链接，直接返回空，让上层走其他解析分支
    arxiv_id = m.group(1).strip("/")
    if arxiv_id.lower().endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    return arxiv_id


def resolve_pdf(paper: PaperSource) -> tuple[Optional[str], str]:
    """把用户输入转成本地 PDF 文件路径。
    返回 (pdf_path, arxiv_id)：路径是本地文件地址，ID 是论文指纹（非 arXiv 则为空）。
    三步策略：① arXiv 链接 → 构造 PDF 地址下载；② 本地 .pdf 路径 → 直接用；③ 其他 URL → 下载后判断"""
    link = paper.link.strip()
    # 比赛合规（domestic）：arXiv/其他 URL 下载都是境外请求，规则禁调；
    # 比赛版只允许"本地上传 PDF"，其余场景走这里返回 None 由上层报错，比赛后 RESEARCH_SOURCE=international 恢复联网
    if RESEARCH_SOURCE != "international" and not (
        link.endswith(".pdf") and os.path.isfile(link)
    ):
        log.warning("合规模式（RESEARCH_SOURCE=%s）仅支持本地上传 PDF，拒绝联网下载：%s", RESEARCH_SOURCE, link)
        return None, ""

    # ① arXiv 链接：构造 PDF 下载地址
    arxiv_id = is_arxiv_url(link)
    if arxiv_id:
        log.info("识别为 arXiv 链接，ID：%s", arxiv_id)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return download_pdf(pdf_url, arxiv_id), arxiv_id

    # ② 本地 PDF 文件：直接返回路径
    if link.lower().endswith(".pdf") and os.path.isfile(link):
        log.info("识别为本地 PDF 文件：%s", link)
        return link, ""

    # ③ 其他 URL：尝试下载
    if link.startswith(("http://", "https://")):
        log.info("识别为其他 URL，尝试下载：%s", link)
        name_hash = hashlib.md5(link.encode()).hexdigest()[:12]
        return download_pdf(link, name_hash), ""

    # 只有标题、没有链接/路径 → 无法获取全文
    log.warning("无法获取 PDF：链接为空或格式不支持（%s）", link)
    return None, ""


def download_pdf(url: str, file_id: str) -> Optional[str]:
    """下载 PDF 到缓存目录，返回本地路径。已缓存则直接返回"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"paper_{file_id}.pdf")

    # 缓存命中：直接返回，不再下载
    if os.path.isfile(cache_path):
        log.info("PDF 缓存命中：%s", cache_path)
        return cache_path

    # 下载 PDF
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            # 检查是否是真正的 PDF（有些链接会重定向到 HTML）
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type.lower() and "pdf" not in content_type.lower():
                log.warning("下载的不是 PDF（Content-Type: %s），可能被重定向到网页：%s", content_type, url)
                return None
            with open(cache_path, "wb") as f:
                f.write(resp.content)
            log.info("PDF 下载完成：%s（%d 字节）", cache_path, len(resp.content))
            return cache_path
    except Exception as e:
        log.warning("PDF 下载失败（%s）：%s", type(e).__name__, url)
        return None


def parse_text(pdf_path: str) -> tuple[str, list[str]]:
    """用 Docling 解析 PDF：提取全文文本 + 导出图表图片。
    返回 (full_text, figure_paths)：文本是 markdown 格式，图是本地文件路径列表"""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as e:
        log.error("Docling 未安装：%s。请运行 pip install docling", e)
        return "", []

    log.info("Docling 开始解析：%s", pdf_path)

    # Docling 配置：开图片提取 + 表格识别
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True  # 把论文里的图表提取成独立图片
    pipeline_options.do_table_structure = True       # 识别表格结构

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    })

    try:
        result = converter.convert(pdf_path)
        full_text = result.document.export_to_markdown()  # 全文输出 Markdown
        log.info("Docling 解析完成，文本长度：%d 字符", len(full_text))
    except Exception as e:
        log.error("Docling 解析失败（%s）：%s", type(e).__name__, pdf_path)
        return "", []

    # 导出图表：把 Docling 提取出的图片保存到缓存目录
    figure_paths = export_figures(result, pdf_path)

    return full_text, figure_paths


def export_figures(result, pdf_path: str) -> list[str]:
    """把 Docling 解析出的图表图片导出到缓存目录，返回本地路径列表。
    Docling 的图片通过 picture_item.get_image(document) 拿 PIL Image 对象（官方推荐方式）"""
    try:
        pictures = result.document.pictures
    except Exception:
        log.info("Docling 未提取到图表（论文可能纯文字）")
        return []

    if not pictures:
        return []

    # 图表单独放一个目录，跟 PDF 缓存分开放
    pdf_name = Path(pdf_path).stem  # 文件名（不含扩展名）
    figure_dir = os.path.join(CACHE_DIR, f"figures_{pdf_name}")
    os.makedirs(figure_dir, exist_ok=True)

    saved: list[str] = []
    # 只用需要导入时才 import，避免 Docling 未安装时 import 报错
    try:
        from docling_core.types.doc import PictureItem
    except ImportError:
        log.warning("docling_core 版本过低，不支持 PictureItem，跳过图表导出")
        return saved

    for i, picture_item in enumerate(pictures):
        try:
            if not isinstance(picture_item, PictureItem):
                continue
            # 用 Docling 官方推荐的方式获取图片：get_image(document)
            img = picture_item.get_image(result.document)
            if img is None:
                continue
            figure_path = os.path.join(figure_dir, f"figure_{i + 1:03d}.png")
            img.save(figure_path, "PNG")
            saved.append(figure_path)
        except Exception as e:
            log.warning("图表 %d 导出失败（%s）", i + 1, type(e).__name__)

    log.info("图表导出完成：%d/%d 张", len(saved), len(pictures))
    return saved


# ============================================================
# 内部辅助函数
# ============================================================


def _understand_figures(figure_paths: list[str]) -> list[str]:
    """用 Qwen-VL 理解图表：每张图生成一段结构化描述。
    单张失败不拖垮全局，失败的那张返回降级描述"""
    notes: list[str] = []
    for i, fpath in enumerate(figure_paths):
        try:
            resp = vision(fpath, DESCRIBE_FIGURE_PROMPT)
            desc = resp.text.strip()
            note = f"图表 {i + 1}：{desc}" if desc else f"图表 {i + 1}：无法自动理解"
            notes.append(note)
            log.info("图表 %d 理解完成", i + 1)
        except Exception as e:
            log.warning("图表 %d 理解失败（%s）", i + 1, type(e).__name__)
            notes.append(f"图表 {i + 1}：理解失败（{type(e).__name__}），请人工查看")
    return notes


def _join_figure_notes(figure_notes: list[str]) -> str:
    """把图表理解描述拼成一段文本，没图就返回占位说明"""
    return "\n".join(figure_notes) if figure_notes else "（本论文未提供图表或图表无法提取）"


# ============================================================
# Stage0：论文结构地图
# ============================================================

def _stage0_structure_map(full_text: str, figure_notes: list[str]) -> dict:
    """Stage0：让 LLM 画"论文地图"（章节作用 + 关键图表 + 论证顺序）。
    非关键路径：JSON 偶发失败时重试 1 次，仍失败返回空 dict，Stage1 仍可继续"""
    prompt = (STAGE0_STRUCTURE_PROMPT
              .replace("{paper_text}", full_text)
              .replace("{figure_notes}", _join_figure_notes(figure_notes)))
    for attempt in range(2):  # 最多试 2 次，JSON 偶发失败时多给一次机会
        try:
            resp = chat(
                system="你是论文结构分析助手。只输出 JSON，不要加任何解释或 Markdown 代码块标记。",
                user=prompt,
                max_tokens=8192,
            )
            data = parse_json(resp.text)
            if data is not None:
                log.info("Stage0 结构地图完成：%d 个章节", len(data.get("sections", [])))
                return data
            log.warning("Stage0 JSON 解析失败（第 %d 次）", attempt + 1)
        except Exception as e:
            log.warning("Stage0 结构解析失败（%s，第 %d 次）", type(e).__name__, attempt + 1)
    log.warning("Stage0 多次失败，使用空结构地图继续")
    return {}


# ============================================================
# Stage1：论文推理知识库（核心）
# ============================================================

def _stage1_reasoning_extract(full_text: str, figure_notes: list[str], structure_map: dict, retry_count: int = 0) -> Optional[dict]:
    """Stage1：LLM 从全文还原"论文推理链"（research_story/design_reasoning/concept_knowledge/
    evidence_chain/experiment_reasoning 等），返回 dict。文本太长超 token 时先压缩再重试，最多 3 次"""
    prompt = (STAGE1_REASONING_PROMPT
              .replace("{paper_text}", full_text)
              .replace("{figure_notes}", _join_figure_notes(figure_notes))
              .replace("{structure_map}", json.dumps(structure_map, ensure_ascii=False, indent=2)))

    try:
        resp = chat(
            system="你是论文推理分析专家。只输出 JSON，不要加任何解释、Markdown 代码块标记或额外文字。",
            user=prompt,
            max_tokens=32768,  # 3 万档够装完整推理知识库；之前 65536 让 flash 模型写超长嵌套 JSON 容易深层出错
        )
        data = parse_json(resp.text)
        if data is None:
            # JSON 解析失败：先压缩原文再重试（更短的输入更可能输出合法 JSON）
            if retry_count >= 3:
                log.error("Stage1 已重试 %d 次仍 JSON 解析失败，停止重试", retry_count)
                return None
            log.warning("Stage1 JSON 解析失败（第 %d 次重试），压缩后重试", retry_count + 1)
            return _stage1_reasoning_extract_truncated(full_text, figure_notes, structure_map, retry_count + 1)
        log.info("Stage1 推理知识库完成")
        return data
    except Exception as e:
        # token 超限类错误：压缩重试
        err_msg = str(e).lower()
        if any(k in err_msg for k in ("token", "context", "length")):
            if retry_count >= 3:
                log.error("Stage1 已重试 %d 次仍 Token 超限，停止重试：%s", retry_count, type(e).__name__)
                return None
            log.info("文本太长超 token 限制（第 %d 次重试），压缩后重试", retry_count + 1)
            return _stage1_reasoning_extract_truncated(full_text, figure_notes, structure_map, retry_count + 1)
        log.warning("Stage1 推理知识库提取失败（%s）", type(e).__name__)
        return None


def _stage1_reasoning_extract_truncated(full_text: str, figure_notes: list[str], structure_map: dict, retry_count: int) -> Optional[dict]:
    """文本太长超 token 限制时：先让 LLM 把全文压缩成事实摘要，再走正常 Stage1 提取"""
    truncate_prompt = TRUNCATE_PROMPT.replace("{paper_text}", full_text)
    try:
        resp = chat(
            system="你是学术论文摘要专家。只输出压缩后的摘要，不要加任何解释。",
            user=truncate_prompt,
            max_tokens=4096,
        )
        compressed = resp.text.strip()
        log.info("文本压缩完成（第 %d 次重试）：原文 %d 字 → 压缩后 %d 字", retry_count, len(full_text), len(compressed))
        return _stage1_reasoning_extract(compressed, figure_notes, structure_map, retry_count)
    except Exception as e:
        log.warning("文本压缩失败（%s）", type(e).__name__)
        return None


# ============================================================
# 一次性生成整篇精读报告（参考 Paper Explainer 设计）
# ============================================================

def _generate_full_report(full_text: str, figure_notes: list[str], stage1: dict) -> Optional[str]:
    """直接拿论文全文 + Stage1 推理知识库，让 LLM 一次性写完整篇精读报告。
    不分章拼接，保证内容连贯；融入 Paper Explainer 的锚点、三类陈述、批判性分析设计。"""
    stage1_json = json.dumps(stage1, ensure_ascii=False, indent=2)
    prompt = (STAGE_REPORT
              .replace("{paper_text}", full_text)
              .replace("{figure_notes}", _join_figure_notes(figure_notes))
              .replace("{stage1_json}", stage1_json))
    try:
        resp = chat(
            system="你是论文精读讲解专家。用中文写一份面向小白的精读报告，术语保留英文。",
            user=prompt,
            max_tokens=65536,  # 整篇报告一次性输出，留足空间
        )
        text = resp.text.strip()
        log.info("整篇精读报告生成完成：%d 字", len(text))
        return text
    except Exception as e:
        log.warning("整篇报告生成失败（%s）", type(e).__name__)
        return None


# ============================================================
# Stage2：教学规划
# ============================================================

def _stage2_planner(stage1: dict) -> Optional[dict]:
    """Stage2：让 LLM 设计教学方案（核心突破/误区/必讲/可跳过/动态章节），返回 dict。
    JSON 偶发失败时重试 1 次，仍失败返回 None（上层用空规划继续）"""
    stage1_json = json.dumps(stage1, ensure_ascii=False, indent=2)
    prompt = STAGE2_PLANNER.replace("{stage1_json}", stage1_json)
    for attempt in range(2):
        try:
            resp = chat(
                system="你是一位擅长讲论文的科研老师。只输出教学方案 JSON，不要加任何解释。",
                user=prompt,
                max_tokens=8192,
            )
            data = parse_json(resp.text)
            if data is not None:
                log.info("Stage2 教学规划完成：%d 个章节", len(data.get("sections", [])))
                return data
            log.warning("Stage2 JSON 解析失败（第 %d 次）", attempt + 1)
        except Exception as e:
            log.warning("Stage2 教学规划失败（%s，第 %d 次）", type(e).__name__, attempt + 1)
    log.warning("Stage2 多次失败，返回空规划")
    return None


# ============================================================
# Stage3：动态分章生成
# ============================================================

def _default_sections() -> list[dict]:
    """Stage2 没给出章节时用的兜底教学顺序（6 段标准推理链）"""
    return [
        {"title": "为什么研究这个问题", "goal": "让读者明白这个问题为什么值得研究"},
        {"title": "以前方法怎么解决", "goal": "介绍旧方法的思路和做法"},
        {"title": "作者核心突破", "goal": "解释作者想通了什么关键点"},
        {"title": "方法设计", "goal": "解释方法为什么这样设计"},
        {"title": "实验验证", "goal": "说明实验如何证明作者观点"},
        {"title": "局限和影响", "goal": "讲清局限和这篇论文改变了什么"},
    ]


def _stage3_generate_sections(stage1: dict, plan: dict) -> Optional[list[dict]]:
    """Stage3：按 Stage2 的动态章节列表并行生成每节内容。
    每章收到完整推理知识库 + 教学规划。单章失败按 5s/10s/20s 退避重试，
    彻底失败返回占位符（不拖垮整篇），只有线程池级错误才返回 None"""
    sections = (plan or {}).get("sections") or _default_sections()
    # 章节数上限 6：避免 LLM 输出过多章节拖慢生成
    sections = sections[:6]
    stage1_json = json.dumps(stage1, ensure_ascii=False, indent=2)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2) if plan else "{}"

    def _call_section(idx: int, sec: dict, max_retries: int = 3) -> dict:
        title = sec.get("title") or f"章节 {idx + 1}"
        goal = sec.get("goal", "")
        prompt = (STAGE3_SECTION
                  .replace("{section_title}", title)
                  .replace("{section_goal}", goal)
                  .replace("{stage1_json}", stage1_json)
                  .replace("{plan}", plan_json))
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = chat(
                    system="只生成指定章节，不要加其他内容。",
                    user=prompt,
                    max_tokens=8192,
                )
                return {"title": title, "content": resp.text.strip()}
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = 5 * (2 ** attempt)  # 5s → 10s → 20s，扛住十几秒的网络抖动
                    log.warning("章节「%s」第 %d 次尝试失败（%s），%d 秒后重试",
                                title, attempt + 1, type(e).__name__, delay)
                    time.sleep(delay)
        # 彻底失败：返回占位符而不是抛异常，避免一篇论文因为一节失败全废
        log.error("章节「%s」重试 %d 次仍失败（%s），降级为占位符",
                  title, max_retries + 1, type(last_error).__name__ if last_error else "unknown")
        return {"title": title, "content": f"（本节生成失败：{type(last_error).__name__ if last_error else 'unknown'}）"}

    try:
        # 并行度上限 3，降低并发触发连接异常的概率
        with ThreadPoolExecutor(max_workers=min(len(sections), 3)) as executor:
            futures = {executor.submit(_call_section, i, sec): i for i, sec in enumerate(sections)}
            results = {}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return [results[i] for i in range(len(sections))]
    except Exception as e:
        log.warning("Stage3 分章生成失败（%s）", type(e).__name__)
        return None


# ============================================================
# Stage4：定点修正
# ============================================================

def _stage4_editor(stage1: dict, plan: dict, sections: list[dict]) -> Optional[dict]:
    """Stage4：定点修正。逐章检查事实/逻辑/教学，只重写有问题的章节并回写，
    不重新理解论文、不重写整篇。JSON 偶发失败时重试 1 次，仍失败返回 None（用 Stage3 原稿）。
    返回 {"issues": [...], "revised_sections": [{"title","content"}]}"""
    stage1_json = json.dumps(stage1, ensure_ascii=False, indent=2)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2) if plan else "{}"
    sections_json = json.dumps(sections, ensure_ascii=False, indent=2)

    prompt = (STAGE4_EDITOR
              .replace("{stage1_json}", stage1_json)
              .replace("{plan}", plan_json)
              .replace("{sections}", sections_json))

    for attempt in range(2):
        try:
            resp = chat(
                system="你是论文精读报告审核编辑。只输出 JSON，不要加任何解释或代码块标记。",
                user=prompt,
                max_tokens=16384,
            )
            data = parse_json(resp.text)
            if data is not None:
                issues = data.get("issues", [])
                revised = data.get("revised_sections", [])
                log.info("Stage4 定点修正完成：%d 个问题，重写 %d 章",
                         len(issues) if isinstance(issues, list) else 0,
                         len(revised) if isinstance(revised, list) else 0)
                return {"issues": issues, "revised_sections": revised}
            log.warning("Stage4 JSON 解析失败（第 %d 次）", attempt + 1)
        except Exception as e:
            log.warning("Stage4 定点修正失败（%s，第 %d 次）", type(e).__name__, attempt + 1)
    log.warning("Stage4 多次失败，使用 Stage3 原稿成稿")
    return None


# ============================================================
# 结果映射：dict → DeepReadReport
# ============================================================

def _as_str_list(value) -> list[str]:
    """把 LLM 可能返回的 list/str/None 统一转成字符串列表"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _serialize_concepts(concepts) -> list[str]:
    """把 concept_knowledge（list of dict）转成大白话字符串列表，给固定字段 concept_explanations 用"""
    out = []
    for c in concepts or []:
        if isinstance(c, dict):
            name = c.get("name", "")
            intuition = c.get("intuition", "") or c.get("problem_solved", "")
            out.append(f"{name}：{intuition}" if name else intuition)
        else:
            out.append(str(c))
    return out


def _serialize_experiments(exp_reasons) -> list[str]:
    """把 experiment_reasoning（list of dict）转成字符串列表，给固定字段 experiments 用"""
    out = []
    for e in exp_reasons or []:
        if isinstance(e, dict):
            parts = [
                ("问题", e.get("question")),
                ("实验", e.get("experiment")),
                ("结果", e.get("result")),
                ("意义", e.get("meaning")),
                ("实际意义", e.get("practical_significance")),
            ]
            out.append("；".join(f"{k}：{v}" for k, v in parts if v))
        else:
            out.append(str(e))
    return out


def _serialize_design(designs) -> str:
    """把 design_reasoning（list of dict）转成一段文字，给固定字段 method 用"""
    lines = []
    for d in designs or []:
        if isinstance(d, dict):
            bits = []
            if d.get("problem"):
                bits.append(f"问题：{d['problem']}")
            if d.get("choice"):
                bits.append(f"选择：{d['choice']}")
            if d.get("why"):
                bits.append(f"原因：{d['why']}")
            lines.append("；".join(bits))
        else:
            lines.append(str(d))
    return "\n".join(lines)


def _serialize_innovations(designs) -> list[str]:
    """从 design_reasoning 提炼创新点列表，给固定字段 paper_innovations 用"""
    out = []
    for d in designs or []:
        if isinstance(d, dict):
            choice = d.get("choice", "")
            why = d.get("why", "")
            out.append(f"{choice}：{why}" if choice and why else (choice or why))
        else:
            out.append(str(d))
    return [x for x in out if x]


def _join_story_problem(story: dict) -> str:
    """把 research_story 拼成一段问题背景，给固定字段 problem_background 用"""
    parts = []
    if story.get("background"):
        parts.append("背景：" + story["background"])
    old = _as_str_list(story.get("old_methods"))
    if old:
        parts.append("旧方法：" + "；".join(old))
    lim = _as_str_list(story.get("old_method_limitations"))
    if lim:
        parts.append("旧方法缺陷：" + "；".join(lim))
    if story.get("research_gap"):
        parts.append("研究空白：" + story["research_gap"])
    if story.get("author_question"):
        parts.append("作者要回答的问题：" + story["author_question"])
    return "\n".join(parts)


def _build_report(stage1: dict, full_report_text: str) -> DeepReadReport:
    """把 Stage1 推理知识库 + LLM 生成的完整报告文本组装成 DeepReadReport。
    full_report 直接用 LLM 一次性生成的文本（不分章拼接），其余固定字段从 Stage1 抽取。"""
    identity = stage1.get("paper_identity", {}) or {}
    authors = identity.get("authors", [])
    authors_str = ", ".join(str(a) for a in authors) if isinstance(authors, list) else (str(authors) if authors else "")

    story = stage1.get("research_story", {}) or {}
    designs = stage1.get("design_reasoning", []) or []
    concepts = stage1.get("concept_knowledge", []) or []
    exp_reasons = stage1.get("experiment_reasoning", []) or []

    # 报告开头加元信息头（论文标题/作者/会议/年份/日期），由代码精确生成，不靠 LLM
    header_parts = []
    title = identity.get("title", "")
    if title:
        header_parts.append(f"# {title}")
    meta = []
    if authors_str:
        meta.append(f"作者：{authors_str}")
    venue = identity.get("venue", "")
    if venue:
        meta.append(f"发表会议/期刊：{venue}")
    year = str(identity.get("year", "")).strip()
    if year:
        meta.append(f"年份：{year}")
    meta.append(f"报告生成日期：{datetime.now().strftime('%Y年%m月%d日')}")
    header_parts.append("\n".join(meta))
    header = "\n".join(header_parts)

    final_report = header + "\n\n" + full_report_text

    one_line = (story.get("new_hypothesis", "")
                or story.get("key_observation", "")
                or story.get("research_gap", ""))

    return DeepReadReport(
        paper_info=PaperInfo(
            title=identity.get("title", ""),
            authors=authors_str,
            year=str(identity.get("year", "")).strip(),
            link=identity.get("link", ""),
        ),
        one_line_summary=one_line,
        problem_background=_join_story_problem(story),
        core_idea=story.get("new_hypothesis", "") or story.get("key_observation", ""),
        concept_explanations=_serialize_concepts(concepts),
        method=_serialize_design(designs),
        paper_innovations=_serialize_innovations(designs),
        experiments=_serialize_experiments(exp_reasons),
        limitations=_as_str_list(stage1.get("author_limitations"))
                    + [f"未提及：{x}" for x in _as_str_list(stage1.get("unmentioned_limitations"))],
        scenario=stage1.get("impact", ""),
        related_directions=[],
        reading_guide=stage1.get("reading_guide", ""),
        full_report=final_report,
        verification_notes="",
    )


# ============================================================
# 历史保存 + 兜底报告
# ============================================================

def _save_to_history(paper: PaperSource, arxiv_id: str, report: DeepReadReport) -> None:
    """把精读结果存进 SQLite 精读历史。非关键路径：失败不影响主流程，只记日志"""
    try:
        paper_key = make_paper_key(
            title=paper.title,
            link=paper.link,
            arxiv_id=arxiv_id,
        )
        record = PaperAnalysis(
            paper_key=paper_key,
            paper_title=report.paper_info.title or paper.title,
            link=paper.link,
            summary=report.one_line_summary,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        save_paper_analysis(record)
        log.info("精读历史已保存：%s", paper_key)
    except Exception as e:
        log.warning("精读历史保存失败（%s），不影响主流程", type(e).__name__)


def _fallback_report(paper: PaperSource, reason: str) -> DeepReadReport:
    """兜底报告：精读过程某一步失败时，返回一个带错误说明的报告"""
    log.warning("精读降级：%s（原因：%s）", paper.title, reason)
    msg = f"精读未完成：{reason}"
    return DeepReadReport(
        paper_info=PaperInfo(title=paper.title, link=paper.link),
        one_line_summary=msg,
        scenario=f"精读流程中断：{reason}。建议人工阅读原文或稍后重试。",
        full_report=msg,
    )
