# 视觉理解模块
# 功能：通过 Qwen2.5-VL-7B-Instruct API 对图片进行内容理解与描述生成
#        用于图文一致性校验和图片内容摘要

from dataclasses import dataclass

from core import llm_client  # 统一模型调用层，vision() 已封装好 Qwen-VL API
from core.logger import get_logger

log = get_logger("image_understanding")  # 本模块日志器，排错时看图片理解调用情况用

# 图片描述提示词：固定模板，逼模型按结构输出，方便后面 BGE 转向量、图文检索用
_DESCRIBE_PROMPT = """你是学术配图解说员。只按格式输出，不要加任何额外文字。

【图片类型】从"架构图/流程图/柱状图/折线图/饼图/表格/热力图/实物照片/示意图/其他"中选一个
【一句话总结】30字以内，只说核心信息
【内容描述】100字以内，说关键元素和它们的关系
【图中文字】只提取图片上肉眼可见的硬性文字或数值（无则写无），只列关键词/数字/标签，没有就写"无"，严禁凭空补充数据
【适用主题】根据图片中明确的标题/坐标轴/图例推导具体学术主题（如“医疗影像分割模型架构”），若主题不明确则写“待定”

参考样例：
【图片类型】架构图
【一句话总结】展示一个三阶段多模态大模型的推理流程
【内容描述】左侧文本编码器+中间对齐模块+右侧生成解码器，箭头指示数据流向
【图中文字】Encoder, Cross-Modal Alignment, Decoder, 准确率 91.2%
【适用主题】多模态大模型架构设计

如果图片模糊、完全损坏、或无法看到任何有效内容，直接回复"图片无法识别"。

禁止自己编造任何数据!!!"""


def describe_image(image: str) -> str:
    """看一张图，返回结构化描述文本
    image 可以是图片 URL 或本地文件路径；API 挂了或图坏了时返回降级描述，不让上游流程崩"""
    log.info("开始理解图片：%s", image)
    try:
        resp = llm_client.vision(image, _DESCRIBE_PROMPT)  # 调 Qwen-VL 看图
        desc = resp.text.strip()                     # 去掉首尾空格
        log.info("图片理解返回描述：%s", desc)
        if not desc:
            log.warning("图片理解返回空描述：%s", image)
            return _fallback_desc(image)              # 空描述时用降级描述兜底
        return desc
    except Exception as e:
        log.warning("图片理解失败（%s），用降级描述兜底：%s", type(e).__name__, image)
        return _fallback_desc(image)


def _fallback_desc(image: str) -> str:
    """降级描述：理解失败时兜底，保证图片还能入库、流程不断，但描述里标注了待人工复核"""
    return f"图片无法自动理解（来源：{image}），请人工查看"


# 图文匹配提示词：让 VLM 判断候选图适不适合作为某章节配图（精排用）
_MATCH_PROMPT = """你是学术配图审核员。判断这张图是否适合作为下面章节的配图。

章节需求：
{requirement}

判断维度（三个都满足才算匹配）：
1. 主题相关：图的内容对应章节讲的学术主题
2. 类型合适：图的类型符合章节需要（如章节要架构图，给流程图就不合适）
3. 信息增量：图能补充文字说不清的内容（如数据趋势、结构关系）

只按以下格式输出，不要加任何额外文字：
【匹配】是/否
【理由】50字以内，说明为什么匹配或不匹配

参考样例：
【匹配】是
【理由】图为Transformer架构图，符合章节"模型结构"需求，能直观展示层间关系

【匹配】否
【理由】图为数据集统计柱状图，章节需要的是模型架构图，类型不符

如果图片模糊、完全损坏、或无法看到任何有效内容，直接回复：
【匹配】否
【理由】图片无法识别"""


@dataclass
class MatchResult:
    """图文匹配结果：VLM 精排的返回"""
    matched: bool       # 图和章节需求匹不匹配
    reason: str         # 一句话理由（为什么匹配 / 为什么不匹配）


def match_image(requirement: str, image: str) -> MatchResult:
    """判断一张图适不适合作为某章节的配图（VLM 精排用）
    image 可以是 URL 或本地路径；API 挂了时倾向放行（matched=True + 警告理由），
    避免所有图被拦掉导致报告没图用"""
    log.info("开始图文匹配：图=%s，需求=%s", image, requirement[:50])
    prompt = _MATCH_PROMPT.format(requirement=requirement)
    try:
        resp = llm_client.vision(image, prompt)     # 调 Qwen-VL 看图
        return _parse_match(resp.text)              # 解析匹配结果
    except Exception as e:
        log.warning("图文匹配失败（%s），放行并标注待复核：%s", type(e).__name__, image)
        return MatchResult(matched=True, reason=f"VLM 精排未执行（{type(e).__name__}），建议人工复核")


def _parse_match(text: str) -> MatchResult:
    """解析 VLM 返回的匹配结果；解析不到【匹配】行时默认放行（True），避免误拦"""
    text = text.strip()
    matched = True       # 默认放行：解析失败也不拦，防止报告没图用
    reason = ""          # 默认空理由，后面没解析到再补兜底语
    # 逐行扫描模型返回的文本，找【匹配】和【理由】两个标签
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("【匹配】"):
            # 把【匹配】标签去掉，剩下的就是模型给的值（是/否/yes/no 等）
            val = line.replace("【匹配】", "").strip()
            # 模型回"是/是的/yes"算匹配，其他都算不匹配
            matched = val.startswith("是") or val.lower().startswith("yes")
        elif line.startswith("【理由】"):
            # 把【理由】标签去掉，剩下的就是理由文字
            reason = line.replace("【理由】", "").strip()
    # 模型没给理由时补一句兜底，让下游知道这张图需要人工看一眼
    if not reason:
        reason = "VLM 未给出明确理由，建议人工复核"
    return MatchResult(matched=matched, reason=reason)
