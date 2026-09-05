# 统一模型调用层
# 作用：把 DeepSeek 文本生成和 Qwen-VL 图片理解封装成两个函数，
#       其他模块只调 chat()/vision()，不直接碰 httpx 和 API 细节

import base64
from dataclasses import dataclass

import httpx

from .config import DEEPSEEK, QWEN_SEARCH, QWEN_VL
from .logger import get_logger

log = get_logger("llm_client")  # 本模块日志器，排错时看 API 调用情况用

RETRY_TIMES = 2  # 网络失败最多重试次数

# 单价：每百万 token 多少钱（输入, 输出），用于估算单次成本
_PRICE = {
    "deepseek": (1.0, 2.0),
    "qwen_vl": (2.0, 5.0),
    "qwen": (2.0, 5.0),
}

# ── 会话级消耗累加器 ──
# 作用：记录"一次任务从头到尾"调了多少 token、花了多少钱。
#       5 个 Agent 各自调 chat()/vision()，这里统一累计，跨挂起/恢复也持续累加，
#       任务结束写任务日志时一次性读取，不需要 Agent 层回传。
_SESSION_INPUT_TOKENS = 0    # 会话累计输入 token
_SESSION_OUTPUT_TOKENS = 0   # 会话累计输出 token
_SESSION_COST = 0.0          # 会话累计花费（元）


def reset_session_usage() -> None:
    """清空会话消耗计数：启动新任务时调用（单篇精读/多 Agent 流程都从这里开始）"""
    global _SESSION_INPUT_TOKENS, _SESSION_OUTPUT_TOKENS, _SESSION_COST
    _SESSION_INPUT_TOKENS = _SESSION_OUTPUT_TOKENS = _SESSION_COST = 0


def get_session_usage() -> tuple[int, int, float]:
    """取会话累计消耗 (输入 token, 输出 token, 花费元)：任务结束写日志用"""
    return _SESSION_INPUT_TOKENS, _SESSION_OUTPUT_TOKENS, round(_SESSION_COST, 4)


@dataclass
class LLMResponse:
    """一次模型调用的返回：文本 + token 用量 + 预估花费"""
    text: str              # 模型生成的文本
    input_tokens: int      # 输入 token 数
    output_tokens: int     # 输出 token 数
    cost: float            # 预估花费（元）


def _accumulate(resp: LLMResponse) -> None:
    """把一次模型调用的消耗累加到会话计数（chat/vision 每次返回前都调）"""
    global _SESSION_INPUT_TOKENS, _SESSION_OUTPUT_TOKENS, _SESSION_COST
    _SESSION_INPUT_TOKENS += resp.input_tokens
    _SESSION_OUTPUT_TOKENS += resp.output_tokens
    _SESSION_COST += resp.cost


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "deepseek") -> float:
    """根据 token 用量估算花费（元），记录任务日志用"""
    price_in, price_out = _PRICE.get(model, _PRICE["deepseek"])
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def chat(system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
    """调 DeepSeek 文本生成：system 是角色设定，user 是提问内容，返回 LLMResponse"""
    log.info("DeepSeek 文本调用开始：model=%s", DEEPSEEK.model)
    url = f"{DEEPSEEK.base_url.rstrip('/')}/chat/completions"    # API 地址
    payload = {
        "model": DEEPSEEK.model,                     # 模型名称
        "messages": [  
            {"role": "system", "content": system},      # 系统消息
            {"role": "user", "content": user},          # 用户消息
        ],
        "max_tokens": max_tokens or DEEPSEEK.max_tokens,   # 最大 token 数
    }
    data = _post(url, payload, DEEPSEEK.api_key, DEEPSEEK.timeout)  # 发 POST 请求
    log.debug("DeepSeek 文本调用返回 token 用量：%s", data["usage"])   # 记录 token 用量
    resp = _to_response(data, "deepseek")             # 解析返回数据
    _accumulate(resp)                                  # 累加到会话消耗（写任务日志用）
    return resp


def vision(image: str, prompt: str) -> LLMResponse:
    """调 Qwen-VL 理解图片：image 是图片 URL 或本地文件路径，prompt 是问它什么"""
    log.info("Qwen-VL 看图调用开始：model=%s", QWEN_VL.model)
    url = f"{QWEN_VL.base_url.rstrip('/')}/chat/completions"    # API 地址
    payload = {
        "model": QWEN_VL.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _to_image_url(image)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": QWEN_VL.max_tokens,
    }
    data = _post(url, payload, QWEN_VL.api_key, QWEN_VL.timeout)
    resp = _to_response(data, "qwen_vl")
    _accumulate(resp)                                  # 累加到会话消耗（写任务日志用）
    return resp


def search(system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
    """境内联网搜索：调阿里云百炼 Qwen 的 enable_search（先联网检索再回答）。
    比赛合规版（domestic）调研模式用它搜真实论文——arXiv/OpenAlex 是境外 API 被禁，
    这里是境内的替代源：请求发国内节点、由阿里自己抓网页，返回结果是检索来的不是模型编的"""
    log.info("境内联网搜索调用开始：model=%s enable_search=true", QWEN_SEARCH.model)
    url = f"{QWEN_SEARCH.base_url.rstrip('/')}/api/v1/services/aigc/text-generation/generation"   # 百炼原生网关
    payload = {
        "model": QWEN_SEARCH.model,
        "input": {"messages": [
            {"role": "system", "content": system},     # 角色设定
            {"role": "user", "content": user},         # 检索指令 + 要整理的论文
        ]},
        "parameters": {
            "enable_search": True,                     # 打开联网检索：先搜真实网页再回答
            "result_format": "message",                # 返回结构用 message 格式，好取正文
            "max_tokens": max_tokens or QWEN_SEARCH.max_tokens,
        },
    }
    data = _post(url, payload, QWEN_SEARCH.api_key, QWEN_SEARCH.timeout)
    text = data["output"]["choices"][0]["message"]["content"]     # 模型生成的正文
    usage = data.get("usage", {})                                 # token 用量
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    resp = LLMResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=estimate_cost(input_tokens, output_tokens, "qwen"),
    )
    _accumulate(resp)                                 # 累加到会话消耗（写任务日志用）
    return resp


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    """发 POST 请求带重试：4xx 不重试直接抛错，5xx/网络错误重试"""
    headers = {"Authorization": f"Bearer {api_key}"}      # API 密钥
    for attempt in range(RETRY_TIMES + 1):        # 最多重试 2 次
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                log.debug("API 请求成功：%s（第 %d 次尝试）", url, attempt + 1)
                return resp.json()
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                log.error("API 请求被拒（%s）：%s，请检查 key 和参数", e.response.status_code, url)
                raise  # 请求本身有问题（key 错、参数错），重试也没用
            log.warning("API 返回 %s，准备重试：%s", e.response.status_code, url)
        except httpx.HTTPError as e:
            log.warning("网络异常（%s），准备重试：%s", type(e).__name__, url)
    log.error("API 调用多次重试仍失败：%s", url)
    raise RuntimeError(f"API 调用多次重试仍失败：{url}")


def _to_response(data: dict, model: str) -> LLMResponse:
    """把 API 返回的 JSON 整理成 LLMResponse，顺带算好花费"""
    text = data["choices"][0]["message"]["content"]     # 模型生成的文本
    usage = data.get("usage", {})                       # token 用量
    input_tokens = usage.get("prompt_tokens", 0)        # 输入 token 数
    output_tokens = usage.get("completion_tokens", 0)   # 输出 token 数
    log.debug("模型返回：输入 %s token，输出 %s token", input_tokens, output_tokens)     # 记录 token 用量
    return LLMResponse(             # 返回 LLMResponse 对象
        text=text,                                  
        input_tokens=input_tokens,                  
        output_tokens=output_tokens,                
        cost=estimate_cost(input_tokens, output_tokens, model),  
    )


def _to_image_url(image: str) -> str:
    """图片转 API 认的格式：http(s) 链接直接用，本地文件转 base64 data url"""
    if image.startswith(("http://", "https://", "data:")):       # 如果是 URL 或 data url，直接返回
        return image
    with open(image, "rb") as f:                        # 读取本地文件内容
        encoded = base64.b64encode(f.read()).decode()   # 编码为 base64 字符串
    return f"data:image/png;base64,{encoded}"           # 返回 base64 data url
