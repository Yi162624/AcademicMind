# 统一模型调用层
# 作用：把 DeepSeek 文本生成和 Qwen-VL 图片理解封装成两个函数，
#       其他模块只调 chat()/vision()，不直接碰 httpx 和 API 细节

import base64
from dataclasses import dataclass

import httpx

from .config import DEEPSEEK, QWEN_VL

RETRY_TIMES = 2  # 网络失败最多重试次数

# 单价：每百万 token 多少钱（输入, 输出），用于估算单次成本
_PRICE = {
    "deepseek": (2.0, 8.0),
    "qwen_vl": (2.0, 8.0),
}


@dataclass
class LLMResponse:
    """一次模型调用的返回：文本 + token 用量 + 预估花费"""
    text: str              # 模型生成的文本
    input_tokens: int      # 输入 token 数
    output_tokens: int     # 输出 token 数
    cost: float            # 预估花费（元）


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "deepseek") -> float:
    """根据 token 用量估算花费（元），记录任务日志用"""
    price_in, price_out = _PRICE.get(model, _PRICE["deepseek"])
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def chat(system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
    """调 DeepSeek 文本生成：system 是角色设定，user 是提问内容，返回 LLMResponse"""
    url = f"{DEEPSEEK.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": DEEPSEEK.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens or DEEPSEEK.max_tokens,
    }
    data = _post(url, payload, DEEPSEEK.api_key, DEEPSEEK.timeout)
    return _to_response(data, "deepseek")


def vision(image: str, prompt: str) -> LLMResponse:
    """调 Qwen-VL 理解图片：image 是图片 URL 或本地文件路径，prompt 是问它什么"""
    url = f"{QWEN_VL.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": QWEN_VL.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _to_image_url(image)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": QWEN_VL.max_tokens,
    }
    data = _post(url, payload, QWEN_VL.api_key, QWEN_VL.timeout)
    return _to_response(data, "qwen_vl")


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    """发 POST 请求带重试：4xx 不重试直接抛错，5xx/网络错误重试"""
    headers = {"Authorization": f"Bearer {api_key}"}
    for _ in range(RETRY_TIMES + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                raise  # 请求本身有问题（key 错、参数错），重试也没用
        except httpx.HTTPError:
            pass
    raise RuntimeError(f"API 调用多次重试仍失败：{url}")


def _to_response(data: dict, model: str) -> LLMResponse:
    """把 API 返回的 JSON 整理成 LLMResponse，顺带算好花费"""
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return LLMResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=estimate_cost(input_tokens, output_tokens, model),
    )


def _to_image_url(image: str) -> str:
    """图片转 API 认的格式：http(s) 链接直接用，本地文件转 base64 data url"""
    if image.startswith(("http://", "https://", "data:")):
        return image
    with open(image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"
