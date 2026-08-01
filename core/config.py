# 统一配置模块
# 作用：全项目的 API key、模型名、预算、路径都在这读，其他模块 import 本文件取值
# 敏感信息放 .env（已 gitignore），模板见 .env.example，没配也有默认值兜底

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # 从项目根目录的 .env 读配置

@dataclass
class ModelConfig:
    """一个大模型 API 的配置项"""
    api_key: str          # API key
    base_url: str         # 接口地址
    model: str            # 模型名
    max_tokens: int       # 单次生成上限
    max_context: int      # 上下文窗口长度
    timeout: float        # 请求超时秒数


def _read(name: str, default: str = "") -> str:
    """读环境变量，没配就用默认值"""
    return os.getenv(name, default)


# ═══════ 文本模型：DeepSeek-V3（所有文本 Agent 共用）═══════
DEEPSEEK = ModelConfig(
    api_key=_read("DEEPSEEK_API_KEY"),
    base_url=_read("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    model=_read("DEEPSEEK_MODEL", "deepseek-chat"),
    max_tokens=int(_read("DEEPSEEK_MAX_TOKENS", "8192")),
    max_context=int(_read("DEEPSEEK_MAX_CONTEXT", "64000")),
    timeout=float(_read("DEEPSEEK_TIMEOUT", "120")),
)

# ═══════ 视觉模型：Qwen2.5-VL-7B（图片/图表理解）═══════
QWEN_VL = ModelConfig(
    api_key=_read("QWEN_VL_API_KEY"),
    base_url=_read("QWEN_VL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    model=_read("QWEN_VL_MODEL", "qwen2.5-vl-7b-instruct"),
    max_tokens=int(_read("QWEN_VL_MAX_TOKENS", "2048")),
    max_context=int(_read("QWEN_VL_MAX_CONTEXT", "32768")),
    timeout=float(_read("QWEN_VL_TIMEOUT", "120")),
)

# ═══════ 预算与路径 ═══════
DAILY_BUDGET = float(_read("DAILY_BUDGET", "20"))       # 每日预算上限（元）
MONTHLY_BUDGET = float(_read("MONTHLY_BUDGET", "200"))  # 每月预算上限（元）
DATA_DIR = _read("DATA_DIR", "data")                    # SQLite / Milvus 数据目录
CACHE_DIR = _read("CACHE_DIR", "data/cache")            # 检索/解析结果缓存目录
