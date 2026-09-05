# 统一配置模块
# 作用：全项目的 API key、模型名、预算、路径都在这读，其他模块 import 本文件取值
# 敏感信息放 .env（已 gitignore），模板见 .env.example，没配也有默认值兜底

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .logger import get_logger

load_dotenv()  # 从项目根目录的 .env 读配置

log = get_logger("config")  # 本模块日志器，提示配置缺失用

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
# 文本模型 DeepSeek：项目里 5 个 Agent 全用它生成文字，靠提示词区分角色
DEEPSEEK = ModelConfig(
    api_key=_read("DEEPSEEK_API_KEY"),                              # API 密钥，从 .env 读，没配也能启动
    base_url=_read("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),  # 接口地址，默认 DeepSeek 官方地址
    model=_read("DEEPSEEK_MODEL", "deepseek-v4-flash"),            # 模型名，默认快速模型 V4-Flash
    max_tokens=int(_read("DEEPSEEK_MAX_TOKENS", "8192")),           # 一次最多生成多少 token，写报告够用
    max_context=int(_read("DEEPSEEK_MAX_CONTEXT", "64000")),        # 上下文窗口（输入+输出合计），预留防超长
    timeout=float(_read("DEEPSEEK_TIMEOUT", "120")),                # 请求超时秒数，防网络卡死
)

# 视觉模型 Qwen-VL：专管看图，图表/截图理解全走它
QWEN_VL = ModelConfig(
    api_key=_read("QWEN_VL_API_KEY"),                              # API 密钥，从 .env 读
    base_url=_read("QWEN_VL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),  # 接口地址，默认阿里云百炼
    model=_read("QWEN_VL_MODEL", "qwen2.5-vl-7b-instruct"),         # 模型名，默认多模态 7B
    max_tokens=int(_read("QWEN_VL_MAX_TOKENS", "4096")),           # 单次生成上限，够图表描述用，留余量防多图超窗
    max_context=int(_read("QWEN_VL_MAX_CONTEXT", "32768")),        # 32K 上下文，图片文字共用，别设太高
    timeout=float(_read("QWEN_VL_TIMEOUT", "120")),                # 请求超时秒数，防网络卡死
)

# ═══════ 配置完整性检查 ═══════
# 没配 key 项目能启动，但调用会失败，提前警告提醒配 .env
if not DEEPSEEK.api_key:
    log.warning("DEEPSEEK_API_KEY 未配置：文本 Agent 调用会失败，请复制 .env.example 为 .env 并填写")
if not QWEN_VL.api_key:
    log.warning("QWEN_VL_API_KEY 未配置：图片理解功能会失败，请复制 .env.example 为 .env 并填写")

# ═══════ 论文检索数据源开关 ═══════
# 用途：比赛合规开关。规则禁调境外 API（arXiv/OpenAlex 全在境外），
# 默认 domestic（境内模式）——论文搜索/PDF 下载全短路；Docling 的布局/表格模型源
# huggingface.co 也是境外，统一指到国内镜像 hf-mirror.com（本地有缓存时直接用缓存、不联网）。
# 只走"本地上传 PDF + 国内大模型"，全程零境外请求；原有境外逻辑原样保留但不会执行；
# 比赛后想恢复境外检索，把 .env 的 RESEARCH_SOURCE 改成 international 即可
RESEARCH_SOURCE = _read("RESEARCH_SOURCE", "domestic")   # domestic=比赛版(禁境外) | international=完整版(可调境外)
if RESEARCH_SOURCE != "international":
    # Docling/transformers 默认从 huggingface.co 拉模型（境外），比赛版指到国内镜像 hf-mirror.com；
    # 本地已缓存模型则 huggingface_hub 直接命中缓存，不发任何网络请求
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ═══════ 预算与路径 ═══════
DAILY_BUDGET = float(_read("DAILY_BUDGET", "20"))       # 每日预算上限（元）
MONTHLY_BUDGET = float(_read("MONTHLY_BUDGET", "200"))  # 每月预算上限（元）

# 项目根目录：用 __file__ 推导（config.py 在 core/ 下，上一级就是项目根），
# 这样不管从哪个目录启动程序，数据/缓存都固定落在项目根下，不会因为 cwd 不同而散落各处
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = _read("DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))  # SQLite / Milvus 数据目录（.env 可覆盖为绝对路径）
CACHE_DIR = _read("CACHE_DIR", os.path.join(_PROJECT_ROOT, "data", "cache"))  # 检索/解析结果缓存目录（.env 可覆盖为绝对路径）
