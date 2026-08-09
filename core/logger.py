# 统一日志模块
# 作用：全项目的日志出口，其他模块一行 import 就能拿到带时间戳的日志器
# 用法：from core.logger import get_logger; log = get_logger("模块名")
# 输出：控制台（开发期直接看）+ 文件 data/logs/app.log（按大小轮转，排错查历史）
# 注意：本模块不 import core 里其他东西，避免循环依赖

import logging
import os
from logging.handlers import RotatingFileHandler

# 日志格式：时间 级别 模块名 消息
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# 项目根目录：用 __file__ 推导（core/ 的上一级），和 config.py 的做法一致，
# 避免从不同目录启动时日志散落各处（本模块不 import config，防止循环依赖）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 日志文件位置（写死放 data/logs，和 config 的 DATA_DIR 默认值一致）
_LOG_DIR = os.path.join(_PROJECT_ROOT, "data", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

_configured = False  # 标记根日志器是否配置过（防止重复加 handler）


def get_logger(name: str = "academicmind") -> logging.Logger:
    """拿一个日志器：第一次调用时完成全局配置，之后直接复用"""
    global _configured
    if not _configured:
        _configure()
        _configured = True
    return logging.getLogger(name)


def _configure() -> None:
    """配置根日志器：控制台输出 + 文件输出，只执行一次"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # 根级别放最低，具体级别由各 handler 管

    # 控制台输出：开发期直接看重要信息
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    # 文件输出：记全量（含 DEBUG），按 1MB 轮转，保留 3 个历史文件
    os.makedirs(_LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file_handler)
