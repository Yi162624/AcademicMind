# 记忆模块：SQLite 存储
# 作用：把"模式偏好（上次用的模式）、单篇精读历史、任务日志"存进本地 SQLite 文件
#       整个 memory/ 里最轻的一个，只用 Python 自带的 sqlite3，不背任何大模型
# 位置：memory/sqlite_store.py，被 main.py 的 save_task_log 和前端（模式恢复/精读历史搜索）调用

import os
import re
import sqlite3
from datetime import datetime

from core.config import DATA_DIR
from core.logger import get_logger
from core.schemas import PaperAnalysis, TaskRecord, UserConfig

log = get_logger("sqlite_store")  # 本模块日志器

# SQLite 数据文件：data/user_config.db（文档 2.1 定的名字，三张表都在这个文件里）
_DB_PATH = os.path.join(DATA_DIR, "user_config.db")


def _connect() -> sqlite3.Connection:
    """开一个 SQLite 连接（每次都新建用完就关，简单可靠）+ 顺手把表建好"""
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)  # data/ 目录不存在就先建
    conn = sqlite3.connect(_DB_PATH)
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """建表：没有才建，重复调用不报错（建表语句写死在脚本里，改动要同步升级脚本）"""
    conn.executescript("""
        -- 用户配置表：一个 user_id 一行，存上次用的模式
        CREATE TABLE IF NOT EXISTS user_config (
            user_id        TEXT PRIMARY KEY,
            preferred_mode TEXT NOT NULL DEFAULT 'survey',
            theme          TEXT NOT NULL DEFAULT 'light'
        );

        -- 单篇精读历史表：一篇论文一行，存精读结论摘要
        -- 主键是 paper_key（论文指纹）：怎么记录就怎么搜，链接/PDF 进来都走同一套标识
        CREATE TABLE IF NOT EXISTS paper_analysis (
            paper_key   TEXT PRIMARY KEY,   -- 论文指纹（arXiv ID/DOI 优先，其次链接，最后标题）
            paper_title TEXT NOT NULL,      -- 论文标题（给人看的）
            link        TEXT NOT NULL DEFAULT '',
            summary     TEXT NOT NULL DEFAULT '',   -- 精读结论摘要
            created_at  TEXT NOT NULL DEFAULT ''    -- 分析时间
        );

        -- 任务日志表：一次任务一行，存耗时/成本/是否被采纳（给开发者分析用）
        CREATE TABLE IF NOT EXISTS task_log (
            task_id      TEXT PRIMARY KEY,
            mode         TEXT NOT NULL DEFAULT 'survey',
            question     TEXT NOT NULL DEFAULT '',
            duration_sec REAL NOT NULL DEFAULT 0,
            token_usage  INTEGER NOT NULL DEFAULT 0,
            cost         REAL NOT NULL DEFAULT 0,
            accepted     INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL DEFAULT ''
        );
    """)
    conn.commit()


# ═══════════ 用户配置（模式偏好）═══════════

# 本地单例标识：模式偏好就存这一行，不区分用户
# （对齐 VS Code 存 settings.json 的思路：设置属于"这台机器"，跟账号/用户无关）
_LOCAL_KEY = "local"


def get_user_config() -> UserConfig:
    """读本地模式偏好（用户上次用的调研/论文模式）；没存过就返回默认值（survey），不报错"""
    conn = _connect()          # 连接数据库
    # 查固定那一行（本地单例，不区分用户）
    try:
        row = conn.execute(     # 执行查询语句
            "SELECT preferred_mode, theme FROM user_config WHERE user_id = ?",
            (_LOCAL_KEY,),      # 绑定参数，防止 SQL 注入
        ).fetchone()            # 取第一行记录
    finally:
        conn.close()            # 关闭数据库连接
    if row is None:
        return UserConfig()     # 没记录 → 默认 survey 模式，前端照常用
    return UserConfig(preferred_mode=row[0], theme=row[1])


def save_user_config(preferred_mode: str) -> None:
    """保存模式偏好：本地固定一行，不区分用户（前端切换模式时调用，下次进页面自动恢复）"""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO user_config (user_id, preferred_mode, theme) VALUES (?, ?, 'light') "
            "ON CONFLICT(user_id) DO UPDATE SET preferred_mode = excluded.preferred_mode",
            (_LOCAL_KEY, preferred_mode),
        )
        conn.commit()
    finally:
        conn.close()


# ═══════════ 单篇精读历史 ═══════════

def make_paper_key(title: str, link: str = "", arxiv_id: str = "") -> str:
    """给论文生成唯一标识：优先论文指纹（arXiv ID/DOI），其次链接，最后标题。
    目的是同一篇论文无论从链接进来还是 PDF 进来，标识都一样，不会记成两条"""
    # PDF 解析出的 arXiv ID（指纹最准，直接用）
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    # 链接论文：先从链接里提取 arXiv ID / DOI，保证和 PDF 提取出的指纹格式一致
    if link:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+)", link)
        if m:
            return f"arxiv:{m.group(1)}"   # 链接里的 arXiv ID
        m = re.search(r"doi\.org/([\w.\-/]+)", link)
        if m:
            return f"doi:{m.group(1)}"     # 链接里的 DOI
        return link.strip()                # 没有指纹，直接拿链接当标识
    return title.strip()                   # 纯 PDF 且没解析出指纹，用标题兜底


def save_paper_analysis(rec: PaperAnalysis) -> None:
    """记录单篇精读结果（单篇 skill 分析完调用）；同一篇论文再分析会覆盖，保持最新"""
    if not rec.created_at:
        rec.created_at = datetime.now().isoformat(timespec="seconds")  # 没传时间就填当前时间
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO paper_analysis (paper_key, paper_title, link, summary, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(paper_key) DO UPDATE SET "
            "paper_title = excluded.paper_title, link = excluded.link, summary = excluded.summary, created_at = excluded.created_at",
            (rec.paper_key, rec.paper_title, rec.link, rec.summary, rec.created_at),
        )
        conn.commit()
    finally:
        conn.close()


def get_paper_analysis(paper_key: str) -> PaperAnalysis | None:
    """按论文指纹查历史精读记录（用户拿 PDF/链接进来，走同一个 make_paper_key 就能搜回）。
    搜到 = 之前分析过（相当于"已读"）；搜不到 = 没分析过（相当于"未读"）"""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT paper_key, paper_title, link, summary, created_at "
            "FROM paper_analysis WHERE paper_key = ?",
            (paper_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return PaperAnalysis(paper_key=row[0], paper_title=row[1], link=row[2], summary=row[3], created_at=row[4])


# ═══════════ 任务日志 ═══════════

def save_task_log(rec: TaskRecord) -> None:
    """记一条任务日志（main.py 每次任务跑完调用；token/cost 等 Agent 报上来再补填）"""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO task_log (task_id, mode, question, duration_sec, token_usage, cost, accepted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rec.task_id, rec.mode, rec.question, rec.duration_sec,
             rec.token_usage, rec.cost, int(rec.accepted), rec.created_at),
        )
        conn.commit()
    finally:
        conn.close()
