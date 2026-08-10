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
from core.schemas import ImageItem, MODE_SURVEY, PaperAnalysis, TaskRecord, UserConfig

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

        -- 图片记忆表：一张图一行，存图片档案 + pHash（去重用）
        -- pHash 存 hex 字符串（64位感知哈希），去重时全量拉出算汉明距离
        CREATE TABLE IF NOT EXISTS image (
            image_id        TEXT PRIMARY KEY,   -- 图片编号（没给就用 URL 当主键）
            url             TEXT NOT NULL DEFAULT '',
            description     TEXT NOT NULL DEFAULT '',
            source          TEXT NOT NULL DEFAULT '',
            subtopic        TEXT NOT NULL DEFAULT '',
            phash           TEXT NOT NULL DEFAULT '',   -- pHash hex 字符串（64位感知哈希），去重用
            image_embedding TEXT NOT NULL DEFAULT ''    -- 描述向量（JSON 字符串），图文检索的"意思"匹配用
        );
    """)
    # 老库升级：image 表没有 image_embedding 列就补上（新库建表已含，重复执行不报错）
    cols = {r[1] for r in conn.execute("PRAGMA table_info(image)")}
    if "image_embedding" not in cols:
        conn.execute("ALTER TABLE image ADD COLUMN image_embedding TEXT NOT NULL DEFAULT ''")
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
    # 数据库里的模式值做一次校验：只认 survey/paper，脏数据回退默认，别让前端拿到非法值
    mode = row[0] if row[0] in ("survey", "paper") else MODE_SURVEY
    return UserConfig(preferred_mode=mode, theme=row[1])


def save_user_config(preferred_mode: str) -> None:
    """保存模式偏好：本地固定一行，不区分用户（前端切换模式时调用，下次进页面自动恢复）"""
    # 模式值先校验：只认 survey/paper，非法值回退默认，别把脏数据写进库里（和读端 get_user_config 对称）
    if preferred_mode not in ("survey", "paper"):
        preferred_mode = MODE_SURVEY
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

def _clean_identifier(raw: str) -> str:
    """清洗从链接里抠出来的指纹：去掉尾部的多余字符（斜杠/.pdf/排版标点），
    保证和 PDF 解析出的干净 ID 完全一致，同一篇论文不会记成两条"""
    s = raw.strip()
    if not s:
        return raw                         # 清洗完变空就返回原样，别把指纹洗没了
    s = s.rstrip("/")                      # 去掉尾部斜杠：abs/2301.12345/ → 2301.12345
    if s.lower().endswith(".pdf"):
        s = s[:-4]                         # 去掉尾部 .pdf：pdf/2301.12345v2.pdf → 2301.12345v2
    return s.rstrip(".,;:()】）")           # 去掉尾部排版标点：doi 链接尾的句号/逗号等


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
            return f"arxiv:{_clean_identifier(m.group(1))}"   # 链接里的 arXiv ID（清洗尾部多余字符）
        m = re.search(r"doi\.org/([\w.\-/]+)", link)
        if m:
            # DOI 大小写不敏感（ISO 标准），统一转小写，避免同一 DOI 记成两条
            return f"doi:{_clean_identifier(m.group(1)).lower()}"     # 链接里的 DOI（清洗尾部多余字符）
        return link.strip()                # 没有指纹，直接拿链接当标识
    return title.strip()                   # 纯 PDF 且没解析出指纹，用标题兜底


def save_paper_analysis(rec: PaperAnalysis) -> None:
    """记录单篇精读结果（单篇 skill 分析完调用）；同一篇论文再分析会覆盖，保持最新"""
    # 论文指纹是主键必须有值：没有说明调用方漏了 make_paper_key，直接报错尽早暴露（fail fast）
    if not rec.paper_key:
        raise ValueError("paper_key 不能为空：请先用 make_paper_key 生成论文指纹")
    # 没传时间就填当前时间（只算个局部变量，不写回入参，别污染调用方的对象）
    created_at = rec.created_at or datetime.now().isoformat(timespec="seconds")
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO paper_analysis (paper_key, paper_title, link, summary, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(paper_key) DO UPDATE SET "
            "paper_title = excluded.paper_title, link = excluded.link, summary = excluded.summary, created_at = excluded.created_at",
            (rec.paper_key, rec.paper_title, rec.link, rec.summary, created_at),
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


# ═══════════ 图片记忆（pHash 去重）═══════════

def _hamming_distance(hex1: str, hex2: str) -> int:
    """算两个 pHash（hex 字符串）的汉明距离：异或后数 1 的个数。
    两个 64 位哈希完全相同距离为 0，完全不同最大为 64"""
    if not hex1 or not hex2:
        return 64  # 有空值就当完全不同，不误判为重复
    try:
        # 两个 hex 先转整数再异或，数 1 的个数就是汉明距离
        return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")
    except ValueError:
        # 传进来的不是合法 hex（脏数据/上游异常），当完全不同处理，别让去重流程崩掉
        return 64


def save_image(item: ImageItem) -> None:
    """存一张图片档案（含 pHash 和描述向量）；同一 image_id 重复存会覆盖（去重更新常用）"""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO image (image_id, url, description, source, subtopic, phash, image_embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(image_id) DO UPDATE SET "
            "url = excluded.url, description = excluded.description, "
            "source = excluded.source, subtopic = excluded.subtopic, "
            "phash = excluded.phash, image_embedding = excluded.image_embedding",
            (item.image_id or item.url, item.url, item.description,
             item.source, item.subtopic, item.phash, item.embedding),
        )
        conn.commit()
    finally:
        conn.close()


def list_images(subtopic: str = "", keyword: str = "") -> list[ImageItem]:
    """按子主题/关键词拉图片清单（图文检索的"召回"阶段用）：
    传 subtopic 就只挑该子主题的图；传 keyword 就在描述里模糊搜；
    都不传就返回全部图片（调研图量小，全量拉出算相似度毫秒级，不用分页）"""
    conn = _connect()
    try:
        sql = "SELECT image_id, url, description, source, subtopic, phash, image_embedding FROM image WHERE 1=1"
        params: list[str] = []
        if subtopic:
            sql += " AND subtopic = ?"     # 子主题精确匹配（图登记时贴的标签）
            params.append(subtopic)
        if keyword:
            sql += " AND description LIKE ?"   # 关键词在描述里模糊搜（兜底召回）
            params.append(f"%{keyword}%")
        rows = conn.execute(sql + " ORDER BY image_id", params).fetchall()
    finally:
        conn.close()
    return [
        ImageItem(
            image_id=row[0], url=row[1], description=row[2],
            source=row[3], subtopic=row[4], phash=row[5], embedding=row[6],
        )
        for row in rows
    ]


def image_exists(image_id: str) -> bool:
    """按 image_id 精确查图片在不在库里（第一级去重：快速预筛同 ID 重复）"""
    conn = _connect()
    try:
        row = conn.execute("SELECT 1 FROM image WHERE image_id = ?", (image_id,)).fetchone()
    finally:
        conn.close()
    return row is not None


def image_exists_by_phash(phash: str, threshold: int = 5) -> bool:
    """按 pHash 汉明距离查图片是否已存在（第二级去重：防同图不同 ID）。
    遍历所有 phash，汉明距离 ≤ threshold 就算重复。
    threshold=5 是社区默认值：只抓几乎完全一样的图，不误杀相似但不同的图"""
    if not phash:
        return False  # 没 pHash 就没法判重，放行让调用方决定
    conn = _connect()
    try:
        rows = conn.execute("SELECT phash FROM image WHERE phash != ''").fetchall()
    finally:
        conn.close()
    for (stored_phash,) in rows:
        if _hamming_distance(phash, stored_phash) <= threshold:
            return True  # 找到一张汉明距离够近的，判重复
    return False


def delete_image(image_id: str) -> None:
    """按 image_id 删一张图片档案（换图/清理不用了的图片）"""
    conn = _connect()
    try:
        conn.execute("DELETE FROM image WHERE image_id = ?", (image_id,))
        conn.commit()
    finally:
        conn.close()
