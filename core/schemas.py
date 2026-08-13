# 数据契约模块
# 作用：全项目统一的数据结构定义，谁传数据都从这儿 import，保证格式一致
# 位置：项目根目录，5个Agent、memory、multimodal、frontend、evaluation 都会引用它
#
# ── 两种任务模式 ──
# "survey" 领域调研：用户提研究问题，输出图文领域报告
# "paper"  论文分析：用户给论文列表，输出综述分析报告
#
# ── 结构清单（按流程顺序）──
#   1 用户输入          → PaperSource（论文分析模式用）
#   2 Planner 产出      → Outline（大纲，调研/分析两种模式字段分开）
#   3 Researcher 产出   → Evidence（文本 TextEvidence + 图片 ImageItem）
#   4 Verifier 产出     → VerifierResult（整体结论）+ Issue（问题清单）
#   5 Writer 产出       → Report（报告，带上 outline 供验证员核对）
#   6 Advisor 产出      → Suggestion（含 Paper 论文）
#   7 单篇精读 skill    → DeepReadStage1（阶段1中间JSON）+ DeepReadReport（八段式报告）
#   8 main.py 汇总      → FinalResult（report/suggestion/deep_read 三选场景，前端只认这一种格式）
#   9 LangGraph 状态    → FlowState（状态机"黑板"，含 HUMAN_IN_LOOP 重试字段）
#  10 记忆层            → SurveyRecord / PaperRecord（向量记忆）/ PaperAnalysis / TaskRecord / UserConfig
#
# 为什么从前往后定：每个结构的字段由"下一棒要消费什么"决定，
#                   先定产出、再定消费，字段才不会对不上。

from __future__ import annotations  # 让注解延迟求值，避免类之间互相引用时报错

from dataclasses import dataclass, field


# ═══════════════════════════════════════
# 任务模式常量：全项目统一用这两个值，别手打字符串
# ═══════════════════════════════════════
MODE_SURVEY = "survey"  # 领域调研模式
MODE_PAPER = "paper"    # 论文分析模式


# ═══════════════════════════════════════
# 1 用户输入：论文分析模式用
# ═══════════════════════════════════════
@dataclass
class PaperSource:
    """用户输入的一篇论文（论文分析模式用）"""
    title: str                  # 论文标题
    link: str = ""              # 论文链接或PDF路径（arXiv链接/URL/本地PDF都行）
    abstract: str = ""          # 摘要（可选，没有就让研究员去读原文）


# ═══════════════════════════════════════
# 2 Planner（规划师）产出：研究大纲
# ═══════════════════════════════════════
@dataclass
class Outline:
    """规划师产出：研究大纲（调研/分析两种模式共用，按模式用对应字段）"""
    mode: str                                   # 任务模式：MODE_SURVEY / MODE_PAPER
    question: str                               # 用户原始问题（两种模式都要）
    # 调研模式（survey）用：
    subtopics: list[str] = field(default_factory=list)          # 默认 3 个子主题（研究员分活用）
    sections: list[str] = field(default_factory=list)           # 报告章节标题（写作者排版用）
    image_requirements: list[str] = field(default_factory=list) # 每章需要的图片类型（研究员找图用）
    # 论文分析模式（paper）用：
    analysis_dimensions: list[str] = field(default_factory=list)     # 每篇论文要分析哪些方面（方法/实验/结论...）
    paper_assignments: list[list[str]] = field(default_factory=list) # 论文怎么分组（每组一个研究员，存论文标题）


# ═══════════════════════════════════════
# 3 Researcher（研究员）产出：图文证据
# ═══════════════════════════════════════
@dataclass
class TextEvidence:
    """文本证据，带来源链接（验证员靠它查引用真假）"""
    content: str                # 证据正文
    source: str                 # 来源链接或出处
    subtopic: str = ""          # 属于哪个子主题/分组


@dataclass
class ImageItem:
    """图片素材，视觉工作记忆里存的就是它（主要调研模式用）"""
    url: str                    # 图片地址
    description: str            # 图片内容描述（视觉理解模块生成）
    source: str                 # 图片来源（报告里要标引用）
    subtopic: str = ""          # 属于哪个子主题
    image_id: str = ""          # 图片唯一编号（报告里引用图片用）
    phash: str = ""             # 图片感知哈希（hex 字符串，64位），pHash 去重用
    embedding: str = ""         # 描述向量（JSON 字符串），图文检索的"意思"匹配用


@dataclass
class Evidence:
    """一个研究员产出：某个子主题/论文分组的证据"""
    subtopic: str                                   # 调研模式=子主题名；分析模式=论文分组名
    texts: list[TextEvidence] = field(default_factory=list)  # 文本证据（分析模式下 source 即论文链接）
    images: list[ImageItem] = field(default_factory=list)    # 图片素材（分析模式通常为空）


# ═══════════════════════════════════════
# 4 Verifier（验证员）产出：检查结论 + 问题清单
# ═══════════════════════════════════════
@dataclass
class Issue:
    """验证员发现的一个质量问题"""
    level: str                  # 严重程度：error=必须改，warning=建议改
    check_type: str             # 检查类型：fact=事实 / citation=引用 / image_text=图文一致 / fidelity=分析忠实度
    message: str                # 问题描述
    location: str = ""          # 问题位置（哪一章/哪张图）


@dataclass
class VerifierResult:
    """验证员整体检查结论（Verifier Agent 的产出）"""
    passed: bool                                        # 是否全部通过（没有 error 级问题就算过）
    issues: list[Issue] = field(default_factory=list)   # 问题清单（通过则为空）
    feedback: str = ""                                  # 综合修改意见（注入 Writer 重写 prompt 用）


# ═══════════════════════════════════════
# 5 Writer（写作者）产出：图文报告
# ═══════════════════════════════════════
@dataclass
class Report:
    """写作者产出：最终报告（调研=图文报告，分析=综述报告）"""
    outline: Outline            # 生成报告用的大纲（验证员核对用）
    html: str                   # 报告正文（HTML）


# ═══════════════════════════════════════
# 6 Advisor（建议Agent）产出：研究方向建议
# ═══════════════════════════════════════
@dataclass
class Paper:
    """一篇推荐的核心论文"""
    title: str                  # 论文标题
    reason: str                 # 推荐理由（一句话）
    link: str = ""              # 论文链接


@dataclass
class Suggestion:
    """建议Agent产出：调研模式=研究方向+论文+行动；分析模式=论文+行动（不给方向）"""
    directions: list[str] = field(default_factory=list)   # 调研模式=3个方向；分析模式留空[]
    papers: list[Paper] = field(default_factory=list)     # 5篇最值得精读的论文（分析模式可为空）
    actions: list[str] = field(default_factory=list)      # 下一步行动清单（建议 3-5 条，Advisor prompt 里强制）


# ═══════════════════════════════════════
# 7 单篇精读 skill（paper_deep_read）产出
# ═══════════════════════════════════════
@dataclass
class PaperInfo:
    """论文元信息：标题/作者/年份/机构/链接（单篇精读用，替代裸 dict）"""
    title: str = ""               # 论文标题
    authors: str = ""             # 所有作者（逗号分隔）
    year: str = ""                # 发表年份
    link: str = ""                # 论文链接或标识


@dataclass
class DeepReadStage1:
    """单篇精读阶段1产出：从论文里提取的关键结构化信息（中间JSON，不直接给用户）
    v2 新增字段：支持分层信息路由和事实核查"""
    paper_info: PaperInfo = field(default_factory=PaperInfo)  # 论文元信息
    one_line_summary: str = ""                                # 一句话总结
    # ── v2 分层 problem 区块 ──
    problem_background: str = ""                              # 问题背景（旧方法怎么做的 + 有什么问题 + 为什么重要）
    problem_old_methods: list[str] = field(default_factory=list)      # [v2] 旧方法列表
    problem_old_method_problems: list[str] = field(default_factory=list) # [v2] 旧方法的具体问题
    problem_paper_examples: list[str] = field(default_factory=list)   # [v2] 论文中的问题示例
    # ── v2 分层 core_idea 区块 ──
    core_idea: str = ""                                       # 核心洞察（大白话：作者想通了什么）
    core_idea_changed_assumption: str = ""                     # [v2] 打破了什么旧假设
    core_idea_paper_evidence: str = ""                         # [v2] 论文原文依据
    concept_explanations: list[str] = field(default_factory=list)  # 核心概念大白话解释
    # ── v2 分层 method 区块 ──
    method: str = ""                                          # 方法全景（结构 + 模块 + 公式直觉）
    method_key_formulas: list[dict] = field(default_factory=list)  # [v2] 关键公式 [{formula, meaning}]
    paper_innovations: list[str] = field(default_factory=list)    # 创新点（列表，每条大白话解释）
    experiments: list[str] = field(default_factory=list)      # 实验分析
    limitations: list[str] = field(default_factory=list)      # 局限性
    analogies: list[str] = field(default_factory=list)        # 生活类比素材
    example_sentences: list[str] = field(default_factory=list)    # 论文中的具体例句
    key_quotes: list[str] = field(default_factory=list)       # 论文原文关键句
    # ── v2 分层 impact 区块 ──
    scenario: str = ""                                        # 适用场景与评价
    impact_paper_claim: str = ""                               # [v2] 论文自己的贡献声明
    related_directions: list[str] = field(default_factory=list) # 关联研究方向
    reading_guide: str = ""                                   # 重点阅读指引
    figure_notes: list[str] = field(default_factory=list)     # 图表理解描述


@dataclass
class DeepReadReport:
    """单篇精读最终产出：大白话教学式精读报告
    v3：full_report 是完整动态叙事 Markdown（真正给小白看的主报告）；
    其余固定字段保留用于兼容 main.py / test_single.py 的结构化读取"""
    paper_info: PaperInfo = field(default_factory=PaperInfo)     # 1 论文元信息
    one_line_summary: str = ""                                   # 2 一句话看懂
    problem_background: str = ""                                 # 3 为什么要研究这个问题
    core_idea: str = ""                                          # 4 核心思想（作者想通了什么）
    concept_explanations: list[str] = field(default_factory=list)  # 5 核心概念大白话解释
    method: str = ""                                             # 6 方法全景
    paper_innovations: list[str] = field(default_factory=list)   # 7 创新点为什么聪明
    experiments: list[str] = field(default_factory=list)         # 8 实验证明了什么
    limitations: list[str] = field(default_factory=list)         # 9 不足与局限
    scenario: str = ""                                           # 10 贡献与影响
    related_directions: list[str] = field(default_factory=list)  # （暂不展示）
    reading_guide: str = ""                                      # 11 重点阅读指引
    full_report: str = ""                                        # [v3] 完整动态叙事报告（主产物）
    verification_notes: str = ""                                 # [v3] Stage4 审核说明（事实/逻辑/教学检查结论）


# ═══════════════════════════════════════
# 8 main.py 汇总：最终结果
# ═══════════════════════════════════════
@dataclass
class FinalResult:
    """最终输出：整个任务的成果，frontend 直接展示它"""
    mode: str                                   # 任务模式（前端按它选渲染方式）
    question: str                               # 用户问题
    report: Report | None = None                # 报告（调研=图文报告，多篇对比=综述报告；单篇精读为空）
    suggestion: Suggestion | None = None        # 建议（方向/论文+行动（分析模式留空）；单篇精读为空）
    deep_read: DeepReadReport | None = None     # 单篇精读八段式报告（仅单篇论文模式有值）
    warning_flags: list[str] = field(default_factory=list)  # 验证未通过的警告标签（展示"请人工复核"）


# ═══════════════════════════════════════
# 9 LangGraph 状态机的"黑板"：所有 Agent 往这里读写
# ═══════════════════════════════════════
@dataclass
class FlowState:
    """LangGraph 状态机全量状态（HUMAN_IN_LOOP 的重试字段都在这）"""
    mode: str = MODE_SURVEY                                        # 任务模式
    question: str = ""                                             # 用户问题（调研模式）或任务描述
    papers: list[PaperSource] = field(default_factory=list)        # 论文列表（论文分析模式用）
    papers_relevant: bool = True                                   # 论文是否彼此相关（paper模式≥2篇，相关性检查节点判断）
    relevance_note: str = ""                                       # 相关性说明：低相关时写明哪些论文差异大（展示给用户/警告用）
    outline: Outline | None = None                                 # 规划师产出的大纲
    outline_retry_count: int = 0                                   # 大纲重试次数（0-1，满1进 HUMAN_IN_LOOP）
    evidences: list[Evidence] = field(default_factory=list)        # 所有研究员产出的证据汇总
    report: Report | None = None                                   # 写作者产出的报告
    report_retry_count: int = 0                                    # 报告重试次数（0-1，满1进 HUMAN_IN_LOOP）
    verifier_feedback: str = ""                                    # 验证员修改意见（注入 Writer 重写 prompt）
    warning_flags: list[str] = field(default_factory=list)         # 未通过项警告标签（1次不过时记录）
    suggestion: Suggestion | None = None                           # 建议 Agent 产出
    human_feedback: str = ""                                       # 用户在 HUMAN_IN_LOOP 输入的修正意见
    final: FinalResult | None = None                               # 最终结果（给前端展示）


# ═══════════════════════════════════════
# 10 记忆层数据结构（不参与主流程，memory 模块单独用）
# ═══════════════════════════════════════
@dataclass
class SurveyRecord:
    """调研模式的历史记录（存 Milvus memory_survey 向量集合）"""
    record_id: str                          # 记录唯一编号（向量主键）
    topic: str                              # 研究主题（生成 embedding 的文本来源）
    report_summary: str                     # 完整报告摘要（向量的 payload）
    embedding: list[float] = field(default_factory=list)  # 主题 embedding 向量
    created_at: str = ""                    # 创建时间


@dataclass
class PaperRecord:
    """论文模式的历史记录（存 Milvus memory_paper 向量集合）"""
    record_id: str                          # 记录唯一编号（向量主键）
    topic: str                              # 论文主题（生成 embedding 的文本来源）
    analysis_conclusion: str                # 分析结论（向量的 payload）
    paper_titles: list[str] = field(default_factory=list)  # 本次涉及哪些论文
    embedding: list[float] = field(default_factory=list)    # 主题 embedding 向量
    created_at: str = ""                    # 创建时间


@dataclass
class PaperAnalysis:
    """单篇论文的精读历史（存 SQLite paper_analysis 表）：分析过一次就记录，
    之后用户用 PDF/链接走同一套标识（make_paper_key）就能搜回历史精读结论"""
    paper_key: str                          # 论文指纹：arXiv ID/DOI 优先，其次链接，最后标题
    paper_title: str                        # 论文标题
    link: str = ""                          # 论文链接
    summary: str = ""                       # 精读结论摘要
    created_at: str = ""                    # 分析时间


@dataclass
class TaskRecord:
    """任务日志：记录一次任务的耗时和成本（存 SQLite task_log 表）"""
    task_id: str                            # 任务唯一编号
    mode: str = MODE_SURVEY                 # 任务模式
    question: str = ""                      # 用户问题/任务描述
    duration_sec: float = 0.0               # 任务耗时（秒）
    token_usage: int = 0                    # token 消耗
    cost: float = 0.0                       # 预估成本（元）
    accepted: bool = False                  # 用户最终是否采纳报告
    created_at: str = ""                    # 完成时间


@dataclass
class UserConfig:
    """本机持久化配置（存 SQLite user_config 表，本地单例，不区分用户）。
    对齐大型软件的通用做法（如 VS Code 的 settings.json）：设置属于"这台机器"，跟账号无关"""
    preferred_mode: str = MODE_SURVEY       # 上次使用的模式，每次进入页面自动恢复
    theme: str = "light"                    # 界面主题（预留，后续扩展用）
