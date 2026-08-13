---
name: paper-deep-read
description: 解析并深度精读单篇学术论文（arXiv 链接、PDF 文件或 URL），还原作者的研究推理链（为什么研究 → 为什么这样设计 → 实验如何证明），输出面向未读原文读者的大白话精读报告。当用户提供单篇论文并要求精读、讲解、分析、总结，或提到"精读""读懂这篇论文""讲一下这篇论文"时使用。
---

# 论文精读 Skill（paper-deep-read）

独立能力模块：输入 1 篇论文，还原其"研究推理链"，输出**动态叙事式精读报告**。不经 5 Agent 协作，由 main.py 的 `_run_deep_read()` 直接调用。

核心目标不是写摘要，而是让没读过原文的小白看懂：为什么研究、旧方法为何失败、作者想通了什么、为什么这样设计、实验如何证明。

## 何时使用

- 用户输入**恰好 1 篇**论文（arXiv 链接 / 本地 PDF / 其他 URL）
- ≥2 篇论文时走 5 Agent 对比流程，**不调用本 skill**

## 工作流（v3 五阶段）

按顺序执行，每步失败按"降级策略"处理，不中断整条链路：

```
① 解析输入（scripts/resolve_input.py）
   arXiv 链接 → 构造 PDF 地址下载；本地 PDF → 直接用；其他 URL → 下载
② 解析 PDF（scripts/parse_pdf.py）
   Docling 统一解析 → 全文 Markdown + 图表 PNG
③ 理解图表（有图才调）
   Qwen-VL 逐张生成结构化描述
④ 五阶段 LLM 编排（skill.py）
   Stage0 结构地图 → Stage1 推理知识库(核心) → Stage2 教学规划
   → Stage3 动态分章(并行) → Stage4 定点修正
⑤ 存精读历史（SQLite paper_analysis 表，非关键路径）
```

### 各阶段职责

| 阶段 | 做什么 | 失败怎么办 |
| :--- | :--- | :--- |
| Stage0 结构地图 | 画出章节作用 + 关键图表 + 论证顺序 | 空地图继续 |
| Stage1 推理知识库 | 还原推理链（research_story / design_reasoning / concept_knowledge / evidence_chain / experiment_reasoning） | token 超限先压缩重试，最多 3 次；仍失败返回兜底报告 |
| Stage2 教学规划 | 决定讲什么：核心突破 / 误区 / 必讲 / 可跳过 / 动态章节 | 空规划 + 默认 8 段章节继续 |
| Stage3 动态分章 | 按章节列表并行生成每节内容（每章 5s/10s/20s 退避重试） | 单章失败降级占位符，线程池级错误才兜底 |
| Stage4 定点修正 | 事实/逻辑/教学三查，只重写有问题的章节 | 用 Stage3 原稿成稿 |

## 输出契约

`DeepReadReport`（定义见 `core/schemas.py`）：
- `full_report`（主产物）：动态叙事 Markdown，章节由教学规划决定，每节按「问题 → 设计 → 原因 → 证据」展开
- `verification_notes`：Stage4 审核说明
- 其余固定字段（`one_line_summary` / `problem_background` / `core_idea` 等）：从推理知识库抽取，兼容结构化读取

## 质量校验清单

生成报告前，对照 [references/quality-checklist.md](references/quality-checklist.md) 逐项自检，重点：不编造、推理链连贯、小白能看懂。

## 降级策略

| 步骤 | 失败原因 | 降级行为 |
| :--- | :--- | :--- |
| PDF 下载 | 链接无效 / 网络超时 | 兜底报告（"无法获取论文全文"） |
| Docling 解析 | PDF 损坏 / 全是扫描图 | 兜底报告（"未提取到有效文本"） |
| Stage0 | LLM / JSON 失败 | 空地图继续 |
| Stage1 | token 超限 / JSON 失败 | 压缩重试最多 3 次，仍失败兜底 |
| Stage2 | LLM 失败 | 空规划 + 默认章节 |
| Stage3 | 单章重试失败 | 占位符；线程池级错误兜底 |
| Stage4 | LLM / JSON 失败 | 用 Stage3 原稿 |
| 图解 | 单张失败 | 记录降级描述 |
| 历史保存 | SQLite 失败 | 记 warning，不影响主流程 |

## 文件结构

- `SKILL.md`：本文件（技能说明书）
- `skill.py`：薄编排（LLM 阶段调用 + 状态流转）
- `prompts.py`：全部 LLM 提示词模板（Python 常量，需 `.replace()` 填参）
- `scripts/resolve_input.py`：输入分流 + PDF 下载 + 缓存（确定性操作）
- `scripts/parse_pdf.py`：Docling 解析 + 图表导出（确定性操作）
- `references/quality-checklist.md`：输出质量校验清单
