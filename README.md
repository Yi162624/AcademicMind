# AcademicMind · 多智能体深度研究助手

基于 **LangGraph + 多 Agent** 的学术深度研究系统。输入一个研究问题（或一组论文），自动完成**领域调研报告 + 研究方向建议 + 核心论文推荐**；单篇论文走独立的"精读 skill"（推理知识库 + 一次性生成），输出带出处锚点与批判性分析的深度精读报告。

> 纯 API 架构：本地不加载任何大模型权重，所有模型能力通过 DeepSeek / Qwen-VL API 调用，8GB 显存即可运行。

## 核心特性

- **5 Agent 流水线**：Planner（规划师）→ Researcher（研究员）→ Writer（写作者）→ Verifier（验证员）→ Advisor（建议 Agent）
- **证据信息流（可追溯）**：Planner 产出研究任务单（ResearchTask）驱动 Researcher；Researcher 提炼结构化证据（findings 结论 + claims 可追溯论断 + sources 来源）；Writer 按证据写并在正文标注 `[E#n]`；Verifier 沿 `[E#n] → Evidence → Claim → Source` 链核对报告真假
- **编号系统分配**：任务（T1/T2）、证据（E1/E2）、来源（S1/S2）编号全部由系统分配，LLM 只能引用、不能自造，杜绝"引用不存在的证据"
- **人类在环（HUMAN_IN_LOOP）**：大纲生成后暂停交用户确认（可反复修改）；报告校验失败且自动重试耗尽后由用户拍板
- **稳健性**：搜索词先经 LLM 精炼成英文关键词再搜（arXiv 不认整句中文）；搜索接口限流（429）时指数退避重试 + 随机抖动错峰
- **单篇精读 skill**：Docling 解析 PDF + Qwen-VL 理解图表 → 推理知识库（还原作者推理链）→ 一次性生成整篇报告，内置防编造锚点约束与批判性分析

## 快速开始

### 1. 环境

- Python ≥ 3.13

```bash
conda env create -f environment.yml   # 精确复现（推荐）
conda activate academicmind
# 或：pip install -r requirements.txt（宽松版本）
```

### 2. 配置 API Key

```bash
cp .env.example .env   # 填入你的 DeepSeek / Qwen-VL API Key
```

`.env` 已被 .gitignore 排除，不会上传。两个 Key 都在 [DeepSeek 开放平台](https://platform.deepseek.com/) 和 [阿里云百炼](https://bailian.console.aliyun.com/) 申请。

### 3. 运行

```bash
# 调研模式：一个研究问题 → 图文调研报告 + 方向建议 + 核心论文
python main.py --mode survey --question "大模型在医疗影像的应用"

# 论文分析模式：多篇论文 → 对比综述
python main.py --mode paper --papers <论文链接1> <论文链接2> ...

# 论文分析模式：单篇论文 → 精读 skill 报告（自动分流，不经多 Agent）
python main.py --mode paper --papers <论文链接>
```

正式使用由 Web 前端（React + FastAPI）调用 `run_task` / `resume_task` 驱动。一键启动：`python app.py`（自动拉起前端 5173 + 后端 8000），浏览器访问 http://localhost:5173。

## 两种模式 + 单篇精读

| 输入 | 走哪条路 | 输出 |
| :--- | :--- | :--- |
| 一个研究问题 | 5 Agent 流水线（调研模式） | 领域报告 + 研究方向 + 核心论文 + 行动清单 |
| ≥2 篇论文 | 5 Agent 流水线（分析模式） | 对比综述报告 + 推荐论文 + 行动清单 |
| 1 篇论文 | 单篇精读 skill | 深度精读报告（背景/方法/实验/批判性分析/局限…） |

## 项目结构

```
AcademicMind/
├── main.py                  # LangGraph 状态机编排：run_task / resume_task（HUMAN_IN_LOOP）
├── agent/                   # 5 个 Agent（规划/研究/写作/验证/建议）
├── core/                    # 数据契约 schemas、统一模型调用 llm_client、JSON 清洗 json_utils、日志 logger
├── memory/                  # SQLite（模式偏好/精读历史/任务日志/图片记忆）+ Milvus Lite 向量记忆
├── multimodal/              # 图片理解（Qwen-VL）、图文检索（BGE + VLM 精排）、视觉工作记忆
├── skills/paper_deep_read/  # 单篇精读 skill（Docling 解析 + 推理知识库 + 一次性生成）
├── frontend/                # Web 前端：FastAPI 网关（server.py）+ React 对话界面（web/）+ 一键启动 app.py
```

## 技术栈

| 模块 | 选型 |
| :--- | :--- |
| 多智能体框架 | LangGraph（interrupt()/Command 实现 HUMAN_IN_LOOP） |
| 文本模型 | DeepSeek API（所有文本 Agent 共用） |
| 视觉模型 | Qwen-VL API（图片/图表理解） |
| PDF 解析 | Docling（IBM 开源，文本 + 表格 + 图片一体化） |
| 向量记忆 | Milvus Lite + BGE-small embedding |
| 结构化存储 | SQLite |

## 注意事项

- 本地不部署任何 LLM/VLM，所有模型能力走 HTTP API；本地仅运行前端、记忆模块、PDF 解析与缓存
- 任务日志记录每次任务的耗时、token 消耗与预估花费（SQLite `task_log` 表）
- `.env`、`data/`、`logs/` 均被 .gitignore 排除，请勿手动加入版本控制

## License

MIT
