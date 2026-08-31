# JSON 解析工具模块
# 作用：把 LLM 返回的"脏"文本清洗解析成 Python dict。
#       LLM 输出经常不干净（代码块包裹、前后夹废话、尾逗号），这里做多层清洗尽量救回来。
# 位置：core/json_utils.py，被 agent/ 下的 5 个 Agent 和 skills/paper_deep_read/skill.py 共用，
#       避免每个模块各写一份重复的解析逻辑。

import json
import re
from typing import Optional


def parse_json(text: str) -> Optional[dict]:
    """把 LLM 返回的文本解析成 JSON dict。
    LLM 输出经常不干净（代码块包裹、前后夹废话、尾逗号），这里做多层清洗，
    尽量把"近似 JSON"救回来，实在不行才返回 None"""
    if not text:
        return None
    text = text.strip()

    # 第一层：剥掉 markdown 代码块（处理 ```json ... ``` 或 ``` ... ```）
    # 用 find 定位 ```，而不是假设它一定在开头/结尾
    fence = text.find("```")
    if fence != -1:
        text = text[fence + 3:]                              # 从 ``` 之后开始截
        text = re.sub(r"^[a-zA-Z]+\s*\n", "", text, count=1)  # 去掉 "json" 这类语言标记
        end = text.rfind("```")                              # 去掉结尾的 ```
        if end != -1:
            text = text[:end]
        text = text.strip()

    # 第二层：截取第一个 { 到最后一个 } 之间的内容（应对 LLM 前后夹带的解释文字）
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    # 第三层：去掉尾逗号（JSON 标准不允许 {..,} 或 [..,]）
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 第四层：直接解析，失败返回 None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
