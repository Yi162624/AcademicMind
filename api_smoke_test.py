# api_smoke_test.py
# 作用：后端 API 接口层冒烟测试脚本（模拟真人逐项测试）
#       覆盖除"论文精读 skill"外的全部接口，只测到"任务启动返回挂起"为止，不跑完整 5-Agent/精读流程
# 运行：python api_smoke_test.py
# 依赖：requests、pypdf（后端同款依赖）

import io
import sys

import requests

# Windows 控制台默认 GBK，先切成 UTF-8 免得打印中文报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = "http://localhost:8000"

# 测试统计：每项记 (名字, 是否通过, 备注)
RESULTS = []


def report(name: str, ok: bool, note: str = ""):
    """记一条测试结果：名字 + 通过与否 + 补充说明"""
    RESULTS.append((name, ok, note))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}  {note}")


def expect(cond: bool, note: str) -> bool:
    """断言辅助：条件成立返回 True，不成立打印原因返回 False"""
    if not cond:
        print(f"      -> 期望不符：{note}")
    return cond


def post_json(path: str, body: dict, timeout: int = 120):
    """POST JSON，返回 (状态码, 解析后的 dict)；异常时返回 (0, {})"""
    try:
        r = requests.post(BASE + path, json=body, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {}
    except Exception as e:
        return 0, {"_error": str(e)}


# ─────────────────────────────────────────────
# 测试 1：健康检查
# ─────────────────────────────────────────────
def test_health():
    """健康检查：/api/health 应返回 {"ok": True}，前端靠它判断后端起没起"""
    try:
        r = requests.get(BASE + "/api/health", timeout=5)
        ok = r.status_code == 200 and r.json().get("ok") is True
        report("健康检查 /api/health", ok, f"HTTP {r.status_code}")
    except Exception as e:
        report("健康检查 /api/health", False, str(e))


# ─────────────────────────────────────────────
# 测试 2：模式偏好读写（SQLite 持久化）
# ─────────────────────────────────────────────
def test_config():
    """模式偏好读写：先存 paper，再读应回 paper，最后恢复 survey 免得影响下次会话"""
    code, data = post_json("/api/config", {"mode": "paper"}, timeout=5)
    ok1 = expect(code == 200 and data.get("mode") == "paper", f"存 paper 失败：{data}")
    r = requests.get(BASE + "/api/config", timeout=5)
    data2 = r.json()
    ok2 = expect(r.status_code == 200 and data2.get("mode") == "paper", f"读回失败：{data2}")
    post_json("/api/config", {"mode": "survey"}, timeout=5)  # 还原成默认
    report("模式偏好读写 /api/config", ok1 and ok2, f"存/读均返回 mode=paper")


# ─────────────────────────────────────────────
# 测试 3：意图分类（simple / research / other 三种都验）
# ─────────────────────────────────────────────
def test_classify():
    """意图分类：闲聊应判 simple，学术问题应判 research，编程求助应判 other"""
    cases = [
        ("你好，在吗", "simple"),
        ("大模型在医疗影像诊断中的应用有哪些主流方法", "research"),
        ("帮我把这段 Python 代码的 bug 找出来", "other"),
    ]
    all_ok = True
    for text, want in cases:
        code, data = post_json("/api/classify", {"text": text}, timeout=30)
        got = data.get("type", "")
        ok = expect(code == 200 and got == want, f"输入「{text}」期望 {want} 实际 {got}（{data.get('reason','')}）")
        all_ok = all_ok and ok
        print(f"      「{text[:20]}…」→ {got}（{data.get('reason','')}）")
    report("意图分类 /api/classify", all_ok, "simple/research/other 三类")


# ─────────────────────────────────────────────
# 测试 4：简单问答直答（一次 LLM）
# ─────────────────────────────────────────────
def test_simple_chat():
    """简单问答：问常识问题，应返回非空的中文回答"""
    code, data = post_json("/api/chat/simple", {"text": "什么是深度学习，用两句话解释"}, timeout=60)
    ans = data.get("answer", "")
    ok = expect(code == 200 and len(ans) > 0, f"回答为空或接口错误：HTTP {code} {data}")
    if ans:
        print(f"      回答：{ans[:80]}…")
    report("简单问答 /api/chat/simple", ok, f"回答 {len(ans)} 字")


# ─────────────────────────────────────────────
# 测试 5：PDF 上传校验（正常 / 垃圾 / 截断 三连）
# ─────────────────────────────────────────────
def _make_valid_pdf() -> bytes:
    """用 pypdf 造一个 1 页的合法 PDF，当"正常文件"测试用"""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_upload():
    """上传接口：合法 PDF 应 200 且返回 title；垃圾字节应 400；截断 PDF 应 400 且提示不完整"""
    valid = _make_valid_pdf()

    # 1) 合法 PDF
    r = requests.post(BASE + "/api/upload", files={"file": ("demo.pdf", valid, "application/pdf")}, timeout=15)
    ok1 = expect(r.status_code == 200 and r.json().get("title") == "demo", f"合法 PDF 上传失败：{r.status_code} {r.text[:100]}")
    if r.status_code == 200:
        print(f"      合法 PDF -> title={r.json().get('title')}")

    # 2) 垃圾字节（不是 PDF）
    r2 = requests.post(BASE + "/api/upload", files={"file": ("fake.txt", b"this is not a pdf at all", "application/pdf")}, timeout=15)
    ok2 = expect(r2.status_code == 400 and "不是有效的 PDF" in r2.json().get("detail", ""), f"垃圾文件应 400：{r2.status_code} {r2.text[:100]}")

    # 3) 截断 PDF（有 %PDF- 头但被砍断，模拟下载中断）
    broken = valid[: len(valid) // 2]  # 只留前半截
    r3 = requests.post(BASE + "/api/upload", files={"file": ("broken.pdf", broken, "application/pdf")}, timeout=15)
    ok3 = expect(r3.status_code == 400 and "不完整" in r3.json().get("detail", ""), f"截断 PDF 应 400 提示不完整：{r3.status_code} {r3.text[:100]}")

    report("PDF 上传校验 /api/upload", ok1 and ok2 and ok3, "正常 200 / 垃圾 400 / 截断 400")


# ─────────────────────────────────────────────
# 测试 6：调研任务启动（只验到"挂起等大纲确认"，不继续跑）
# ─────────────────────────────────────────────
def test_task_start_survey():
    """调研任务启动：给一个真实调研问题，期望返回 interrupt(stage=outline) 让前端弹确认卡片。
    只验启动这一跳，不 resume，不跑完整流程，控制耗时"""
    body = {
        "mode": "survey",
        "question": "大语言模型在金融风控领域的应用现状综述",
        "papers": [],
        "thread_id": "smoke-test-001",
    }
    code, data = post_json("/api/task/start", body, timeout=180)
    if code == 0:
        report("调研启动 /api/task/start", False, f"请求异常：{data.get('_error','')}")
        return
    rtype = data.get("type", "")
    stage = (data.get("data") or {}).get("stage", "") if isinstance(data.get("data"), dict) else ""
    note = f"type={rtype} stage={stage} thread_id={data.get('thread_id','')}"
    # 允许两种结果：挂起等确认（正常），或直接出结果（大纲不需确认）；都不算错
    ok = code == 200 and rtype in ("interrupt", "result")
    if rtype == "result":
        # result 说明大纲直接被接受跑到底了，耗时较长；也视作通过但注明
        note += "（直接出 result，未走确认）"
    report("调研启动 /api/task/start", ok, note)


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AcademicMind 后端 API 冒烟测试（不含论文精读 skill）")
    print("=" * 60)
    test_health()
    test_config()
    test_classify()
    test_simple_chat()
    test_upload()
    test_task_start_survey()
    print("-" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"结果汇总：{passed}/{len(RESULTS)} 通过")
    for name, ok, note in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {note}")
    sys.exit(0 if passed == len(RESULTS) else 1)
