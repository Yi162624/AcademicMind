# 临时测试：单篇论文精读
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.schemas import PaperSource
from skills.paper_deep_read.skill import run_deep_read

paper = PaperSource(
    title="Attention Is All You Need",
    link="https://arxiv.org/abs/1706.03762",
)

print("正在精读：", paper.title)
report = run_deep_read(paper)

# 打印论文基本信息
pi = report.paper_info
print("\n" + "=" * 70)
print("  论文基本信息")
print("=" * 70)
print(f"标题: {pi.title}")
print(f"作者: {pi.authors}")
print(f"年份: {pi.year}")
print(f"链接: {pi.link}")

# 打印 full_report 主产物（AI 直接生成的教学叙事报告）
print("\n" + "=" * 70)
print("  精读报告（full_report 主产物）")
print("=" * 70)
print(report.full_report)

# 打印定点修正记录（Stage4 检查出的问题，可选）
if report.verification_notes:
    print("\n" + "=" * 70)
    print("  定点修正记录")
    print("=" * 70)
    print(report.verification_notes)
