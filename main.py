"""
LLM-Eval-Bench —— AI 自动化评估工具（无梯子版）
================================================
流程：读取测试题 → DeepSeek 答题 → Ragas 评分 → 打印成绩单

所有网络请求均走国内可访问的地址：
- DeepSeek API: api.deepseek.com（国内直连）
- Embedding 模型: ModelScope 镜像下载（不走 HuggingFace）
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

# Windows 中文终端强制 UTF-8，避免 emoji/特殊字符乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录加入路径，确保 src 模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

from src.llm_client import get_deepseek_llm, generate_answer
from src.evaluator import get_embeddings, run_evaluation

# 加载 .env：先在项目目录找，再去上级目录（C:\Users\Sophie）找
env_path = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# ─────────────────────────────────────────────────
# 1. 加载测试数据
# ─────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "test_cases.json")

if not os.path.exists(DATA_FILE):
    print(f"[ERROR] 测试数据文件不存在: {DATA_FILE}")
    sys.exit(1)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    test_cases = json.load(f)

print(f"[数据] 已加载 {len(test_cases)} 道测试题")

# ─────────────────────────────────────────────────
# 2. 初始化 DeepSeek LLM
# ─────────────────────────────────────────────────
print("[LLM] 正在初始化 DeepSeek 客户端...")
llm = get_deepseek_llm(temperature=0.1, timeout=120)

# ─────────────────────────────────────────────────
# 3. AI 答题阶段
# ─────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print("  第一阶段：DeepSeek 答题中...")
print(f"{'─' * 50}")

questions = []
ground_truths = []
answers = []

for i, item in enumerate(test_cases, 1):
    q = item["question"]
    gt = item["ground_truth"]

    print(f"\n[{i}/{len(test_cases)}] {q[:50]}...")
    ans = generate_answer(llm, q)

    questions.append(q)
    ground_truths.append(gt)
    answers.append(ans)

    print(f"   → 答案长度: {len(ans)} 字符")

# ─────────────────────────────────────────────────
# 4. 构建 Ragas Dataset
# ─────────────────────────────────────────────────
# faithfulness 需要 contexts（用标准答案作为参考上下文）
# answer_correctness 需要 ground_truth
contexts = [[gt] for gt in ground_truths]

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths,
    "contexts": contexts,            # faithfulness 依赖此字段
})

print(f"\n[数据] Ragas Dataset 构建完成，共 {len(dataset)} 条")

# ─────────────────────────────────────────────────
# 5. 加载 Embedding 模型（ModelScope 镜像）
# ─────────────────────────────────────────────────
print("[Embedding] 正在加载模型（通过 ModelScope 镜像，无需梯子）...")
embeddings = get_embeddings()

# ─────────────────────────────────────────────────
# 6. Ragas 评估
# ─────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print("  第二阶段：RAGAS 评估打分中...")
print(f"{'─' * 50}")

try:
    result = run_evaluation(dataset, llm, embeddings)
except Exception as e:
    print(f"[ERROR] 评估失败: {e}")
    print(f"   错误类型: {type(e).__name__}")
    print("   请检查：")
    print("   1. DEEPSEEK_API_KEY 是否有效")
    print("   2. Embedding 模型是否已下载")
    print("   3. 网络是否正常（api.deepseek.com 是否可访问）")
    sys.exit(1)

# ─────────────────────────────────────────────────
# 7. 打印成绩单表格
# ─────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("                    >> LLM 评估成绩单 <<")
print(f"{'=' * 72}")

# 将结果转为 DataFrame
df = result.to_pandas()

# 添加题号和问题摘要列
df.insert(0, "题号", list(range(1, len(test_cases) + 1)))
df.insert(1, "问题摘要", [q[:30] + "…" if len(q) > 30 else q for q in questions])

# 格式化小数显示
metric_cols = [c for c in df.columns if c not in ("题号", "问题摘要")]
for col in metric_cols:
    df[col] = df[col].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v))

print(df.to_string(index=False))

# 打印平均分（只计算数值列）
print(f"\n{'─' * 72}")
for col in metric_cols:
    vals = result.to_pandas()[col]
    try:
        avg = vals.mean()
        print(f"  {col} 平均分: {avg:.4f}")
    except (TypeError, AttributeError):
        pass  # 跳过非数值列
print(f"{'=' * 72}")

print("\n[完成] 评估完成!")
print("   指标说明：")
print("   • faithfulness     — 答案是否忠实于标准答案")
print("   • answer_relevancy — 答案是否紧扣问题")
print("   • answer_correctness — 答案与标准答案比对是否正确")
