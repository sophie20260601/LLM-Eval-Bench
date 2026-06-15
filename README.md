# LLM-Eval-Bench

AI 回答质量自动化评估工具 —— 基于 RAGAS 的三维评估平台。

[![CI](https://github.com/sophie20260601/LLM-Eval-Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/sophie20260601/LLM-Eval-Bench/actions)

## 功能

RAGAS 框架驱动的 LLM 评估流水线，覆盖 3 个领域 30 道标准测试题：

| 指标 | 评估维度 | 方式 |
|------|---------|------|
| Faithfulness | 答案是否忠实于标准答案 | LLM 判断 |
| Answer Relevancy | 答案是否切题 | Embedding 相似度 |
| Answer Correctness | 答案是否正确 | LLM 逐条比对 |

流程：读取测试题 → DeepSeek 答题 → RAGAS 评分 → 输出成绩单

## 特点

- **国内直连**：DeepSeek API（答题 + 评估）+ ModelScope 镜像（Embedding 模型），无需翻墙
- **零配置**：`pip install` 后创建 `.env` 填入 API Key 即可启动
- **CI/CD**：push 自动验证数据完整性，手动触发全量评估

## 快速开始

```bash
git clone https://github.com/sophie20260601/LLM-Eval-Bench.git
cd LLM-Eval-Bench
pip install -r requirements.txt
# 编辑 .env 填入 DEEPSEEK_API_KEY
python main.py
```

## CI/CD

- **Push 自动触发**：数据格式验证 + 模块导入检查（无需 API Key）
- **手动触发**：[Actions 页面](https://github.com/sophie20260601/LLM-Eval-Bench/actions) → Run workflow → 勾选 Run full evaluation → 运行全量评估

## 技术栈

Python · RAGAS · LangChain · DeepSeek API · ModelScope · Sentence-Transformers · HuggingFace Datasets · GitHub Actions
