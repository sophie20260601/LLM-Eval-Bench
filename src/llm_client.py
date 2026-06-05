"""
DeepSeek LLM 客户端（国内直连，无需代理）
==========================================
使用 langchain_openai 封装 DeepSeek API，兼容 OpenAI SDK。
"""

import os
import sys
from langchain_openai import ChatOpenAI


def get_deepseek_llm(temperature: float = 0.1, timeout: int = 30) -> ChatOpenAI:
    """
    创建并返回 DeepSeek LLM 实例。

    关键配置：
    - base_url 使用 api.deepseek.com，国内直连无需代理
    - api_key 从环境变量 DEEPSEEK_API_KEY 读取
    - 所有请求设置超时防止卡死
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 DEEPSEEK_API_KEY")
        print("   请在 .env 文件中配置：DEEPSEEK_API_KEY=sk-xxx")
        print("   或在当前终端执行：set DEEPSEEK_API_KEY=sk-xxx")
        sys.exit(1)

    return ChatOpenAI(
        model="deepseek-chat",                         # DeepSeek 对话模型
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",        # 国内直连，兼容 OpenAI SDK 格式
        temperature=temperature,
        timeout=timeout,                               # 150s 超时（Ragas 并发调用多）
        max_retries=2,
    )


def generate_answer(llm: ChatOpenAI, question: str) -> str:
    """使用 DeepSeek 生成单个问题的回答，超时 30 秒"""
    response = llm.invoke(
        f"你是 Python 编程专家。请详细回答以下问题，给出准确、完整的答案：\n\n{question}"
    )
    return response.content.strip()
