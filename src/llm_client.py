"""
DeepSeek LLM 客户端（国内直连，无需代理）
==========================================
使用 langchain_openai 封装 DeepSeek API，兼容 OpenAI SDK。
强制 n=1 因为 DeepSeek 不支持多版本生成（n>1）。
"""

import os
import sys
from langchain_openai import ChatOpenAI


class DeepSeekChatOpenAI(ChatOpenAI):
    """DeepSeek 专用 LLM，强制 n=1，拦截 ragas 传入的 n>1 请求。

    ragas 的 answer_correctness 指标内部固定请求 n=3 次生成，
    DeepSeek API 只支持 n=1，直接拦截避免 400 错误。
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["n"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_deepseek_llm(temperature: float = 0.1, timeout: int = 30) -> DeepSeekChatOpenAI:
    """
    创建并返回 DeepSeek LLM 实例。

    关键配置：
    - base_url 使用 api.deepseek.com，国内直连无需代理
    - api_key 从环境变量 DEEPSEEK_API_KEY 读取
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 DEEPSEEK_API_KEY")
        print("   请在 .env 文件中配置：DEEPSEEK_API_KEY=sk-xxx")
        print("   或在当前终端执行：set DEEPSEEK_API_KEY=sk-xxx")
        sys.exit(1)

    return DeepSeekChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
    )


def generate_answer(llm: DeepSeekChatOpenAI, question: str) -> str:
    """使用 DeepSeek 生成单个问题的回答"""
    response = llm.invoke(
        f"你是 Python 编程专家。请详细回答以下问题，给出准确、完整的答案：\n\n{question}"
    )
    return response.content.strip()
