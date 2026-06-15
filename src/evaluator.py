"""
RAGAS 评估器（无梯子环境）
==========================
核心要点：
1. Embedding 模型通过 ModelScope（魔搭）下载，不走 HuggingFace Hub
2. 所有评估指标使用 DeepSeek API 作为 LLM
3. 评估前检查本地缓存，模型不存在时给出清晰提示
"""

import os
import sys
from datasets import Dataset

# ── 1. 获取 Embedding 模型（ModelScope 镜像，无需梯子） ──


def get_embeddings():
    """
    从 ModelScope 下载 Embedding 模型，避免访问 HuggingFace Hub。
    下载后缓存在 ./embeddings_cache，下次直接加载不重复下载。
    """
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings_cache")

    # ── 方案A：用 ModelScope 下载到本地缓存 ──
    try:
        from modelscope import snapshot_download

        print(f"[ModelScope] 正在检查/下载模型: {model_id}")
        local_path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision="master",
        )
        print(f"[ModelScope] 模型就绪: {local_path}")

        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=local_path,               # 使用本地路径，不触发 HuggingFace 下载
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings

    except ImportError:
        print("[ERROR] 未安装 modelscope，Embedding 模型下载失败。")
        print("   请运行以下命令后重试：")
        print("   pip install modelscope")
        print("   modelscope download --model_id sentence-transformers/all-MiniLM-L6-v2")
        sys.exit(1)

    except Exception as e:
        print(f"[ModelScope] 下载失败，尝试使用 HuggingFace 国内镜像... ({type(e).__name__})")

        # ── 方案B：降级使用 HuggingFace 国内镜像 ──
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=model_id,
                cache_folder=cache_dir,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[HF-Mirror] Embedding 模型加载成功（使用 hf-mirror.com）")
            return embeddings

        except Exception as e2:
            print(f"[ERROR] Embedding 模型加载完全失败: {e2}")
            print("   请手动安装并下载模型：")
            print("   1. pip install modelscope")
            print("   2. modelscope download --model_id sentence-transformers/all-MiniLM-L6-v2")
            sys.exit(1)


# ── 2. 组装 Ragas 评估器 ──


def run_evaluation(dataset: Dataset, llm, embeddings) -> dict:
    """
    执行 Ragas 评估，包含三个指标：
    - faithfulness：答案是否忠实于标准答案（基于 LLM）
    - answer_relevancy：答案是否切题（基于 Embedding + LLM）
    - answer_correctness：答案是否正确（基于 LLM + 标准答案比对）

    所有 LLM 调用走 DeepSeek，所有 Embedding 走本地模型。
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    # answer_correctness 在 ragas 0.4.x 中的位置可能有变化，做兼容处理
    try:
        from ragas.metrics import answer_correctness
    except ImportError:
        from ragas.metrics import factual_correctness as answer_correctness

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    # 限制并发 worker 数量，避免 DeepSeek API 超时
    # 默认 max_workers=16 会导致 15 个任务同时砸向 API
    # CI 环境（GitHub Actions 在美国）到 DeepSeek（国内）延迟高，
    # 自动降为单 worker + 5 分钟超时，避免跨国网络超时雪崩
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    max_w = 1 if is_ci else 3
    timeout_s = 300 if is_ci else 180
    if is_ci:
        print("[CI 检测] 已启用单 worker 模式，避免跨国超时")
    run_config = RunConfig(max_workers=max_w, timeout=timeout_s)

    print("\n[RAGAS] 开始评估...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=run_config,
    )
    print("[RAGAS] 评估完成")

    return result
