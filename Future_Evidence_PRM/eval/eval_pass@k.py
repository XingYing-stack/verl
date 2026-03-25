from datasets import load_dataset
import os
import argparse

import pandas as pd
from ..utils import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent tree sampling for SearchR1-like data")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/nfsdata/fanshengda/verl/Future_Evidence_PRM/dataset/hotpotqa_validation.parquet",
    )
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent questions")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://localhost:8888/v1",
        help="OpenAI-compatible base URL for LLM server (e.g., SGLang)",
    )
    parser.add_argument("--llm_model", type=str, default="Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument("--llm_temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--llm_max_tokens", type=int, default=4096, help="Max new tokens per completion")
    parser.add_argument("--llm_timeout_s", type=int, default=120, help="LLM request timeout seconds")
    parser.add_argument(
        "--retriever_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Retriever HTTP endpoint (Search-R1-like)",
    )
    parser.add_argument(
        "--near_miss_prm_url",
        type=str,
        default="http://localhost:6001/v1",
        help="OpenAI-compatible base URL for LLM server (e.g., SGLang)",
    )

    parser.add_argument("--retriever_topk", type=int, default=3, help="Retriever topk")
    parser.add_argument("--retriever_timeout_s", type=int, default=30, help="Retriever timeout seconds")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="dada",
        help="API key for OpenAI-compatible LLM server Authorization header.",
    )
    parser.add_argument("--k", type=int, default=8, help="pass@k")

    args = parser.parse_args()

    df = pd.read_parquet("./dataset/hotpotqa_validation.parquet")


if __name__ == "__main__":
    main()
