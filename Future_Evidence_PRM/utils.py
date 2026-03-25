import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - library should exist in runtime image
    SentenceTransformer = None  # type: ignore[assignment]
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch


import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from queue import Queue

import pandas as pd
import requests
from openai import OpenAI
import atexit
from tqdm import tqdm


SENTENCE_MODEL_PATH = "/nfsdata/fanshengda/models/AI-ModelScope/all-MiniLM-L6-v2"


def _format_arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, sort_keys=True)
    except (TypeError, ValueError):
        return str(arguments)


@lru_cache(maxsize=1)
def _get_sentence_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for tool_call_similarity but is not installed."
        )
    return SentenceTransformer(SENTENCE_MODEL_PATH)


@lru_cache(maxsize=4096)
def _embed_argument_text(arg_text: str) -> np.ndarray:
    model = _get_sentence_model()
    embedding = model.encode(
        [arg_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0]


def _extract_tool_calls(trace: str) -> List[Dict[str, Any]]:
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    raw_calls = re.findall(pattern, trace)
    tool_calls: List[Dict[str, Any]] = []
    for raw_call in raw_calls:
        try:
            tool_calls.append(json.loads(raw_call))
        except json.JSONDecodeError:
            continue
    return tool_calls


# 结合SentenceTransformer对工具调用参数进行匹配，相似度超过阈值即视为同一调用。
def tool_call_similarity(trace1: str, trace2: str, threshold: float = 0.75) -> float:
    tool_calls1 = _extract_tool_calls(trace1)
    tool_calls2 = _extract_tool_calls(trace2)

    if not tool_calls1 and not tool_calls2:
        return 1.0
    if not tool_calls1 or not tool_calls2:
        return 0.0

    def _enrich(tool_calls: List[Dict[str, Any]]) -> List[Tuple[str, np.ndarray]]:
        enriched: List[Tuple[str, np.ndarray]] = []
        for call in tool_calls:
            name = str(call.get("name", ""))
            arguments = _format_arguments(call.get("arguments"))
            embedding = _embed_argument_text(arguments)
            enriched.append((name, embedding))
        return enriched

    enriched1 = _enrich(tool_calls1)
    enriched2 = _enrich(tool_calls2)

    matched_indices: set[int] = set()
    matches = 0

    for name1, emb1 in enriched1:
        best_idx = None
        best_sim = -1.0

        for idx, (name2, emb2) in enumerate(enriched2):
            if idx in matched_indices:
                continue
            if name1 != name2:
                continue

            similarity = float(np.dot(emb1, emb2))
            if similarity >= threshold and similarity > best_sim:
                best_sim = similarity
                best_idx = idx

        if best_idx is not None:
            matches += 1
            matched_indices.add(best_idx)

    union_size = len(enriched1) + len(enriched2) - matches
    if union_size == 0:
        return 1.0
    return matches / union_size


def build_pair_prompt(trace1, trace2):
    return

# 默认端口: qwen-2.5-7B instruct 服务在 8888，wiki 检索服务在 8000

def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_parquet_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path}")
    return pd.read_parquet(path)


_client_local = threading.local()


def _get_openai_client(base_url: str, api_key: Optional[str]) -> OpenAI:
    """Return a thread-local OpenAI client to avoid leaking many HTTP connections."""
    client: Optional[OpenAI] = getattr(_client_local, "client", None)
    cached_base = getattr(_client_local, "base_url", None)
    cached_key = getattr(_client_local, "api_key", None)
    if client is not None and cached_base == base_url and cached_key == api_key:
        return client
    # Create and cache per-thread client
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    _client_local.client = client
    _client_local.base_url = base_url
    _client_local.api_key = api_key
    return client


def openai_chat_completions(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    n: int = 1,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    timeout: int = 60,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Call OpenAI-compatible Chat Completions via official SDK; return assistant messages with tool_calls."""
    client = _get_openai_client(base_url, api_key)

    def _single_call() -> Dict[str, Any]:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        if not resp.choices:
            return {"role": "assistant", "content": "", "tool_calls": []}
        msg = resp.choices[0].message
        out: Dict[str, Any] = {"role": msg.role or "assistant", "content": msg.content or ""}
        out["tool_calls"] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                out["tool_calls"].append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                )
        return out

    outputs: List[Dict[str, Any]] = []
    for _ in range(max(1, n)):
        outputs.append(_single_call())
    return outputs


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def parse_tool_call(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract the first tool_call JSON dict if present and valid.

    Expected content (SearchTool): {"name": "search", "arguments": {"query_list": [...]}}
    Returns the parsed dict or None.
    """
    m = TOOL_CALL_RE.search(raw_text)
    if not m:
        return None
    snippet = m.group(1).strip()
    # Try strict JSON first
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        # Fallback: try to salvage minimal fields
        # Extract query_list heuristically if possible
        qlist: Optional[List[str]] = None
        qlist_match = re.search(r"\"query_list\"\s*:\s*(\[[^\]]*\])", snippet)
        if qlist_match:
            try:
                qlist = json.loads(qlist_match.group(1))
            except Exception:
                qlist = None
        name_match = re.search(r"\"name\"\s*:\s*\"(.*?)\"", snippet)
        name = name_match.group(1) if name_match else "search"
        if qlist:
            return {"name": name, "arguments": {"query_list": qlist}}
        return None


def contains_answer(raw_text: str) -> bool:
    return ANSWER_RE.search(raw_text) is not None


def call_retriever(
    retrieval_url: str,
    query_list: List[str],
    topk: int = 3,
    timeout: int = 30,
) -> str:
    """Call the retriever and return the plain formatted result string to place inside <tool_response>.

    We reuse perform_single_search_batch from verl.tools.utils.search_r1_like_utils which
    handles retries and formatting.
    """
    # perform_single_search_batch returns (json_str, metadata)
    result_text, _meta = perform_single_search_batch(
        retrieval_service_url=retrieval_url, query_list=query_list, topk=topk, concurrent_semaphore=None, timeout=timeout
    )
    try:
        data = json.loads(result_text)
        formatted = data.get("result", "")
        if not isinstance(formatted, str):
            formatted = str(formatted)
        return formatted
    except Exception:
        # Fallback to raw string
        return result_text


@dataclass
class TrajectoryRecord:
    query_index: int
    question: str
    data_source: str
    ground_truth: Any
    messages: List[Dict[str, Any]]
    depth: int
    terminated: bool
    leaf_output: Optional[str] = None


@dataclass
class SamplingConfig:
    llm_base_url: str = "http://127.0.0.1:8888"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 1024
    llm_timeout_s: int = 120
    branching_factor: int = 2
    max_depth: int = 4
    concurrency: int = 32
    retriever_url: str = "http://127.0.0.1:8000/retrieve"
    retriever_topk: int = 3
    retriever_timeout_s: int = 30
    pause_between_calls_s: float = 0.0  # optional small sleep to be gentle
    openai_api_key: str = "dada"
    tools: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries",
                            },
                            "query": {"type": "string", "description": "Single query"},
                        },
                        "required": ["query_list"],
                    },
                },
            }
        ]
    )


def _to_native(obj: Any) -> Any:
    """Best-effort conversion of parquet-loaded nested values to native Python objects.

    Handles pyarrow scalars (as_py), pyarrow arrays (to_pylist), JSON strings/bytes.
    """
    # Already native
    if isinstance(obj, (list, dict)):
        return obj
    # PyArrow scalar types
    if hasattr(obj, "as_py") and callable(getattr(obj, "as_py")):
        try:
            return obj.as_py()
        except Exception:
            pass
    # PyArrow list-like
    if hasattr(obj, "to_pylist") and callable(getattr(obj, "to_pylist")):
        try:
            return obj.to_pylist()
        except Exception:
            pass
    # JSON string/bytes
    if isinstance(obj, (str, bytes)):
        try:
            s = obj.decode("utf-8") if isinstance(obj, bytes) else obj
            return json.loads(s)
        except Exception:
            return obj
    return obj


def _extract_question_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    # Find the last user message and try to split by "Question:" pattern
    user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_contents:
        return None
    last = user_contents[-1]
    if not isinstance(last, str):
        last = str(last)
    if "Question:" in last:
        return last.split("Question:")[-1].strip()
    return last.strip() if last else None

