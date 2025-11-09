import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - library should exist in runtime image
    SentenceTransformer = None  # type: ignore[assignment]


SENTENCE_MODEL_PATH = "/workspace/fanshengda/models/sentence-transformers/all-MiniLM-L6-v2"


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
