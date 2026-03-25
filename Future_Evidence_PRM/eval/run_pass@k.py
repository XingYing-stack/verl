from __future__ import annotations

"""
Sample k SearchR1-like trajectories per question and save them for later best-of-k evaluation.

Output: JSONL (one line per trajectory sample), each record contains:
- query_index, sample_index
- question, data_source, ground_truth (if present in extra_info.tools_kwargs.search.create_kwargs.ground_truth)
- terminated, stop_reason, answer_text (parsed from <answer>...</answer> if present)
- messages: OpenAI-style messages including tool call/response turns when available
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# NOTE: This file name contains "@", so it cannot be run via `python -m ...`.
# We support direct execution via a defensive import fallback.
try:
    from Future_Evidence_PRM.utils import (  # type: ignore[import-not-found]
        SamplingConfig,
        _extract_question_from_messages,
        _now_str,
        _to_native,
        call_retriever,
        contains_answer,
        ensure_dir,
        openai_chat_completions,
        parse_tool_call,
        read_parquet_dataset,
    )
except Exception:  # pragma: no cover
    import sys

    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from Future_Evidence_PRM.utils import (  # type: ignore[import-not-found]
        SamplingConfig,
        _extract_question_from_messages,
        _now_str,
        _to_native,
        call_retriever,
        contains_answer,
        ensure_dir,
        openai_chat_completions,
        parse_tool_call,
        read_parquet_dataset,
    )


ANSWER_SPAN_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _extract_answer_text(raw_text: str) -> Optional[str]:
    matches = ANSWER_SPAN_RE.findall(raw_text or "")
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_base_messages(prompt: Any) -> List[Dict[str, Any]]:
    prompt = list(prompt)
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except Exception:
            prompt = None
    if not isinstance(prompt, list):
        return []
    base_messages: List[Dict[str, Any]] = []
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(role, str):
            continue
        if not isinstance(content, str):
            content = str(content)
        base_messages.append({"role": role, "content": content})
    return base_messages


def rollout_single_trajectory(
    base_messages: List[Dict[str, Any]],
    *,
    cfg: SamplingConfig,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], bool, str, Optional[str]]:
    messages: List[Dict[str, Any]] = [dict(m) for m in base_messages]
    last_assistant_content: Optional[str] = None

    for _step_id in range(max_steps):
        try:
            assistant = openai_chat_completions(
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                messages=messages,
                n=1,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                timeout=cfg.llm_timeout_s,
                api_key=cfg.openai_api_key,
                tools=cfg.tools,
            )[0]
        except Exception as exc:
            err_msg = {"role": "assistant", "content": f"<error> LLM error: {exc} </error>"}
            messages.append(err_msg)
            return messages, False, "llm_error", None

        content = assistant.get("content", "") or ""
        tool_calls = assistant.get("tool_calls", []) or []
        last_assistant_content = content

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if contains_answer(content):
            return messages, True, "answer", _extract_answer_text(content)

        structured_calls: List[Dict[str, Any]] = []
        for tc in tool_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            structured_calls.append(
                {
                    "id": tc.get("id") if isinstance(tc, dict) else None,
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", {}),
                }
            )

        legacy_mode = False
        if not structured_calls:
            legacy = parse_tool_call(content)
            if legacy:
                legacy_mode = True
                structured_calls = [{"id": None, "name": legacy.get("name"), "arguments": legacy.get("arguments", {})}]

        search_calls = [c for c in structured_calls if c.get("name") == "search"]
        if not search_calls:
            continue

        for call in search_calls:
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            qlist = args.get("query_list")
            if not qlist and "query" in args:
                qlist = [args["query"]]
            if isinstance(qlist, str):
                query_list = [qlist]
            elif isinstance(qlist, list):
                query_list = [str(x) for x in qlist]
            else:
                query_list = []

            if not query_list:
                continue

            try:
                retrieved = call_retriever(
                    retrieval_url=cfg.retriever_url,
                    query_list=query_list,
                    topk=cfg.retriever_topk,
                    timeout=cfg.retriever_timeout_s,
                )
            except Exception as exc:
                retrieved = f"Search error: {exc}"

            if legacy_mode:
                tool_resp_block = f"<tool_response>\n{retrieved}\n</tool_response>"
                messages.append({"role": "user", "content": tool_resp_block})
            else:
                tool_msg: Dict[str, Any] = {"role": "tool", "content": retrieved, "name": "search"}
                if call.get("id"):
                    tool_msg["tool_call_id"] = call["id"]
                messages.append(tool_msg)

            if cfg.pause_between_calls_s > 0:
                time.sleep(cfg.pause_between_calls_s)

    return messages, False, "max_steps", _extract_answer_text(last_assistant_content or "")


def _extract_ground_truth(extra_info: Any) -> Any:
    extra_info = _to_native(extra_info)
    if not isinstance(extra_info, dict):
        return None
    try:
        tools_kwargs = extra_info.get("tools_kwargs", {})
        gt = tools_kwargs.get("search", {}).get("create_kwargs", {}).get("ground_truth")
        gt = _to_native(gt)
        if isinstance(gt, dict) and "target" in gt:
            try:
                gt["target"] = list(gt["target"])
            except Exception:
                pass
        return gt
    except Exception:
        return None


def _iter_work_items(start: int, end: int, *, k: int) -> Iterable[Tuple[int, int]]:
    for idx in range(start, end):
        for sample_index in range(k):
            yield idx, sample_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample k SearchR1-like trajectories per question and save messages.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/nfsdata/fanshengda/verl/Future_Evidence_PRM/input/searchR1_hotpotqa_validation/test.parquet",
        help="Path to SearchR1-like parquet (must include prompt + extra_info).",
    )
    parser.add_argument("--output_dir", type=str, default="./output/pass_at_k", help="Directory to save JSONL.")
    parser.add_argument("--start", type=int, default=0, help="Start row index (inclusive)")
    parser.add_argument("--end", type=int, default=10000, help="End row index (exclusive); -1 means all")
    parser.add_argument("--k", type=int, default=8, help="Sample k trajectories per question")
    parser.add_argument("--max_steps", type=int, default=4, help="Max LLM steps per trajectory")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent trajectories")

    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://172.16.1.40:8888/v1",
        help="OpenAI-compatible base URL for LLM server (e.g., SGLang).",
    )
    parser.add_argument("--llm_model", type=str, default="Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument("--llm_temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--llm_max_tokens", type=int, default=4096, help="Max new tokens per completion")
    parser.add_argument("--llm_timeout_s", type=int, default=120, help="LLM request timeout seconds")
    parser.add_argument(
        "--retriever_url",
        type=str,
        default="http://172.16.1.40:8000/retrieve",
        help="Retriever HTTP endpoint (Search-R1-like).",
    )
    parser.add_argument("--retriever_topk", type=int, default=3, help="Retriever topk")
    parser.add_argument("--retriever_timeout_s", type=int, default=30, help="Retriever timeout seconds")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="dada",
        help="API key for OpenAI-compatible LLM server Authorization header.",
    )
    parser.add_argument("--pause_between_calls_s", type=float, default=0.0, help="Optional sleep between calls")

    args = parser.parse_args()

    df = read_parquet_dataset(args.dataset_path)
    start = max(0, int(args.start))
    end = int(args.end)
    if end < 0 or end > len(df):
        end = len(df)
    if start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, len={len(df)}")
    if args.k <= 0:
        raise ValueError("--k must be > 0")

    ensure_dir(args.output_dir)
    out_fp = os.path.join(
        args.output_dir,
        f"pass_at_k_rollouts_{os.path.basename(args.dataset_path).split('.')[0]}_{start}_{end}_k{args.k}_{_now_str()}.jsonl",
    )
    print(f"Writing rollouts to: {out_fp}")
    # Ensure each run starts from a clean output file (writer uses append mode).
    with open(out_fp, "w", encoding="utf-8"):
        pass

    cfg = SamplingConfig(
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_temperature=float(args.llm_temperature),
        llm_max_tokens=int(args.llm_max_tokens),
        llm_timeout_s=int(args.llm_timeout_s),
        concurrency=int(args.concurrency),
        retriever_url=args.retriever_url,
        retriever_topk=int(args.retriever_topk),
        retriever_timeout_s=int(args.retriever_timeout_s),
        pause_between_calls_s=float(args.pause_between_calls_s),
        openai_api_key=args.openai_api_key,
    )

    lines_q: Queue[Optional[str]] = Queue(maxsize=10000)

    def _writer_worker() -> None:
        with open(out_fp, "a", encoding="utf-8") as f:
            while True:
                item = lines_q.get()
                if item is None:
                    lines_q.task_done()
                    break
                f.write(item)
                lines_q.task_done()

    writer_thread = threading.Thread(target=_writer_worker, name="writer", daemon=True)
    writer_thread.start()

    def worker(item: Tuple[int, int]) -> None:
        idx, sample_index = item
        row = df.iloc[idx]

        prompt = row.get("prompt")
        base_messages = _normalize_base_messages(prompt)
        if not base_messages:
            rec = {
                "query_index": idx,
                "sample_index": sample_index,
                "error": "empty_prompt",
            }
            lines_q.put(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        extra_info = _to_native(row.get("extra_info", {}))
        data_source = row.get("data_source", "")
        question = extra_info.get("question") if isinstance(extra_info, dict) else None
        if not question:
            question = _extract_question_from_messages(base_messages) or ""

        ground_truth = _extract_ground_truth(extra_info)

        messages, terminated, stop_reason, answer_text = rollout_single_trajectory(
            base_messages=base_messages,
            cfg=cfg,
            max_steps=int(args.max_steps),
        )

        rec = {
            "query_index": idx,
            "sample_index": sample_index,
            "question": question,
            "data_source": data_source,
            "ground_truth": ground_truth,
            "terminated": terminated,
            "stop_reason": stop_reason,
            "answer_text": answer_text,
            "messages": messages,
            "meta": {
                "dataset_path": args.dataset_path,
                "llm_base_url": cfg.llm_base_url,
                "llm_model": cfg.llm_model,
                "llm_temperature": cfg.llm_temperature,
                "llm_max_tokens": cfg.llm_max_tokens,
                "retriever_url": cfg.retriever_url,
                "retriever_topk": cfg.retriever_topk,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        }
        lines_q.put(json.dumps(rec, ensure_ascii=False) + "\n")

    work_items = list(_iter_work_items(start, end, k=int(args.k)))
    with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
        futures = [ex.submit(worker, it) for it in work_items]
        with tqdm(total=len(futures), desc="Sampling", unit="traj", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # pragma: no cover - defensive for runtime data quirks
                    pbar.write(f"Worker failed: {exc}")
                finally:
                    pbar.update(1)

    lines_q.join()
    lines_q.put(None)
    writer_thread.join()
    print("Done.")


if __name__ == "__main__":
    main()
