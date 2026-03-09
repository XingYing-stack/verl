#!/usr/bin/env python3
"""
Convert summary model JSONL outputs into SFT parquet format.

Input JSONL example: fetch_merged.jsonl
Output parquet is compatible with verl.utils.dataset.sft_dataset.SFTDataset.
"""

import argparse
import json
import os
from typing import Any, Iterable

import pandas as pd

from verl.utils import hf_tokenizer


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_apply_chat_template_kwargs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --apply_chat_template_kwargs: {raw}") from exc


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _build_prompt(messages: list[dict[str, Any]]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _coerce_text(messages[0].get("content"))
    parts: list[str] = []
    for msg in messages:
        role = _coerce_text(msg.get("role", "user"))
        content = _coerce_text(msg.get("content"))
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _strip_eos(text: str, eos_token: str | None) -> str:
    if not eos_token:
        return text
    if text.endswith(eos_token):
        return text[: -len(eos_token)]
    if text.endswith(eos_token + "\n"):
        return text[: -(len(eos_token) + 1)]
    return text


def _build_response(
    prompt: str,
    assistant_message: dict[str, Any],
    tokenizer,
    apply_chat_template_kwargs: dict[str, Any],
) -> str:
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [assistant_message],
        add_generation_prompt=False,
        tokenize=False,
        **apply_chat_template_kwargs,
    )
    if full_text.startswith(prompt_text):
        assistant_text = full_text[len(prompt_text) :]
    else:
        idx = full_text.rfind(prompt_text)
        if idx != -1:
            assistant_text = full_text[idx + len(prompt_text) :]
        else:
            assistant_text = _coerce_text(assistant_message.get("content"))
            reasoning = _coerce_text(assistant_message.get("reasoning_content"))
            if reasoning:
                assistant_text = f"{reasoning}\n{assistant_text}" if assistant_text else reasoning
    return _strip_eos(assistant_text, tokenizer.eos_token)


def _extract_assistant_message(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response", {})
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    assistant_message = {
        "role": "assistant",
        "content": _coerce_text(message.get("content")),
    }
    reasoning = _coerce_text(message.get("reasoning_content"))
    if reasoning:
        assistant_message["reasoning_content"] = reasoning
    return assistant_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to SFT parquet.")
    parser.add_argument("--input_path", required=True, help="Path to input JSONL file.")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to output parquet file. Defaults to input path with .parquet suffix.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path used to apply chat template.",
    )
    parser.add_argument(
        "--apply_chat_template_kwargs",
        default=None,
        help="JSON string of kwargs passed to tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom code for tokenizer.",
    )

    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        base, _ = os.path.splitext(args.input_path)
        output_path = f"{base}.parquet"

    apply_chat_template_kwargs = _parse_apply_chat_template_kwargs(args.apply_chat_template_kwargs)
    tokenizer = hf_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    rows: list[dict[str, str]] = []
    skipped = 0
    for record in _iter_jsonl(args.input_path):
        messages = record.get("request", {}).get("messages") or []
        if not messages:
            skipped += 1
            continue
        prompt = _build_prompt(messages)
        assistant_message = _extract_assistant_message(record)
        response = _build_response(prompt, assistant_message, tokenizer, apply_chat_template_kwargs)
        rows.append({"prompt": prompt, "response": response})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}. Skipped {skipped} empty records.")
