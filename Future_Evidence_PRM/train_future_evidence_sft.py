"""
Train a decoder-only model with future-evidence distillation SFT.

Each JSONL row is expected to contain:
- `messages`: student input messages
- `response` or `response_json`: distilled teacher target

The current target format is structured JSON, typically:
{
  "step_assessments": {
    "steps": {"<assistant_index>": "...", ...}
  },
  "future_implications": {
    "steps": {
      "<assistant_index>": {
        "directions": ["...", "..."],
        "branch_point": "...",
        "recoverability": "easy|possible|hard|none",
        "revealing_signal": "..."
      }
    }
  },
  "explanations": {
    "steps": {"<assistant_index>": "...", ...}
  },
  "step_labels": {"<assistant_index>": -1|0|1, ...},
  "final_label": -1|1
}

The model sees the full trajectory prompt in `messages`, and SFT loss is applied
only on the distilled `response`.

Example:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch Future_Evidence_PRM/train_future_evidence_sft.py \
  --train_jsonl /nfsdata/fanshengda/verl/input_data/future_evidence_sft/qwen3_30b_a3b_hotpotqa_future_evidence_sft_train.jsonl \
  --model_name_or_path /nfsdata/fanshengda/models/Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir /nfsdata/fanshengda/verl/prm_ckpts/qwen3_4b_future_evidence_sft_30BA3B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_SWANLAB_API_KEY = "WoZrF9qolYJjzYBCfArih"
DEFAULT_SWANLAB_PROJ_NAME = "Foresight-PRM"
DEFAULT_DEBUG_SAMPLE_COUNT = 10


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


def _to_native(obj: Any) -> Any:
    if isinstance(obj, (list, dict)):
        return obj
    if isinstance(obj, (str, bytes)):
        text = obj.decode("utf-8") if isinstance(obj, bytes) else obj
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)
        return obj
    if hasattr(obj, "as_py") and callable(getattr(obj, "as_py")):
        return obj.as_py()
    if hasattr(obj, "to_pylist") and callable(getattr(obj, "to_pylist")):
        return obj.to_pylist()
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    return obj


def _safe_native(obj: Any) -> Any:
    obj = _to_native(obj)
    if isinstance(obj, dict):
        return {str(key): _safe_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_safe_native(item) for item in obj]
    return obj


def _parse_json_kwargs(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON string: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got: {type(parsed)}")
    return parsed


def _resolve_chat_template_kwargs(model_name_or_path: str, raw_kwargs: str) -> Dict[str, Any]:
    kwargs = _parse_json_kwargs(raw_kwargs)
    if kwargs:
        return kwargs
    if "qwen3" in model_name_or_path.lower():
        return {"enable_thinking": False}
    return {}


def _get_response_content(record: Dict[str, Any]) -> str:
    response = record.get("response")
    if isinstance(response, str) and response.strip():
        text = response
    else:
        response_json = _safe_native(record.get("response_json"))
        if response_json is None:
            raise ValueError("Each record must contain `response` or `response_json`.")
        text = json.dumps(response_json, ensure_ascii=False, indent=2)
    return text


def _split_indices_by_query(records: List[Dict[str, Any]], eval_ratio: float, seed: int) -> tuple[List[int], List[int]]:
    query_values = [record.get("query_index") for record in records]
    if all(value is None for value in query_values):
        indices = np.arange(len(records))
        if len(indices) <= 1:
            return indices.tolist(), []
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        eval_size = min(max(1, int(round(len(indices) * eval_ratio))), len(indices) - 1)
        return indices[eval_size:].tolist(), indices[:eval_size].tolist()

    unique_queries = []
    seen = set()
    for value in query_values:
        if value in seen:
            continue
        seen.add(value)
        unique_queries.append(value)
    if len(unique_queries) <= 1:
        return list(range(len(records))), []

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_queries)
    eval_size = min(max(1, int(round(len(unique_queries) * eval_ratio))), len(unique_queries) - 1)
    eval_queries = set(unique_queries[:eval_size])

    train_indices: List[int] = []
    eval_indices: List[int] = []
    for row_index, query_index in enumerate(query_values):
        if query_index in eval_queries:
            eval_indices.append(row_index)
        else:
            train_indices.append(row_index)
    return train_indices, eval_indices


class FutureEvidenceSFTDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        *,
        max_length: int,
        truncation: str,
        apply_chat_template_kwargs: Dict[str, Any],
    ) -> None:
        self.records = [_safe_native(record) for record in records]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.apply_chat_template_kwargs = apply_chat_template_kwargs

    def __len__(self) -> int:
        return len(self.records)

    def _build_sample(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        messages = _safe_native(record.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise ValueError("Each record must contain non-empty `messages`.")

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.apply_chat_template_kwargs,
        )
        response_content = _get_response_content(record)
        full_text = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response_content}],
            tokenize=False,
            add_generation_prompt=False,
            **self.apply_chat_template_kwargs,
        )

        if self.tokenizer.is_fast:
            full_encoding = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            input_ids = full_encoding["input_ids"][0]
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            response_start = len(prompt_text)
            offsets = full_encoding["offset_mapping"][0].tolist()
            for token_index, (start, _) in enumerate(offsets):
                if int(start) < response_start:
                    labels[token_index] = -100
        else:
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            full_ids = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            input_ids = full_ids
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            if prompt_ids.size(0) <= full_ids.size(0) and torch.equal(full_ids[: prompt_ids.size(0)], prompt_ids):
                labels[: prompt_ids.size(0)] = -100
            else:
                response_text = response_content
                if self.tokenizer.eos_token and not response_text.endswith(self.tokenizer.eos_token):
                    response_text = response_text + self.tokenizer.eos_token
                response_ids = self.tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                input_ids = torch.cat([prompt_ids, response_ids], dim=0)
                labels = torch.cat(
                    [
                        torch.full((prompt_ids.size(0),), -100, dtype=torch.long),
                        response_ids.clone(),
                    ],
                    dim=0,
                )
                attention_mask = torch.ones_like(input_ids)

        if input_ids.size(0) > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                labels = labels[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise ValueError(f"Sequence length exceeds max_length={self.max_length}")

        if not torch.any(labels != -100):
            response_only_text = response_content
            if self.tokenizer.eos_token and not response_only_text.endswith(self.tokenizer.eos_token):
                response_only_text = response_only_text + self.tokenizer.eos_token
            response_ids = self.tokenizer(response_only_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            keep_length = min(response_ids.size(0), self.max_length)
            input_ids = response_ids[-keep_length:] if self.truncation == "left" else response_ids[:keep_length]
            labels = input_ids.clone()
            attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._build_sample(self.records[idx])

    def preview(self, idx: int) -> Dict[str, str]:
        record = self.records[idx]
        messages = _safe_native(record.get("messages"))
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.apply_chat_template_kwargs,
        )
        response_content = _get_response_content(record)
        full_text = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response_content}],
            tokenize=False,
            add_generation_prompt=False,
            **self.apply_chat_template_kwargs,
        )
        sample = self._build_sample(record)
        input_text = self.tokenizer.decode(sample["input_ids"].tolist(), skip_special_tokens=False)
        return {
            "query_index": str(record.get("query_index")),
            "trajectory_sample_id": str(record.get("trajectory_sample_id")),
            "prompt_text": prompt_text,
            "full_text": full_text,
            "input_text": input_text,
            "response_text": response_content,
            "input_length": str(int(sample["input_ids"].size(0))),
            "supervised_tokens": str(int((sample["labels"] != -100).sum().item())),
        }


class DataCollatorForFutureEvidenceSFT:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must have either pad_token_id or eos_token_id.")
            pad_token_id = self.tokenizer.eos_token_id

        input_ids = pad_sequence([feature["input_ids"] for feature in features], batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(
            [feature["attention_mask"] for feature in features], batch_first=True, padding_value=0
        )
        labels = pad_sequence([feature["labels"] for feature in features], batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _debug_print_samples(title: str, dataset: FutureEvidenceSFTDataset, indices: List[int], limit: int = DEFAULT_DEBUG_SAMPLE_COUNT) -> None:
    print(f"\n===== {title} =====")
    if not indices:
        print("(empty)")
        return
    for sample_idx, row_index in enumerate(indices[:limit]):
        preview = dataset.preview(row_index)
        print(
            f"[sample {sample_idx}] row_index={row_index} query_index={preview['query_index']} "
            f"trajectory_sample_id={preview['trajectory_sample_id']}"
        )
        print(f"input_length={preview['input_length']} supervised_tokens={preview['supervised_tokens']}")
        print("prompt_text=")
        print(preview["prompt_text"])
        print("full_text=")
        print(preview["full_text"])
        print("input_text=")
        print(preview["input_text"])
        print("response_text=")
        print(preview["response_text"])
        print("-----")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future-evidence distillation SFT model")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=20000)
    parser.add_argument("--truncation", type=str, default="left", choices=["left", "right", "error"])
    parser.add_argument("--apply_chat_template_kwargs", type=str, default="")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SWANLAB_API_KEY", DEFAULT_SWANLAB_API_KEY)
    os.environ.setdefault("SWANLAB_PROJ_NAME", DEFAULT_SWANLAB_PROJ_NAME)

    apply_chat_template_kwargs = _resolve_chat_template_kwargs(args.model_name_or_path, args.apply_chat_template_kwargs)
    print(f"apply_chat_template_kwargs={json.dumps(apply_chat_template_kwargs, ensure_ascii=False)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = list(_iter_jsonl(Path(args.train_jsonl)))
    if not train_records:
        raise ValueError(f"No records found in {args.train_jsonl}")

    train_source = FutureEvidenceSFTDataset(
        train_records,
        tokenizer,
        max_length=args.max_length,
        truncation=args.truncation,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    if args.eval_jsonl:
        eval_records = list(_iter_jsonl(Path(args.eval_jsonl)))
        eval_source = FutureEvidenceSFTDataset(
            eval_records,
            tokenizer,
            max_length=args.max_length,
            truncation=args.truncation,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        train_dataset = train_source
        eval_dataset = eval_source
        train_indices = list(range(len(train_source)))
        eval_indices = list(range(len(eval_source)))
    else:
        train_indices, eval_indices = _split_indices_by_query(train_records, args.eval_ratio, args.seed)
        train_dataset = Subset(train_source, train_indices)
        eval_dataset = Subset(train_source, eval_indices) if eval_indices else None

    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    print(f"train_size={len(train_dataset)} eval_size={0 if eval_dataset is None else len(eval_dataset)}")
    _debug_print_samples("Train Samples", train_source, train_indices, limit=DEFAULT_DEBUG_SAMPLE_COUNT)
    if args.eval_jsonl:
        _debug_print_samples("Eval Samples", eval_source, eval_indices, limit=DEFAULT_DEBUG_SAMPLE_COUNT)
    elif eval_indices:
        _debug_print_samples("Eval Samples", train_source, eval_indices, limit=DEFAULT_DEBUG_SAMPLE_COUNT)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_cache": False,
    }
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["attn_implementation"] = DEFAULT_ATTN_IMPLEMENTATION

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=Path(args.output_dir).name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=8,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        report_to=["swanlab"],
        save_strategy="steps",
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForFutureEvidenceSFT(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
