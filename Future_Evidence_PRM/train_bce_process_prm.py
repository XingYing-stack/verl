"""
Train a decoder-only PRM with multi-marker BCE supervision.

One trajectory becomes one training sample.
If a trajectory has 5 assistant steps, the sample has 5 supervised marker positions.


accelerate launch Future_Evidence_PRM/train_bce_process_prm.py \
--train_parquet input_data/bce_process/hotpotqa_traj_bce_train.parquet \
--eval_parquet input_data/bce_process/hotpotqa_traj_bce_validation.parquet \
--model_name_or_path /nfsdata/fanshengda/models/Qwen/Qwen3-4B-Instruct-2507 \
--output_dir ./prm_ckpts/hotpotqa_bce_prm
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_REPORT_TO = "swanlab"
DEFAULT_SWANLAB_API_KEY = "WoZrF9qolYJjzYBCfArih"
DEFAULT_SWANLAB_PROJ_NAME = "Foresight-PRM"


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


class TrajectoryBCEDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        *,
        marker_token: str,
        max_length: int,
        truncation: str,
    ) -> None:
        self.dataframe = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

        self.messages = [_safe_native(item) for item in self.dataframe["prompt"].tolist()]
        self.tools = [_safe_native(item) for item in self.dataframe["tools"].tolist()] if "tools" in self.dataframe.columns else [None] * len(self.dataframe)
        self.step_labels = [_safe_native(item) for item in self.dataframe["step_labels"].tolist()]

        self.marker_token_ids = self.tokenizer.encode(marker_token, add_special_tokens=False)
        if len(self.marker_token_ids) != 1:
            raise ValueError(f"Marker token must map to exactly one token, got {self.marker_token_ids}")
        self.marker_token_id = self.marker_token_ids[0]

    def __len__(self) -> int:
        return len(self.messages)

    def _segment_tokens(
        self,
        messages: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        *,
        is_assistant: bool,
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[int]:
        if start_idx > 0:
            previous_text = self.tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
            if is_assistant:
                previous_text_with_prompt = self.tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )
        else:
            previous_text = ""
            previous_text_with_prompt = ""

        current_text = self.tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

        if is_assistant:
            generation_prompt_text = previous_text_with_prompt[len(previous_text):]
            generation_prompt_tokens = self.tokenizer.encode(generation_prompt_text, add_special_tokens=False)
            message_tokens = self.tokenizer.encode(
                current_text[len(previous_text_with_prompt):],
                add_special_tokens=False,
            )
            return generation_prompt_tokens + message_tokens

        return self.tokenizer.encode(current_text[len(previous_text):], add_special_tokens=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        messages = self.messages[idx]
        tools = self.tools[idx]
        step_labels = [int(label) for label in self.step_labels[idx]]

        input_ids: List[int] = []
        marker_labels: List[int] = []

        step_index = 0
        message_index = 0
        pending_step_label: Optional[int] = None
        while message_index < len(messages):
            message = messages[message_index]
            role = message["role"]

            if role == "assistant":
                tokens = self._segment_tokens(messages, message_index, message_index + 1, is_assistant=True, tools=tools)
                input_ids.extend(tokens)
                marker_labels.extend([-100] * len(tokens))

                next_role = messages[message_index + 1]["role"] if message_index + 1 < len(messages) else None
                if next_role == "tool":
                    if pending_step_label is not None:
                        raise ValueError(f"Sample {idx} has two assistant steps waiting for tool observations")
                    pending_step_label = step_labels[step_index]
                    step_index += 1
                elif next_role is None:
                    input_ids.append(self.marker_token_id)
                    marker_labels.append(step_labels[step_index])
                    step_index += 1
                else:
                    input_ids.append(self.marker_token_id)
                    marker_labels.append(step_labels[step_index])
                    step_index += 1
                message_index += 1
                continue

            if role == "tool":
                end_index = message_index + 1
                while end_index < len(messages) and messages[end_index]["role"] == "tool":
                    end_index += 1
                tokens = self._segment_tokens(messages, message_index, end_index, is_assistant=False, tools=tools)
                input_ids.extend(tokens)
                marker_labels.extend([-100] * len(tokens))
                if pending_step_label is None:
                    raise ValueError(f"Sample {idx} has tool observation without preceding assistant step")
                input_ids.append(self.marker_token_id)
                marker_labels.append(pending_step_label)
                pending_step_label = None
                message_index = end_index
                continue

            tokens = self._segment_tokens(messages, message_index, message_index + 1, is_assistant=False, tools=tools)
            input_ids.extend(tokens)
            marker_labels.extend([-100] * len(tokens))
            message_index += 1

        if pending_step_label is not None:
            raise ValueError(f"Sample {idx} ended with an unfinished assistant step")
        if step_index != len(step_labels):
            raise ValueError(f"Expected {len(step_labels)} assistant steps, got {step_index}")

        attention_mask = [1] * len(input_ids)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        if len(input_ids) > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                marker_labels = marker_labels[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                marker_labels = marker_labels[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise ValueError(f"Sequence length {len(input_ids)} exceeds max_length={self.max_length}")
        else:
            pad_size = self.max_length - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * pad_size)
            marker_labels = marker_labels + ([-100] * pad_size)
            attention_mask = attention_mask + ([0] * pad_size)

        if max(marker_labels) < 0:
            raise ValueError(f"Sample {idx} has no supervised marker after truncation")

        position_ids = [position * mask for position, mask in enumerate(attention_mask)]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "step_labels": torch.tensor(marker_labels, dtype=torch.float32),
        }


def _debug_print_dataframe_rows(title: str, dataset: TrajectoryBCEDataset, row_indices: List[int], tokenizer, limit: int = 2) -> None:
    print(f"\n===== {title} =====")
    if not row_indices:
        print("(empty)")
        return

    for sample_idx, row_index in enumerate(row_indices[:limit]):
        row = dataset.dataframe.iloc[row_index]
        prompt = _safe_native(row["prompt"])
        tools = _safe_native(row["tools"]) if "tools" in row else None
        step_labels = _safe_native(row["step_labels"])
        rendered_text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        sample = dataset[row_index]
        valid_length = int(sample["attention_mask"].sum().item())
        decoded_input = tokenizer.decode(sample["input_ids"][:valid_length].tolist(), skip_special_tokens=False)
        print(f"[sample {sample_idx}] row_index={row_index} query_index={row.get('query_index')} tree_id={row.get('tree_id')}")
        print(f"step_labels={step_labels}")
        print("rendered_prompt=")
        print(rendered_text)
        print("rendered_input_with_markers=")
        print(decoded_input)
        print("-----")


class DataCollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([feature["input_ids"] for feature in features]),
            "attention_mask": torch.stack([feature["attention_mask"] for feature in features]),
            "position_ids": torch.stack([feature["position_ids"] for feature in features]),
            "step_labels": torch.stack([feature["step_labels"] for feature in features]),
        }


class BCETrainer(Trainer):
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        step_labels = inputs.pop("step_labels")
        outputs = model(**inputs)
        logits = outputs.logits.float().squeeze(-1)

        valid_mask = step_labels >= 0
        valid_logits = logits[valid_mask]
        valid_labels = step_labels[valid_mask]
        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        step_labels = inputs.pop("step_labels")
        inputs = self._prepare_inputs(inputs)
        step_labels = step_labels.to(inputs["input_ids"].device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.float().squeeze(-1)
            valid_mask = step_labels >= 0
            valid_logits = logits[valid_mask]
            valid_labels = step_labels[valid_mask]
            loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, step_labels)


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    logits = np.asarray(eval_pred.predictions).reshape(-1)
    labels = np.asarray(eval_pred.label_ids).reshape(-1)
    valid_mask = labels >= 0
    logits = logits[valid_mask]
    labels = labels[valid_mask].astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def safe_div(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0,
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "mean_prob": float(probs.mean()) if probs.size else 0.0,
    }


def _split_indices_by_query(dataframe: pd.DataFrame, eval_ratio: float, seed: int) -> tuple[List[int], List[int]]:
    if "query_index" not in dataframe.columns:
        indices = np.arange(len(dataframe))
        if len(indices) <= 1:
            return indices.tolist(), []
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        eval_size = min(max(1, int(round(len(indices) * eval_ratio))), len(indices) - 1)
        return indices[eval_size:].tolist(), indices[:eval_size].tolist()

    unique_queries = dataframe["query_index"].drop_duplicates().tolist()
    if len(unique_queries) <= 1:
        return list(range(len(dataframe))), []
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_queries)
    eval_size = min(max(1, int(round(len(unique_queries) * eval_ratio))), len(unique_queries) - 1)
    eval_queries = set(unique_queries[:eval_size])

    train_indices: List[int] = []
    eval_indices: List[int] = []
    for row_index, query_index in enumerate(dataframe["query_index"].tolist()):
        if query_index in eval_queries:
            eval_indices.append(row_index)
        else:
            train_indices.append(row_index)
    return train_indices, eval_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-marker BCE process PRM")
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--eval_parquet", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=16000)
    parser.add_argument("--truncation", type=str, default="left", choices=["left", "left", "right"])
    parser.add_argument("--marker_token", type=str, default="<extra_0>")

    parser.add_argument("--per_device_train_batch_size", type=int, default=6)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SWANLAB_API_KEY", DEFAULT_SWANLAB_API_KEY)
    os.environ.setdefault("SWANLAB_PROJ_NAME", DEFAULT_SWANLAB_PROJ_NAME)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added = tokenizer.add_tokens([args.marker_token])
    print(f"Added {added} marker tokens")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "num_labels": 1,
        "use_cache": False,
        "attn_implementation": DEFAULT_ATTN_IMPLEMENTATION,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, **model_kwargs)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    train_source = TrajectoryBCEDataset(
        args.train_parquet,
        tokenizer,
        marker_token=args.marker_token,
        max_length=args.max_length,
        truncation=args.truncation,
    )

    if args.eval_parquet:
        train_dataset = train_source
        eval_dataset = TrajectoryBCEDataset(
            args.eval_parquet,
            tokenizer,
            marker_token=args.marker_token,
            max_length=args.max_length,
            truncation=args.truncation,
        )
        train_indices = list(range(len(train_source.dataframe)))
        eval_indices = list(range(len(eval_dataset.dataframe)))
    else:
        train_indices, eval_indices = _split_indices_by_query(train_source.dataframe, args.eval_ratio, args.seed)
        train_dataset = Subset(train_source, train_indices)
        eval_dataset = Subset(train_source, eval_indices) if eval_indices else None

    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    print(f"train_size={len(train_dataset)} eval_size={0 if eval_dataset is None else len(eval_dataset)}")
    _debug_print_dataframe_rows("Train Samples", train_source, train_indices, tokenizer)
    if args.eval_parquet:
        _debug_print_dataframe_rows("Eval Samples", eval_dataset, eval_indices, tokenizer)
    elif eval_indices:
        _debug_print_dataframe_rows("Eval Samples", train_source, eval_indices, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        report_to=DEFAULT_REPORT_TO,
        eval_strategy="steps" if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model="f1" if has_eval else None,
        greater_is_better=True if has_eval else None,
    )

    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
