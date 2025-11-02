# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for AgentCPMMultiTurnSFTDataset.
"""



import json
from pathlib import Path

import pytest
import torch

from verl.utils.dataset.agentcpm_multiturn_sft_dataset import (
    AgentCPMMultiTurnSFTDataset,
    convert_sharegpt_to_conversation,
)


class DummyChatTokenizer:
    """Lightweight tokenizer that mimics the HF chat API used by the dataset."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.tools_calls: list = []
        self.enable_thinking_calls: list = []

    @staticmethod
    def _render(messages, add_generation_prompt):
        parts: list[str] = []
        for message in messages:
            parts.append(f"<<{message['role']}>>")
            parts.append(message["content"])
            parts.append("\n")
        if add_generation_prompt:
            parts.append("<<assistant>>")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # Unused in the dummy implementation
        if not text:
            return []
        return list(text.encode("utf-8"))

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        filtered = [int(token) for token in tokens if int(token) != self.pad_token_id]
        if not filtered:
            return ""
        return bytes(filtered).decode("utf-8")

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize: bool = False,
        return_tensors: str | None = None,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
        **kwargs,
    ):
        del kwargs  # Unused in the dummy implementation
        self.tools_calls.append(tools)
        self.enable_thinking_calls.append(enable_thinking)
        rendered = self._render(messages, add_generation_prompt)
        if not tokenize:
            return rendered
        if return_tensors not in (None, "pt"):
            raise ValueError("Dummy tokenizer only supports return_tensors='pt'")
        encoded = self.encode(rendered)
        return torch.tensor([encoded], dtype=torch.long)


def _write_sharegpt_json(tmp_path: Path) -> tuple[Path, list[dict]]:
    sample_data = [
        {
            "system": "You are a meticulous researcher.",
            "conversations": [
                {"from": "human", "value": "Which movement matched all the conditions?"},
                {
                    "from": "gpt",
                    "value": "<think>Check leader timeline.</think> <answer>Hechalutz.</answer>",
                },
                {"from": "human", "value": "When did the leader return to the USSR?"},
                {"from": "gpt", "value": "The leader returned in 1927 to resume underground work."},
                {"from": "human", "value": ""},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search authoritative sources.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        {
            "system": "Keep answers concise and polite.",
            "conversations": [
                {"from": "human", "value": "Say hello."},
                {"from": "gpt", "value": "<think>Plan greeting.</think> <answer>Hello!</answer>"},
            ],
            "tools": [],
        },
    ]
    json_path = tmp_path / "agentcpm_sample.json"
    json_path.write_text(json.dumps(sample_data), encoding="utf-8")
    return json_path, sample_data


def test_agentcpm_multiturn_dataset_masks_and_padding(tmp_path: Path) -> None:
    json_path, raw_data = _write_sharegpt_json(tmp_path)
    tokenizer = DummyChatTokenizer()
    config = {"max_length": 1024, "truncation": "error", "enable_thinking": False}
    dataset = AgentCPMMultiTurnSFTDataset(json_files=str(json_path), tokenizer=tokenizer, config=config)

    assert len(dataset) == len(raw_data)

    sample = dataset[0]
    expected_keys = {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    assert expected_keys == set(sample)

    for key in expected_keys:
        tensor = sample[key]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.shape == sample["input_ids"].shape

    seq_length = sample["input_ids"].shape[0]
    assert seq_length == config["max_length"]
    effective_length = int(sample["attention_mask"].sum().item())
    assert effective_length > 0
    if effective_length < seq_length:
        pad_slice = slice(effective_length, None)
        assert torch.all(sample["attention_mask"][pad_slice] == 0)
        assert torch.all(sample["loss_mask"][pad_slice] == 0)
        assert torch.all(sample["input_ids"][pad_slice] == tokenizer.pad_token_id)

    assistant_tokens = sample["input_ids"][sample["loss_mask"] == 1]
    assistant_text = tokenizer.decode(assistant_tokens)
    assert "Hechalutz" in assistant_text
    assert "returned in 1927" in assistant_text

    non_assistant_tokens = sample["input_ids"][sample["loss_mask"] == 0]
    non_assistant_text = tokenizer.decode(non_assistant_tokens)
    assert raw_data[0]["system"] in non_assistant_text
    for turn in raw_data[0]["conversations"]:
        content = turn["value"]
        if not content:
            continue
        if turn["from"] == "gpt":
            assert content in assistant_text
            assert content not in non_assistant_text
        else:
            assert content in non_assistant_text
            assert content not in assistant_text

    assert any(call == raw_data[0]["tools"] for call in tokenizer.tools_calls)
    assert all(flag is False for flag in tokenizer.enable_thinking_calls if flag is not None)


def test_agentcpm_multiturn_dataset_truncation_error(tmp_path: Path) -> None:
    json_path, _ = _write_sharegpt_json(tmp_path)
    tokenizer = DummyChatTokenizer()
    config = {"max_length": 16, "truncation": "error"}
    dataset = AgentCPMMultiTurnSFTDataset(json_files=str(json_path), tokenizer=tokenizer, config=config)

    with pytest.raises(ValueError):
        _ = dataset[0]


def test_agentcpm_multiturn_dataset_filters_overlong_samples(tmp_path: Path) -> None:
    json_path, raw_data = _write_sharegpt_json(tmp_path)
    tokenizer = DummyChatTokenizer()

    long_messages = convert_sharegpt_to_conversation(raw_data[0])
    short_messages = convert_sharegpt_to_conversation(raw_data[1])

    long_len = tokenizer.apply_chat_template(long_messages, tokenize=True, return_tensors="pt").shape[-1]
    short_len = tokenizer.apply_chat_template(short_messages, tokenize=True, return_tensors="pt").shape[-1]

    assert long_len > short_len

    max_prompt_length = short_len + 5
    config = {
        "max_length": 256,
        "truncation": "right",
        "enable_thinking": False,
        "filter_overlong_prompts": True,
        "max_prompt_length": max_prompt_length,
    }

    dataset = AgentCPMMultiTurnSFTDataset(json_files=str(json_path), tokenizer=tokenizer, config=config)

    assert len(dataset) == 1
    sample = dataset[0]
    assistant_text = tokenizer.decode(sample["input_ids"][sample["loss_mask"] == 1])
    assert "Hello" in assistant_text
