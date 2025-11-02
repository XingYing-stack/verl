from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import types


# Import the RL simulation entry from the smoke script
import sys
from pathlib import Path

# Ensure repo root is on sys.path for importing from scripts/
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.smoke_test_initialize_from_config import _simulate_rl_training  # type: ignore  # noqa: E402


@dataclass
class _Resp:
    text: str


class _FakeSchema:
    def __init__(self) -> None:
        self._schema = {
            "function": {
                "parameters": {
                    "properties": {
                        "foo": {"type": "string"},
                    },
                    "required": ["foo"],
                }
            }
        }

    def model_dump(self) -> Dict[str, Any]:  # mimic pydantic model
        return self._schema


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[Dict[str, Any]] = []

    def get_openai_tool_schema(self) -> _FakeSchema:
        return _FakeSchema()

    async def create(self) -> Tuple[str, Dict[str, Any]]:
        return f"inst-{self.name}", {}

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs: Any):  # noqa: ANN401
        # Capture the context monitor passed from the simulation
        self.calls.append({
            "instance_id": instance_id,
            "parameters": parameters,
            "context_monitor": kwargs.get("context_monitor"),
        })
        return _Resp(text=f"ok-{self.name}"), None, {}

    async def release(self, instance_id: str) -> None:
        return None


def test_simulate_context_warning_rl_loop_exercises_tools():
    # Two tools, both should be exercised each step
    tools = [_FakeTool("t1"), _FakeTool("t2")]

    episodes = 2
    steps_per_episode = 3
    tools_per_step = 2
    # pick a budget and ratio so that the ramp can cross threshold
    base_context = {"current_tokens": 20000, "max_tokens": 40000, "warning_ratio": 0.9}

    asyncio.run(
        _simulate_rl_training(
            tools,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            tools_per_step=tools_per_step,
            base_context=base_context,
        )
    )

    # Each tool should be called exactly episodes * steps_per_episode times
    expected_calls = episodes * steps_per_episode
    for t in tools:
        assert len(t.calls) == expected_calls, f"{t.name} expected {expected_calls} calls, got {len(t.calls)}"

        # Ensure context monitor is present and at least one step exceeds threshold
        cms = [c.get("context_monitor") for c in t.calls]
        assert all(isinstance(cm, dict) for cm in cms)
        max_tokens = base_context["max_tokens"]
        threshold = int(max_tokens * base_context["warning_ratio"])  # type: ignore[arg-type]
        assert any(int(cm["current_tokens"]) >= threshold for cm in cms if isinstance(cm, dict))
