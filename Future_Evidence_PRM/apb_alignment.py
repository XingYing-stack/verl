from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


DEFAULT_SEARCH_TOOLS: List[Dict[str, Any]] = [
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
                    "query": {
                        "type": "string",
                        "description": "Single query",
                    },
                },
                "required": ["query_list"],
            },
        },
    }
]


APB_LABEL_SCHEME = """+1: Correct and effective.
    The step is factually correct given the information available at that time
    and clearly moves the task closer to successful completion by:
    (i) correctly invoking a tool or interpreting tool outputs, or
    (ii) introducing valid constraints, decisions, or information that
         reduces the remaining uncertainty of the task.

 0: Neutral or exploratory.
    The step is reasonable but has limited or unclear impact on task progress.
    This includes exploratory reasoning, redundant restatements, partial planning,
    or cases where the correctness is debatable given the available evidence.
    Tool calls that fail due to external reasons (e.g., timeout, 404), when the
    attempt itself is reasonable, are typically labeled 0.

-1: Incorrect or harmful.
    The step contains factual errors, misinterprets tool outputs, violates
    constraints, repeats failed actions without a meaningful change in strategy,
    fabricates tool results or evidence, or otherwise pushes the trajectory away
    from successful completion."""


APB_COMMON_RULES = """- Only assistant messages are labeled. User and tool messages serve only as evidence.
- Any step labeled -1 triggers a cumulative penalty: all subsequent assistant steps
  in the same workflow should also be labeled -1, unless one of the following holds:
    (i) the assistant explicitly acknowledges and corrects the earlier mistake, or
    (ii) the assistant produces a subsequent step that no longer depends on the
         incorrect assumption and effectively resumes progress toward the task.
- Repeating the same failed action without a meaningful change in parameters or
  strategy typically transitions from 0 to -1.
- If an incorrect statement does not affect any subsequent reasoning or actions
  and is not relied upon later, it may be labeled 0; otherwise, it should be labeled -1.
- Any violation of the policies or requirements specified in the system prompt results in a score of -1, except for certain output-formatting norms. The following behaviors are considered acceptable and do not incur penalties: providing a text response simultaneously with a tool call, not conducting reasoning before a tool call, failing to encapsulate reasoning content within <think>...</think> tags, responding to the user while executing a function call, or executing multiple parallel tool calls.
- A score of +1 is assigned if the entire conversation is initiated by the assistant and its first message is a greeting; this exemption applies only to the first message.
- Upon user request, if the assistant executes specific instructions, a score of +1 shall be awarded, notwithstanding any deviation from the overarching objective."""


FUTURE_IMPLICATION_SPEC = """For every labeled assistant step, also provide a concise mechanism-level
future_implication with these fields:
- directions: a list of 1 or 2 plausible future trends that this step may induce
  over the next few steps. Do not restate the current observation or quote tool
  outputs; summarize likely trajectory tendencies instead.
- branch_point: the main assumption, decision, or condition that later determines
  whether the trajectory progresses or derails.
- recoverability: one of easy, possible, hard, none.
- revealing_signal: what later observation, tool feedback, or verification signal
  would expose whether the step is well-grounded."""


STEP_ASSESSMENT_SPEC = """For every labeled assistant step, also provide a concise step_assessment:
- step_assessment: a short local judgment of the current step itself.
- It must use only the information available up to that step, plus an immediately
  following tool observation if that tool observation is part of current_step_messages.
- It must not rely on hidden future continuations, ground-truth answers, or later
  branch outcomes."""


HINDSIGHT_STEP_ICL_EXAMPLES: List[Dict[str, Any]] = [
    {
        "step_assessment": (
            "The assistant correctly synthesizes information from both search results to provide the answer "
            "'Delhi'. The answer is factually correct, properly formatted in the required <answer> tags, "
            "and directly addresses the question."
        ),
        "future_implication": {
            "directions": ["The task will terminate successfully with the correct answer provided"],
            "branch_point": "None - the task is complete with the correct answer",
            "recoverability": "easy",
            "revealing_signal": "Task termination with success status",
        },
        "explanation": (
            "The assistant correctly extracts 'Delhi' from the search results and provides the answer in the "
            "required format. This completes the task successfully."
        ),
        "step_label": 1,
    },
    {
        "step_assessment": (
            "The assistant conducts a redundant search for 'head office of The Oberoi Group' despite already "
            "having this exact information from the previous search result (Doc 1 already stated 'head office "
            "in Delhi'). This is unnecessary duplication."
        ),
        "future_implication": {
            "directions": [
                "The assistant will likely synthesize the redundant search results and still arrive at the correct answer 'Delhi'",
                "The trajectory wastes one step but remains on track to succeed",
            ],
            "branch_point": (
                "Whether the assistant recognizes it already has sufficient information or continues searching unnecessarily"
            ),
            "recoverability": "easy",
            "revealing_signal": "The next assistant message should provide the answer 'Delhi' without further tool calls",
        },
        "explanation": (
            "This search is redundant because the previous search already clearly stated 'The Oberoi Group is "
            "a hotel group with its head office in Delhi.' While not harmful, it doesn't advance the task."
        ),
        "step_label": 0,
    },
]


TRAJECTORY_SYSTEM_PROMPT = f"""You are a strict but fair trajectory annotator for tool-use agents.

You will be given one complete trajectory consisting of system, user, assistant,
and tool messages, together with the tool definitions.

Your task is to label EACH assistant message (each assistant message constitutes
one Step) using the following scheme:

{APB_LABEL_SCHEME}

Important rules:
{APB_COMMON_RULES}
- For step_label, judge each step strictly based on the information available up to
  that point in the trajectory.
- Avoid outcome bias: do not reward or punish a step solely because later steps
  succeed or fail.
- The labeled unit is still the assistant message at the given index. If that
  assistant message is immediately followed by a tool observation caused by the
  same action, you may treat that tool observation as part of the step-local
  consequence when judging the step.
- step_assessment must capture the local quality of the step itself before any
  future-evidence synthesis.
- explanation must justify the current step_label and stay grounded in evidence that
  is visible from the trajectory itself.
- future_implication must summarize 1 or 2 mechanism-level future trends of the
  current step. Do not repeat the current observation, and do not copy
  branch-specific later events.

{STEP_ASSESSMENT_SPEC}
{FUTURE_IMPLICATION_SPEC}

After labeling all assistant steps, also assign a label to:

FINAL_RESULT:
+1: The overall task is successfully completed.
-1: The task fails due to incorrect reasoning, tool misuse, or unresolved errors.

Return JSON only with exactly this schema:
{{
  "step_assessments": {{"steps": {{"<assistant_index>": "short local judgment", ...}}}},
  "future_implications": {{
    "steps": {{
      "<assistant_index>": {{
        "directions": ["...", "..."],
        "branch_point": "...",
        "recoverability": "easy|possible|hard|none",
        "revealing_signal": "..."
      }},
      ...
    }}
  }},
  "explanations": {{
    "steps": {{"<assistant_index>": "short reason for humans", ...}}
  }},
  "step_labels": {{"<assistant_index>": -1|0|1, ...}},
  "final_label": -1|1
}}"""


HINDSIGHT_STEP_SYSTEM_PROMPT = f"""You are a strict but fair trajectory annotator for tool-use agents.

You will be given one current step inside a trajectory, together with
the preceding context, tool definitions, the ground-truth answer(s), and several
hidden future continuations from the same prefix.

Judge ONLY the single current step defined by current_step_messages, but use the
same three-way label semantics as follows:

{APB_LABEL_SCHEME}

Important rules:
{APB_COMMON_RULES}
- The current step must be judged in present tense as it is taken.
- current_step_messages defines the full current step. If it contains an
  immediately following tool observation, treat that tool observation as part of
  the same local step context rather than as hidden future evidence.
- You are a privileged hindsight teacher, not a blind local annotator.
- step_assessment must describe the local quality of the current step using only
  messages_before_current_step and current_step_messages.
- Use the ground-truth answer(s) and hidden future continuations only to infer
  what the current step already implies about future progress, future failure,
  recoverability, and later revealing signals.
- Avoid outcome bias: do not reward or punish a step solely because a later branch
  succeeds or fails. Hidden evidence must point to a concrete property already
  present in the current step.
- If hindsight evidence does not reveal a concrete latent property of the current
  step, default to the best trajectory-only judgment, usually 0 when uncertain.
- explanation must justify the current step_label using evidence understandable from
  the visible trajectory itself. Do not mention trajectory numbers, "future", hidden
  evidence, or later branch-specific details unless absolutely necessary.
- future_implication should compress the future evidence into 1 or 2
  mechanism-level future trends rather than narrating what literally happens
  later or repeating the current observation.

{STEP_ASSESSMENT_SPEC}
{FUTURE_IMPLICATION_SPEC}

Return JSON only with exactly this schema:
{{
  "step_assessment": "short local judgment",
  "future_implication": {{
    "directions": ["...", "..."],
    "branch_point": "...",
    "recoverability": "easy|possible|hard|none",
    "revealing_signal": "..."
  }},
  "explanation": "short reason for humans",
  "step_label": -1|0|1
}}"""


TRAJECTORY_USER_INSTRUCTIONS = """Label every index in assistant_message_indices.

Rules:
- step_assessments.steps MUST contain ALL assistant indices as strings.
- future_implications.steps MUST contain ALL assistant indices as strings.
- Each future_implications.steps entry MUST contain directions, branch_point,
  recoverability, and revealing_signal.
- explanations.steps MUST contain ALL assistant indices as strings.
- step_labels MUST contain ALL assistant indices as strings.
- Emit top-level keys in this exact order: step_assessments, future_implications,
  explanations, step_labels, final_label.
- Keep each step_assessment concise (<= 2 sentences).
- Keep each explanation concise (<= 2 sentences).
- directions MUST be a list of 1 or 2 short strings.
- Keep each future_implication field concise (prefer <= 1 sentence per field).
- final_label MUST be either -1 or 1.
- Return JSON only."""


HINDSIGHT_STEP_USER_INSTRUCTIONS = """Judge the single current step in current_step_messages.

Rules:
- step_assessment MUST be a non-empty string.
- step_label MUST be one of -1, 0, 1.
- future_implication MUST be an object.
- future_implication MUST contain directions, branch_point,
  recoverability, and revealing_signal.
- explanation MUST be a non-empty string.
- Emit keys in this exact order: step_assessment, future_implication,
  explanation, step_label.
- Keep the step_assessment concise (<= 2 sentences).
- Keep the explanation concise (<= 2 sentences).
- directions MUST be a list of 1 or 2 short strings.
- Keep each future_implication field concise (prefer <= 1 sentence per field).
- Use REFERENCE_ANNOTATION_EXAMPLES only as style and decision references; do not
  copy their facts into the current annotation.
- Return JSON only."""


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


def safe_native(obj: Any) -> Any:
    obj = _to_native(obj)
    if isinstance(obj, dict):
        return {str(key): safe_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [safe_native(item) for item in obj]
    return obj


def ensure_tools(tools: Any) -> List[Dict[str, Any]]:
    tools = safe_native(tools)
    if isinstance(tools, list) and tools:
        return [tool for tool in tools if isinstance(tool, dict)]
    return safe_native(DEFAULT_SEARCH_TOOLS)


def assistant_message_indices(messages: Any) -> List[int]:
    messages = safe_native(messages)
    if not isinstance(messages, list):
        return []
    return [idx for idx, msg in enumerate(messages) if isinstance(msg, dict) and msg.get("role") == "assistant"]


def build_trajectory_payload(
    *,
    question: Any,
    task_description: Any,
    tools: Any,
    messages: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[int]]:
    messages = safe_native(messages)
    indices = assistant_message_indices(messages)
    payload = {
        "question": question,
        "task_description": task_description,
        "tools": ensure_tools(tools),
        "messages": list(enumerate(messages)),
        "assistant_message_indices": indices,
        "notes": {
            "step_definition": (
                "Each labeled Step is anchored on one message with role=='assistant'. "
                "If an assistant message is immediately followed by a tool observation, "
                "treat that tool observation as step-local evidence, but keep the label "
                "attached to the assistant index."
            ),
            "output_requirements": (
                "Return JSON with step_assessments, future_implications, explanations, step_labels, and final_label."
            ),
        },
    }
    return payload, indices


def build_trajectory_prompt_messages(
    *,
    question: Any,
    task_description: Any,
    tools: Any,
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], List[int]]:
    payload, indices = build_trajectory_payload(
        question=question,
        task_description=task_description,
        tools=tools,
        messages=messages,
    )
    prompt = [
        {"role": "system", "content": TRAJECTORY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TRAJECTORY_USER_INSTRUCTIONS + "\n\nTRAJECTORY_JSON:\n" + json.dumps(payload, ensure_ascii=False),
        },
    ]
    return prompt, indices


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    json_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = json_block_pattern.findall(text)
    if matches:
        json_str = matches[-1].strip()
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    end = text.rfind("}")
    if end < 0:
        raise ValueError("model output is not JSON")

    brace_count = 0
    start = -1
    for i in range(end, -1, -1):
        if text[i] == "}":
            brace_count += 1
        elif text[i] == "{":
            brace_count -= 1
            if brace_count == 0:
                start = i
                break
    if start < 0:
        raise ValueError("model output is not JSON")

    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output JSON is not an object")
    return obj


def coerce_step_label(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError("step label must be -1/0/1, got bool")
    if isinstance(val, int):
        out = val
    elif isinstance(val, str) and val.strip() in {"-1", "0", "1"}:
        out = int(val.strip())
    else:
        raise ValueError(f"step label must be -1/0/1, got {val!r}")
    if out not in (-1, 0, 1):
        raise ValueError(f"step label must be -1/0/1, got {out}")
    return out


def coerce_final_label(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError("final label must be -1/1, got bool")
    if isinstance(val, int):
        out = val
    elif isinstance(val, str) and val.strip() in {"-1", "1"}:
        out = int(val.strip())
    else:
        raise ValueError(f"final label must be -1/1, got {val!r}")
    if out not in (-1, 1):
        raise ValueError(f"final label must be -1/1, got {out}")
    return out


def coerce_recoverability(val: Any) -> str:
    if not isinstance(val, str):
        raise ValueError(f"recoverability must be a string, got {type(val).__name__}")
    normalized = val.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "easy": "easy",
        "recoverable_easy": "easy",
        "possible": "possible",
        "recoverable": "possible",
        "medium": "possible",
        "hard": "hard",
        "difficult": "hard",
        "unlikely": "hard",
        "none": "none",
        "irreversible": "none",
        "not_recoverable": "none",
        "impossible": "none",
    }
    if normalized not in mapping:
        raise ValueError(f"recoverability must be one of easy/possible/hard/none, got {val!r}")
    return mapping[normalized]


def coerce_text_field(val: Any, *, field_name: str) -> str:
    if val is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    text = val if isinstance(val, str) else str(val)
    text = text.strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def coerce_directions(val: Any) -> List[str]:
    if isinstance(val, list):
        directions = [
            coerce_text_field(item, field_name=f"future_implication.directions[{idx}]")
            for idx, item in enumerate(val)
        ]
    else:
        raise ValueError("future_implication.directions must be a list of strings")

    directions = [direction for direction in directions if direction]
    if not 1 <= len(directions) <= 2:
        raise ValueError(f"future_implication.directions must contain 1 or 2 items, got {len(directions)}")
    return directions


def normalize_future_implication_entry(entry: Any) -> Dict[str, Any]:
    entry = safe_native(entry)
    if not isinstance(entry, dict):
        raise ValueError("future_implication entry must be an object")
    return {
        "directions": coerce_directions(entry.get("directions")),
        "branch_point": coerce_text_field(entry.get("branch_point"), field_name="future_implication.branch_point"),
        "recoverability": coerce_recoverability(entry.get("recoverability")),
        "revealing_signal": coerce_text_field(
            entry.get("revealing_signal"),
            field_name="future_implication.revealing_signal",
        ),
    }


def normalize_step_only_annotation(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step_assessment": coerce_text_field(
            parsed.get("step_assessment"),
            field_name="step_assessment",
        ),
        "future_implication": normalize_future_implication_entry(parsed.get("future_implication")),
        "explanation": coerce_text_field(
            parsed.get("explanation"),
            field_name="explanation",
        ),
        "step_label": coerce_step_label(parsed.get("step_label")),
    }


def normalize_trajectory_annotation(
    parsed: Dict[str, Any],
    *,
    assistant_indices: List[int],
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, Dict[str, Any]], int, Dict[str, str]]:
    expected_keys = [str(idx) for idx in assistant_indices]

    step_assessments_raw = parsed.get("step_assessments")
    if not isinstance(step_assessments_raw, dict):
        raise ValueError("missing step_assessments object")
    step_assessment_steps_raw = step_assessments_raw.get("steps")
    if not isinstance(step_assessment_steps_raw, dict):
        raise ValueError("missing step_assessments.steps object")
    step_assessments: Dict[str, str] = {}
    for key in expected_keys:
        if key not in step_assessment_steps_raw:
            raise ValueError(f"missing step_assessment for assistant index {key}")
        step_assessments[key] = coerce_text_field(
            step_assessment_steps_raw[key],
            field_name=f"step_assessments.steps[{key}]",
        )

    step_labels_raw = parsed.get("step_labels")
    if not isinstance(step_labels_raw, dict):
        raise ValueError("missing step_labels object")

    step_labels: Dict[str, int] = {}
    for key in expected_keys:
        if key not in step_labels_raw:
            raise ValueError(f"missing step label for assistant index {key}")
        step_labels[key] = coerce_step_label(step_labels_raw[key])

    future_implications_raw = parsed.get("future_implications")
    if not isinstance(future_implications_raw, dict):
        raise ValueError("missing future_implications object")
    future_steps_raw = future_implications_raw.get("steps")
    if not isinstance(future_steps_raw, dict):
        raise ValueError("missing future_implications.steps object")
    future_implications: Dict[str, Dict[str, Any]] = {}
    for key in expected_keys:
        if key not in future_steps_raw:
            raise ValueError(f"missing future implication for assistant index {key}")
        future_implications[key] = normalize_future_implication_entry(future_steps_raw[key])

    final_label = coerce_final_label(parsed.get("final_label"))

    explanations_raw = parsed.get("explanations")
    if not isinstance(explanations_raw, dict):
        raise ValueError("missing explanations object")
    steps_raw = explanations_raw.get("steps")
    if not isinstance(steps_raw, dict):
        raise ValueError("missing explanations.steps object")

    explanations: Dict[str, str] = {}
    for key in expected_keys:
        if key not in steps_raw:
            raise ValueError(f"missing explanation for assistant index {key}")
        explanations[key] = coerce_text_field(
            steps_raw[key],
            field_name=f"explanations.steps[{key}]",
        )

    return step_assessments, step_labels, future_implications, final_label, explanations


def build_trajectory_response_json(
    *,
    step_assessments: Dict[str, str],
    step_future_implications: Dict[str, Dict[str, Any]],
    step_explanations: Dict[str, str],
    step_labels: Dict[str, int],
    final_label: int,
) -> Dict[str, Any]:
    return {
        "step_assessments": {
            "steps": {
                str(key): coerce_text_field(value, field_name=f"step_assessments[{key}]")
                for key, value in step_assessments.items()
            }
        },
        "future_implications": {
            "steps": {
                str(key): normalize_future_implication_entry(value)
                for key, value in step_future_implications.items()
            }
        },
        "explanations": {
            "steps": {
                str(key): coerce_text_field(value, field_name=f"step_explanations[{key}]")
                for key, value in step_explanations.items()
            }
        },
        "step_labels": {str(key): int(value) for key, value in step_labels.items()},
        "final_label": int(final_label),
    }
