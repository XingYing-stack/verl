#!/usr/bin/env python3
# coding: utf-8
"""Smoke test for tool initialization via config with lightweight REST calls.

The script performs three actions:
  1. Load tools using ``initialize_tools_from_config``
  2. Dump the discovered tool schemas (names should now be the bare tool ids)
  3. Issue a synthetic request to each tool using heuristic payloads so that the
     REST manager code path is exercised even if the backend only returns an error

Additionally, this script supports a focused case study for the browser agent:
  - Case study: call the ``fetch_url`` tool with provided URLs and a purpose, and
    show the response text (which may include an LLM-generated summary if your
    browser_agent is enabled in config).

Example usage (from repo root):

    python3 scripts/smoke_test_initialize_from_config.py \
        --config examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml \
        --max-tools 5

The ``--max-tools`` flag limits how many tools we try to execute. Use
``--dry-run`` to only print schemas without issuing calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

from verl.tools.rest_tool import RESTMCPTool
from verl.tools.utils.tool_registry import initialize_tools_from_config


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
logger = logging.getLogger("smoke_init_mcp")

def _default_value(field: str, schema: Mapping[str, Any]) -> Any:
    enum_vals = schema.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        return enum_vals[0]

    typ = schema.get("type")
    if typ == "string":
        if field == "code":
            return "print('hello world')"
        if field in {"url", "uri"}:
            return "https://example.com"
        if field.lower().endswith("id"):
            return "fake-id"
        return f"fake-{field}"
    if typ in {"integer", "number"}:
        return 1
    if typ == "boolean":
        return True
    if typ == "array":
        return []
    if typ == "object":
        return {}

    # Fallback when type info is missing
    return None


def _build_payload(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    props = parameters.get("properties", {}) if parameters else {}
    required = parameters.get("required", []) if parameters else []
    payload: Dict[str, Any] = {}

    # Always populate required fields; optionally pre-fill a couple of useful optional ones
    candidate_fields = set(props.keys())
    candidate_fields.update(required)

    for name in candidate_fields:
        field_schema = props.get(name, {})
        if name not in required and not field_schema:
            continue
        payload[name] = _default_value(name, field_schema)
    return payload


async def _exercise_tool(tool: RESTMCPTool, payload: Dict[str, Any]) -> None:
    instance_id, _ = await tool.create()
    try:
        resp, _, metrics = await tool.execute(instance_id, payload)
        logger.info("tool=%s response=%s metrics=%s", tool.name, resp.text[:160], metrics)
    except Exception as exc:  # noqa: BLE001 - smoke script should never crash on tool errors
        logger.error("tool=%s failed with %s", tool.name, exc, exc_info=True)
    finally:
        try:
            await tool.release(instance_id)
        except Exception:  # noqa: BLE001 - best effort cleanup
            logger.debug("tool=%s release failed", tool.name, exc_info=True)


async def _exercise_tools(tool_list: list[RESTMCPTool], max_tools: int | None, dry_run: bool) -> None:
    if dry_run:
        logger.info("Dry-run mode: skipping tool execution")
        return

    for idx, tool in enumerate(tool_list, start=1):
        if max_tools is not None and idx > max_tools:
            break

        schema_dict = tool.get_openai_tool_schema().model_dump()
        payload = _build_payload(schema_dict["function"].get("parameters", {}))
        logger.info("[%d] exercising tool=%s payload=%s", idx, tool.name, json.dumps(payload, ensure_ascii=False))
        await _exercise_tool(tool, payload)


async def _run_case_study_fetch_url(
    tool_list: list[RESTMCPTool], urls: List[str], purpose: str, timeout_s: int | None
) -> None:
    # Find the fetch_url tool (bare or namespaced)
    target_names = {"fetch_url"}
    # also accept names ending with .fetch_url just in case
    fetch_tool: RESTMCPTool | None = None
    for t in tool_list:
        if t.name in target_names or t.name.endswith(".fetch_url"):
            fetch_tool = t
            break

    if fetch_tool is None:
        logger.error("Could not find 'fetch_url' in loaded tools. Tools: %s", [t.name for t in tool_list])
        return

    schema_dict = fetch_tool.get_openai_tool_schema().model_dump()
    params = schema_dict.get("function", {}).get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}

    # Ensure payload matches schema: url should be an array, purpose a string
    payload: Dict[str, Any] = {
        "url": urls,
        "purpose": purpose
    }

    # Attach optional fields if present in schema with reasonable defaults
    for optional_name in (set(props.keys()) - set(["url", "purpose"])):
        try:
            payload.setdefault(optional_name, _default_value(optional_name, props.get(optional_name, {})))
        except Exception:
            continue

    # Respect an optional timeout override if the REST tool supports it
    if isinstance(timeout_s, int) and timeout_s > 0:
        try:
            fetch_tool.timeout = timeout_s
        except Exception:
            pass

    logger.info("[case-study] exercising tool=%s payload=%s", fetch_tool.name, json.dumps(payload, ensure_ascii=False))
    instance_id, _ = await fetch_tool.create()
    try:
        resp, _, metrics = await fetch_tool.execute(instance_id, payload)
        print("\n===== fetch_url response (truncated by tool if configured) =====\n")
        print(resp.text)
        print("\n================ metrics ================\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        try:
            await fetch_tool.release(instance_id)
        except Exception:
            logger.debug("tool=%s release failed", fetch_tool.name, exc_info=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test AgentCPM REST MCP tool config")
    default_config = Path("/workspace/fanshengda/verl/examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml")
    parser.add_argument("--config", "-c", type=Path, default=default_config, help="path to tools config yaml")
    parser.add_argument("--max-tools", type=int, default=5, help="limit number of tools to execute")
    parser.add_argument("--dry-run", action="store_true", help="only print schemas, skip tool execution")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="number of concurrent initialize_tools_from_config calls",
    )
    # Case study: fetch_url + LLM summary
    parser.add_argument("--case-study-fetch-url", action="store_true", help="run a focused case study with fetch_url")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="URL to fetch (repeatable). If omitted, uses https://example.com",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        default="",
        help="Purpose for visiting the page(s), used by the browser agent summarizer.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional per-call timeout seconds for the REST tool call.",
    )
    args = parser.parse_args()

    def _load_once(run_id: int) -> Tuple[int, List[RESTMCPTool]]:
        logger.info("[%d] Loading tools from %s", run_id, args.config)
        tool_list_local = initialize_tools_from_config(str(args.config))
        logger.info("[%d] Loaded %d tools", run_id, len(tool_list_local))
        return run_id, tool_list_local

    tool_lists: List[Tuple[int, List[RESTMCPTool]]] = []
    errors: List[Tuple[int, Exception]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(_load_once, idx): idx for idx in range(1, max(1, args.concurrency) + 1)}
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result = future.result()
                tool_lists.append(result)
            except Exception as exc:  # noqa: BLE001 - surface init issues
                errors.append((run_id, exc))


    for run_id, exc in errors:
        logger.error("[%d] initialization failed: %s", run_id, exc, exc_info=True)

    if not tool_lists:
        logger.error("All concurrent initializations failed; exiting")
        logger.warning("No tools initialized; exiting")
        return

    tool_lists.sort(key=lambda item: item[0])
    primary_tools = tool_lists[0][1]
    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in primary_tools]
    print("tool_schemas\n", tool_schemas)

    if not primary_tools:
        logger.warning("Primary tool list empty; exiting")
        return

    if not hasattr(RESTMCPTool, "_rest_manager") or RESTMCPTool._rest_manager is None:
        logger.warning("RESTMCPTool manager missing; skipping execution phase")
        return

    # Run case study if requested, otherwise run generic exercise
    if args.case_study_fetch_url:
        urls = args.urls if args.urls else ["https://arxiv.org/pdf/2510.26474"]
        try:
            asyncio.run(_run_case_study_fetch_url(primary_tools, urls, args.purpose, args.timeout))
        except KeyboardInterrupt:
            pass
    else:
        asyncio.run(_exercise_tools(primary_tools, args.max_tools, args.dry_run))


if __name__ == "__main__":
    main()