#!/usr/bin/env python3
# coding: utf-8
"""Smoke test for tool initialization via config with lightweight REST calls.

The script performs three actions:
  1. Load tools using ``initialize_tools_from_config``
  2. Dump the discovered tool schemas (names should now be the bare tool ids)
  3. Issue a synthetic request to each tool using heuristic payloads so that the
     REST manager code path is exercised even if the backend only returns an error

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test AgentCPM REST MCP tool config")
    default_config = Path("examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml")
    parser.add_argument("--config", "-c", type=Path, default=default_config, help="path to tools config yaml")
    parser.add_argument("--max-tools", type=int, default=5, help="limit number of tools to execute")
    parser.add_argument("--dry-run", action="store_true", help="only print schemas, skip tool execution")
    args = parser.parse_args()

    logger.info("Loading tools from %s", args.config)
    tool_list = initialize_tools_from_config(str(args.config))

    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    print("tool_schemas", tool_schemas)

    if not tool_list:
        logger.warning("No tools initialized; exiting")
        return

    if not hasattr(RESTMCPTool, "_rest_manager") or RESTMCPTool._rest_manager is None:
        logger.warning("RESTMCPTool manager missing; skipping execution phase")
        return

    asyncio.run(_exercise_tools(tool_list, args.max_tools, args.dry_run))


if __name__ == "__main__":
    main()
