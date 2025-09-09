#!/usr/bin/env python3
# coding: utf-8
"""
Test script for initialize_tools_from_config invocation.

Usage:
  # from repo root
  python3 scripts/test_initialize_agentcpm_mcp_tool.py --config path/to/tools.yaml

If you don't pass --config, it defaults to "./tools.yaml".
The script will try to find and import initialize_tools_from_config inside the verl/tools package
by scanning Python modules under verl/tools.
"""

import argparse
import importlib
import logging
import sys
import traceback
from pathlib import Path
import json
from verl.tools.utils.tool_registry import initialize_tools_from_config
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("test_init_mcp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", type=str, default="./examples/sglang_multiturn/config/tool_config/agentcpm_mcp_tool_config.yaml", help="path to tools config (tools.yaml)")
    args = p.parse_args()


    tool_list = initialize_tools_from_config(args.config)
    # print('tool_list', tool_list)

    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    
    # tool_map = {tool.name: tool for tool in tool_list}

    print('tool_schemas', tool_schemas)

    # sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]



if __name__ == "__main__":
    main()
