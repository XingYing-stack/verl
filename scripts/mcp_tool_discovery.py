#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖：
  pip install httpx

说明：
  - 按顺序调用 MCP Manager 的 /servers、/tools、/server/{server}/tools
  - 统计并打印服务器与工具数量
  - 生成 openai_tools（OpenAI function calling 兼容格式）
"""

import asyncio
import json
from typing import Dict, List, Any

import httpx


async def fetch_json(client: httpx.AsyncClient, url: str) -> Any:
    resp = await client.get(url, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


async def main():
    manager_url = "http://101.6.41.97:18080/container/21ee0059c73a/mcpapi"

    servers: Dict[str, Dict] = {}
    tools_by_server: Dict[str, List[Dict]] = {}
    all_tools: List[Dict] = []
    openai_tools: List[Dict] = []

    async with httpx.AsyncClient() as client:
        # 1) 获取服务器列表
        servers_data = await fetch_json(client, f"{manager_url}/servers")
        server_names = servers_data.get("servers", []) or []
        servers = {name: {} for name in server_names}
        print(f"获取到 {len(servers)} 个MCP服务器")

        # 2) 获取所有工具（聚合视角）
        tools_data = await fetch_json(client, f"{manager_url}/tools")
        all_tools = tools_data.get("tools", []) or []
        print(f"获取到 {len(all_tools)} 个MCP工具（聚合）")

        # 3) 按服务器获取工具，并构建 openai_tools
        for server in servers.keys():
            try:
                server_tools_data = await fetch_json(client, f"{manager_url}/server/{server}/tools")
                tools_list = server_tools_data.get("tools", []) or []
                tools_by_server[server] = tools_list
                print(f"服务器 '{server}' 上有 {len(tools_list)} 个工具")

                # 转成 OpenAI function 调用格式
                for tool in tools_list:
                    if "function" in tool and isinstance(tool["function"], dict):
                        fn = tool["function"]
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": fn.get("name"),
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {})
                            }
                        })
            except httpx.HTTPError as e:
                print(f"[警告] 拉取服务器 '{server}' 工具失败：{e}")

    # —— 输出汇总 —— #
    print("\n=== 汇总 ===")
    print(f"服务器数：{len(servers)}")
    print(f"聚合工具总数（/tools）：{len(all_tools)}")
    print(f"按服务器工具总数：{sum(len(v) for v in tools_by_server.values())}")
    print(f"OpenAI 兼容工具条目数：{len(openai_tools)}")

    # 如需查看具体内容，取消注释：
    # print("\nopenai_tools 示例：")
    # print(json.dumps(openai_tools[:3], ensure_ascii=False, indent=2))

    # 如果你需要把结果给到其他模块用，也可以返回它们


    print(openai_tools)
    return {
        "servers": servers,
        "all_tools": all_tools,
        "tools_by_server": tools_by_server,
        "openai_tools": openai_tools,
    }


if __name__ == "__main__":
    asyncio.run(main())
