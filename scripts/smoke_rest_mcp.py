import asyncio
import json
import os
from omegaconf import OmegaConf

from verl.tools.rest_tool import RESTMCPTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.utils.mcp_clients.AgentCPMMCPClientManager import MCPManager

MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8088/mcpapi")

async def main():
    mgr = MCPManager(manager_url=MCP_BASE, timeout=20)
    ok = await mgr.initialize()
    if not ok:
        raise SystemExit("MCPManager 初始化失败")
    RESTMCPTool._rest_manager = mgr

    schemas = await mgr.fetch_tool_schemas(None)
    print(f"发现 {len(schemas)} 个工具")
    print("示例工具：", [s["function"]["name"] for s in schemas[:5]])

    # 找到 mcp-code-executor.execute_code
    target = "mcp-code-executor.execute_code"
    schema_dict = next(s for s in schemas if s["function"]["name"] == target)
    tool_schema = OpenAIFunctionToolSchema.model_validate(schema_dict)
    tool = RESTMCPTool(config={"timeout": 20}, tool_schema=tool_schema)

    instance_id, _ = await tool.create()
    resp, _, metrics = await tool.execute(instance_id, {"code": "print(40+2)"})
    print("执行输出：", resp.text[:200], "...")
    print("metrics：", metrics)

    # 并发 5 次
    async def _one(i):
        r, _, _ = await tool.execute(instance_id, {"code": f"print({i}*2)"})
        return r.text

    outs = await asyncio.gather(*[_one(i) for i in range(5)])
    print("并发返回条数：", len(outs))

if __name__ == "__main__":
    asyncio.run(main())
