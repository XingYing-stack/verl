import json, logging, os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RESTMCPTool(BaseTool):
    """
    与 BaseTool 风格一致的 REST 兜底工具类：
      - self.name = "<server>.<tool>"（来自 schema）
      - create/execute/release 生命周期与现有一致
      - execute() 通过 AgentCPMMCPClineManager.MCPManager.call_tool(...) 走 REST
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = config.get("timeout", 30)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    async def _call(self, instance_id: str, parameters: dict[str, Any]) -> tuple[str, dict]:
        # 从我们刚写的 REST Manager 调用（保持现有 import 习惯）
        from verl.tools.utils.mcp_clients.AgentCPMMCPClientManager import MCPManager as RESTManager

        # 约定：RESTManager 是全局单例或在外层初始化后已存在 —— 如果不是，你也可以做成模块级全局
        # 这里直接用类属性/单例缓存以避免重复初始化
        if not hasattr(RESTMCPTool, "_rest_manager") or RESTMCPTool._rest_manager is None:
            raise RuntimeError("REST MCPManager is not initialized. Call initialize_agentcpm_mcp_tool first.")

        mgr: RESTManager = RESTMCPTool._rest_manager
        result = await mgr.call_tool(self.name, parameters, timeout=self.timeout)
        # 我们把原始 JSON 压成字符串返回（保持 execute 返回文本）
        # 也可在此解析结构，提纯你要的字段
        return (result.content[0].text if result.content else ""), {"status": result.status}

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        if not parameters:
            msg = "parameters is empty"
            logger.error(msg)
            return ToolResponse(text=json.dumps({"error": msg}, ensure_ascii=False)), 0.0, {}

        try:
            text, meta = await self._call(instance_id, parameters)
            self._instance_dict[instance_id]["reward"].append(text.strip())
            metrics = {"status": meta.get("status", "unknown")}
            return ToolResponse(text=text), 0.0, metrics
        except Exception as e:
            err = json.dumps({"error": f"Tool exec failed: {e}"}, ensure_ascii=False)
            return ToolResponse(text=err), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # 你原实现是返回字符串数组；若需要数值，这里可以自定义
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
