import json, logging, httpx, asyncio
from typing import Dict, List, Any, Optional



logger = logging.getLogger(__name__)

class _TextPart:
    """让返回结果长得像 MCP Content 里的一条 text 分片，便于后续统一解析。"""
    __slots__ = ("type", "text")
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _CallResult:
    """与 MCPBaseTool 期望的 call_result.content 对齐（即便我们不会用 MCPBaseTool）。"""
    __slots__ = ("status", "content", "raw")
    def __init__(self, status: str, text: str, raw: Any):
        self.status = status
        self.content = [_TextPart(text)]
        self.raw = raw


def _normalize_params(params: dict | None) -> dict:
    if not params:
        return {"type": "object", "properties": {}, "required": []}
    out = dict(params)
    if "required" not in out:
        out["required"] = []
    return out

class MCPManager:
    """
    仅通过 /mcpapi 的 REST 控制面工作：
      - GET /servers
      - GET /tools  以及 /server/{name}/tools
      - POST /tool/{server.tool}
    暴露的 API 与你原来的 Manager 保持一致：initialize / fetch_tool_schemas / call_tool
    """
    rootServerName = "mcpServers"
    initialized = False

    def __init__(self, manager_url: str = "http://127.0.0.1:8088/mcpapi", timeout: int = 30):
        self.base = manager_url.rstrip("/")
        self.timeout = timeout
        self.http = httpx.AsyncClient(timeout=timeout)
        self.servers: List[str] = []
        self.tools_by_server: Dict[str, List[dict]] = {}
        self.all_tools: List[dict] = []
        # tool_name -> True（REST 下无需多 client，直接 POST）
        self.tool_client_mapping: Dict[str, bool] = {}

    async def initialize(self) -> bool:
        try:
            # 1) 发现 servers
            r = await self.http.get(f"{self.base}/servers")
            r.raise_for_status()
            self.servers = r.json().get("servers", [])
            logger.info(f"REST MCP servers: {self.servers}")

            # 2) 拉所有工具（总表 + 分服务器；二者任一即可）
            #   /tools 里的 function.name 已经是 "server.tool" 形式，可直接做注册名
            tr = await self.http.get(f"{self.base}/tools")
            tr.raise_for_status()
            self.all_tools = tr.json().get("tools", [])

            for s in self.servers:
                try:
                    ts = await self.http.get(f"{self.base}/server/{s}/tools")
                    ts.raise_for_status()
                    self.tools_by_server[s] = ts.json().get("tools", [])
                except Exception as e:
                    logger.warning(f"GET /server/{s}/tools failed: {e}")
                    self.tools_by_server[s] = []

            # TODO:真的是这样的吗
            # 3) 建立 tool 映射
            for item in self.all_tools:
                if "function" in item:
                    name = item["function"]["name"]  # 已是 <server>.<tool>
                    self.tool_client_mapping[name] = True

            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Initialize MCPManager(REST) failed: {e}")
            return False

    # —— VERL registry 需要的 schema 列表（OpenAI function 形状）
    async def fetch_tool_schemas(self, tool_selected_list: Optional[List[str]]) -> List[dict]:
        out: List[dict] = []
        for item in self.all_tools:
            if "function" not in item:
                continue
            fn = item["function"]
            name = fn.get("name")
            if tool_selected_list and name not in tool_selected_list:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": name,  # 形如 "mcp-code-executor.execute_code"
                    "description": fn.get("description", ""),
                    "parameters": _normalize_params(fn.get("parameters", {}))
                }
            })
        return out

    # —— 执行工具（POST /tool/{server.tool}）
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], timeout: Optional[int] = None) -> _CallResult:
        assert self.initialized, "MCPManager not initialized"
        if tool_name not in self.tool_client_mapping:
            raise ValueError(f"Unknown tool: {tool_name}")

        url = f"{self.base}/tool/{tool_name}"
        try:
            r = await self.http.post(url, json=parameters, timeout=timeout or self.timeout)
            r.raise_for_status()
            data = r.json()
            status = data.get("status", "success")
            # 尽量把“看得见”的文本塞到 content 里；保留原始体在 raw，供上层需要时取用
            text = json.dumps(data, ensure_ascii=False)
            return _CallResult(status=status, text=text, raw=data)
        except httpx.TimeoutException:
            return _CallResult(status="error", text=json.dumps({"error": "timeout"}), raw=None)
        except httpx.HTTPStatusError as e:
            return _CallResult(status="error", text=json.dumps({"error": f"http {e.response.status_code}", "detail": e.response.text}, ensure_ascii=False), raw=None)
        except Exception as e:
            return _CallResult(status="error", text=json.dumps({"error": str(e)}, ensure_ascii=False), raw=None)