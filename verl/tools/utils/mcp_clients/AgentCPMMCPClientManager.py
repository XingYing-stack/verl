import json, logging, httpx, asyncio
from typing import Dict, List, Any, Optional
from openai import OpenAI
import tiktoken  # type: ignore
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

    def __init__(
        self,
        manager_url: str = "http://127.0.0.1:8088/mcpapi",
        timeout: int = 60,
        retries: int = 5,
        browser_agent: Optional[dict] = None,
    ):
        self.base = manager_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        # 不在此处持久化 AsyncClient，避免跨事件循环复用导致绑定错误。
        # 在每次请求中临时创建 AsyncClient（async with httpx.AsyncClient(...)).
        # 这样可以彻底避免 “Event bound to a different event loop” 问题。
        self.http = None  # deprecated: kept for backward compatibility, not used
        self.servers: List[str] = []
        self.tools_by_server: Dict[str, List[dict]] = {}
        self.all_tools: List[dict] = []
        # 记录工具与 server 的映射，以支持对外暴露纯 tool 名称时仍能找到 server
        self.tool_client_mapping: Dict[str, Optional[str]] = {}
        # 保留带 server 前缀的原始名字，便于兼容旧调用方式
        self.full_tool_name_mapping: Dict[str, Optional[str]] = {}
        # 浏览器处理器配置
        ba = browser_agent or {}
        self.browser_agent_enabled: bool = bool(ba.get("enable", False))
        self.browser_agent_tool_names: List[str] = list(ba.get("tool_names", []))
        self.browser_agent_url: Optional[str] = ba.get("browser_agent_url")
        try:
            self.browser_agent_timeout: Optional[float] = float(ba.get("timeout")) if ba.get("timeout") is not None else None
        except Exception:
            self.browser_agent_timeout = None
        # OpenAI-compatible credentials
        self.browser_agent_key: Optional[str] = ba.get("browser_agent_key")
        self.browser_agent_model_name: Optional[str] = ba.get("browser_agent_model_name")


        self.enc = tiktoken.get_encoding("cl100k_base")

    async def _get_with_retries(self, url: str) -> httpx.Response:
        retries = self.retries
        """Perform GET with simple retry to handle transient failures."""
        last_exc: Optional[Exception] = None
        # 为了避免跨事件循环绑定问题，这里每次调用都临时创建 AsyncClient
        async with httpx.AsyncClient(timeout=None) as client:
            for attempt in range(1, retries + 1):
                try:
                    # 为每次尝试设置一次总超时，保持与 POST 一致的行为
                    response = await asyncio.wait_for(
                        client.get(url),
                        timeout=float(self.timeout),
                    )
                    response.raise_for_status()
                    return response
                except (asyncio.TimeoutError, Exception) as exc:
                    last_exc = exc
                    logger.warning(
                        "GET %s attempt %d/%d failed: %s",
                        url,
                        attempt,
                        retries,
                        exc,
                    )
                    if attempt == retries:
                        raise
                    await asyncio.sleep(0.2 * attempt)
        raise last_exc  # type: ignore[misc]

    async def initialize(self) -> bool:
        try:
            # 1) 发现 servers
            self.tool_client_mapping.clear()
            self.full_tool_name_mapping.clear()
            r = await self._get_with_retries(f"{self.base}/servers")
            self.servers = r.json().get("servers", [])
            logger.info(f"REST MCP servers: {self.servers}")

            # 2) 拉所有工具（总表 + 分服务器；二者任一即可）
            #   /tools 里的 function.name 已经是 "server.tool" 形式，可直接做注册名
            tr = await self._get_with_retries(f"{self.base}/tools")
            self.all_tools = tr.json().get("tools", [])

            for s in self.servers:
                try:
                    ts = await self._get_with_retries(f"{self.base}/server/{s}/tools")
                    self.tools_by_server[s] = ts.json().get("tools", [])
                except Exception as e:
                    logger.warning(f"GET /server/{s}/tools failed: {e}")
                    self.tools_by_server[s] = []

            # 3) 建立 tool 映射
            for item in self.all_tools:
                if "function" not in item:
                    continue
                raw_name = item["function"].get("name")
                if not raw_name:
                    continue

                server_name: Optional[str] = None
                tool_name = raw_name
                if "." in raw_name:
                    server_name, _, tool_name = raw_name.partition(".")
                # 记录完整名称，兼容旧逻辑
                self.full_tool_name_mapping[raw_name] = server_name

                existing_server = self.tool_client_mapping.get(tool_name)
                if existing_server is not None and existing_server != server_name:
                    logger.warning(
                        "Tool name collision detected for '%s' (servers: %s, %s); keeping the first mapping.",
                        tool_name,
                        existing_server,
                        server_name,
                    )
                    continue
                self.tool_client_mapping[tool_name] = server_name

            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Initialize MCPManager(REST) failed: {e}")
            return False

    # —— VERL registry 需要的 schema 列表（OpenAI function 形状）
    async def fetch_tool_schemas(self, tool_selected_list: Optional[List[str]]) -> List[dict]:
        out: List[dict] = []
        selected_set = set(tool_selected_list) if tool_selected_list else None
        for item in self.all_tools:
            if "function" not in item:
                continue
            fn = item["function"]
            raw_name = fn.get("name")
            if not raw_name:
                continue
            server_name: Optional[str] = None
            tool_name = raw_name
            if "." in raw_name:
                server_name, _, tool_name = raw_name.partition(".")

            if selected_set is not None and tool_name not in selected_set and raw_name not in selected_set:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": fn.get("description", ""),
                    "parameters": _normalize_params(fn.get("parameters", {}))
                }
            })
        return out

    # —— 执行工具（POST /tool/{server.tool}）
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> _CallResult:
        retries = self.retries
        assert self.initialized, "MCPManager not initialized"

        resolved_name = None
        server_name: Optional[str] = None

        if tool_name in self.tool_client_mapping:
            server_name = self.tool_client_mapping[tool_name]
            resolved_name = f"{server_name}.{tool_name}" if server_name else tool_name
        elif tool_name in self.full_tool_name_mapping:
            server_name = self.full_tool_name_mapping[tool_name]
            resolved_name = tool_name
        elif "." in tool_name:
            srv, _, bare = tool_name.partition(".")
            if bare in self.tool_client_mapping:
                expected_server = self.tool_client_mapping[bare]
                if expected_server is not None and expected_server != srv:
                    logger.warning(
                        "Tool '%s' requested with server '%s' but registered under '%s'; proceeding with requested server.",
                        bare,
                        srv,
                        expected_server,
                    )
                server_name = srv
                resolved_name = tool_name
            else:
                resolved_name = tool_name
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        server_name, tool_name = resolved_name.split('.')
        url = f"{self.base}/tool/{resolved_name}"
        last_exc: Optional[Exception] = None
        # 每次调用临时创建 AsyncClient，避免跨 loop 复用
        async with httpx.AsyncClient(timeout=None) as client:
            for attempt in range(1, retries + 1):
                try:
                    # 使用 asyncio.wait_for 包裹以实现“总超时”，避免仅依赖 httpx 的分阶段超时配置
                    overall_timeout = float(timeout or self.timeout)
                    r = await asyncio.wait_for(
                        client.post(url, json=parameters),
                        timeout=overall_timeout,
                    )
                    r.raise_for_status()
                    data = r.json()
                    status = data.get("status", "success")
                    # 默认将原始 JSON 序列化作为文本返回
                    text = json.dumps(data, ensure_ascii=False)

                    # ====================如启用浏览器处理器，并且命中工具名单，则尝试调用外部 LLM 生成摘要==========================
                    if self.browser_agent_enabled and self._should_process_with_browser_agent(tool_name):
                        try:
                            processed_text = await self._maybe_process_with_browser_agent(tool_name, parameters, data)
                            if isinstance(processed_text, str) and processed_text.strip():
                                text = processed_text
                        except Exception as proc_exc:
                            logger.warning(f"Browser agent processing failed: {proc_exc}")
                    # ==============================================================================
                    return _CallResult(status=status, text=text, raw=data)

                except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                    last_exc = e
                    logger.warning(
                        "POST %s attempt %d/%d timeout: %s",
                        url,
                        attempt,
                        retries,
                        e,
                    )
                except httpx.HTTPStatusError as e:
                    last_exc = e
                    logger.warning(
                        "POST %s attempt %d/%d HTTP error: %s",
                        url,
                        attempt,
                        retries,
                        e,
                    )
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "POST %s attempt %d/%d failed: %s",
                        url,
                        attempt,
                        retries,
                        e,
                    )
                # 达到这里说明本次尝试失败
                if attempt == retries:
                    break
                await asyncio.sleep(5 * attempt)

        # 最终失败，按异常类型返回
        if isinstance(last_exc, (httpx.TimeoutException, asyncio.TimeoutError)):
            return _CallResult(status="error", text=json.dumps({"error": "timeout"}), raw=None)
        if isinstance(last_exc, httpx.HTTPStatusError):
            e = last_exc
            assert isinstance(e, httpx.HTTPStatusError)
            return _CallResult(
                status="error",
                text=json.dumps({"error": f"http {e.response.status_code}", "detail": e.response.text}, ensure_ascii=False),
                raw=None,
            )
        return _CallResult(status="error", text=json.dumps({"error": str(last_exc) if last_exc else "unknown"}, ensure_ascii=False), raw=None)

    def _should_process_with_browser_agent(self, tool_name: str) -> bool:
        try:
            if not self.browser_agent_tool_names:
                return False
            return tool_name in self.browser_agent_tool_names
        except Exception:
            return False

    def _extract_purpose(self, parameters: Dict[str, Any]) -> str:
        # 兼容大小写或拼写错误：purpose / propose
        for k in ("purpose", "propose", "Purpose"):
            v = parameters.get(k)
            if isinstance(v, str):
                return v
        return ""

    def _extract_raw_content_from_data(self, data: Any) -> str:
        # 尽力从返回体中提取可用文本
        try:
            if isinstance(data, dict):
                # 常见形状：{"content": "..."}
                if isinstance(data.get("content"), str):
                    return data.get("content")  # type: ignore[return-value]
                # 嵌套: data.data.content
                d = data.get("data")
                if isinstance(d, dict) and isinstance(d.get("content"), str):
                    return d.get("content")  # type: ignore[return-value]
                # 尝试取最长的字符串值
                longest = ""
                for v in data.values():
                    if isinstance(v, str) and len(v) > len(longest):
                        longest = v
                if longest:
                    return longest
            elif isinstance(data, str):
                return data
        except Exception:
            pass
        return ""

    def _build_browser_messages(self, raw_content: str, tool_name: str, purpose: Optional[str]) -> list[dict]:
        # 参考 Browser Processor V5 的提示结构，并在输入前做 Token 截断
        def _truncate(txt: str, max_tokens: int = 120000) -> str:
            if not isinstance(txt, str) or not txt:
                return ""
            try:
                toks = self.enc.encode(txt)
                if len(toks) <= max_tokens:
                    return txt
                return self.enc.decode(toks[:max_tokens])
            except Exception:
                return txt

        raw_content = _truncate(raw_content)

        # "Summarize the key points and extract evidence relevant to the goal."
        # 仅使用 purpose 作为目标导向上下文
        context_awareness_prompt = (
            "\nIMPORTANT CONTEXT:\n"
            f"- The agent's IMMEDIATE PURPOSE for this page is: \"{purpose}\"\n\n"
            "Your primary task is to extract and summarize information that is DIRECTLY RELEVANT to the IMMEDIATE PURPOSE.\n"
        ) if purpose else (
            "\nIMPORTANT CONTEXT:\n"
            f"- The agent's IMMEDIATE PURPOSE for this page is: Summarize the key points and extract evidence.\n\n"
            "Your primary task is to extract and summarize information that is DIRECTLY RELEVANT to the IMMEDIATE PURPOSE.\n"
        )

        user_prompt = f"""
Please process the following webpage or local file content and user goal to extract relevant information:

## **Webpage/Local file Content** {raw_content}

## **User Goal**

{context_awareness_prompt}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format: You MUST use Markdown with the following headings:**
## Rational
(Your analysis of relevance here)

## Evidence
(Your extracted evidence here)

## Summary
(Your final summary here)
"""

        return [{"role": "user", "content": user_prompt}]

    async def _call_browser_llm(self, messages: list[dict]) -> str:
        # 使用 OpenAI Python SDK 访问 OpenAI 兼容的 /v1 接口
        if not self.browser_agent_url or not self.browser_agent_model_name:
            return ""
        try:
            client = OpenAI(api_key=self.browser_agent_key, base_url=self.browser_agent_url)
            # 将同步 SDK 调用放到线程池，外层用 wait_for 做总超时
            def _invoke():
                resp = client.chat.completions.create(
                    model=self.browser_agent_model_name, messages=messages, temperature=0.0, top_p=1.0,
                    n=1, frequency_penalty=0.0, presence_penalty=0.0, logit_bias={}
                )
                try:
                    if resp and resp.choices and resp.choices[0].message.content:
                        return resp.choices[0].message.content
                    if resp and resp.choices and resp.choices[0].message.reasoning_content:
                        return resp.choices[0].message.reasoning_content
                    # OpenAI>=1.0 返回对象：choices[0].message.content
                    return ""
                except Exception:
                    # 退化处理为 dict 访问
                    try:
                        d = resp if isinstance(resp, dict) else resp.model_dump()  # type: ignore[attr-defined]
                        return d.get("choices", [{}])[0].get("message", {}).get("content", "")
                    except Exception:
                        return ""

            effective_timeout = float(self.browser_agent_timeout or self.timeout)
            return await asyncio.wait_for(asyncio.to_thread(_invoke), timeout=effective_timeout)
        except Exception as e:
            logger.warning(f"call browser LLM failed: {e}")
            return ""

    async def _maybe_process_with_browser_agent(self, tool_name: str, parameters: Dict[str, Any], data: Any) -> Optional[str]:
        # 仅解析 purpose，用于目标导向摘要
        purpose = self._extract_purpose(parameters)
        if not purpose:
            logger.warning(f"无法从 参数 {parameters} 中解析 'purpose'。")

        raw_content = self._extract_raw_content_from_data(data)
        if not raw_content:
            return None

        # 特判 fetch_url：可能含多个页面，用分隔符切分
        if tool_name == "fetch_url" and "\n=======\n" in raw_content:
            import re
            pages = [p for p in raw_content.split("\n=======\n") if p.strip()]
            async def process_block(i: int, block: str) -> Optional[str]:
                url = None
                try:
                    m = re.match(r"The content from (https?://[^\s]+):", block)
                    url = m.group(1) if m else None
                except Exception:
                    url = None
                messages = self._build_browser_messages(block, tool_name, purpose)
                summary = await self._call_browser_llm(messages)
                if not summary:
                    return None
                prefix = f"URL: {url}" if url else f"PAGE_{i+1}"
                return f"{prefix}\n\nSummary:\n{summary}"

            results = await asyncio.gather(*[process_block(i, b) for i, b in enumerate(pages)], return_exceptions=True)
            parts: list[str] = []
            for res in results:
                if isinstance(res, str) and res:
                    parts.append(res)
            if parts:
                return "\n\n---\n\n".join(parts)
            return None

        # 单页/普通工具：直接处理
        messages = self._build_browser_messages(raw_content, tool_name, purpose)
        summary = await self._call_browser_llm(messages)
        return summary if summary else None
