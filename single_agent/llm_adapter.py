"""LLM 适配器。

设计要点:
- 统一 OpenAI / Anthropic 两种协议,通过 ``api_format`` 自动切换。
- 兼容 ``tools/llm`` 包(默认后端);允许注入任何带 ``_http._client`` 的 client。
- 不依赖 LLMClient 内部的 ``_parse``(纯 tool_calls 响应会抛错)——自己 parse。
- 把 LLM 调用放到 ``run_in_executor``,因为底层 httpx 是同步的。
- 错误 raise(``LLMCallError``)而不是吞成字符串返回,由上层决定。

子类化/注入:
    >>> class FakeLLM(LLMAdapter):
    ...     async def chat_with_tools(self, messages, *, tools=None, **kw):
    ...         return LLMResult(content="", tool_calls=[
    ...             ToolCall(id="1", name="done", arguments={"answer": "hi"}),
    ...         ])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("single_agent.llm_adapter")


class LLMCallError(Exception):
    """LLM 调用失败(网络、鉴权、解析等)。"""


@dataclass
class ToolCall:
    """统一的工具调用表示。"""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    """统一的 LLM 调用结果。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMAdapter:
    """LLM 客户端抽象。

    默认实现走 ``tools/llm`` 包(它管理 token、provider 切换、协议分发);
    如果只想用纯 httpx,可以重写 ``_chat_with_tools_sync``。

    协议格式由 ``api_format`` 决定(``"openai"`` 或 ``"anthropic"``)。
    """

    api_format: str = "openai"

    def __init__(
        self,
        provider: str = "",
        *,
        timeout: float = 60.0,
        model: str = "",
        api_format: str | None = None,
    ):
        self.provider_name = provider
        self.timeout = timeout
        self.model = model
        if api_format:
            self.api_format = api_format

        # 试图加载 tools/llm 作为默认后端(可选)。
        self._backend_client: Any = None
        self._backend_http: Any = None
        self._base_url: str = ""
        if provider:
            self._init_backend(provider, timeout, model)

    # ------------------------------------------------------------------
    # 后端初始化: 默认尝试 tools/llm;失败也不报错,留给子类自管
    # ------------------------------------------------------------------
    def _init_backend(self, provider: str, timeout: float, model: str) -> None:
        try:
            # tools/llm 包路径注入。两种布局都兼容:
            #   <tools>/llm/llm/__init__.py   (llm 是项目目录,内含同名包)
            #   <tools>/llm/__init__.py       (llm 包直接在 tools 下)
            # 只插入真正含 llm 包的目录——否则 namespace package 会把
            # sys.modules['llm'] 污染成空壳,后续正确路径也救不回来。
            _here = os.path.dirname(os.path.abspath(__file__))
            _tools_root = os.path.dirname(os.path.dirname(_here))
            for cand in (os.path.join(_tools_root, "llm"), _tools_root):
                if os.path.isfile(os.path.join(cand, "llm", "__init__.py")):
                    if cand not in sys.path:
                        sys.path.insert(0, cand)
                    break
            from llm import LLMClient  # type: ignore
            from llm.providers import PROVIDERS  # type: ignore
        except Exception as exc:  # pragma: no cover - 缺包时由调用方注入
            log.debug("LLMAdapter: tools/llm 后端不可用 (%s); 等待显式注入", exc)
            return

        try:
            client = LLMClient(provider=provider, timeout=timeout)
            meta = PROVIDERS[provider]
            self._backend_client = client
            self._backend_http = client.current._http
            self._base_url = (
                client.current.base_url or meta["base_url"]
            ).rstrip("/")
            self.model = model or client.current.model or meta["default_model"]
            self.api_format = client.current.api_format or self.api_format
            self.provider_name = provider
            log.info(
                "LLMAdapter init provider=%s api_format=%s model=%s",
                provider, self.api_format, self.model,
            )
        except Exception as exc:  # pragma: no cover - 配置错误时只打日志
            log.warning("LLMAdapter 后端初始化失败 (%s); 请注入自定义 chat_with_tools", exc)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    async def chat_with_tools(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """异步调 LLM,可能返回工具调用。子类可重写此方法完全自定义。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._chat_with_tools_sync,
            messages, tools, tool_choice, temperature, max_tokens,
        )

    def _chat_with_tools_sync(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | dict,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResult:
        if self._backend_http is None:
            raise LLMCallError(
                "LLMAdapter has no backend; subclass and override chat_with_tools"
            )
        if self.api_format == "openai":
            payload = self._build_openai_payload(
                messages, tools, tool_choice, temperature, max_tokens,
            )
        elif self.api_format == "anthropic":
            payload = self._build_anthropic_payload(
                messages, tools, tool_choice, temperature, max_tokens,
            )
        else:
            raise LLMCallError(f"unsupported api_format: {self.api_format!r}")

        url = self._endpoint_url()
        try:
            resp = self._backend_http._client.post(
                url, json=payload, headers=self._backend_http._headers(),
            )
        except Exception as exc:
            raise LLMCallError(f"request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMCallError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except Exception as exc:
            raise LLMCallError(f"bad JSON response: {exc}") from exc
        return self._parse_response(data)

    def close(self) -> None:
        """关闭底层 httpx client。"""
        if self._backend_client is not None:
            try:
                self._backend_client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 协议构造
    # ------------------------------------------------------------------
    def _endpoint_url(self) -> str:
        if self.api_format == "anthropic":
            return f"{self._base_url}/v1/messages"
        return f"{self._base_url}/chat/completions"

    def _build_openai_payload(
        self, messages, tools, tool_choice, temperature, max_tokens,
    ) -> dict:
        payload: dict = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        return payload

    def _build_anthropic_payload(
        self, messages, tools, tool_choice, temperature, max_tokens,
    ) -> dict:
        system_parts: list[str] = []
        new_messages: list[dict] = []
        pending_tool_results: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content if isinstance(content, str) else str(content))
                continue
            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else str(content),
                })
                continue
            if role == "assistant":
                if pending_tool_results:
                    new_messages.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []
                tool_calls = msg.get("tool_calls") or []
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args or {},
                    })
                new_messages.append({"role": "assistant", "content": blocks})
                continue
            if pending_tool_results:
                new_messages.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []
            new_messages.append({"role": "user", "content": content})
        if pending_tool_results:
            new_messages.append({"role": "user", "content": pending_tool_results})
        if not new_messages:
            raise LLMCallError("anthropic: no user/assistant messages after conversion")

        payload: dict = {
            "model": self.model,
            "messages": new_messages,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = self._convert_tools_to_anthropic(tools)
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "any":
                payload["tool_choice"] = {"type": "any"}
            else:
                payload["tool_choice"] = {"type": "tool", "name": str(tool_choice)}
        return payload

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
        out = []
        for t in tools:
            fn = t.get("function") or t
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def _parse_response(self, data: dict) -> LLMResult:
        if self.api_format == "openai":
            content, calls = self._parse_openai(data)
        elif self.api_format == "anthropic":
            content, calls = self._parse_anthropic(data)
        else:
            raise LLMCallError(f"cannot parse api_format: {self.api_format!r}")
        return LLMResult(content=content, tool_calls=calls, raw=data)

    @staticmethod
    def _parse_openai(raw: dict) -> tuple[str, list[ToolCall]]:
        parts: list[str] = []
        calls: list[ToolCall] = []
        choices = raw.get("choices") or []
        if not choices:
            return "", calls
        msg = (choices[0] or {}).get("message") or {}
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {"_raw": str(args_raw)[:200]}
            calls.append(ToolCall(
                id=tc.get("id", ""),
                name=name,
                arguments=args if isinstance(args, dict) else {},
            ))
        return "".join(parts).strip(), calls

    @staticmethod
    def _parse_anthropic(raw: dict) -> tuple[str, list[ToolCall]]:
        parts: list[str] = []
        calls: list[ToolCall] = []
        for block in raw.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input") or {},
                ))
        return "".join(parts).strip(), calls