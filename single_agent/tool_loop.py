"""Claude Code 风格主循环。

messages = [system, user, assistant(tool_calls), tool, assistant, tool, ...]
循环: LLM.chat_with_tools → 解析 tool_calls → 执行 → 把 tool 结果加回 messages → 下一轮
直到 LLM 调用 done 或无 tool_calls 或 step 上限。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Awaitable, Callable

from .llm_adapter import LLMAdapter, LLMResult, LLMCallError
from .prompts import SYSTEM_LONG, SYSTEM_SHORT
from .tools import ToolRegistry

log = logging.getLogger("single_agent.tool_loop")


LogCallback = Callable[[str, str, str, str | None], Awaitable[None]]


class ToolLoop:
    """主循环。

    长业务循环直到 done/无 tool_calls/step 上限;短业务单轮。
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tools: ToolRegistry,
        *,
        max_steps: int = 30,
        system_long: str = SYSTEM_LONG,
        system_short: str = SYSTEM_SHORT,
        system_overlay: str = "",
        on_log: LogCallback | None = None,
        llm_semaphore: asyncio.Semaphore | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.system_long = system_long
        self.system_short = system_short
        self.system_overlay = system_overlay
        self.on_log = on_log
        # 按次串行化 LLM 调用(防自并发限流);只在单次 chat 期间持有,
        # 工具执行阶段释放,短任务得以在长任务的间隙穿插。
        self.llm_semaphore = llm_semaphore

    @contextlib.asynccontextmanager
    async def _llm_gate(self):
        if self.llm_semaphore is None:
            yield
        else:
            async with self.llm_semaphore:
                yield

    # ----------------------------------------------------------- overlay
    def set_system_overlay(self, overlay: str) -> None:
        """运行时热更新 overlay(下次 chat 用新值,不影响 in-flight)。"""
        self.system_overlay = overlay

    def _system(self, base: str) -> str:
        if not self.system_overlay:
            return base
        return f"{base}\n\n{self.system_overlay}"

    # ----------------------------------------------------------- long
    async def run_long_task(self, task_id: str, user_text: str) -> str:
        """跑长业务,返回最终答案。"""
        messages: list[dict] = [
            {"role": "system", "content": self._system(self.system_long)},
            {"role": "user", "content": user_text},
        ]
        await self._log("state_change", "internal", "long task start", task_id)
        for step in range(self.max_steps):
            try:
                async with self._llm_gate():
                    resp = await self.llm.chat_with_tools(
                        messages, tools=self.tools.specs(short_mode=False),
                    )
            except LLMCallError as exc:
                await self._log("llm_call", "internal", f"error: {exc}", task_id)
                return f"[LLM error: {exc}]"

            await self._log(
                "llm_call", "internal",
                f"step={step} content={(resp.content or '')[:60]!r} "
                f"tool_calls={len(resp.tool_calls)}",
                task_id,
            )

            if not resp.has_tool_calls:
                await self._log("state_change", "internal", "done via content", task_id)
                return resp.content or "(empty)"

            # assistant 消息回填
            messages.append(self._assistant_message(resp))

            for tc in resp.tool_calls:
                await self._log(
                    "tool_call", tc.name,
                    f"args={json.dumps(tc.arguments, ensure_ascii=False)[:80]}",
                    task_id,
                )
                if tc.name == "done":
                    answer = str(tc.arguments.get("answer", ""))
                    await self._log("state_change", "internal", "done via tool", task_id)
                    return answer
                # 执行工具
                result = await self.tools.execute(tc.name, tc.arguments)
                await self._log(
                    "tool_call", tc.name,
                    f"result={json.dumps(result, ensure_ascii=False)[:80]}",
                    task_id,
                )
                messages.append(self._tool_message(tc.id, tc.name, result))

        await self._log(
            "state_change", "internal",
            f"max_steps={self.max_steps} reached", task_id,
        )
        return f"[exceeded max_steps={self.max_steps}]"

    # ----------------------------------------------------------- short
    async def run_short_task(self, task_id: str, user_text: str) -> str:
        """跑短业务(单轮 + done)。"""
        messages: list[dict] = [
            {"role": "system", "content": self._system(self.system_short)},
            {"role": "user", "content": user_text},
        ]
        await self._log("state_change", "internal", "short task start", task_id)
        try:
            async with self._llm_gate():
                resp = await self.llm.chat_with_tools(
                    messages, tools=self.tools.specs(short_mode=True),
                )
        except LLMCallError as exc:
            await self._log("llm_call", "internal", f"error: {exc}", task_id)
            return f"[LLM error: {exc}]"
        await self._log(
            "llm_call", "internal",
            f"short content={(resp.content or '')[:60]!r}", task_id,
        )
        if resp.has_tool_calls:
            for tc in resp.tool_calls:
                if tc.name == "done":
                    return str(tc.arguments.get("answer", ""))
        return resp.content or "(empty)"

    # ----------------------------------------------------------- helpers
    def _assistant_message(self, resp: LLMResult) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.has_tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
        return msg

    @staticmethod
    def _tool_message(tc_id: str, tc_name: str, result: Any) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tc_id,
            "name": tc_name,
            "content": json.dumps(result, ensure_ascii=False),
        }

    async def _log(
        self, kind: str, peer: str, snippet: str, task_id: str | None,
    ) -> None:
        if self.on_log is None:
            return
        try:
            await self.on_log(kind, peer, snippet, task_id)
        except Exception:
            log.exception("on_log callback failed")