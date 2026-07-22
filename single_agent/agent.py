"""SingleAgent —— 顶层封装。

对外只暴露:
- ``run(text, kind="long")``           直接 await 返回答案(短业务同步接口)
- ``run_long(text)`` / ``run_short(text)`` 同上,显式区分
- ``submit(item)``                     异步提交任务,不阻塞(用 TaskQueue)
- ``on_event(cb)``                     订阅事件流(供多 agent 协调层)
- ``set_overlay(text)``                热更新 system overlay
- ``register_tool(tool)``              动态添加/替换工具
- ``start()`` / ``stop()`` / ``close()`` 生命周期

注入点(便于多 agent 协作时定制):
- ``llm``: 自定义 LLMAdapter(默认用 ``tools/llm`` 后端)
- ``tools``: 自定义 ToolRegistry(默认 ``build_default_registry``)
- ``memory``: 自定义 Memory(默认新建)

示例::

    import asyncio
    from pathlib import Path
    from single_agent import SingleAgent, AgentConfig

    async def main():
        agent = SingleAgent(AgentConfig(
            agent_id="alpha",
            provider="minimax",
            workspace=Path("./workspace"),
            log_file=Path("./logs/alpha.jsonl"),
        ))
        await agent.start()
        try:
            print(await agent.run("用一句话介绍 Python"))
        finally:
            await agent.close()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable

from .config import AgentConfig
from .llm_adapter import LLMAdapter
from .memory import EventLogger, Memory
from .prompts import SYSTEM_LONG as DEFAULT_SYSTEM_LONG
from .prompts import SYSTEM_SHORT as DEFAULT_SYSTEM_SHORT
from .task_queue import TaskItem, TaskQueue
from .tool_loop import ToolLoop
from .tools import ToolContext, ToolRegistry, build_default_registry

log = logging.getLogger("single_agent.agent")

EventCallback = Callable[[dict], Awaitable[None]]


class SingleAgent:
    """封装良好的单 agent。

    完整生命周期:
        agent = SingleAgent(cfg)
        await agent.start()       # 启动内部 task_queue + event_logger
        answer = await agent.run("...")
        await agent.close()       # 优雅退出
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        llm: LLMAdapter | None = None,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
        event_logger: EventLogger | None = None,
    ):
        self.config = config
        self.memory = memory or Memory()
        self.event_logger = event_logger or EventLogger(
            agent_id=config.agent_id,
            log_file=config.log_file,
        )

        # LLM(允许注入;默认自己 new 一个)
        self.llm = llm or LLMAdapter(
            provider=config.provider,
            timeout=config.llm_timeout_s,
            model=config.llm_model,
        )

        # 工具(允许注入;默认用内置 4 件套)
        ctx = ToolContext(workspace=config.workspace, agent_id=config.agent_id)
        self.tools = tools or build_default_registry(ctx)

        # system prompts: 用户注入优先,否则用内置
        sys_long = config.system_long or DEFAULT_SYSTEM_LONG
        sys_short = config.system_short or DEFAULT_SYSTEM_SHORT

        self.loop = ToolLoop(
            self.llm, self.tools,
            max_steps=config.max_steps,
            system_long=sys_long,
            system_short=sys_short,
            system_overlay=self.memory.overlay,
            on_log=self._on_loop_log,
        )

        self.queue = TaskQueue(
            long_runner=self._run_long,
            short_runner=self._run_short,
            llm_semaphore=asyncio.Semaphore(1),
        )
        self.queue.on_task_done(self._on_task_done)

        # 外部事件订阅者列表
        self._event_subs: list[EventCallback] = []
        self._started = False

    # ===========================================================
    # 生命周期
    # ===========================================================
    async def start(self) -> None:
        if self._started:
            return
        self.event_logger.start()
        self.queue.start()
        self._started = True
        log.info("SingleAgent started agent_id=%s provider=%s",
                 self.config.agent_id, self.config.provider)

    async def close(self) -> None:
        if not self._started:
            return
        await self.queue.stop()
        await self.event_logger.stop()
        if self.config.auto_close_llm:
            # LLM close 是同步,放到线程池避免阻塞 shutdown
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.llm.close)
            except RuntimeError:
                self.llm.close()
        self._started = False
        log.info("SingleAgent closed agent_id=%s", self.config.agent_id)

    # ===========================================================
    # 同步 API: 直接返回答案(默认 long)
    # ===========================================================
    async def run(self, text: str, *, kind: str = "long") -> str:
        task_id = f"inline-{uuid.uuid4().hex[:8]}"
        return await self._run_inline(task_id, text, kind)

    async def run_long(self, text: str) -> str:
        return await self.run(text, kind="long")

    async def run_short(self, text: str) -> str:
        return await self.run(text, kind="short")

    async def _run_inline(self, task_id: str, text: str, kind: str) -> str:
        prev_state = self.memory.state
        prev_task = self.memory.current_task_id
        self.memory.set_state(
            "long_busy" if kind == "long" else "short_busy", task_id,
        )
        try:
            if kind == "long":
                return await self.loop.run_long_task(task_id, text)
            return await self.loop.run_short_task(task_id, text)
        finally:
            self.memory.set_state(prev_state, prev_task)

    # ===========================================================
    # 异步 API: 提交任务,不阻塞(给上层 dispatcher / 多 agent 用)
    # ===========================================================
    async def submit(
        self,
        text: str,
        *,
        kind: str = "long",
        task_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """提交任务,立即返回 task_id(答案通过 ``on_task_done``/事件回调送达)。"""
        if not self._started:
            await self.start()
        tid = task_id or f"q-{uuid.uuid4().hex[:8]}"
        await self.queue.submit(TaskItem(
            task_id=tid, text=text, kind=kind, meta=meta,
        ))
        return tid

    # ===========================================================
    # 工具 / Overlay 动态更新
    # ===========================================================
    def register_tool(self, tool) -> None:
        """动态注册/替换工具。"""
        self.tools.register(tool)

    def set_overlay(self, overlay: str) -> None:
        """热更新 system overlay;下次 LLM chat 生效。"""
        self.memory.set_overlay(overlay)
        self.loop.set_system_overlay(overlay)

    # ===========================================================
    # 事件订阅(供多 agent 协调层)
    # ===========================================================
    def on_event(self, cb: EventCallback) -> None:
        """订阅事件流(每条日志条目都会回调一次)。

        适合:
        - 多 agent 协作时,把事件汇总到 dispatcher
        - UI 实时展示
        - 测试断言

        也可直接 ``event_logger.set_emit()`` 注册单个回调。
        """
        self._event_subs.append(cb)

    # ===========================================================
    # 内部: task_queue 的 runner / done
    # ===========================================================
    async def _run_long(self, item: TaskItem) -> str:
        self.memory.set_state("long_busy", item.task_id)
        try:
            return await self.loop.run_long_task(item.task_id, item.text)
        finally:
            self.memory.set_state("idle", None)

    async def _run_short(self, item: TaskItem) -> str:
        self.memory.set_state("short_busy", item.task_id)
        try:
            return await self.loop.run_short_task(item.task_id, item.text)
        finally:
            self.memory.set_state("idle", None)

    async def _on_task_done(self, item: TaskItem, answer: str, status: str) -> None:
        # 走 EventLogger(本地 + emit)
        await self.event_logger.log(
            "task_done", "internal",
            f"task done status={status} answer={answer[:60]!r}",
            task_id=item.task_id,
        )
        # 走外部订阅
        for cb in list(self._event_subs):
            try:
                await cb({
                    "kind": "task_done",
                    "task_id": item.task_id,
                    "kind_kind": item.kind,
                    "answer": answer,
                    "status": status,
                    "agent_id": self.config.agent_id,
                })
            except Exception:
                log.exception("event subscriber raised")

    async def _on_loop_log(
        self, kind: str, peer: str, snippet: str, task_id: str | None,
    ) -> None:
        await self.event_logger.log(kind, peer, snippet, task_id=task_id)

    # ===========================================================
    # 上下文管理
    # ===========================================================
    async def __aenter__(self) -> "SingleAgent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()