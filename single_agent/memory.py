"""Memory —— 日志 + system overlay + 事件总线。

三件事:
1. ``EventLogger``: 写本地 jsonl(可选)+ 推回调。异步队列消费。
2. ``Memory``: 维护 ``system_overlay``(可热更新,下次 LLM 调用生效)。
3. 简单的 ``TaskHandle`` 暴露当前任务状态(idle/long_busy/short_busy)。

设计为可被外部多 agent 协调层挂钩(``EventLogger.set_emit`` 注册回调即可)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger("single_agent.memory")


# ============================================================
# EventLogger
# ============================================================


EmitCallback = Callable[[dict], Awaitable[None]]


class EventLogger:
    """异步事件日志。

    - 本地 jsonl 落盘(可选)
    - 异步推送给注册的回调(可选,例如推 server / 推父 agent)
    - snippet 截前 100 字符(避免日志爆)
    """

    def __init__(
        self,
        agent_id: str,
        log_file: Path | None = None,
        *,
        queue_maxsize: int = 10_000,
    ):
        self.agent_id = agent_id
        self.log_file = Path(log_file) if log_file else None
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=queue_maxsize)
        self._emit: EmitCallback | None = None
        self._task: asyncio.Task | None = None
        self._stop = False

    # ----------------------------------------------------------- hooks
    def set_emit(self, cb: EmitCallback) -> None:
        """注册一个外部回调(每条事件都会调用)。"""
        self._emit = cb

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop = False  # 支持 stop() 后重新 start()
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ----------------------------------------------------------- API
    async def log(
        self,
        kind: str,
        peer: str,
        snippet: str,
        *,
        task_id: str | None = None,
        write_local: bool = True,
    ) -> None:
        snip = (snippet or "")[:100]
        entry = {
            "ts": time.time(),
            "agent_id": self.agent_id,
            "kind": kind,
            "peer": peer,
            "snippet": snip,
            "task_id": task_id,
        }
        if write_local and self.log_file is not None:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as exc:
                log.warning("local log write failed: %s", exc)
        try:
            # put_nowait: 队列满时丢弃而不是阻塞调用方
            # (await put 在满队列上会永远挂起,QueueFull 只有 put_nowait 才抛)
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            log.warning("event queue full, dropping kind=%s", kind)

    async def _worker(self) -> None:
        while not self._stop:
            try:
                entry = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                if self._emit is not None:
                    await self._emit(entry)
            except Exception as exc:
                log.warning("emit callback failed: %s", exc)
            finally:
                self._queue.task_done()


# ============================================================
# Memory —— overlay + 状态
# ============================================================


@dataclass
class Memory:
    """维护 system overlay(可热更新)+ 当前任务状态。

    Attributes:
        overlay: 追加到 system prompt 后的额外内容(knowledge_anchors 等)。
        state: 当前任务状态(``idle`` / ``long_busy`` / ``short_busy``)。
        current_task_id: 当前正在跑的 task_id(若有)。
    """

    overlay: str = ""
    state: str = "idle"
    current_task_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def set_overlay(self, overlay: str) -> None:
        self.overlay = overlay

    def set_state(self, state: str, task_id: str | None = None) -> None:
        self.state = state
        self.current_task_id = task_id

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "current_task_id": self.current_task_id,
            "overlay_chars": len(self.overlay),
            **self.extra,
        }