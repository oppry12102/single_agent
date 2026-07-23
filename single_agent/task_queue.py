"""任务调度: 长业务单槽 + 短业务 FIFO。

设计目标:
- 长业务(``kind="long"``)同时只跑一个(单槽),后续 long 任务排队等待。
- 短业务(``kind="short"``)FIFO 串行(可并发多 agent 协作时按顺序快速回答)。
- LLM 调用的串行化(防自并发限流)由 ``ToolLoop.llm_semaphore`` 按次调用负责,
  因此长任务执行工具的间隙,短任务可以穿插调用 LLM,不会被长任务饿死。
- ``on_task_done`` 回调暴露 ``(item, answer, status)``,用于多 agent 协调层回包。
  回调在锁外执行,慢订阅者不会阻塞后续任务。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

log = logging.getLogger("single_agent.task_queue")


@dataclass
class TaskItem:
    """任务条目。

    Attributes:
        task_id: 唯一 id(调用方给)。
        text: 用户文本。
        kind: ``"long"`` / ``"short"``。
        meta: 任意附加数据(可挂原始 envelope / sender / 等)。
    """

    task_id: str
    text: str
    kind: str = "long"
    meta: dict[str, Any] | None = None


Runner = Callable[[TaskItem], Awaitable[str]]
DoneCallback = Callable[[TaskItem, str, str], Awaitable[None]]


class TaskQueue:
    """Agent 端任务分发。

    用法::

        async def long_runner(item: TaskItem) -> str:
            return await loop.run_long_task(item.task_id, item.text)

        q = TaskQueue(long_runner=long_runner, short_runner=short_runner)
        q.on_task_done(my_callback)
        q.start()
        await q.submit(TaskItem(task_id="t1", text="hello", kind="long"))
        ...
        await q.stop()
    """

    def __init__(
        self,
        *,
        long_runner: Runner,
        short_runner: Runner,
    ):
        self.long_runner = long_runner
        self.short_runner = short_runner
        self.long_in_flight = asyncio.Lock()
        self.short_queue: asyncio.Queue[TaskItem] = asyncio.Queue()
        self._short_worker_task: asyncio.Task | None = None
        self._long_tasks: set[asyncio.Task] = set()
        self._on_task_done: DoneCallback | None = None
        self._stop = False

    # ----------------------------------------------------------- hooks
    def on_task_done(self, cb: DoneCallback) -> None:
        """注册任务完成回调: ``async def cb(item, answer, status)``。"""
        self._on_task_done = cb

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop = False  # 支持 stop() 后重新 start()
        if self._short_worker_task is None:
            self._short_worker_task = asyncio.create_task(self._short_worker())

    async def stop(self) -> None:
        self._stop = True
        # 取消所有 in-flight long tasks
        long_tasks = list(self._long_tasks)
        for t in long_tasks:
            t.cancel()
        if long_tasks:
            await asyncio.gather(*long_tasks, return_exceptions=True)
        if self._short_worker_task is not None:
            self._short_worker_task.cancel()
            try:
                await self._short_worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self._short_worker_task = None

    # ----------------------------------------------------------- submit
    async def submit(self, item: TaskItem) -> None:
        """提交一个任务(立即返回,任务在后台跑)。"""
        if item.kind == "long":
            t = asyncio.create_task(self._run_long(item))
            self._long_tasks.add(t)
            t.add_done_callback(self._long_tasks.discard)
        else:
            await self.short_queue.put(item)

    # ----------------------------------------------------------- workers
    async def _run_long(self, item: TaskItem) -> None:
        answer: str
        status: str
        async with self.long_in_flight:
            try:
                answer = await self.long_runner(item)
                status = "ok"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("long task %s failed: %s", item.task_id, exc)
                answer = f"[error: {exc}]"
                status = "error"
        # 回调在锁外执行: 慢订阅者不阻塞排队的下一个 long 任务
        await self._finish(item, answer, status)

    async def _short_worker(self) -> None:
        while not self._stop:
            try:
                item = await self.short_queue.get()
            except asyncio.CancelledError:
                break
            try:
                answer = await self.short_runner(item)
                await self._finish(item, answer, "ok")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.exception("short task %s failed: %s", item.task_id, exc)
                await self._finish(item, f"[error: {exc}]", "error")
            finally:
                self.short_queue.task_done()

    async def _finish(self, item: TaskItem, answer: str, status: str) -> None:
        if self._on_task_done is not None:
            try:
                await self._on_task_done(item, answer, status)
            except Exception:
                log.exception("on_task_done failed")