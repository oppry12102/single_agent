"""测试 TaskQueue。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from single_agent import TaskItem, TaskQueue


# ============================================================
# helpers
# ============================================================


def make_queue(long_runner=None, short_runner=None):
    async def default_long(item: TaskItem) -> str:
        await asyncio.sleep(0.01)
        return f"long:{item.text}"

    async def default_short(item: TaskItem) -> str:
        await asyncio.sleep(0.01)
        return f"short:{item.text}"

    return TaskQueue(
        long_runner=long_runner or default_long,
        short_runner=short_runner or default_short,
        llm_semaphore=asyncio.Semaphore(1),
    )


# ============================================================
# 长业务单槽
# ============================================================


def test_long_serialized():
    """两个 long 任务必须串行执行。"""
    order: list[str] = []

    async def slow_long(item: TaskItem) -> str:
        order.append(f"start:{item.task_id}")
        await asyncio.sleep(0.05)
        order.append(f"end:{item.task_id}")
        return f"L:{item.task_id}"

    async def fast_long(item: TaskItem) -> str:
        order.append(f"start:{item.task_id}")
        await asyncio.sleep(0.0)
        order.append(f"end:{item.task_id}")
        return f"L:{item.task_id}"

    runners = iter([slow_long, fast_long])

    async def pick(item):
        return await next(runners)(item)

    async def main():
        q = make_queue(long_runner=pick)
        done_results = []
        q.on_task_done(lambda i, a, s: done_results.append((i.task_id, a, s)))
        q.start()
        await q.submit(TaskItem(task_id="a", text="A", kind="long"))
        await q.submit(TaskItem(task_id="b", text="B", kind="long"))
        # 等所有 long 完成
        while len(done_results) < 2:
            await asyncio.sleep(0.01)
        await q.stop()
        return order, done_results

    order, results = asyncio.run(main())
    # start:a ... end:a ... start:b ... end:b
    assert order == ["start:a", "end:a", "start:b", "end:b"]
    assert sorted(results) == [("a", "L:a", "ok"), ("b", "L:b", "ok")]


# ============================================================
# 短业务 FIFO
# ============================================================


def test_short_fifo_order():
    seen: list[str] = []

    async def short(item: TaskItem) -> str:
        seen.append(item.task_id)
        return f"S:{item.task_id}"

    async def main():
        q = make_queue(short_runner=short)
        results = []
        q.on_task_done(lambda i, a, s: results.append(i.task_id))
        q.start()
        for i in range(5):
            await q.submit(TaskItem(task_id=f"s{i}", text="x", kind="short"))
        # 等所有 short 完成
        while len(results) < 5:
            await asyncio.sleep(0.01)
        await q.stop()

    asyncio.run(main())
    assert seen == ["s0", "s1", "s2", "s3", "s4"]


# ============================================================
# on_task_done 回调
# ============================================================


def test_on_task_done_called():
    async def main():
        q = make_queue()
        results = []
        q.on_task_done(lambda i, a, s: results.append((i.task_id, a, s)))
        q.start()
        await q.submit(TaskItem(task_id="t", text="x", kind="long"))
        while not results:
            await asyncio.sleep(0.01)
        await q.stop()
        assert results == [("t", "long:x", "ok")]

    asyncio.run(main())


def test_on_task_done_long_error_status():
    async def bad_long(item: TaskItem) -> str:
        raise RuntimeError("oops")

    async def main():
        q = make_queue(long_runner=bad_long)
        results = []
        q.on_task_done(lambda i, a, s: results.append((i.task_id, a, s)))
        q.start()
        await q.submit(TaskItem(task_id="e", text="x", kind="long"))
        while not results:
            await asyncio.sleep(0.01)
        await q.stop()
        assert len(results) == 1
        tid, ans, status = results[0]
        assert tid == "e"
        assert status == "error"
        assert "oops" in ans

    asyncio.run(main())


def test_on_task_done_short_error_status():
    async def bad_short(item: TaskItem) -> str:
        raise RuntimeError("bad")

    async def main():
        q = make_queue(short_runner=bad_short)
        results = []
        q.on_task_done(lambda i, a, s: results.append((i.task_id, a, s)))
        q.start()
        await q.submit(TaskItem(task_id="e", text="x", kind="short"))
        while not results:
            await asyncio.sleep(0.01)
        await q.stop()
        assert results[0][2] == "error"

    asyncio.run(main())


# ============================================================
# stop 取消 in-flight long
# ============================================================


def test_stop_cancels_inflight_long():
    started = asyncio.Event()
    cancel_done = asyncio.Event()

    async def hanging_long(item: TaskItem) -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancel_done.set()
            raise
        return "should not reach"

    async def main():
        q = make_queue(long_runner=hanging_long)
        q.start()
        await q.submit(TaskItem(task_id="h", text="x", kind="long"))
        await started.wait()
        # 立刻 stop
        await q.stop()
        assert cancel_done.is_set()

    asyncio.run(main())


# ============================================================
# 不启动也能 submit(后续 start 会消费)
# ============================================================


def test_submit_after_start():
    async def main():
        q = make_queue()
        results = []
        q.on_task_done(lambda i, a, s: results.append(i.task_id))
        # 不 start, 直接 submit -> submit 内部会立即 spawn long task
        # 长任务会等 long_in_flight lock(无需 start)
        await q.submit(TaskItem(task_id="x", text="x", kind="long"))
        while not results:
            await asyncio.sleep(0.01)
        # 最后 start 再 stop(cleanup)
        q.start()
        await q.stop()

    asyncio.run(main())