"""测试 Memory + EventLogger。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from single_agent import EventLogger, Memory


# ============================================================
# Memory
# ============================================================


def test_memory_default_state():
    m = Memory()
    assert m.state == "idle"
    assert m.current_task_id is None
    assert m.overlay == ""


def test_memory_set_overlay():
    m = Memory()
    m.set_overlay("hello world")
    assert m.overlay == "hello world"


def test_memory_set_state():
    m = Memory()
    m.set_state("long_busy", "t1")
    assert m.state == "long_busy"
    assert m.current_task_id == "t1"
    m.set_state("idle")
    assert m.state == "idle"
    assert m.current_task_id is None


def test_memory_snapshot():
    m = Memory()
    m.set_state("long_busy", "t1")
    m.set_overlay("xy" * 50)
    snap = m.snapshot()
    assert snap["state"] == "long_busy"
    assert snap["current_task_id"] == "t1"
    assert snap["overlay_chars"] == 100


# ============================================================
# EventLogger: 本地落盘 + emit 回调
# ============================================================


def test_event_logger_writes_local_jsonl(tmp_path: Path):
    log_file = tmp_path / "log.jsonl"
    el = EventLogger(agent_id="a", log_file=log_file)

    async def main():
        await el.log("k1", "p", "snippet here", task_id="t1")
        await el.log("k2", "p", "another", task_id="t2")
        el.start()
        # 给 worker 一点时间把 emit(无)消化掉
        await asyncio.sleep(0.05)
        await el.stop()

    asyncio.run(main())
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    e0 = json.loads(lines[0])
    assert e0["agent_id"] == "a"
    assert e0["kind"] == "k1"
    assert e0["task_id"] == "t1"
    assert e0["snippet"] == "snippet here"


def test_event_logger_emit_callback(tmp_path: Path):
    received: list[dict] = []

    async def emit(entry):
        received.append(entry)

    el = EventLogger(agent_id="a", log_file=None)
    el.set_emit(emit)

    async def main():
        el.start()
        await el.log("k", "peer", "msg", task_id="t")
        # 等 worker 消化
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.02)
        await el.stop()

    asyncio.run(main())
    assert len(received) >= 1
    assert received[0]["kind"] == "k"
    assert received[0]["task_id"] == "t"


def test_event_logger_snippet_truncated(tmp_path: Path):
    el = EventLogger(agent_id="a", log_file=None)
    received: list[dict] = []

    async def emit(entry):
        received.append(entry)

    el.set_emit(emit)

    async def main():
        el.start()
        await el.log("k", "p", "x" * 500)
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.02)
        await el.stop()

    asyncio.run(main())
    assert len(received[0]["snippet"]) == 100


def test_event_logger_no_local_no_emit(tmp_path: Path):
    el = EventLogger(agent_id="a", log_file=None)

    async def main():
        el.start()
        await el.log("k", "p", "x")
        await el.stop()

    # 不应抛错
    asyncio.run(main())


def test_event_logger_emit_callback_exception_does_not_crash(tmp_path: Path):
    async def bad_emit(entry):
        raise RuntimeError("emit bad")

    el = EventLogger(agent_id="a", log_file=None)
    el.set_emit(bad_emit)

    async def main():
        el.start()
        await el.log("k", "p", "x")
        await asyncio.sleep(0.05)
        await el.stop()

    asyncio.run(main())  # 不应抛


def test_event_logger_double_start_safe(tmp_path: Path):
    el = EventLogger(agent_id="a", log_file=None)

    async def main():
        el.start()
        el.start()  # idempotent
        await el.stop()

    asyncio.run(main())


def test_event_logger_stop_without_start(tmp_path: Path):
    el = EventLogger(agent_id="a", log_file=None)

    async def main():
        await el.stop()  # 不应抛

    asyncio.run(main())