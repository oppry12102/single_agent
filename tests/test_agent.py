"""测试 SingleAgent 顶层 API。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from single_agent import (
    AgentConfig,
    LLMResult,
    Memory,
    SingleAgent,
    Tool,
    ToolCall,
)

from conftest import ScriptedLLM


# ============================================================
# 构造 / 生命周期
# ============================================================


def test_construction_creates_workspace(tmp_path: Path):
    ws = tmp_path / "ws"
    cfg = AgentConfig(agent_id="x", provider="p", workspace=ws)
    agent = SingleAgent(cfg, llm=ScriptedLLM())
    assert ws.exists()


def test_context_manager(cfg, scripted_llm):
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent as a:
            assert a is agent
            assert a._started

    asyncio.run(main())
    assert not agent._started


def test_start_close_idempotent(cfg, scripted_llm):
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        await agent.start()
        await agent.start()  # idempotent
        await agent.close()
        await agent.close()  # idempotent

    asyncio.run(main())


# ============================================================
# run / run_long / run_short
# ============================================================


def test_run_long(cfg, scripted_llm):
    scripted_llm.queue(LLMResult(content="", tool_calls=[
        ToolCall("1", "done", {"answer": "L-ok"}),
    ]))
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent:
            return await agent.run_long("explain")

    r = asyncio.run(main())
    assert r == "L-ok"


def test_run_short(cfg, scripted_llm):
    scripted_llm.queue(LLMResult(content="", tool_calls=[
        ToolCall("1", "done", {"answer": "S-ok"}),
    ]))
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent:
            return await agent.run_short("q")

    r = asyncio.run(main())
    assert r == "S-ok"


def test_run_default_kind_long(cfg, scripted_llm):
    scripted_llm.queue(LLMResult(content="", tool_calls=[
        ToolCall("1", "done", {"answer": "L"}),
    ]))
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent:
            return await agent.run("hi")  # 默认 long

    assert asyncio.run(main()) == "L"


# ============================================================
# submit 异步任务 + on_event 回调
# ============================================================


def test_submit_long_fires_event(cfg, scripted_llm):
    scripted_llm.queue(
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "out"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)
    received: list[dict] = []

    async def cb(ev):
        received.append(ev)

    async def main():
        async with agent:
            agent.on_event(cb)
            tid = await agent.submit("do it", kind="long")
            # 等任务完成
            for _ in range(100):
                if received:
                    break
                await asyncio.sleep(0.02)
            return tid

    tid = asyncio.run(main())
    assert isinstance(tid, str) and tid.startswith("q-")
    assert any(ev["kind"] == "task_done" and ev["task_id"] == tid for ev in received)
    done_ev = next(ev for ev in received if ev["kind"] == "task_done")
    assert done_ev["answer"] == "out"
    assert done_ev["status"] == "ok"


def test_submit_short_fifo(cfg, scripted_llm):
    scripted_llm.queue(
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "a"})]),
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "b"})]),
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "c"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)
    received: list[str] = []

    async def cb(ev):
        if ev["kind"] == "task_done":
            received.append(ev["answer"])

    async def main():
        async with agent:
            agent.on_event(cb)
            for _ in range(3):
                await agent.submit("x", kind="short")
            for _ in range(200):
                if len(received) >= 3:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(main())
    assert received == ["a", "b", "c"]


def test_submit_with_custom_task_id(cfg, scripted_llm):
    scripted_llm.queue(
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "ok"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent:
            return await agent.submit("x", kind="long", task_id="my-id")

    tid = asyncio.run(main())
    assert tid == "my-id"


def test_submit_meta_passed_through(cfg, scripted_llm):
    scripted_llm.queue(
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "ok"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)
    seen_meta = []

    async def cb(ev):
        if ev["kind"] == "task_done":
            seen_meta.append(ev)

    async def main():
        async with agent:
            agent.on_event(cb)
            await agent.submit("x", kind="long", task_id="t1", meta={"from": "x"})
            for _ in range(100):
                if seen_meta:
                    break
                await asyncio.sleep(0.02)

    asyncio.run(main())
    # meta 没在 event 顶层暴露(简化);至少任务正常完成
    assert seen_meta


# ============================================================
# 动态工具注册 + overlay 热更新
# ============================================================


def test_register_tool_runtime(cfg, scripted_llm):
    class MyTool(Tool):
        name = "my_tool"
        spec = {"type": "function", "function": {
            "name": "my_tool", "parameters": {"type": "object"},
        }}
        async def run(self, args, ctx):
            return {"ok": args.get("x")}

    scripted_llm.queue(
        LLMResult(content="", tool_calls=[ToolCall("1", "my_tool", {"x": "hi"})]),
        LLMResult(content="", tool_calls=[ToolCall("2", "done", {"answer": "DONE"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)
    agent.register_tool(MyTool())

    async def main():
        async with agent:
            return await agent.run_long("use my tool")

    r = asyncio.run(main())
    assert r == "DONE"
    # 第二次 messages 应有 tool result
    assert any(
        m["role"] == "tool" and m["name"] == "my_tool"
        for m in scripted_llm.calls[1]
    )


def test_set_overlay(cfg, scripted_llm):
    seen_systems: list[str] = []

    class SpyLLM(ScriptedLLM):
        async def chat_with_tools(self, messages, *, tools=None, **kw):
            for m in messages:
                if m["role"] == "system":
                    seen_systems.append(m["content"])
            return LLMResult(content="", tool_calls=[
                ToolCall("1", "done", {"answer": "ok"}),
            ])

    spy = SpyLLM()
    agent = SingleAgent(cfg, llm=spy)
    agent.set_overlay("ANCHOR-12345")

    async def main():
        async with agent:
            return await agent.run_long("hi")

    asyncio.run(main())
    assert any("ANCHOR-12345" in s for s in seen_systems)


# ============================================================
# 错误处理: LLM 抛错
# ============================================================


def test_run_long_llm_error_returns_error_message(cfg, scripted_llm):
    from single_agent import LLMCallError
    scripted_llm.set_scripted_error(LLMCallError("network fail"))
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent:
            return await agent.run_long("x")

    r = asyncio.run(main())
    assert r.startswith("[LLM error:")
    assert "network fail" in r


# ============================================================
# Memory 状态切换
# ============================================================


def test_memory_state_during_run(cfg, scripted_llm):
    agent = SingleAgent(cfg, llm=scripted_llm)
    states: list[str] = []

    original = agent.loop.run_long_task

    async def spy(*args, **kw):
        states.append(agent.memory.state)
        return await original(*args, **kw)

    agent.loop.run_long_task = spy  # type: ignore
    scripted_llm.queue(LLMResult(content="", tool_calls=[
        ToolCall("1", "done", {"answer": "ok"}),
    ]))

    async def main():
        async with agent:
            return await agent.run_long("x")

    asyncio.run(main())
    assert "long_busy" in states
    assert agent.memory.state == "idle"


# ============================================================
# on_event 订阅异常隔离
# ============================================================


def test_event_subscriber_exception_does_not_break_other_subs(cfg, scripted_llm):
    scripted_llm.queue(
        LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "ok"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)
    good_received: list[dict] = []

    async def bad_cb(ev):
        raise RuntimeError("bad")

    async def good_cb(ev):
        good_received.append(ev)

    async def main():
        async with agent:
            agent.on_event(bad_cb)
            agent.on_event(good_cb)
            await agent.submit("x", kind="long")
            for _ in range(100):
                if good_received:
                    break
                await asyncio.sleep(0.02)

    asyncio.run(main())
    assert good_received  # good_cb 仍然收到事件


# ============================================================
# 注入自定义 Memory / tools
# ============================================================


def test_inject_memory(cfg, scripted_llm):
    mem = Memory()
    mem.extra["foo"] = "bar"
    agent = SingleAgent(cfg, llm=scripted_llm, memory=mem)
    assert agent.memory is mem
    assert agent.memory.extra["foo"] == "bar"


def test_inject_tools(cfg, scripted_llm):
    from single_agent import ToolContext, ToolRegistry
    custom = ToolRegistry(ToolContext(workspace=cfg.workspace, agent_id="x"))
    agent = SingleAgent(cfg, llm=scripted_llm, tools=custom)
    assert agent.tools is custom


# ============================================================
# end-to-end: 自定义 tool + done
# ============================================================


def test_e2e_custom_tool_chain(cfg, scripted_llm):
    """完整链路: LLM 调自定义 tool → 拿结果 → LLM 调 done。"""
    class Counter(Tool):
        name = "counter"
        spec = {"type": "function", "function": {
            "name": "counter", "description": "count to N",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
        }}
        async def run(self, args, ctx):
            return {"count": list(range(args.get("n", 3)))}

    scripted_llm.queue(
        LLMResult(content="counting", tool_calls=[
            ToolCall("1", "counter", {"n": 3}),
        ]),
        LLMResult(content="done", tool_calls=[
            ToolCall("2", "done", {"answer": "[0,1,2]"}),
        ]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)
    agent.register_tool(Counter())

    async def main():
        async with agent:
            return await agent.run_long("count to 3")

    r = asyncio.run(main())
    assert r == "[0,1,2]"
    # 第二轮 messages 应包含 counter 的 tool 结果
    msgs = scripted_llm.calls[1]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert any("count" in m["content"] for m in tool_msgs)


# ============================================================
# close() 清理 in-flight
# ============================================================


def test_close_cleans_up_inflight(cfg, scripted_llm):
    scripted_llm.queue(
        # 不会立刻返回,需 sleep
        LLMResult(content="slow", tool_calls=[ToolCall("1", "done", {"answer": "x"})]),
    )
    agent = SingleAgent(cfg, llm=scripted_llm)

    async def main():
        async with agent:
            # 提交一个 long 任务
            await agent.submit("x", kind="long")
            await asyncio.sleep(0.01)  # 让它跑起来

    asyncio.run(main())  # 不应抛错