"""测试 ToolLoop 主循环。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from single_agent import LLMAdapter, LLMResult, ToolCall
from single_agent.tool_loop import ToolLoop
from single_agent.tools import ToolContext, build_default_registry

from conftest import ScriptedLLM


def _loop(cfg, scripted) -> ToolLoop:
    ctx = ToolContext(workspace=cfg.workspace, agent_id=cfg.agent_id)
    tools = build_default_registry(ctx)
    return ToolLoop(scripted, tools, max_steps=cfg.max_steps)


# ============================================================
# run_long_task: 一次 done
# ============================================================


def test_long_done_first_step(cfg):
    scripted = ScriptedLLM([
        LLMResult(content="", tool_calls=[
            ToolCall("1", "done", {"answer": "ok"}),
        ]),
    ])
    lp = _loop(cfg, scripted)
    r = asyncio.run(lp.run_long_task("t1", "hi"))
    assert r == "ok"
    assert len(scripted.calls) == 1


# ============================================================
# run_long_task: 工具调用 → done
# ============================================================


def test_long_tool_then_done(cfg):
    # 第一次 LLM 调 run_shell; 第二次 done
    scripted = ScriptedLLM([
        LLMResult(content="let me check", tool_calls=[
            ToolCall("1", "run_shell", {"cmd": "echo a"}),
        ]),
        LLMResult(content="got it", tool_calls=[
            ToolCall("2", "done", {"answer": "a"}),
        ]),
    ])
    lp = _loop(cfg, scripted)
    r = asyncio.run(lp.run_long_task("t2", "what is a?"))
    assert r == "a"
    assert len(scripted.calls) == 2

    # 第二次 messages 应包含 tool result
    msgs = scripted.calls[1]
    assert any(m["role"] == "tool" and m["tool_call_id"] == "1" for m in msgs)


# ============================================================
# run_long_task: 无 tool_calls → 直接返回 content
# ============================================================


def test_long_no_tool_calls_returns_content(cfg):
    scripted = ScriptedLLM([
        LLMResult(content="just text", tool_calls=[]),
    ])
    lp = _loop(cfg, scripted)
    r = asyncio.run(lp.run_long_task("t3", "hi"))
    assert r == "just text"


# ============================================================
# run_long_task: LLM 报错 → [LLM error: ...]
# ============================================================


def test_long_llm_error_returns_error_message(cfg):
    from single_agent import LLMCallError
    scripted = ScriptedLLM()
    scripted.set_scripted_error(LLMCallError("boom"))
    lp = _loop(cfg, scripted)
    r = asyncio.run(lp.run_long_task("t4", "hi"))
    assert r.startswith("[LLM error:")
    assert "boom" in r


# ============================================================
# run_long_task: 达到 max_steps
# ============================================================


def test_long_max_steps_exceeded(cfg):
    # 一直调 run_shell(echo 1),永远不调 done,触发 step 上限
    script = [
        LLMResult(content="loop", tool_calls=[
            ToolCall(str(i), "run_shell", {"cmd": "echo 1"}),
        ])
        for i in range(10)
    ]
    scripted = ScriptedLLM(script)
    lp = _loop(cfg, scripted)
    lp.max_steps = 3
    r = asyncio.run(lp.run_long_task("t5", "loop"))
    assert "exceeded max_steps=3" in r


# ============================================================
# run_short_task
# ============================================================


def test_short_done(cfg):
    scripted = ScriptedLLM([
        LLMResult(content="", tool_calls=[
            ToolCall("1", "done", {"answer": "short ok"}),
        ]),
    ])
    lp = _loop(cfg, scripted)
    r = asyncio.run(lp.run_short_task("t6", "hi"))
    assert r == "short ok"


def test_short_returns_content_when_no_tool(cfg):
    scripted = ScriptedLLM([
        LLMResult(content="inline answer", tool_calls=[]),
    ])
    lp = _loop(cfg, scripted)
    r = asyncio.run(lp.run_short_task("t7", "hi"))
    assert r == "inline answer"


def test_short_filters_long_tools(cfg):
    """短业务不应暴露 read_file 等重型工具(specs(short_mode=True))。"""
    scripted = ScriptedLLM([
        LLMResult(content="should not see read_file", tool_calls=[]),
    ])
    lp = _loop(cfg, scripted)
    asyncio.run(lp.run_short_task("t8", "x"))
    # 调用 LLM 时传入的 tools 仅含 done
    tools_arg = scripted.calls[0]  # type: ignore[index]
    # calls 存的是 messages; 这里改用 mock 验证通过 specs() 调用本身
    # 用一个 spy LLM 验证传入的 tools
    seen_tools: list = []

    class SpyLLM(ScriptedLLM):
        async def chat_with_tools(self, messages, *, tools=None, **kw):
            seen_tools.append(tools)
            return LLMResult(content="x", tool_calls=[])

    spy = SpyLLM()
    ctx = ToolContext(workspace=cfg.workspace, agent_id=cfg.agent_id)
    tools = build_default_registry(ctx)
    lp2 = ToolLoop(spy, tools, max_steps=3)
    asyncio.run(lp2.run_short_task("t9", "x"))
    assert seen_tools  # 调用了 LLM
    # 短业务: 只暴露 done
    assert all(t["function"]["name"] == "done" for t in seen_tools[0])


# ============================================================
# overlay
# ============================================================


def test_overlay_appended_to_system(cfg):
    seen_systems: list[str] = []

    class SpyLLM(ScriptedLLM):
        async def chat_with_tools(self, messages, *, tools=None, **kw):
            for m in messages:
                if m["role"] == "system":
                    seen_systems.append(m["content"])
            return LLMResult(content="x", tool_calls=[])

    spy = SpyLLM()
    ctx = ToolContext(workspace=cfg.workspace, agent_id=cfg.agent_id)
    tools = build_default_registry(ctx)
    lp = ToolLoop(spy, tools, max_steps=3)
    lp.set_system_overlay("EXTRA KNOWLEDGE")
    asyncio.run(lp.run_long_task("t10", "x"))
    assert any("EXTRA KNOWLEDGE" in s for s in seen_systems)


def test_set_system_overlay_hot_update(cfg):
    lp = _loop(cfg, ScriptedLLM())
    assert lp.system_overlay == ""
    lp.set_system_overlay("hello")
    assert lp.system_overlay == "hello"


# ============================================================
# on_log 回调
# ============================================================


def test_on_log_callback(cfg):
    logs: list[tuple[str, str, str]] = []

    async def cb(kind, peer, snippet, task_id):
        logs.append((kind, peer, snippet[:60]))

    scripted = ScriptedLLM([
        LLMResult(content="", tool_calls=[
            ToolCall("1", "done", {"answer": "ok"}),
        ]),
    ])
    ctx = ToolContext(workspace=cfg.workspace, agent_id=cfg.agent_id)
    tools = build_default_registry(ctx)
    lp = ToolLoop(scripted, tools, max_steps=3, on_log=cb)
    asyncio.run(lp.run_long_task("t11", "hi"))
    kinds = [k for k, _, _ in logs]
    assert "state_change" in kinds
    assert "llm_call" in kinds
    assert "tool_call" in kinds


def test_on_log_callback_exception_does_not_crash(cfg):
    async def bad_cb(*args, **kw):
        raise RuntimeError("cb error")

    scripted = ScriptedLLM([
        LLMResult(content="", tool_calls=[
            ToolCall("1", "done", {"answer": "ok"}),
        ]),
    ])
    ctx = ToolContext(workspace=cfg.workspace, agent_id=cfg.agent_id)
    tools = build_default_registry(ctx)
    lp = ToolLoop(scripted, tools, max_steps=3, on_log=bad_cb)
    r = asyncio.run(lp.run_long_task("t12", "hi"))
    assert r == "ok"  # 回调失败不应影响主流程


# ============================================================
# 回归: llm_semaphore 按次串行化 LLM 调用(而不是整个任务)
# ============================================================


def test_llm_semaphore_serializes_single_calls(cfg):
    """共享 sem 时并发 peak=1;不传 sem 时 peak=2(调用可重叠)。"""

    async def run_pair(sem) -> int:
        in_call = 0
        peak = 0

        class SlowLLM(LLMAdapter):
            def __init__(self):
                super().__init__(provider="")

            async def chat_with_tools(self, messages, *, tools=None, **kw):
                nonlocal in_call, peak
                in_call += 1
                peak = max(peak, in_call)
                await asyncio.sleep(0.05)
                in_call -= 1
                return LLMResult(content="", tool_calls=[
                    ToolCall("1", "done", {"answer": "a"}),
                ])

        ctx = ToolContext(workspace=cfg.workspace, agent_id=cfg.agent_id)
        tools = build_default_registry(ctx)
        lp = ToolLoop(SlowLLM(), tools, llm_semaphore=sem)
        await asyncio.gather(
            lp.run_long_task("t1", "x"),
            lp.run_short_task("t2", "y"),
        )
        return peak

    assert asyncio.run(run_pair(asyncio.Semaphore(1))) == 1
    assert asyncio.run(run_pair(None)) == 2