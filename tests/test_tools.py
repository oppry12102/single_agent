"""测试工具层。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from single_agent import Tool, ToolContext, ToolRegistry, build_default_registry


# ============================================================
# Registry
# ============================================================


def _echo_tool():
    class EchoTool(Tool):
        name = "echo"
        spec = {
            "type": "function", "function": {
                "name": "echo",
                "description": "echo",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
        async def run(self, args, ctx):
            return {"echo": args.get("x", "")}
    return EchoTool()


def test_register_and_execute(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    reg.register(_echo_tool())
    r = asyncio.run(reg.execute("echo", {"x": "hi"}))
    assert r == {"echo": "hi"}


def test_execute_unknown_tool_returns_error(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    r = asyncio.run(reg.execute("nope", {}))
    assert "error" in r and "unknown" in r["error"]


def test_execute_tool_raises_returns_error(tmp_path: Path):
    class Boom(Tool):
        name = "boom"
        spec = {"type": "function", "function": {"name": "boom", "parameters": {}}}
        async def run(self, args, ctx):
            raise RuntimeError("kaboom")
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    reg.register(Boom())
    r = asyncio.run(reg.execute("boom", {}))
    assert "error" in r
    assert "RuntimeError" in r["error"]
    assert "kaboom" in r["error"]


def test_short_mode_filters_specs(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    reg.register(_echo_tool())
    reg.register(_echo_tool())  # noqa - 想 register 第二个不覆盖
    # specs(short_mode=True) 仅返回允许的工具(done 默认)
    reg.allow_in_short_mode("echo")
    assert {t["function"]["name"] for t in reg.specs(short_mode=True)} == {"echo"}
    assert {t["function"]["name"] for t in reg.specs(short_mode=False)} >= {"echo"}


def test_short_mode_rejects_execute(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    reg.register(_echo_tool())
    r = asyncio.run(reg.execute("echo", {"x": "y"}, short_mode=True))
    assert "error" in r and "not allowed" in r["error"]


def test_unregister(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    reg.register(_echo_tool())
    assert "echo" in reg.names()
    reg.unregister("echo")
    assert "echo" not in reg.names()


def test_register_empty_name_raises(tmp_path: Path):
    class Nameless(Tool):
        name = ""
        spec = {}
        async def run(self, args, ctx): return {}
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    with pytest.raises(ValueError):
        reg.register(Nameless())


def test_protocol_like_accepts_non_subclass(tmp_path: Path):
    """ToolLike 是 Protocol——任何有 name/spec/run 的对象都可注册。"""
    class Duck:
        name = "duck"
        spec = {"type": "function", "function": {"name": "duck", "parameters": {}}}
        async def run(self, args, ctx): return {"quack": True}
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = ToolRegistry(ctx)
    reg.register(Duck())
    r = asyncio.run(reg.execute("duck", {}))
    assert r == {"quack": True}


# ============================================================
# Built-in tools
# ============================================================


def test_read_file_success(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("read_file", {"path": "hello.txt"}))
    assert r["content"] == "hello world"
    assert r["bytes"] == 11


def test_read_file_not_found(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("read_file", {"path": "missing.txt"}))
    assert "error" in r and "not found" in r["error"]


def test_read_file_escape_blocked(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("read_file", {"path": "../etc/passwd"}))
    assert "error" in r


def test_write_file_success(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("write_file", {"path": "a/b/c.txt", "content": "x"}))
    assert "bytes" in r
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "x"


def test_write_file_escape_blocked(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("write_file", {"path": "../x.txt", "content": "x"}))
    assert "error" in r


def test_run_shell_success(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("run_shell", {"cmd": "echo hi"}))
    assert r["exit_code"] == 0
    assert "hi" in r["output"]


def test_run_shell_failure_exit_code(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("run_shell", {"cmd": "exit 7"}))
    assert r["exit_code"] == 7


def test_run_shell_timeout(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("run_shell", {"cmd": "sleep 5", "timeout_s": 1}))
    assert "error" in r
    assert "timeout" in r["error"]


def test_run_shell_empty_cmd(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    r = asyncio.run(reg.execute("run_shell", {"cmd": "  "}))
    assert "error" in r


def test_done_tool_returns_answer(tmp_path: Path):
    from single_agent.tools import DoneTool
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    r = asyncio.run(DoneTool().run({"answer": "OK"}, ctx))
    assert r == {"answer": "OK"}


def test_default_registry_has_four_tools(tmp_path: Path):
    ctx = ToolContext(workspace=tmp_path, agent_id="a")
    reg = build_default_registry(ctx)
    names = set(reg.names())
    assert {"read_file", "write_file", "run_shell", "done"} <= names