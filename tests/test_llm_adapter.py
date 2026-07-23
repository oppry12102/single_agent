"""测试 LLMAdapter。

重点:
- OpenAI / Anthropic payload 构造
- OpenAI / Anthropic 响应 parse(含纯 tool_calls 无 content 边界)
- 协议分发
- LLMCallError 抛错(不吞)
- 后端不可用时通过子类化覆盖
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from single_agent import LLMAdapter, LLMCallError, LLMResult, ToolCall


# ============================================================
# Payload 构造
# ============================================================


def test_build_openai_payload_basic():
    a = LLMAdapter(provider="", api_format="openai")
    a.model = "m1"
    p = a._build_openai_payload(
        [{"role": "user", "content": "hi"}],
        tools=None, tool_choice="auto", temperature=0.5, max_tokens=64,
    )
    assert p["model"] == "m1"
    assert p["messages"] == [{"role": "user", "content": "hi"}]
    assert p["temperature"] == 0.5
    assert p["max_tokens"] == 64
    assert "tools" not in p


def test_build_openai_payload_with_tools():
    a = LLMAdapter(provider="", api_format="openai")
    a.model = "m1"
    p = a._build_openai_payload(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="required", temperature=None, max_tokens=None,
    )
    assert p["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert p["tool_choice"] == "required"
    assert "temperature" not in p


def test_build_anthropic_payload_system_extracted():
    a = LLMAdapter(provider="", api_format="anthropic")
    a.model = "m1"
    msgs = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hi"},
    ]
    p = a._build_anthropic_payload(
        msgs, tools=None, tool_choice="auto", temperature=None, max_tokens=128,
    )
    assert p["system"] == "sys prompt"
    assert p["model"] == "m1"
    assert p["max_tokens"] == 128
    assert p["messages"] == [{"role": "user", "content": "hi"}]


def test_build_anthropic_payload_tool_role_converted_to_user_block():
    a = LLMAdapter(provider="", api_format="anthropic")
    a.model = "m1"
    msgs = [
        {"role": "user", "content": "do"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "f", "arguments": json.dumps({"x": 1})}},
        ]},
        {"role": "tool", "tool_call_id": "a", "name": "f",
         "content": json.dumps({"ok": True})},
    ]
    p = a._build_anthropic_payload(
        msgs, tools=None, tool_choice="auto", temperature=None, max_tokens=None,
    )
    # 期望: 3 条 messages: user / assistant(带 tool_use) / user(带 tool_result)
    assert len(p["messages"]) == 3
    assert p["messages"][1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in p["messages"][1]["content"])
    assert p["messages"][2]["role"] == "user"
    assert any(b["type"] == "tool_result" for b in p["messages"][2]["content"])
    assert p["messages"][2]["content"][0]["tool_use_id"] == "a"


def test_build_anthropic_payload_empty_after_conversion_raises():
    a = LLMAdapter(provider="", api_format="anthropic")
    a.model = "m1"
    with pytest.raises(LLMCallError):
        a._build_anthropic_payload(
            [{"role": "system", "content": "only sys"}],
            tools=None, tool_choice="auto", temperature=None, max_tokens=None,
        )


def test_convert_tools_to_anthropic():
    out = LLMAdapter._convert_tools_to_anthropic([
        {"type": "function", "function": {
            "name": "f", "description": "d", "parameters": {"type": "object"},
        }},
    ])
    assert out == [{
        "name": "f", "description": "d", "input_schema": {"type": "object"},
    }]


# ============================================================
# Parse
# ============================================================


def test_parse_openai_text_only():
    c, calls = LLMAdapter._parse_openai({
        "choices": [{"message": {"content": "hi"}}],
    })
    assert c == "hi"
    assert calls == []


def test_parse_openai_tool_calls_only_no_content():
    """曾经的边界 bug: content=None,tool_calls 非空。"""
    c, calls = LLMAdapter._parse_openai({
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "1", "function": {
                "name": "done", "arguments": json.dumps({"answer": "ok"}),
            }},
        ]}}],
    })
    assert c == ""
    assert len(calls) == 1
    assert calls[0].name == "done"
    assert calls[0].arguments == {"answer": "ok"}


def test_parse_openai_bad_json_arguments():
    c, calls = LLMAdapter._parse_openai({
        "choices": [{"message": {"content": "", "tool_calls": [
            {"id": "1", "function": {
                "name": "f", "arguments": "not json",
            }},
        ]}}],
    })
    assert len(calls) == 1
    assert calls[0].arguments.get("_raw", "").startswith("not json")


def test_parse_openai_empty_choices():
    c, calls = LLMAdapter._parse_openai({})
    assert c == "" and calls == []
    c, calls = LLMAdapter._parse_openai({"choices": []})
    assert c == "" and calls == []


def test_parse_anthropic_text_only():
    c, calls = LLMAdapter._parse_anthropic({
        "content": [{"type": "text", "text": "hi"}],
    })
    assert c == "hi" and calls == []


def test_parse_anthropic_tool_use():
    c, calls = LLMAdapter._parse_anthropic({
        "content": [
            {"type": "text", "text": "doing "},
            {"type": "tool_use", "id": "1", "name": "done", "input": {"answer": "x"}},
        ],
    })
    assert c == "doing"
    assert len(calls) == 1
    assert calls[0].name == "done"
    assert calls[0].arguments == {"answer": "x"}


def test_parse_anthropic_empty():
    c, calls = LLMAdapter._parse_anthropic({})
    assert c == "" and calls == []


# ============================================================
# Endpoint & protocol dispatch
# ============================================================


def test_endpoint_url_openai():
    a = LLMAdapter(provider="", api_format="openai")
    a._base_url = "https://x/y"
    assert a._endpoint_url() == "https://x/y/chat/completions"


def test_endpoint_url_anthropic():
    a = LLMAdapter(provider="", api_format="anthropic")
    a._base_url = "https://x/y"
    assert a._endpoint_url() == "https://x/y/v1/messages"


def test_parse_response_dispatch():
    a = LLMAdapter(provider="", api_format="openai")
    out = a._parse_response({
        "choices": [{"message": {"content": "hi"}}],
    })
    assert isinstance(out, LLMResult)
    assert out.content == "hi"


def test_unsupported_api_format_raises():
    a = LLMAdapter(provider="", api_format="bogus")
    a.model = "m"
    # _build_*_payload 不做 api_format 校验(由 dispatcher 上层做)
    # 但 _parse_response 必须 raise
    with pytest.raises(LLMCallError):
        a._parse_response({})


# ============================================================
# 后端不可用时,chat_with_tools 应该 raise
# ============================================================


def test_chat_with_tools_no_backend_raises():
    a = LLMAdapter(provider="", api_format="openai")
    # _backend_http 一定是 None(没有 provider 注入)
    assert a._backend_http is None
    import asyncio
    with pytest.raises(LLMCallError):
        asyncio.run(a.chat_with_tools([{"role": "user", "content": "hi"}]))


# ============================================================
# 子类化: 自定义 chat_with_tools
# ============================================================


def test_subclass_override_chat_with_tools():
    class MyLLM(LLMAdapter):
        async def chat_with_tools(self, messages, *, tools=None, **kw):
            return LLMResult(content="override", tool_calls=[])

    import asyncio
    a = MyLLM(provider="", api_format="openai")
    r = asyncio.run(a.chat_with_tools([{"role": "user", "content": "x"}]))
    assert r.content == "override"


# ============================================================
# close() 不抛
# ============================================================


def test_close_safe_with_no_backend():
    a = LLMAdapter(provider="", api_format="openai")
    a.close()  # 不应抛
    assert a._backend_client is None


# ============================================================
# 回归: 无 PYTHONPATH 时后端路径注入必须生效
# (bug: 注入的是 tools/agent 和 tools/agent/llm,无法 import llm 包)
# ============================================================


def test_backend_sys_path_injection(monkeypatch):
    import sys

    import single_agent.llm_adapter as la

    agent_root = Path(la.__file__).resolve().parents[1]  # <tools>/agent
    tools_root = Path(la.__file__).resolve().parents[2]  # <tools>
    if not (tools_root / "llm" / "llm" / "__init__.py").is_file():
        pytest.skip("tools/llm 项目不存在,跳过")

    # 模拟没有 PYTHONPATH 的干净环境
    monkeypatch.setattr(
        sys, "path",
        [p for p in sys.path if p not in (str(agent_root), str(tools_root))],
    )
    for mod in [m for m in sys.modules if m == "llm" or m.startswith("llm.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    # provider 不存在会在 client init 阶段失败(被捕获),
    # 但 `from llm import ...` 这一步必须成功
    LLMAdapter(provider="nonexistent-provider")
    assert "llm" in sys.modules
    assert hasattr(sys.modules["llm"], "LLMClient")