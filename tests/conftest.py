"""pytest 公共 fixture。"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# 把 single_agent/ 加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from single_agent import (  # noqa: E402
    AgentConfig,
    EventLogger,
    LLMAdapter,
    LLMResult,
    Memory,
    SingleAgent,
    ToolCall,
    ToolContext,
    ToolRegistry,
    build_default_registry,
)
from single_agent.tools import Tool  # noqa: E402


# ============================================================
# 可编程 LLM: 让测试按脚本驱动 LLM 返回
# ============================================================


class ScriptedLLM(LLMAdapter):
    """按脚本返回 LLMResult,逐次消费。

    用法::

        scripted = ScriptedLLM([
            LLMResult(content="", tool_calls=[ToolCall("1", "done", {"answer": "ok"})]),
            LLMResult(content="", tool_calls=[ToolCall("1", "read_file", {"path": "x.txt"})]),
            ...
        ])
    """

    def __init__(self, script: list[LLMResult] | None = None):
        # 跳过父类的后端初始化(provider='')
        super().__init__(provider="", api_format="openai")
        self.script: list[LLMResult] = list(script or [])
        self.calls: list[list[dict]] = []

    def queue(self, *results: LLMResult) -> None:
        self.script.extend(results)

    async def chat_with_tools(
        self, messages, *, tools=None, tool_choice="auto",
        temperature=None, max_tokens=None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError(
                f"ScriptedLLM ran out of script (got {len(self.calls)} calls)"
            )
        return self.script.pop(0)

    def set_scripted_error(self, exc: Exception | None = None) -> None:
        """下一次 chat 抛 ``exc``(默认 ``LLMCallError``)。"""
        cls = type(exc) if exc else None
        if cls is None:
            from single_agent import LLMCallError
            self.script.insert(0, _ErrorStub(LLMCallError("scripted error")))
        else:
            self.script.insert(0, _ErrorStub(exc))


class _ErrorStub(LLMResult):
    """占位:ScriptedLLM 看到 _ErrorStub 就抛对应异常。"""

    def __init__(self, exc: Exception):
        super().__init__(content="")
        self._exc = exc

    def __getattr__(self, name):  # pragma: no cover - 占位
        raise self._exc


# 修复 ScriptedLLM.set_scripted_error: 实现一个独立抛出机制
async def _scripted_chat_raise(self, messages, *, tools=None, **kw):
    self.calls.append(list(messages))
    if not self.script:
        raise AssertionError("ScriptedLLM ran out of script")
    item = self.script.pop(0)
    if isinstance(item, _ErrorStub):
        raise item._exc
    return item


# patch 方法
ScriptedLLM.chat_with_tools = _scripted_chat_raise  # type: ignore


# ============================================================
# 临时目录 / 配置
# ============================================================


@pytest.fixture
def tmpdir_path(tmp_path: Path) -> Path:
    """pytest tmp_path 的别名(更短)。"""
    return tmp_path


@pytest.fixture
def cfg(tmp_path: Path) -> AgentConfig:
    """默认 config,workspace/log_file 都在 tmp_path 下。"""
    return AgentConfig(
        agent_id="test-agent",
        provider="mock",
        workspace=tmp_path / "workspace",
        log_file=tmp_path / "logs" / "test.jsonl",
        max_steps=10,
    )


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def agent(cfg: AgentConfig, scripted_llm: ScriptedLLM) -> SingleAgent:
    a = SingleAgent(cfg, llm=scripted_llm)
    return a