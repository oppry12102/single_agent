"""single_agent —— 封装良好的单 agent 框架。

设计目标：
1. 单一入口 ``SingleAgent`` 类,对外只暴露 ``run()`` / ``run_short()`` / ``submit()``。
2. 可插拔 LLM: 默认使用 ``tools/llm`` 包(OpenAI/Anthropic 双协议),允许注入 mock/自定义 client。
3. 可插拔 Tools: 注册表 + Tool 基类,内置 ``read_file / write_file / run_shell / done``。
4. 可插拔 Memory/Log: 本地 jsonl + 可选回调 + system_overlay 热更新。
5. 可插拔 Transport: 不强制 WebSocket,提供 ``on_event`` hook 给上层多 agent 协调使用。
6. 零强依赖: 仅需 ``httpx``、``pyyaml``(可选);``tools/llm`` 是可选后端。

典型用法::

    from single_agent import SingleAgent, AgentConfig

    agent = SingleAgent(AgentConfig(
        agent_id="alpha",
        provider="minimax",
        workspace=Path("./workspace"),
    ))
    answer = await agent.run("总结一下当前目录有哪些 py 文件")

测试用法(注入 mock LLM)::

    from single_agent import SingleAgent, AgentConfig, LLMAdapter, ToolCall, LLMResult

    class FakeLLM(LLMAdapter):
        async def chat_with_tools(self, messages, *, tools=None, **kw):
            return LLMResult(content="hello", tool_calls=[
                ToolCall(id="t1", name="done", arguments={"answer": "ok"}),
            ])

    agent = SingleAgent(AgentConfig(agent_id="t", provider="minimax",
                                    workspace=Path("/tmp")), llm=FakeLLM(provider="x"))
    assert await agent.run("hi") == "ok"

详见 README.md。
"""

from .config import AgentConfig
from .llm_adapter import LLMAdapter, LLMResult, ToolCall, LLMCallError
from .tools import Tool, ToolRegistry, ToolContext, build_default_registry
from .tool_loop import ToolLoop
from .task_queue import TaskQueue, TaskItem
from .memory import Memory, EventLogger
from .agent import SingleAgent

__all__ = [
    "AgentConfig",
    "LLMAdapter",
    "LLMResult",
    "ToolCall",
    "LLMCallError",
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "build_default_registry",
    "ToolLoop",
    "TaskQueue",
    "TaskItem",
    "Memory",
    "EventLogger",
    "SingleAgent",
]