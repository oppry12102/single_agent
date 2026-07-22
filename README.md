# single_agent —— 封装良好的单 agent 框架

> 参考 [`ht_world/agent`](../ht_world/agent) 多 agent 协作系统里的单 agent 实现,
> 抽取核心、剥离 server 依赖、加上充分测试,作为后续多 agent 协作项目的基础。

**核心目标**: 把 Claude Code 风格的"多轮 LLM + 工具调用循环"封装成一个 Python 包,
让上层多 agent 协调层只关心"派任务 / 收答案",不关心内部循环细节。

---

## 特性

- **单入口 `SingleAgent`**:`run() / run_long() / run_short() / submit()` 四种用法覆盖同步/异步。
- **可插拔 LLM**:`LLMAdapter` 抽象,默认使用 [`tools/llm`](../llm) 包(OpenAI/Anthropic 双协议),允许注入 mock/自定义 client,便于测试和替换后端。
- **可插拔 Tools**:`Tool` 基类 + `ToolRegistry`,内置 `read_file / write_file / run_shell / done` 四个工具,支持沙箱(防 `../` 越界)。
- **可插拔 Memory**: `Memory` + `EventLogger`,本地 jsonl + 异步事件回调 + system overlay 热更新。
- **零强外部依赖**:仅需 `httpx`(LLM HTTP);不依赖 `shared/protocol.py` / WebSocket / Server,可在任意进程内运行。
- **充分测试**:95 个 pytest 用例覆盖各组件 happy path + 异常 + 边界 + 并发。

---

## 目录结构

```
tools/agent/
├── README.md                       # 本文档
├── single_agent/                   # 主包
│   ├── __init__.py                 # 公共 API 导出
│   ├── agent.py                    # SingleAgent 主类
│   ├── llm_adapter.py              # LLM 适配器(OpenAI/Anthropic)
│   ├── tool_loop.py                # Claude Code 风格主循环
│   ├── tools.py                    # Tool 基类 + Registry + 内置工具
│   ├── task_queue.py               # 长/短任务调度
│   ├── memory.py                   # EventLogger + Memory
│   ├── config.py                   # AgentConfig
│   └── prompts/
│       ├── system_long.md
│       └── system_short.md
├── tests/                          # pytest 测试
│   ├── conftest.py
│   ├── test_agent.py
│   ├── test_config.py
│   ├── test_llm_adapter.py
│   ├── test_memory.py
│   ├── test_task_queue.py
│   ├── test_tool_loop.py
│   └── test_tools.py
└── example.py                      # 一键运行示例
```

---

## 安装

`single_agent` 是纯 Python 包,无强制依赖。

```bash
# 仅用 httpx(走自定义 LLM)
pip install httpx

# 默认 LLM 后端需要 tools/llm
pip install httpx pyyaml  # tools/llm 本身也只需 httpx

# 测试
pip install pytest pytest-asyncio
```

把 `tools/agent` 父目录加入 `PYTHONPATH` 即可 `import single_agent`:

```bash
export PYTHONPATH=/home/htao/work/tools:$PYTHONPATH
```

---

## 快速上手

### 1. 最小例子: 一次性问答

```python
import asyncio
from pathlib import Path
from single_agent import SingleAgent, AgentConfig

async def main():
    agent = SingleAgent(AgentConfig(
        agent_id="alpha",
        provider="minimax",         # minimax / glm / kimi / 自定义
        workspace=Path("./workspace"),
        log_file=Path("./logs/alpha.jsonl"),
    ))
    async with agent:
        answer = await agent.run("用一句话介绍 Python")
    print(answer)

asyncio.run(main())
```

### 2. 短业务(单轮,不调工具)

```python
async with agent:
    print(await agent.run_short("1+1=?"))
```

### 3. 长业务(多轮 + 工具)

```python
async with agent:
    # agent 可自行决定调 read_file/run_shell,最终 done 提交答案
    print(await agent.run_long("统计 workspace 下有多少 .py 文件"))
```

### 4. 异步提交(任务后台跑,通过事件回调收答案)

适合多 agent 协作场景(上层 dispatcher 把任务塞过来,agent 自己异步处理):

```python
agent = SingleAgent(cfg)
agent.on_event(lambda ev: print("event:", ev))

async with agent:
    task_id = await agent.submit("...大型任务...", kind="long")
    # 答案会通过 on_event 回调送达,标记 kind="task_done"
```

---

## 配置

`AgentConfig` 字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `agent_id` | str | agent 唯一标识(多 agent 时用于区分) |
| `provider` | str | LLM 提供商:`minimax` / `glm` / `kimi` / 自定义 |
| `workspace` | Path | 工具沙箱根目录 |
| `log_file` | Path? | 本地 jsonl 日志路径,None = 不落盘 |
| `max_steps` | int | 主循环最大步数(防死循环),默认 30 |
| `llm_timeout_s` | float | 单次 LLM HTTP 超时,默认 60 |
| `llm_model` | str | 覆盖 provider 默认模型,空 = 走默认 |
| `system_long` / `system_short` | str | 自定义系统 prompt;空 = 走内置 `prompts/*.md` |
| `auto_close_llm` | bool | `close()` 时是否关闭 httpx client |

从 YAML 加载:

```python
from single_agent import AgentConfig
cfg = AgentConfig.from_yaml("./agent.yaml")
```

YAML 范例:

```yaml
agent_id: alpha
provider: minimax
workspace: ./workspace
log_file: ./logs/alpha.jsonl
max_steps: 30
llm_timeout_s: 60
```

---

## 自定义工具

继承 `Tool` 基类或直接 duck-type(只要有 `name` / `spec` / `async run()`):

```python
from single_agent import Tool, ToolContext

class WebSearchTool(Tool):
    name = "web_search"
    spec = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "用关键词搜索网页,返回前 5 条标题 + 摘要",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    async def run(self, args, ctx):
        # 你的搜索逻辑
        return {"results": ["...", "..."]}

# 注册
agent.register_tool(WebSearchTool())

# 短业务也想暴露?tools.allow_in_short_mode("web_search")
```

---

## 自定义 LLM

### 方式 A: 注入自定义 `LLMAdapter` 子类(推荐用于测试)

```python
from single_agent import LLMAdapter, LLMResult, ToolCall

class FakeLLM(LLMAdapter):
    async def chat_with_tools(self, messages, *, tools=None, **kw):
        return LLMResult(content="", tool_calls=[
            ToolCall("1", "done", {"answer": "fake answer"}),
        ])

agent = SingleAgent(cfg, llm=FakeLLM())
```

### 方式 B: 接入新的 LLM 服务

```python
class MyLLM(LLMAdapter):
    api_format = "openai"  # 或 "anthropic"
    def __init__(self):
        super().__init__(provider="")  # 跳过默认后端
        self.model = "my-model"
        self._base_url = "https://my-llm.example.com"
        self._backend_http = ...  # 你的 httpx client
```

---

## 多 agent 协作扩展点

`SingleAgent` 已为多 agent 协作场景预留几个关键 hook:

| Hook | 用法 |
|---|---|
| `agent.submit(text, kind, task_id, meta)` | 异步提交任务,立即返回 `task_id`;答案走 `on_event` |
| `agent.on_event(cb)` | 订阅事件流(`task_done` / `tool_call` / `llm_call` 等) |
| `agent.event_logger.set_emit(cb)` | 直接接管事件推送(推 server / 推 UI) |
| `agent.set_overlay(text)` | 热更新 system overlay(多 agent 共享知识层) |
| `agent.register_tool(tool)` | 动态加工具(如 `call_agent` 工具调其他 agent) |
| `agent.memory.set_state(...)` | 上层可读 agent 当前状态(idle / long_busy) |

### 典型多 agent 协作模式

```python
# alpha agent 配置 + 注册 call_agent 工具(自己实现)
class CallAgentTool(Tool):
    name = "call_agent"
    spec = { ... }

    async def run(self, args, ctx):
        target = args["target"]
        text = args["text"]
        # 通过 ctx.extra["call_fn"] 路由到对应 agent
        result = await ctx.extra["call_fn"](target, text)
        return {"answer": result}

alpha = SingleAgent(alpha_cfg, llm=alpha_llm)
beta  = SingleAgent(beta_cfg,  llm=beta_llm)
alpha.tools.register(CallAgentTool())
alpha.tools.ctx.extra["call_fn"] = lambda target, text: (
    beta.run(text) if target == "beta" else asyncio.sleep(0)
)
```

更复杂的协调(投票、leader/follower、lease 等)由上层 dispatcher 负责,
本框架不绑死任何拓扑。

---

## API 速查

```python
class SingleAgent:
    def __init__(config, *, llm=None, tools=None, memory=None, event_logger=None)
    async def start() -> None
    async def close() -> None              # 优雅退出

    # 同步接口(立即返回答案)
    async def run(text, *, kind="long") -> str
    async def run_long(text) -> str
    async def run_short(text) -> str

    # 异步接口(后台跑,通过回调收答案)
    async def submit(text, *, kind="long", task_id=None, meta=None) -> str

    # 动态更新
    def register_tool(tool) -> None
    def set_overlay(text) -> None

    # 事件
    def on_event(cb) -> None               # cb: async def cb(event: dict)

    # 上下文管理
    async def __aenter__() -> SingleAgent
    async def __aexit__(...) -> None
```

`event` 字典结构:

```json
{
  "kind": "task_done | llm_call | tool_call | state_change",
  "task_id": "...",
  "agent_id": "alpha",
  "answer": "...",      // task_done 时
  "status": "ok | error",
  "kind_kind": "long | short"   // task_done 时
}
```

---

## 与 ht_world 原实现的差异

| 维度 | ht_world `agent/` | 本包 `single_agent/` |
|---|---|---|
| 入口 | `agent/main.py` + `asyncio.run` | `SingleAgent` 类(async context manager) |
| 配置 | YAML(必填 `server_url` 等) | YAML / dict / 直接构造 |
| LLM 后端 | 必接 `tools/llm` | 必接 `tools/llm`(可注入 mock) |
| 网络层 | 强制 WebSocket ↔ server | **无**;用 `submit()` + `on_event()` 由上层路由 |
| 协议 | 依赖 `shared/protocol.py` 的 Envelope | 无协议依赖 |
| 工具 | 5 件(read/write/run_shell/call_agent/done) | 4 件(去掉 call_agent,由上层自实现) |
| 心跳 / 日志 | 必推 server | 本地落盘 + 可选回调 |
| 测试 | 集成 smoke | 95 个 pytest 单元 + 集成 |

**保留的设计**:
- Claude Code 风格多轮 LLM + 工具循环(`ToolLoop`)
- 长业务单槽 + 短业务 FIFO + LLM semaphore(`TaskQueue`)
- 防 `../` 越界的文件沙箱(`_safe_under`)
- system overlay 追加到 system prompt 的机制(`ToolLoop.set_system_overlay`)
- 协议无关的 LLMResult / ToolCall 表示(`LLMAdapter`)

**剥离的依赖**:
- `shared/protocol.py` Envelope — 多 agent 协作时由上层 dispatcher 负责消息路由
- `Connection` WebSocket — 由上层 dispatcher 实现
- `Heartbeat` — 可选;若需要,直接订阅 `event_logger` 的 emit
- `mesh_router` / `peer_connection` — 由上层 dispatcher 实现 P2P

---

## 测试

```bash
cd /home/htao/work/tools/agent
python3 -m pytest tests/ -v
```

测试覆盖:

| 文件 | 测试数 | 范围 |
|---|---|---|
| `test_config.py` | 7 | 构造、dict/yaml 加载、缺失依赖 |
| `test_llm_adapter.py` | 20 | OpenAI/Anthropic payload 构造 + 解析、协议分发、错误注入、子类化 |
| `test_tools.py` | 19 | registry、内置 4 工具、越界防护、Protocol duck-type |
| `test_tool_loop.py` | 12 | 长/短业务主循环、done/无 tool_calls/max_steps、overlay、on_log |
| `test_task_queue.py` | 7 | 长单槽、短 FIFO、错误状态、cancel cleanup |
| `test_memory.py` | 11 | Memory 状态切换、EventLogger 落盘/emit/截断/异常隔离 |
| `test_agent.py` | 19 | SingleAgent 生命周期、submit/on_event、动态工具、热更新 overlay、错误处理 |
| **合计** | **95** | |

测试用 `ScriptedLLM`(可编程 LLM,conftest 提供)避免真实 HTTP 调用,
执行时间约 6 秒。

---

## 设计要点

1. **零全局状态**:`SingleAgent` 实例之间无隐式共享;线程安全 = asyncio 单事件循环。
2. **可注入性优于可继承性**:每个核心组件(`LLMAdapter` / `ToolRegistry` / `Memory` /
   `EventLogger` / `TaskQueue`)都可通过构造函数注入,而不是用 monkey-patch。
3. **同步/异步双接口**:`run()` 直接 await 返回答案(脚本/CLI 友好);`submit()` 异步派任务
   (多 agent 协作友好)。两者底层共用同一 `ToolLoop`。
4. **错误显式 raise**:LLM 错误以 `LLMCallError` 抛出,由 `ToolLoop` 捕获并包成 `[LLM error: ...]`;
   工具错误以 `{"error": "..."}` 返回(LLM 可读、可修复)。
5. **沙箱一致**:`read_file` / `write_file` 都走 `_safe_under`(先 `resolve()` 再 `relative_to`)
   防 `../` 字面前缀绕过。

---

## License

内部项目,无外部 License 约束。