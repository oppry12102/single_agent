"""single_agent 一键示例。

展示:
1. 直接构造 SingleAgent + 异步 context manager
2. run_short / run_long 两种业务
3. submit + on_event 异步模式
4. 自定义工具 + register_tool
5. set_overlay 热更新

运行::

    PYTHONPATH=/home/htao/work/tools python3 /home/htao/work/tools/agent/example.py

需要 ``tools/llm`` 包以及对应的 API key(``~/.config/llm-client/config.json`` 或环境变量)。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 让自己可以 import single_agent
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from single_agent import AgentConfig, SingleAgent, Tool, ToolContext


# ============================================================
# 自定义工具: 列出 workspace 下所有 .py 文件
# ============================================================


class ListPyFilesTool(Tool):
    name = "list_py_files"

    spec = {
        "type": "function",
        "function": {
            "name": "list_py_files",
            "description": (
                "列出 workspace 下所有 .py 文件相对路径。"
                "用于在回答'这个项目有哪些代码文件'之类的问题时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }

    async def run(self, args: dict, ctx: ToolContext) -> dict:
        py_files = sorted(
            str(p.relative_to(ctx.workspace))
            for p in ctx.workspace.rglob("*.py")
            if p.is_file()
        )
        return {"count": len(py_files), "files": py_files[:50]}


# ============================================================
# 主示例
# ============================================================


async def main() -> None:
    workspace = ROOT / "workspace"
    log_dir = ROOT / "logs"
    workspace.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    cfg = AgentConfig(
        agent_id="demo",
        provider="minimax",       # 替换成你配置好的 provider
        workspace=workspace,
        log_file=log_dir / "demo.jsonl",
        max_steps=10,
    )

    async with SingleAgent(cfg) as agent:
        # 1) 注册自定义工具
        agent.register_tool(ListPyFilesTool())

        # 2) 短业务: 一句话问答
        print(">>> run_short")
        ans = await agent.run_short("用一句话介绍 Python")
        print(ans)
        print()

        # 3) 长业务: 多轮 + 工具
        print(">>> run_long")
        ans = await agent.run_long(
            "统计 workspace 下有多少 .py 文件,然后用一句话告诉我"
        )
        print(ans)
        print()

        # 4) 异步提交 + 事件回调
        print(">>> submit + on_event")
        events: list[dict] = []

        async def on_event(ev):
            events.append(ev)

        agent.on_event(on_event)

        await agent.submit("写一首关于秋天的五言绝句", kind="short")
        await agent.submit("再写一首关于春天的", kind="short")

        # 等所有事件到位
        for _ in range(200):
            done_count = sum(
                1 for ev in events
                if ev["kind"] == "task_done"
            )
            if done_count >= 2:
                break
            await asyncio.sleep(0.05)

        for ev in events:
            if ev["kind"] == "task_done":
                print(f"  [{ev['task_id']}] {ev['answer']}")


if __name__ == "__main__":
    asyncio.run(main())