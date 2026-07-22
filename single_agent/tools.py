"""工具集 —— 基类 + 注册表 + 默认实现。

设计:
- ``Tool`` 是 dataclass 风格的基类;子类必须实现 ``name``、``spec``、``run``。
- ``ToolContext`` 携带 workspace / agent_id / 扩展钩子(``mesh`` 等)。
- ``ToolRegistry`` 管理所有工具;``specs(short_mode)`` 决定哪些工具暴露给 LLM。
- ``build_default_registry`` 组装默认工具集(read_file / write_file / run_shell / done)。
- 工具执行不抛异常;失败统一返回 ``{"error": ...}``,让 LLM 可以理解并修正。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ============================================================
# ToolContext
# ============================================================


@dataclass
class ToolContext:
    """工具执行上下文。

    Attributes:
        workspace: 工具可写的根目录。
        agent_id: 当前 agent id(用于沙箱/命名空间)。
        extra: 任意附加上下文(多 agent 协作时挂 mesh router 等)。
    """

    workspace: Path
    agent_id: str
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)


# ============================================================
# Tool 基类
# ============================================================


@runtime_checkable
class ToolLike(Protocol):
    """最小工具协议——只要有 name/spec/run 即可(不强制继承 Tool)。"""

    name: str
    spec: dict

    async def run(self, args: dict, ctx: ToolContext) -> dict: ...


class Tool:
    """工具基类(可选继承)。"""

    name: str = ""
    spec: dict = {}

    async def run(self, args: dict, ctx: ToolContext) -> dict:
        raise NotImplementedError


# ============================================================
# Registry
# ============================================================


class ToolRegistry:
    """工具注册表。

    ``short_mode`` 控制短业务时(单轮)允许暴露给 LLM 的工具集合;
    默认仅 ``done``,防止 LLM 在短业务里读文件绕弯。
    """

    def __init__(
        self,
        ctx: ToolContext,
        *,
        short_mode_allowed: set[str] | None = None,
    ):
        self.ctx = ctx
        self._tools: dict[str, ToolLike] = {}
        self._short_mode_allowed: set[str] = short_mode_allowed or {"done"}

    # ----------------------------------------------------------- registry
    def register(self, tool: ToolLike) -> None:
        if not tool.name:
            raise ValueError("tool.name must be non-empty")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolLike | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def allow_in_short_mode(self, name: str) -> None:
        self._short_mode_allowed.add(name)

    # ----------------------------------------------------------- specs
    def specs(self, short_mode: bool = False) -> list[dict]:
        if short_mode:
            return [
                t.spec for t in self._tools.values()
                if t.name in self._short_mode_allowed
            ]
        return [t.spec for t in self._tools.values()]

    # ----------------------------------------------------------- execute
    async def execute(
        self,
        name: str,
        args: dict,
        *,
        short_mode: bool = False,
    ) -> dict:
        if short_mode and name not in self._short_mode_allowed:
            return {"error": f"tool {name!r} not allowed in short mode"}
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name!r}"}
        try:
            return await tool.run(args, self.ctx)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


# ============================================================
# 内置工具
# ============================================================


def _safe_under(base: Path, target: Path) -> bool:
    """target 必须真正在 base 内(防 ../ 越界)。先 resolve 再比较。"""
    try:
        base_r = base.resolve()
        target_r = target.resolve()
        target_r.relative_to(base_r)
        return True
    except (ValueError, OSError):
        return False


class ReadFileTool(Tool):
    name = "read_file"
    spec = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "读取 workspace 下的文件。相对路径。"
                "可带 max_bytes(默认 1MB)限制大小。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer", "default": 1048576},
                },
                "required": ["path"],
            },
        },
    }

    async def run(self, args: dict, ctx: ToolContext) -> dict:
        rel = str(args.get("path", "")).lstrip("/")
        max_bytes = int(args.get("max_bytes", 1_048_576))
        if not rel:
            return {"error": "path is required"}
        target = (ctx.workspace / rel).resolve()
        if not _safe_under(ctx.workspace, target):
            return {"error": f"path escapes workspace: {args.get('path')}"}
        if not target.exists():
            return {"error": f"file not found: {rel}"}
        if not target.is_file():
            return {"error": f"not a file: {rel}"}
        try:
            data = target.read_bytes()[:max_bytes]
            return {
                "path": rel,
                "content": data.decode("utf-8", errors="replace"),
                "bytes": len(data),
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class WriteFileTool(Tool):
    name = "write_file"
    spec = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "写入文本到 workspace。相对路径(强制 sandbox 在 workspace 内)。"
                "会自动创建父目录。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }

    async def run(self, args: dict, ctx: ToolContext) -> dict:
        rel = str(args.get("path", ""))
        content = str(args.get("content", ""))
        if not rel:
            return {"error": "path is required"}
        # 先 resolve 再判断(用 lstrip('/') 即可,.. 由 _safe_under 处理)
        clean = rel.lstrip("/")
        target = (ctx.workspace / clean).resolve()
        if not _safe_under(ctx.workspace, target):
            return {"error": f"path escapes workspace: {args.get('path')}"}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"path": rel, "bytes": len(content.encode("utf-8"))}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class RunShellTool(Tool):
    name = "run_shell"
    spec = {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "执行 shell 命令(默认 30s 超时,输出截断到 10KB),"
                "工作目录为 workspace。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 30},
                },
                "required": ["cmd"],
            },
        },
    }

    MAX_OUTPUT_BYTES = 10 * 1024

    async def run(self, args: dict, ctx: ToolContext) -> dict:
        cmd = str(args.get("cmd", "")).strip()
        if not cmd:
            return {"error": "cmd is required"}
        timeout_s = int(args.get("timeout_s", 30))
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(ctx.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"error": f"timeout after {timeout_s}s", "cmd": cmd}
            out_bytes = stdout[: self.MAX_OUTPUT_BYTES]
            output = out_bytes.decode("utf-8", errors="replace")
            truncated = len(stdout) > self.MAX_OUTPUT_BYTES
            return {
                "cmd": cmd,
                "exit_code": proc.returncode,
                "output": output,
                "truncated": truncated,
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class DoneTool(Tool):
    name = "done"
    spec = {
        "type": "function",
        "function": {
            "name": "done",
            "description": "提交最终答案,结束当前任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            },
        },
    }

    async def run(self, args: dict, ctx: ToolContext) -> dict:
        # done 工具的返回值会被 ToolLoop 特殊处理,这里只回显
        return {"answer": str(args.get("answer", ""))}


# ============================================================
# 默认注册表构造
# ============================================================


def build_default_registry(ctx: ToolContext) -> ToolRegistry:
    """组装默认工具集(read/write/run_shell/done)。"""
    reg = ToolRegistry(ctx)
    reg.register(ReadFileTool())
    reg.register(WriteFileTool())
    reg.register(RunShellTool())
    reg.register(DoneTool())
    return reg