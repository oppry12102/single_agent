"""Agent 配置。

不依赖任何外部服务;每个字段都有合理默认值,通常只需要指定 ``agent_id`` +
``provider`` + ``workspace`` 三项即可。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    """Agent 运行配置。

    Attributes:
        agent_id: 全局唯一 id(多 agent 协作时用来区分)。
        provider: LLM 提供商标识(``minimax`` / ``glm`` / ``kimi`` / 自定义)。
        workspace: 工具可写的根目录(``run_shell`` 在此 cwd;``write_file`` 写入此目录)。
        log_file: 本地 jsonl 日志路径;留 None 关闭本地落盘。
        max_steps: 主循环最大迭代步数(防止工具死循环)。
        llm_timeout_s: 单次 LLM HTTP 超时秒数。
        llm_model: 覆盖 provider 默认模型;留空走 provider 默认。
        heartbeat_interval_s: 心跳周期(预留 hook,本类不强制使用)。
        system_long / system_short: 系统 prompt;留空走内置 prompts/*.md。
        tools: 注册到 ``ToolRegistry`` 的工具集合;None 表示 ``build_default_registry``。
        auto_close_llm: ``close()`` 时是否自动关闭底层 httpx client。
    """

    agent_id: str
    provider: str
    workspace: Path
    log_file: Path | None = None
    max_steps: int = 30
    llm_timeout_s: float = 60.0
    llm_model: str = ""
    heartbeat_interval_s: float = 5.0
    system_long: str = ""
    system_short: str = ""
    auto_close_llm: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.log_file is not None:
            self.log_file = Path(self.log_file).resolve()
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """从 dict 创建(便于从 yaml/JSON 加载)。"""
        ws = data.get("workspace", "./workspace")
        log = data.get("log_file")
        return cls(
            agent_id=data["agent_id"],
            provider=data["provider"],
            workspace=Path(ws),
            log_file=Path(log) if log else None,
            max_steps=int(data.get("max_steps", 30)),
            llm_timeout_s=float(data.get("llm_timeout_s", 60.0)),
            llm_model=str(data.get("llm_model") or data.get("model", "")),
            heartbeat_interval_s=float(data.get("heartbeat_interval_s", 5.0)),
            system_long=str(data.get("system_long", "")),
            system_short=str(data.get("system_short", "")),
            auto_close_llm=bool(data.get("auto_close_llm", True)),
            extra=dict(data.get("extra", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """从 YAML 加载(可选依赖 pyyaml)。"""
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for from_yaml(); install with `pip install pyyaml`"
            ) from exc
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))