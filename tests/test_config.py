"""测试 config.py。"""

from __future__ import annotations

from pathlib import Path

import pytest

from single_agent import AgentConfig


def test_post_init_creates_workspace(tmp_path: Path):
    ws = tmp_path / "ws"
    cfg = AgentConfig(agent_id="x", provider="p", workspace=ws)
    assert ws.exists()
    assert ws.is_dir()


def test_post_init_creates_log_parent(tmp_path: Path):
    log = tmp_path / "a" / "b" / "log.jsonl"
    cfg = AgentConfig(agent_id="x", provider="p", workspace=tmp_path, log_file=log)
    assert log.parent.exists()


def test_log_file_optional(tmp_path: Path):
    cfg = AgentConfig(agent_id="x", provider="p", workspace=tmp_path)
    assert cfg.log_file is None


def test_from_dict_minimal(tmp_path: Path):
    cfg = AgentConfig.from_dict({
        "agent_id": "a", "provider": "p", "workspace": str(tmp_path),
    })
    assert cfg.agent_id == "a"
    assert cfg.provider == "p"
    assert cfg.max_steps == 30  # default
    assert cfg.llm_timeout_s == 60.0  # default


def test_from_dict_full(tmp_path: Path):
    cfg = AgentConfig.from_dict({
        "agent_id": "a",
        "provider": "p",
        "workspace": str(tmp_path),
        "log_file": str(tmp_path / "l.jsonl"),
        "max_steps": 5,
        "llm_timeout_s": 12.5,
        "llm_model": "x-model",
        "heartbeat_interval_s": 2.0,
    })
    assert cfg.max_steps == 5
    assert cfg.llm_timeout_s == 12.5
    assert cfg.llm_model == "x-model"
    assert cfg.heartbeat_interval_s == 2.0


def test_from_yaml(tmp_path: Path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "agent_id: a\nprovider: p\nworkspace: ./ws\nmax_steps: 7\n",
        encoding="utf-8",
    )
    cfg = AgentConfig.from_yaml(yaml_file)
    assert cfg.agent_id == "a"
    assert cfg.max_steps == 7
    assert cfg.workspace.exists()


def test_from_yaml_missing_dep(tmp_path: Path, monkeypatch):
    # 通过 monkeypatch 临时屏蔽 yaml
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **kw):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        AgentConfig.from_yaml(tmp_path / "x.yaml")