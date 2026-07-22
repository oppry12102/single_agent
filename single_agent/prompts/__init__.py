"""默认 system prompt 模板。

可通过 ``AgentConfig.system_long / system_short`` 覆盖。
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


SYSTEM_LONG = (_PROMPTS_DIR / "system_long.md").read_text(encoding="utf-8")
SYSTEM_SHORT = (_PROMPTS_DIR / "system_short.md").read_text(encoding="utf-8")