from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.config_loader import PROJECT_ROOT


class PromptLoader:
    def __init__(self, prompt_dir: str | Path | None = None) -> None:
        self.prompt_dir = Path(prompt_dir) if prompt_dir else PROJECT_ROOT / "prompts"

    def load(self, template_name: str) -> str:
        path = self.prompt_dir / template_name
        if path.suffix != ".md":
            path = path.with_suffix(".md")
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, template_name: str, variables: dict[str, Any]) -> str:
        rendered = self.load(template_name)
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered.strip()

    def unresolved_variables(self, text: str) -> list[str]:
        return sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)))
