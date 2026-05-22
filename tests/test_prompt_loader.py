from pathlib import Path

from src.prompts.prompt_loader import PromptLoader


def test_prompt_loader_renders_variables(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "sample.md").write_text("Hello {{NAME}} from {{PLACE}}.", encoding="utf-8")

    loader = PromptLoader(prompt_dir)

    assert loader.render("sample", {"NAME": "Ada", "PLACE": "CouncilAI"}) == "Hello Ada from CouncilAI."


def test_prompt_loader_reports_unresolved_variables(tmp_path: Path) -> None:
    loader = PromptLoader(tmp_path)

    assert loader.unresolved_variables("{{ONE}} {{TWO}} {{ONE}}") == ["ONE", "TWO"]
