from pydantic import ValidationError

from src.orchestration.state import DebateState


def test_state_defaults_are_isolated() -> None:
    first = DebateState(user_prompt="A", mode="silent", max_iterations=1, active_models=["chatgpt", "claude"])
    second = DebateState(user_prompt="B", mode="observable", max_iterations=2, active_models=["gemini", "claude"])

    first.initial_answers["chatgpt"] = "answer"
    first.errors.append("warning")

    assert second.initial_answers == {}
    assert second.errors == []


def test_record_turn_adds_error_to_state() -> None:
    state = DebateState(user_prompt="A", mode="silent", max_iterations=1, active_models=["chatgpt", "claude"])

    state.record_turn("chatgpt", 1, "critique", "prompt", "", "timeout")

    assert len(state.turns) == 1
    assert state.errors == ["chatgpt critique: timeout"]


def test_mode_validation() -> None:
    try:
        DebateState(user_prompt="A", mode="debug", max_iterations=1, active_models=["chatgpt", "claude"])  # type: ignore[arg-type]
    except ValidationError:
        return

    raise AssertionError("invalid mode should fail validation")
