from src.orchestration.state import DebateState
from src.storage.sqlite_store import SQLiteStore, generate_title_from_prompt


def _settings(db_path) -> dict:
    return {
        "logging": {
            "save_sqlite": True,
            "sqlite_path": str(db_path),
        }
    }


def test_generate_title_from_prompt_uses_first_prompt_text() -> None:
    assert generate_title_from_prompt("how are you doing") == "how are you doing"


def test_generate_title_from_prompt_truncates_long_prompt() -> None:
    prompt = "explain database recommendation for my project with tradeoffs and implementation steps"

    assert generate_title_from_prompt(prompt) == "explain database recommendation for my project with tradeoff..."


def test_generate_title_from_prompt_collapses_newlines_and_spaces() -> None:
    assert generate_title_from_prompt("how\n\nare    you\tdoing") == "how are you doing"


def test_save_and_list_chat_history(tmp_path) -> None:
    store = SQLiteStore(_settings(tmp_path / "debates.db"))
    state = DebateState(
        run_id="run-1",
        user_prompt="how are you doing",
        mode="normal",
        max_iterations=1,
        active_models=["chatgpt"],
    )
    state.record_turn("chatgpt", 0, "initial_answer", "prompt", "I am doing well.")

    store.save_run_summary(state)
    history = store.list_chat_history()

    assert len(history) == 1
    assert history[0]["run_id"] == "run-1"
    assert history[0]["title"] == "how are you doing"
    assert history[0]["user_prompt"] == "how are you doing"
    assert history[0]["status"] == "completed"


def test_get_run_detail_returns_prompt_final_answer_and_turns(tmp_path) -> None:
    store = SQLiteStore(_settings(tmp_path / "debates.db"))
    state = DebateState(
        run_id="run-2",
        user_prompt="compare normal and debate mode",
        mode="debate",
        max_iterations=2,
        active_models=["chatgpt", "gemini"],
        final_answer="Final synthesis text.",
    )
    state.record_turn("chatgpt", 0, "initial_answer", "prompt 1", "answer 1")
    state.record_turn("gemini", 0, "initial_answer", "prompt 2", "answer 2")

    store.save_run_summary(state)
    detail = store.get_run_detail("run-2")

    assert detail is not None
    assert detail["run"]["run_id"] == "run-2"
    assert detail["run"]["user_prompt"] == "compare normal and debate mode"
    assert detail["run"]["final_answer"] == "Final synthesis text."
    assert [turn["model_name"] for turn in detail["turns"]] == ["chatgpt", "gemini"]
    assert [turn["response"] for turn in detail["turns"]] == ["answer 1", "answer 2"]


def test_schema_backfills_title_for_existing_runs(tmp_path) -> None:
    db_path = tmp_path / "debates.db"
    store = SQLiteStore(_settings(db_path))
    with store._connect() as conn:
        conn.execute(
            """
            CREATE TABLE runs(
                id TEXT PRIMARY KEY,
                created_at TEXT,
                user_prompt TEXT,
                mode TEXT,
                max_iterations INTEGER,
                final_answer TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO runs(id, created_at, user_prompt, mode, max_iterations, final_answer)
            VALUES('old-run', '2026-01-01T00:00:00Z', 'old prompt title', 'normal', 1, NULL)
            """
        )

    history = store.list_chat_history()

    assert history[0]["run_id"] == "old-run"
    assert history[0]["title"] == "old prompt title"
    assert history[0]["status"] == "completed"
    assert history[0]["updated_at"] == "2026-01-01T00:00:00Z"
