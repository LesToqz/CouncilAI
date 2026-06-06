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
    assert history[0]["chat_id"] == "run-1"
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


def test_second_run_reuses_existing_chat_without_new_history_item(tmp_path) -> None:
    store = SQLiteStore(_settings(tmp_path / "debates.db"))
    first = DebateState(
        run_id="run-1",
        user_prompt="first prompt",
        mode="normal",
        max_iterations=1,
        active_models=["chatgpt"],
    )
    second = DebateState(
        run_id="run-2",
        chat_id="run-1",
        user_prompt="second prompt",
        mode="normal",
        max_iterations=1,
        active_models=["gemini"],
    )

    store.save_run_summary(first)
    store.save_run_summary(second)

    history = store.list_chat_history()
    detail = store.get_chat_detail("run-1")

    assert len(history) == 1
    assert history[0]["chat_id"] == "run-1"
    assert history[0]["title"] == "first prompt"
    assert detail is not None
    assert [run_detail["run"]["run_id"] for run_detail in detail["runs"]] == ["run-1", "run-2"]
    assert [run_detail["run"]["user_prompt"] for run_detail in detail["runs"]] == ["first prompt", "second prompt"]


def test_mixed_modes_are_stored_per_run_inside_one_chat(tmp_path) -> None:
    store = SQLiteStore(_settings(tmp_path / "debates.db"))
    normal = DebateState(
        run_id="normal-run",
        user_prompt="normal prompt",
        mode="normal",
        max_iterations=1,
        active_models=["chatgpt"],
    )
    debate = DebateState(
        run_id="debate-run",
        chat_id="normal-run",
        user_prompt="debate prompt",
        mode="debate",
        max_iterations=2,
        active_models=["chatgpt", "claude"],
        final_answer="debate final",
    )

    store.save_run_summary(normal)
    store.save_run_summary(debate)
    detail = store.get_chat_detail("normal-run")

    assert detail is not None
    runs = [run_detail["run"] for run_detail in detail["runs"]]
    assert [run["mode"] for run in runs] == ["normal", "debate"]
    assert runs[0]["final_answer"] is None
    assert runs[1]["final_answer"] == "debate final"


def test_debate_then_normal_does_not_inherit_debate_mode(tmp_path) -> None:
    store = SQLiteStore(_settings(tmp_path / "debates.db"))
    debate = DebateState(
        run_id="debate-run",
        user_prompt="debate prompt",
        mode="debate",
        max_iterations=2,
        active_models=["chatgpt", "claude"],
        final_answer="debate final",
    )
    normal = DebateState(
        run_id="normal-run",
        chat_id="debate-run",
        user_prompt="normal prompt",
        mode="normal",
        max_iterations=1,
        active_models=["gemini"],
    )

    store.save_run_summary(debate)
    store.save_run_summary(normal)
    detail = store.get_chat_detail("debate-run")

    assert detail is not None
    runs = [run_detail["run"] for run_detail in detail["runs"]]
    assert [run["mode"] for run in runs] == ["debate", "normal"]
    assert runs[0]["final_answer"] == "debate final"
    assert runs[1]["final_answer"] is None


def test_delete_chat_removes_chat_runs_and_turns(tmp_path) -> None:
    store = SQLiteStore(_settings(tmp_path / "debates.db"))
    first = DebateState(
        run_id="run-1",
        user_prompt="first prompt",
        mode="normal",
        max_iterations=1,
        active_models=["chatgpt"],
    )
    second = DebateState(
        run_id="run-2",
        chat_id="run-1",
        user_prompt="second prompt",
        mode="normal",
        max_iterations=1,
        active_models=["gemini"],
    )
    first.record_turn("chatgpt", 0, "initial_answer", "prompt", "answer")
    second.record_turn("gemini", 0, "initial_answer", "prompt", "answer")

    store.save_run_summary(first)
    store.save_run_summary(second)
    store.delete_chat("run-1")

    assert store.list_chat_history() == []
    assert store.get_chat_detail("run-1") is None
    assert store.get_run_detail("run-1") is None
    assert store.get_run_detail("run-2") is None


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
    assert history[0]["chat_id"] == "old-run"
    assert history[0]["title"] == "old prompt title"
    assert history[0]["status"] == "completed"
    assert history[0]["updated_at"] == "2026-01-01T00:00:00Z"

    detail = store.get_chat_detail("old-run")

    assert detail is not None
    assert [run_detail["run"]["run_id"] for run_detail in detail["runs"]] == ["old-run"]
