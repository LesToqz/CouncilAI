from __future__ import annotations

import sqlite3
from pathlib import Path

from src.orchestration.state import DebateState
from src.utils.timing import utc_now_iso


class SQLiteStore:
    def __init__(self, settings: dict) -> None:
        logging_settings = settings.get("logging", {})
        self.enabled = bool(logging_settings.get("save_sqlite", True))
        self.path = Path(logging_settings.get("sqlite_path", "data/sqlite/debates.db"))

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def ensure_schema(self) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs(
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
                CREATE TABLE IF NOT EXISTS turns(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model TEXT,
                    iteration INTEGER,
                    phase TEXT,
                    prompt TEXT,
                    response TEXT,
                    error TEXT,
                    created_at TEXT
                )
                """
            )

    def save(self, state: DebateState) -> None:
        if not self.enabled:
            return

        self.ensure_schema()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(
                    id, created_at, user_prompt, mode, max_iterations, final_answer
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    state.run_id,
                    state.created_at,
                    state.user_prompt,
                    state.mode,
                    state.max_iterations,
                    state.final_answer,
                ),
            )
            conn.execute("DELETE FROM turns WHERE run_id = ?", (state.run_id,))
            conn.executemany(
                """
                INSERT INTO turns(
                    run_id, model, iteration, phase, prompt, response, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        state.run_id,
                        turn.model,
                        turn.iteration,
                        turn.phase,
                        turn.prompt,
                        turn.response,
                        turn.error,
                        utc_now_iso(),
                    )
                    for turn in state.turns
                ],
            )
