from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.orchestration.state import DebateState, ModelTurn
from src.utils.timing import utc_now_iso


def generate_title_from_prompt(prompt: str, limit: int = 60) -> str:
    normalized = " ".join((prompt or "").split())
    if not normalized:
        return "Untitled chat"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


class SQLiteStore:
    def __init__(self, settings: dict) -> None:
        logging_settings = settings.get("logging", {})
        self.enabled = bool(logging_settings.get("save_sqlite", True))
        self.path = Path(logging_settings.get("sqlite_path", "data/sqlite/debates.db"))

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        self.ensure_schema()

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
                    final_answer TEXT,
                    title TEXT,
                    status TEXT,
                    updated_at TEXT,
                    metadata_json TEXT
                )
                """
            )
            self._add_column_if_missing(conn, "runs", "title", "TEXT")
            self._add_column_if_missing(conn, "runs", "status", "TEXT")
            self._add_column_if_missing(conn, "runs", "updated_at", "TEXT")
            self._add_column_if_missing(conn, "runs", "metadata_json", "TEXT")
            self._backfill_run_summaries(conn)
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_run_id ON turns(run_id)")

    def save(self, state: DebateState) -> None:
        self.save_run_summary(state)

    def save_run_summary(self, state: DebateState, status: str | None = None) -> None:
        if not self.enabled:
            return

        self.ensure_schema()
        updated_at = utc_now_iso()
        run_status = status or self._status_for_state(state)
        metadata = {
            "active_models": state.active_models,
            "current_iteration": state.current_iteration,
            "errors": state.errors,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(
                    id, created_at, user_prompt, mode, max_iterations, final_answer,
                    title, status, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.run_id,
                    state.created_at,
                    state.user_prompt,
                    state.mode,
                    state.max_iterations,
                    state.final_answer,
                    generate_title_from_prompt(state.user_prompt),
                    run_status,
                    updated_at,
                    json.dumps(metadata, ensure_ascii=False),
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

    def list_chat_history(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        self.ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id AS run_id,
                    COALESCE(NULLIF(title, ''), user_prompt, 'Untitled chat') AS title,
                    user_prompt,
                    mode,
                    status,
                    created_at,
                    updated_at
                FROM runs
                ORDER BY COALESCE(updated_at, created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        self.ensure_schema()
        with self._connect() as conn:
            run = conn.execute(
                """
                SELECT
                    id AS run_id,
                    COALESCE(NULLIF(title, ''), user_prompt, 'Untitled chat') AS title,
                    user_prompt,
                    final_answer,
                    mode,
                    max_iterations,
                    status,
                    created_at,
                    updated_at,
                    metadata_json
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                return None

            turns = conn.execute(
                """
                SELECT
                    id AS turn_id,
                    run_id,
                    model AS model_name,
                    model,
                    iteration,
                    phase,
                    prompt,
                    response,
                    error,
                    created_at
                FROM turns
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()

        return {
            "run": dict(run),
            "turns": [dict(turn) for turn in turns],
        }

    def delete_run(self, run_id: str) -> None:
        if not self.enabled:
            return

        self.ensure_schema()
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def state_from_detail(self, detail: dict[str, Any]) -> DebateState:
        run = detail["run"]
        metadata: dict[str, Any] = {}
        if run["metadata_json"]:
            try:
                metadata = json.loads(run["metadata_json"])
            except json.JSONDecodeError:
                metadata = {}
        turns = [
            ModelTurn(
                model=turn["model"],
                iteration=int(turn["iteration"] or 0),
                phase=turn["phase"] or "",
                prompt=turn["prompt"] or "",
                response=turn["response"] or "",
                error=turn["error"],
            )
            for turn in detail.get("turns", [])
        ]
        active_models = metadata.get("active_models") or [
            model for model in ("chatgpt", "gemini", "claude") if any(turn.model == model for turn in turns)
        ]
        mode = str(run["mode"] or "normal").lower()
        if mode not in {"normal", "debate"}:
            mode = "normal"
        errors = [f"{turn.model} {turn.phase}: {turn.error}" for turn in turns if turn.error]
        return DebateState(
            run_id=run["run_id"],
            created_at=run["created_at"] or utc_now_iso(),
            user_prompt=run["user_prompt"] or "",
            mode=mode,
            max_iterations=int(run["max_iterations"] or 1),
            active_models=active_models,
            turns=turns,
            final_answer=run["final_answer"],
            errors=errors,
        )

    def _status_for_state(self, state: DebateState) -> str:
        has_response = bool(state.final_answer) or any(turn.response for turn in state.turns)
        if state.errors and not has_response:
            return "failed"
        return "completed"

    def _add_column_if_missing(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_run_summaries(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, user_prompt, created_at, title, status, updated_at
            FROM runs
            WHERE title IS NULL OR title = '' OR status IS NULL OR status = '' OR updated_at IS NULL OR updated_at = ''
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE runs
                SET title = COALESCE(NULLIF(title, ''), ?),
                    status = COALESCE(NULLIF(status, ''), 'completed'),
                    updated_at = COALESCE(NULLIF(updated_at, ''), created_at, ?)
                WHERE id = ?
                """,
                (
                    generate_title_from_prompt(row["user_prompt"] or ""),
                    utc_now_iso(),
                    row["id"],
                ),
            )
