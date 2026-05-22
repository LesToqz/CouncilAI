from __future__ import annotations

import json
from pathlib import Path

from src.orchestration.state import DebateState
from src.utils.timing import filename_timestamp, utc_now_iso


class DebateJSONLLogger:
    def __init__(self, settings: dict) -> None:
        logging_settings = settings.get("logging", {})
        self.enabled = bool(logging_settings.get("save_jsonl", True))
        self.log_dir = Path(logging_settings.get("jsonl_dir", "data/debate_logs"))

    def write(self, state: DebateState) -> Path | None:
        if not self.enabled:
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{filename_timestamp()}_{state.run_id}_debate.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for turn in state.turns:
                payload = turn.model_dump() if hasattr(turn, "model_dump") else turn.dict()
                payload["timestamp"] = utc_now_iso()
                payload["run_id"] = state.run_id
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path
