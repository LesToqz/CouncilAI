from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.utils.timing import utc_now_iso


class ModelTurn(BaseModel):
    model: str
    iteration: int
    phase: str
    prompt: str
    response: str
    error: str | None = None


class DebateState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=utc_now_iso)
    user_prompt: str
    mode: Literal["silent", "observable"]
    max_iterations: int
    active_models: list[str]
    current_iteration: int = 0
    initial_answers: dict[str, str] = Field(default_factory=dict)
    critiques: dict[str, list[str]] = Field(default_factory=dict)
    refinements: dict[str, list[str]] = Field(default_factory=dict)
    turns: list[ModelTurn] = Field(default_factory=list)
    final_answer: str | None = None
    errors: list[str] = Field(default_factory=list)

    def latest_answer_for(self, model_key: str) -> str:
        refinements = self.refinements.get(model_key) or []
        if refinements:
            return refinements[-1]
        return self.initial_answers.get(model_key, "")

    def record_turn(
        self,
        model: str,
        iteration: int,
        phase: str,
        prompt: str,
        response: str,
        error: str | None = None,
    ) -> None:
        self.turns.append(
            ModelTurn(
                model=model,
                iteration=iteration,
                phase=phase,
                prompt=prompt,
                response=response,
                error=error,
            )
        )
        if error:
            self.errors.append(f"{model} {phase}: {error}")
