import asyncio

from src.orchestration.nodes import DebateNodeRunner
from src.orchestration.state import DebateState


class BarrierAdapter:
    def __init__(self, model_key: str, started: list[str], all_started: asyncio.Event, expected: int) -> None:
        self.model_key = model_key
        self.started = started
        self.all_started = all_started
        self.expected = expected

    async def ask(self, prompt: str) -> str:
        self.started.append(self.model_key)
        if len(self.started) == self.expected:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), timeout=0.25)
        return f"{self.model_key} answered: {prompt[:12]}"


def test_initial_answer_round_fans_out_to_all_models() -> None:
    async def run() -> None:
        settings = {
            "model_sites": {
                "chatgpt": {"name": "ChatGPT", "role": "strategist"},
                "gemini": {"name": "Gemini", "role": "researcher"},
                "claude": {"name": "Claude", "role": "critic"},
            },
            "logging": {"save_jsonl": False, "save_sqlite": False},
        }
        started: list[str] = []
        all_started = asyncio.Event()
        runner = DebateNodeRunner(settings)
        runner.adapters = {
            model_key: BarrierAdapter(model_key, started, all_started, expected=3)
            for model_key in ("chatgpt", "gemini", "claude")
        }
        state = DebateState(
            user_prompt="Explain normal mode",
            mode="normal",
            max_iterations=1,
            active_models=["chatgpt", "gemini", "claude"],
        )

        result = await runner.initial_answer_round(state)

        assert set(started) == {"chatgpt", "gemini", "claude"}
        assert result.active_models == ["chatgpt", "gemini", "claude"]
        assert set(result.initial_answers) == {"chatgpt", "gemini", "claude"}

    asyncio.run(run())
