import asyncio

from src.browser.gemini_adapter import GeminiAdapter


class FakeGeminiAdapter(GeminiAdapter):
    def __init__(self, responses: list[str], counts: list[int]) -> None:
        super().__init__(
            model_key="gemini",
            site={"url": "https://gemini.google.com"},
            context=None,
            settings={
                "browser": {
                    "timeout_ms": 200,
                    "response_poll_interval_seconds": 0.001,
                    "response_stable_polls": 2,
                }
            },
        )
        self.responses = responses
        self.counts = counts
        self.response_index = 0
        self.count_index = 0

    async def extract_latest_response(self) -> str:
        value = self.responses[min(self.response_index, len(self.responses) - 1)]
        self.response_index += 1
        return value

    async def _count_model_responses(self) -> int:
        value = self.counts[min(self.count_index, len(self.counts) - 1)]
        self.count_index += 1
        return value

    async def _has_active_response_indicator(self) -> bool:
        return False


def test_gemini_wait_accepts_changed_text_when_response_count_is_reused() -> None:
    adapter = FakeGeminiAdapter(
        responses=["old answer", "new answer", "new answer", "new answer"],
        counts=[1, 1, 1, 1],
    )
    adapter._response_count_before_send = 1

    asyncio.run(adapter.wait_for_response_complete(previous_response="old answer"))
