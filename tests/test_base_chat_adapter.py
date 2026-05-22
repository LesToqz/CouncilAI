import asyncio

import pytest

from src.browser.base_chat_adapter import BaseChatAdapter, EmptyResponseError


class FakeAdapter(BaseChatAdapter):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(
            model_key="fake",
            site={"url": "https://example.test"},
            context=None,
            settings={
                "browser": {
                    "timeout_ms": 200,
                    "response_poll_interval_seconds": 0.001,
                    "response_stable_polls": 2,
                },
                "debate": {"min_response_chars": 2},
            },
        )
        self.responses = responses
        self.index = 0

    async def extract_latest_response(self) -> str:
        value = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return value

    async def _wait_while_stop_button_visible(self) -> None:
        return None


def test_wait_for_response_requires_text_to_change() -> None:
    adapter = FakeAdapter(["old response"])

    with pytest.raises(EmptyResponseError, match="did not change"):
        asyncio.run(adapter.wait_for_response_complete(previous_response="old response"))


def test_wait_for_response_accepts_new_stable_text() -> None:
    adapter = FakeAdapter(["old response", "new response", "new response", "new response"])

    asyncio.run(adapter.wait_for_response_complete(previous_response="old response"))


def test_prompt_echo_is_rejected() -> None:
    adapter = FakeAdapter(
        [
            "You are participating in a multi-model debate.\n\nOriginal user question:",
            "You are participating in a multi-model debate.\n\nOriginal user question:",
        ]
    )

    with pytest.raises(EmptyResponseError, match="submitted prompt"):
        asyncio.run(
            adapter.wait_for_response_complete(
                previous_response="old response",
                submitted_prompt="You are participating in a multi-model debate.\n\nOriginal user question:",
            )
        )
