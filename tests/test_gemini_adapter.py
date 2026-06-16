import asyncio

from src.browser.gemini_adapter import GeminiAdapter


class FakeElement:
    def __init__(self, text: str) -> None:
        self.text = text

    async def inner_html(self) -> str:
        return self.text

    async def text_content(self) -> str:
        return self.text


class FakeLocator:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    async def count(self) -> int:
        return len(self.texts)

    def nth(self, index: int) -> FakeElement:
        return FakeElement(self.texts[index])


class FakePage:
    def __init__(self, selector_texts: dict[str, list[str]]) -> None:
        self.selector_texts = selector_texts

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self.selector_texts.get(selector, []))


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


class FakeAskGeminiAdapter(FakeGeminiAdapter):
    async def ensure_logged_in(self) -> bool:
        return True

    async def send_prompt(self, prompt: str) -> None:
        self._last_submitted_prompt = prompt
        self._last_completed_response = ""


def test_gemini_wait_accepts_changed_text_when_response_count_is_reused() -> None:
    adapter = FakeGeminiAdapter(
        responses=["old answer", "new answer", "new answer", "new answer"],
        counts=[1, 1, 1, 1],
    )
    adapter._response_count_before_send = 1

    asyncio.run(adapter.wait_for_response_complete(previous_response="old answer"))


def test_gemini_ask_uses_stable_response_when_final_extract_is_prompt_echo() -> None:
    adapter = FakeAskGeminiAdapter(
        responses=[
            "old answer",
            "new answer",
            "new answer",
            "new answer",
            "summarize the result",
        ],
        counts=[1, 1, 1, 1, 1],
    )
    adapter._response_count_before_send = 1

    response = asyncio.run(adapter.ask("summarize the result"))

    assert response == "new answer"


def test_gemini_extract_accepts_reused_response_block() -> None:
    adapter = GeminiAdapter(
        model_key="gemini",
        site={"url": "https://gemini.google.com"},
        context=None,
        settings={},
        page=FakePage({"model-response": ["new critique answer"]}),
    )
    adapter._response_counts_before_send = {"model-response": 1}
    adapter._last_submitted_prompt = "critique this answer"

    response = asyncio.run(adapter.extract_latest_response())

    assert response == "new critique answer"


def test_gemini_extract_rejects_reused_prompt_echo() -> None:
    adapter = GeminiAdapter(
        model_key="gemini",
        site={"url": "https://gemini.google.com"},
        context=None,
        settings={},
        page=FakePage({"model-response": ["critique this answer"]}),
    )
    adapter._response_counts_before_send = {"model-response": 1}
    adapter._last_submitted_prompt = "critique this answer"

    response = asyncio.run(adapter.extract_latest_response())

    assert response == ""
