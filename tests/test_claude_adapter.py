import asyncio

from src.browser.claude_adapter import ClaudeAdapter


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


class FakeClaudeAdapter(ClaudeAdapter):
    def __init__(self, responses: list[str], counts: list[int]) -> None:
        super().__init__(
            model_key="claude",
            site={"url": "https://claude.ai"},
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

    async def _count_assistant_blocks(self) -> int:
        value = self.counts[min(self.count_index, len(self.counts) - 1)]
        self.count_index += 1
        return value

    async def _wait_while_stop_button_visible(self) -> None:
        return None


class FakeAskClaudeAdapter(FakeClaudeAdapter):
    async def ensure_logged_in(self) -> bool:
        return True

    async def send_prompt(self, prompt: str) -> None:
        self._last_submitted_prompt = prompt
        self._last_completed_response = ""


def test_claude_wait_accepts_changed_text_when_response_count_is_reused() -> None:
    adapter = FakeClaudeAdapter(
        responses=["old answer", "new critique", "new critique", "new critique"],
        counts=[1, 1, 1, 1],
    )
    adapter._response_count_before_send = 1

    asyncio.run(adapter.wait_for_response_complete(previous_response="old answer"))


def test_claude_wait_caches_only_new_suffix_when_container_combines_turns() -> None:
    adapter = FakeClaudeAdapter(
        responses=[
            "old answer",
            "old answer\n\nnew critique",
            "old answer\n\nnew critique",
            "old answer\n\nnew critique",
        ],
        counts=[1, 1, 1, 1],
    )
    adapter._response_count_before_send = 1

    asyncio.run(adapter.wait_for_response_complete(previous_response="old answer"))

    assert adapter._last_completed_response == "new critique"


def test_claude_ask_uses_stable_response_when_final_extract_is_prompt_echo() -> None:
    adapter = FakeAskClaudeAdapter(
        responses=[
            "old answer",
            "new critique",
            "new critique",
            "new critique",
            "critique this answer",
        ],
        counts=[1, 1, 1, 1, 1],
    )
    adapter._response_count_before_send = 1

    response = asyncio.run(adapter.ask("critique this answer"))

    assert response == "new critique"


def test_claude_extract_accepts_reused_response_block() -> None:
    adapter = ClaudeAdapter(
        model_key="claude",
        site={"url": "https://claude.ai"},
        context=None,
        settings={},
        page=FakePage({"[data-message-role='assistant']": ["new critique answer"]}),
    )
    adapter._response_count_before_send = 1
    adapter._last_submitted_prompt = "critique this answer"

    response = asyncio.run(adapter.extract_latest_response())

    assert response == "new critique answer"


def test_claude_extract_rejects_reused_prompt_echo() -> None:
    adapter = ClaudeAdapter(
        model_key="claude",
        site={"url": "https://claude.ai"},
        context=None,
        settings={},
        page=FakePage({"[data-message-role='assistant']": ["critique this answer"]}),
    )
    adapter._response_count_before_send = 1
    adapter._last_submitted_prompt = "critique this answer"

    response = asyncio.run(adapter.extract_latest_response())

    assert response == ""
