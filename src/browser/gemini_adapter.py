from __future__ import annotations

import asyncio
import time

from src.browser.base_chat_adapter import BaseChatAdapter, EmptyResponseError


class GeminiAdapter(BaseChatAdapter):
    PROMPT_SELECTORS = [
        "rich-textarea div[contenteditable='true']",
        "div.ql-editor",
        "div[contenteditable='true']",
        "textarea",
    ]
    RESPONSE_SELECTORS = [
        "model-response",
        "model-response message-content",
        "message-content",
        ".model-response-text",
        ".markdown",
        "[data-response-index]",
    ]
    STOP_SELECTORS = [
        "button[aria-label*='Stop']",
        "button:has-text('Stop response')",
        "button:has-text('Stop')",
    ]
    SUBMIT_SELECTORS = [
        "button[aria-label*='Send message']",
        "button[aria-label*='Send']",
        "button[aria-label*='Submit']",
        "button.send-button",
        "button:has-text('Send')",
    ]
    ASSISTANT_BLOCK_SELECTORS = [
        "model-response",
        "model-response message-content",
        "message-content",
        ".model-response-text",
        "[data-response-index]",
        ".markdown",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_count_before_send = 0
        self._response_counts_before_send: dict[str, int] = {}

    async def send_prompt(self, prompt: str) -> None:
        self._response_counts_before_send = await self._count_response_blocks_by_selector()
        self._response_count_before_send = max(self._response_counts_before_send.values(), default=0)
        await super().send_prompt(prompt)

    async def extract_latest_response(self) -> str:
        if self.page is None:
            return ""

        tried_selectors = set()
        for selector in self.ASSISTANT_BLOCK_SELECTORS:
            tried_selectors.add(selector)
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
            except Exception:
                continue

            previous_count = self._response_counts_before_send.get(selector, 0)
            for index in range(count - 1, -1, -1):
                if index < previous_count:
                    break
                text = await self._extract_clean_text(locator.nth(index))
                if text:
                    return text

        for selector in self.RESPONSE_SELECTORS:
            if selector in tried_selectors:
                continue
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
            except Exception:
                continue
            cleaned = [await self._extract_clean_text(locator.nth(index)) for index in range(count)]
            non_empty = [text for text in cleaned if text]
            if non_empty:
                return non_empty[-1]
        return ""

    async def _count_model_responses(self) -> int:
        counts = await self._count_response_blocks_by_selector()
        return max(counts.values(), default=0)

    async def _count_response_blocks_by_selector(self) -> dict[str, int]:
        if self.page is None:
            return {}
        counts = {}
        for selector in self.ASSISTANT_BLOCK_SELECTORS:
            try:
                counts[selector] = await self.page.locator(selector).count()
            except Exception:
                continue
        return counts

    async def _has_active_response_indicator(self) -> bool:
        if self.page is None:
            return False

        for selector in self.STOP_SELECTORS:
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
                for index in range(count):
                    candidate = locator.nth(index)
                    if await candidate.is_visible():
                        return True
            except Exception:
                continue

        try:
            busy = self.page.locator("[aria-busy='true'], [data-is-streaming='true']")
            count = await busy.count()
            for index in range(count):
                if await busy.nth(index).is_visible():
                    return True
            return False
        except Exception:
            return False

    async def wait_for_response_complete(
        self,
        previous_response: str | None = None,
        submitted_prompt: str | None = None,
    ) -> None:
        timeout_ms = int(self.settings.get("browser", {}).get("timeout_ms", 120000))
        poll_interval = float(self.settings.get("browser", {}).get("response_poll_interval_seconds", 1.0))
        stable_polls_required = int(self.settings.get("browser", {}).get("response_stable_polls", 3))
        deadline = time.monotonic() + timeout_ms / 1000

        last_text = ""
        stable_count = 0
        saw_new = False

        while time.monotonic() < deadline:
            current_count = await self._count_model_responses()
            if current_count > self._response_count_before_send:
                saw_new = True

            current_text = await self.extract_latest_response()

            if submitted_prompt and self._looks_like_prompt_echo(current_text, submitted_prompt):
                await asyncio.sleep(poll_interval)
                continue

            # Gemini sometimes streams a later turn through a reused container
            # or a new selector family. In that case the count may not increase,
            # but changed stable text is still a valid new response.
            if current_text and current_text != (previous_response or ""):
                saw_new = True

            if saw_new and current_text and current_text != (previous_response or ""):
                if current_text == last_text and current_text.strip():
                    stable_count += 1
                else:
                    stable_count = 0
                    last_text = current_text

                if stable_count >= stable_polls_required and not await self._has_active_response_indicator():
                    return

            await asyncio.sleep(poll_interval)

        if not last_text.strip() or last_text == (previous_response or ""):
            raise EmptyResponseError(f"{self.model_key}: Gemini response did not appear after prompt submission")

        if submitted_prompt and self._looks_like_prompt_echo(last_text, submitted_prompt):
            raise EmptyResponseError(f"{self.model_key}: latest text looks like the submitted prompt, not a response")
