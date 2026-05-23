from __future__ import annotations

import asyncio
import time

from src.browser.base_chat_adapter import BaseChatAdapter


class GeminiAdapter(BaseChatAdapter):
    PROMPT_SELECTORS = [
        "rich-textarea div[contenteditable='true']",
        "div.ql-editor",
        "div[contenteditable='true']",
        "textarea",
    ]
    RESPONSE_SELECTORS = [
        # Gemini marks model-turn containers; target only model (not user) messages.
        "model-response message-content",
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

    # ── Override extract_latest_response ─────────────────────────────
    # Gemini wraps each model reply inside a <model-response> container.
    # We count how many model-response elements exist *before* we send a
    # prompt so we can always grab text from the very last one, avoiding
    # stale results from earlier turns.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_count_before_send: int = 0

    async def send_prompt(self, prompt: str) -> None:
        """Snapshot the current number of model-response blocks, then send."""
        self._response_count_before_send = await self._count_model_responses()
        await super().send_prompt(prompt)

    async def extract_latest_response(self) -> str:
        """Return text only from the newest model-response block.

        Gemini keeps every turn on the same page, so we must avoid
        returning text from an older reply.  The approach:
        1.  Try ``model-response message-content`` to get model-only
            message blocks (skips user messages).
        2.  Among those blocks, pick the **last** one whose index is
            >= ``_response_count_before_send`` (i.e. it appeared after
            we submitted the latest prompt).
        3.  Fall back to the generic selectors in ``RESPONSE_SELECTORS``
            for forward-compatibility with DOM changes.
        """
        if self.page is None:
            return ""

        # ── Primary: scoped to model-response containers ────────────
        model_msg_selector = "model-response message-content"
        try:
            all_msgs = self.page.locator(model_msg_selector)
            count = await all_msgs.count()
            if count > 0:
                # Walk from newest to oldest, pick the first that is new.
                for idx in range(count - 1, -1, -1):
                    if idx < self._response_count_before_send:
                        break  # everything from here is old
                    text = await self._extract_clean_text(all_msgs.nth(idx))
                    if text:
                        return text
        except Exception:
            pass

        # ── Fallback: remaining selectors (skip the one we just tried) ──
        for selector in self.RESPONSE_SELECTORS:
            if selector == model_msg_selector:
                continue
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
            except Exception:
                continue
            cleaned = [await self._extract_clean_text(locator.nth(idx)) for idx in range(count)]
            non_empty = [t for t in cleaned if t]
            if non_empty:
                return non_empty[-1]
        return ""

    async def _count_model_responses(self) -> int:
        """Return how many ``model-response message-content`` elements exist."""
        if self.page is None:
            return 0
        try:
            return await self.page.locator("model-response message-content").count()
        except Exception:
            return 0

    async def wait_for_response_complete(
        self,
        previous_response: str | None = None,
        submitted_prompt: str | None = None,
    ) -> None:
        """Gemini-specific wait: first wait for a *new* model-response
        block to appear (index >= _response_count_before_send), then
        wait for the text inside it to stabilise."""

        timeout_ms = int(self.settings.get("browser", {}).get("timeout_ms", 120000))
        poll_interval = float(self.settings.get("browser", {}).get("response_poll_interval_seconds", 1.0))
        stable_polls_required = int(self.settings.get("browser", {}).get("response_stable_polls", 3))
        deadline = time.monotonic() + timeout_ms / 1000

        last_text = ""
        stable_count = 0
        saw_new = False

        while time.monotonic() < deadline:
            # Let the stop button disappear first (model still generating).
            await self._wait_while_stop_button_visible()

            current_count = await self._count_model_responses()
            if current_count > self._response_count_before_send:
                saw_new = True

            current_text = await self.extract_latest_response()

            # Guard against picking up the prompt echo.
            if submitted_prompt and self._looks_like_prompt_echo(current_text, submitted_prompt):
                await asyncio.sleep(poll_interval)
                continue

            if saw_new and current_text and current_text != (previous_response or ""):
                if current_text == last_text and current_text.strip():
                    stable_count += 1
                else:
                    stable_count = 0
                    last_text = current_text

                if stable_count >= stable_polls_required:
                    return

            await asyncio.sleep(poll_interval)

        # Timeout fallback ─ same checks as the base class.
        if not last_text.strip() or last_text == (previous_response or ""):
            from src.browser.base_chat_adapter import EmptyResponseError
            raise EmptyResponseError(f"{self.model_key}: Gemini response did not appear after prompt submission")

        if submitted_prompt and self._looks_like_prompt_echo(last_text, submitted_prompt):
            from src.browser.base_chat_adapter import EmptyResponseError
            raise EmptyResponseError(f"{self.model_key}: latest text looks like the submitted prompt, not a response")
