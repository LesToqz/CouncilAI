from __future__ import annotations

import asyncio
import time

from src.browser.base_chat_adapter import BaseChatAdapter


class ClaudeAdapter(BaseChatAdapter):
    """Adapter for claude.ai with custom response-tracking logic.

    Claude keeps all conversation turns on a single page, so the base
    class ``extract_latest_response`` cannot reliably distinguish old
    replies from new ones.  This adapter mirrors the approach used by
    ``GeminiAdapter``: it snapshots the number of assistant-message
    containers *before* a prompt is sent, then waits for a new container
    to appear and reads text only from it.
    """

    PROMPT_SELECTORS = [
        # Claude uses a ProseMirror contenteditable inside a fieldset.
        "fieldset div[contenteditable='true'].ProseMirror",
        "fieldset div[contenteditable='true']",
        "div[contenteditable='true'].ProseMirror",
        "div[contenteditable='true'][role='textbox']",
        "div[contenteditable='true']",
        "textarea",
    ]

    RESPONSE_SELECTORS = [
        # Primary: scoped to assistant-only message containers.
        "[data-is-streaming]",
        "[data-message-role='assistant']",
        "[data-testid='message-content']",
        ".font-claude-message",
        ".prose",
        ".markdown",
        "article",
    ]

    STOP_SELECTORS = [
        "button[aria-label*='Stop']",
        "button:has-text('Stop')",
        "button:has-text('Stop response')",
    ]

    SUBMIT_SELECTORS = [
        "button[aria-label='Send Message']",
        "button[aria-label*='Send']",
        "fieldset button[type='submit']",
        "fieldset button:not([aria-label='Attach']):not([aria-label='Add content'])",
        "button:has-text('Send')",
    ]

    # ── Selectors for locating individual assistant-message blocks ───
    # We try multiple patterns because Claude.ai's DOM evolves.  The
    # adapter walks through these in order and uses the first one that
    # returns at least one element.
    _ASSISTANT_BLOCK_SELECTORS = [
        "[data-is-streaming]",
        "[data-message-role='assistant']",
        "[data-testid='message-content']",
        ".font-claude-message",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_count_before_send: int = 0

    # ── Override send_prompt ─────────────────────────────────────────
    async def send_prompt(self, prompt: str) -> None:
        """Snapshot the current number of assistant-message blocks, then send."""
        self._response_count_before_send = await self._count_assistant_blocks()
        await super().send_prompt(prompt)

    # ── Override extract_latest_response ─────────────────────────────
    async def extract_latest_response(self) -> str:
        """Return text only from the newest assistant-message block.

        Claude keeps every turn on the same page, so we must avoid
        returning text from an older reply.  The approach:
        1.  Try each selector in ``_ASSISTANT_BLOCK_SELECTORS`` to find
            assistant-only message containers.
        2.  Among those, pick the **last** one whose index is
            >= ``_response_count_before_send`` (i.e. it appeared after
            we submitted the latest prompt).
        3.  Fall back to the generic ``RESPONSE_SELECTORS`` list for
            forward-compatibility with DOM changes.
        """
        if self.page is None:
            return ""

        # ── Primary: scoped to assistant-message containers ──────────
        for selector in self._ASSISTANT_BLOCK_SELECTORS:
            try:
                all_blocks = self.page.locator(selector)
                count = await all_blocks.count()
                if count > 0:
                    # Walk from newest to oldest, pick the first that is new.
                    for idx in range(count - 1, -1, -1):
                        if idx < self._response_count_before_send:
                            break  # everything from here is old
                        text = await self._extract_clean_text(all_blocks.nth(idx))
                        if text:
                            return text
            except Exception:
                continue

        # ── Fallback: generic RESPONSE_SELECTORS ─────────────────────
        for selector in self.RESPONSE_SELECTORS:
            # Skip selectors we already tried above.
            if selector in self._ASSISTANT_BLOCK_SELECTORS:
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

    # ── Override wait_for_response_complete ───────────────────────────
    async def wait_for_response_complete(
        self,
        previous_response: str | None = None,
        submitted_prompt: str | None = None,
    ) -> None:
        """Claude-specific wait: first wait for a *new* assistant-message
        block to appear (count increases), then wait for the text inside
        it to stabilise."""

        timeout_ms = int(self.settings.get("browser", {}).get("timeout_ms", 120000))
        min_chars = int(self.settings.get("debate", {}).get("min_response_chars", 100))
        poll_interval = float(
            self.settings.get("browser", {}).get("response_poll_interval_seconds", 1.0)
        )
        stable_polls_required = int(
            self.settings.get("browser", {}).get("response_stable_polls", 3)
        )
        deadline = time.monotonic() + timeout_ms / 1000

        last_text = ""
        stable_count = 0
        saw_new = False

        while time.monotonic() < deadline:
            # Let the stop button disappear first (model still generating).
            await self._wait_while_stop_button_visible()

            current_count = await self._count_assistant_blocks()
            if current_count > self._response_count_before_send:
                saw_new = True

            current_text = await self.extract_latest_response()

            # Guard against picking up the prompt echo.
            if submitted_prompt and self._looks_like_prompt_echo(
                current_text, submitted_prompt
            ):
                await asyncio.sleep(poll_interval)
                continue

            if saw_new and current_text and current_text != (previous_response or ""):
                if current_text == last_text and len(current_text) >= min_chars:
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

            raise EmptyResponseError(
                f"{self.model_key}: Claude response did not appear after prompt submission"
            )

        if submitted_prompt and self._looks_like_prompt_echo(
            last_text, submitted_prompt
        ):
            from src.browser.base_chat_adapter import EmptyResponseError

            raise EmptyResponseError(
                f"{self.model_key}: latest text looks like the submitted prompt, not a response"
            )

    # ── Private helpers ──────────────────────────────────────────────
    async def _count_assistant_blocks(self) -> int:
        """Return how many assistant-message containers currently exist.

        Tries each selector in ``_ASSISTANT_BLOCK_SELECTORS`` and returns
        the count from the first one that matches any elements.
        """
        if self.page is None:
            return 0
        for selector in self._ASSISTANT_BLOCK_SELECTORS:
            try:
                count = await self.page.locator(selector).count()
                if count > 0:
                    return count
            except Exception:
                continue
        return 0
