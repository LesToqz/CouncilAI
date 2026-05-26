from __future__ import annotations

import asyncio
import time
from typing import Any

from src.utils.text_cleaner import clean_response_html, clean_response_text


class ChatAdapterError(RuntimeError):
    pass


class NotLoggedInError(ChatAdapterError):
    pass


class PromptBoxNotFoundError(ChatAdapterError):
    pass


class EmptyResponseError(ChatAdapterError):
    pass


class BaseChatAdapter:
    PROMPT_SELECTORS = [
        "textarea",
        "div[contenteditable='true']",
    ]
    RESPONSE_SELECTORS = [
        "[data-message-author-role='assistant']",
        ".markdown",
        "article",
    ]
    STOP_SELECTORS = [
        "button[aria-label*='Stop']",
        "button:has-text('Stop')",
    ]
    SUBMIT_SELECTORS = [
        "button[data-testid='send-button']",
        "button[aria-label*='Send']",
        "button:has-text('Send')",
    ]
    LOGIN_HINT_SELECTORS = [
        "text=/log in|sign in|continue with/i",
        "button:has-text('Log in')",
        "button:has-text('Sign in')",
    ]

    def __init__(self, model_key: str, site: dict, context: Any, settings: dict, page: Any | None = None) -> None:
        self.model_key = model_key
        self.site = site
        self.context = context
        self.settings = settings
        self.page = page

    async def open(self) -> None:
        if self.page is not None:
            return

        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        await self.page.goto(self.site["url"], wait_until="domcontentloaded")

    async def ensure_logged_in(self) -> bool:
        if self.page is None:
            await self.open()
        prompt = await self._first_visible_locator(self.PROMPT_SELECTORS, timeout_seconds=12)
        if prompt is not None:
            return True

        login_hint = await self._first_visible_locator(self.LOGIN_HINT_SELECTORS, timeout_seconds=2)
        return login_hint is None

    async def send_prompt(self, prompt: str) -> None:
        if self.page is None:
            await self.open()
        prompt_box = await self._first_visible_locator(self.PROMPT_SELECTORS, timeout_seconds=20)
        if prompt_box is None:
            raise PromptBoxNotFoundError(f"{self.model_key}: prompt box not found")

        await prompt_box.click()
        try:
            await prompt_box.fill(prompt)
        except Exception:
            await self.page.keyboard.press("Control+A")
            await self.page.keyboard.press("Backspace")
            await self.page.keyboard.insert_text(prompt)
        await self._submit_prompt()

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
        saw_new_response = False

        while time.monotonic() < deadline:
            await self._wait_while_stop_button_visible()
            current_text = await self.extract_latest_response()
            is_new_response = bool(current_text.strip()) and current_text != (previous_response or "")
            if submitted_prompt and self._looks_like_prompt_echo(current_text, submitted_prompt):
                is_new_response = False

            if is_new_response:
                saw_new_response = True

            if saw_new_response and current_text == last_text and current_text.strip():
                stable_count += 1
            else:
                stable_count = 0
                last_text = current_text

            if stable_count >= stable_polls_required:
                return
            await asyncio.sleep(poll_interval)

        if not last_text.strip() or last_text == (previous_response or ""):
            raise EmptyResponseError(f"{self.model_key}: response did not change after prompt submission")

        if submitted_prompt and self._looks_like_prompt_echo(last_text, submitted_prompt):
            raise EmptyResponseError(f"{self.model_key}: latest text looks like the submitted prompt, not a response")

    async def extract_latest_response(self) -> str:
        if self.page is None:
            return ""

        for selector in self.RESPONSE_SELECTORS:
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
            except Exception:
                continue
            cleaned = []
            for index in range(count):
                cleaned.append(await self._extract_clean_text(locator.nth(index)))
            non_empty = [text for text in cleaned if text]
            if non_empty:
                return non_empty[-1]
        return ""

    async def ask(self, prompt: str) -> str:
        is_logged_in = await self.ensure_logged_in()
        if not is_logged_in:
            raise NotLoggedInError(f"{self.model_key}: user is not logged in")

        previous_response = await self.extract_latest_response()
        await self.send_prompt(prompt)
        await self.wait_for_response_complete(previous_response=previous_response, submitted_prompt=prompt)
        response = await self.extract_latest_response()
        if not response.strip():
            raise EmptyResponseError(f"{self.model_key}: empty response")
        if response == previous_response:
            raise EmptyResponseError(f"{self.model_key}: latest response is unchanged from the previous turn")
        if self._looks_like_prompt_echo(response, prompt):
            raise EmptyResponseError(f"{self.model_key}: latest text is the submitted prompt, not a response")
        return response

    async def _first_visible_locator(
        self,
        selectors: list[str],
        timeout_seconds: float = 5,
    ) -> Any | None:
        if self.page is None:
            return None

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            for selector in selectors:
                try:
                    locator = self.page.locator(selector)
                    count = await locator.count()
                    for index in range(count - 1, -1, -1):
                        candidate = locator.nth(index)
                        if await candidate.is_visible():
                            return candidate
                except Exception:
                    continue
            await asyncio.sleep(0.25)
        return None

    async def _wait_while_stop_button_visible(self, max_wait_seconds: float = 300) -> None:
        """Block until no stop-generation button is visible on the page.

        Previously this checked once and returned after a 1-second sleep even
        if the button was still visible.  Now it loops until the button is
        gone (or *max_wait_seconds* elapses).
        """
        if self.page is None:
            return
        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            any_visible = False
            for selector in self.STOP_SELECTORS:
                try:
                    locator = self.page.locator(selector).first
                    if await locator.count() and await locator.is_visible():
                        any_visible = True
                        break
                except Exception:
                    continue
            if not any_visible:
                return
            await asyncio.sleep(1)

    async def _submit_prompt(self) -> None:
        submit_button = await self._first_enabled_locator(self.SUBMIT_SELECTORS, timeout_seconds=3)
        if submit_button is not None:
            await submit_button.click()
            return
        await self.page.keyboard.press("Enter")

    async def _first_enabled_locator(
        self,
        selectors: list[str],
        timeout_seconds: float = 5,
    ) -> Any | None:
        if self.page is None:
            return None

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            for selector in selectors:
                try:
                    locator = self.page.locator(selector)
                    count = await locator.count()
                    for index in range(count - 1, -1, -1):
                        candidate = locator.nth(index)
                        if await candidate.is_visible() and await candidate.is_enabled():
                            return candidate
                except Exception:
                    continue
            await asyncio.sleep(0.25)
        return None

    async def _extract_clean_text(self, locator: Any) -> str:
        try:
            markup = await locator.inner_html()
            cleaned = clean_response_html(markup)
            if cleaned:
                return cleaned
        except Exception:
            pass

        try:
            return clean_response_text(await locator.text_content() or "")
        except Exception:
            return ""

    def _looks_like_prompt_echo(self, text: str, prompt: str) -> bool:
        normalized_text = " ".join(text.split())
        normalized_prompt = " ".join(prompt.split())
        if not normalized_text:
            return False
        # Exact match — definitely an echo.
        if normalized_text == normalized_prompt:
            return True
        # The extracted text is a short leading fragment of the prompt
        # (e.g. the user-message bubble is still rendering).
        # Only flag this when the text is clearly much shorter than the prompt
        # so we don't accidentally flag a genuine long response.
        if (
            len(normalized_text) > 40
            and len(normalized_text) < len(normalized_prompt) * 0.6
            and normalized_prompt.startswith(normalized_text)
        ):
            return True
        # Specific debate-system-prompt markers — these appear at the start of
        # the injected instructions and should never open a real reply.
        prompt_markers = (
            "You are participating in a multi-model debate.",
            "You are the final synthesizer in a multi-model debate.",
            "## Critique",
            "## Refinement",
        )
        return any(normalized_text.startswith(marker) for marker in prompt_markers)
