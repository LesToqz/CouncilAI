from __future__ import annotations

from src.browser.base_chat_adapter import BaseChatAdapter


class ChatGPTAdapter(BaseChatAdapter):
    PROMPT_SELECTORS = [
        "[data-testid='prompt-textarea']",
        "#prompt-textarea",
        "div[contenteditable='true']",
        "textarea",
    ]
    RESPONSE_SELECTORS = [
        "[data-message-author-role='assistant']",
        "article[data-testid^='conversation-turn'] .markdown",
        ".markdown",
        "article",
    ]
    STOP_SELECTORS = [
        "[data-testid='stop-button']",
        "button[aria-label*='Stop']",
        "button:has-text('Stop generating')",
        "button:has-text('Stop')",
    ]
