from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as markdownify_html


_ARTIFACT_PATTERNS = [
    r"^\s*(copy|share|retry|regenerate|thumbs up|thumbs down)\s*$",
    r"^\s*(good response|bad response)\s*$",
]

_INTERNAL_PROMPT_PATTERNS = [
    r"^\s*you are participating in a multi-model debate\.\s*$",
    r"^\s*you are the final synthesizer in a multi-model debate\.\s*$",
    r"^\s*councilai private debate context\..*$",
    r"^\s*councilai final synthesis task\..*$",
]

_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\([^)]*\)")


def clean_response_text(text: str) -> str:
    lines = []
    normalized = _MARKDOWN_IMAGE_PATTERN.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    for line in normalized.split("\n"):
        stripped = line.strip()
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in _ARTIFACT_PATTERNS):
            continue
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in _INTERNAL_PROMPT_PATTERNS):
            continue
        lines.append(line.rstrip())

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def clean_response_html(markup: str) -> str:
    """Convert rendered model output HTML into clean Markdown.

    Reading only ``text_content`` from model pages flattens rich output such
    as tables. Converting the assistant message HTML first preserves headings,
    lists, code blocks, links, and tables well enough for Gradio Markdown.
    """
    soup = BeautifulSoup(markup or "", "html.parser")
    for tag in soup.find_all(["button", "img", "picture", "script", "source", "style", "svg"]):
        tag.decompose()

    markdown = markdownify_html(
        str(soup),
        bullets="-",
        heading_style="ATX",
        strip=["canvas", "form", "input", "textarea"],
    )
    return clean_response_text(markdown)
