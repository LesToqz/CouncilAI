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
_MODEL_PREFIX_LINE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:gemini\s+(?:said|says)|claude\s+responded)(?:\*\*)?\s*:?\s*$",
    re.IGNORECASE,
)
_MODEL_PREFIX_INLINE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:gemini\s+(?:said|says)|claude\s+responded)(?:\*\*)?\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


def clean_response_text(text: str) -> str:
    lines = []
    normalized = _MARKDOWN_IMAGE_PATTERN.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    for line in normalized.split("\n"):
        stripped = line.strip()
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in _ARTIFACT_PATTERNS):
            continue
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in _INTERNAL_PROMPT_PATTERNS):
            continue
        if _MODEL_PREFIX_LINE_PATTERN.match(stripped):
            continue
        inline_prefix = _MODEL_PREFIX_INLINE_PATTERN.match(line)
        if inline_prefix:
            lines.append(inline_prefix.group(1).rstrip())
            continue
        lines.append(line.rstrip())

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _remove_duplicate_leading_title(cleaned)


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


def _remove_duplicate_leading_title(text: str) -> str:
    lines = text.splitlines()
    non_empty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    if len(non_empty_indexes) < 2:
        return text

    first_index, second_index = non_empty_indexes[:2]
    first = lines[first_index]
    second = lines[second_index]
    if _normalized_title_line(first) != _normalized_title_line(second):
        return text

    first_score = _title_format_score(first)
    second_score = _title_format_score(second)
    remove_index = first_index if second_score > first_score else second_index
    del lines[remove_index]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _normalized_title_line(line: str) -> str:
    normalized = line.strip()
    normalized = re.sub(r"^#{1,6}\s+", "", normalized)
    normalized = re.sub(r"^\*\*(.+?)\*\*$", r"\1", normalized)
    normalized = re.sub(r"^__(.+?)__$", r"\1", normalized)
    return " ".join(normalized.split()).casefold()


def _title_format_score(line: str) -> int:
    stripped = line.strip()
    if re.match(r"^#{1,6}\s+", stripped):
        return 3
    if re.match(r"^(?:\*\*|__).+(?:\*\*|__)$", stripped):
        return 2
    return 1
