from src.utils.text_cleaner import clean_response_html, clean_response_text


def test_clean_response_text_removes_ui_artifacts() -> None:
    assert clean_response_text("Answer\nCopy\nShare") == "Answer"


def test_clean_response_text_preserves_code_indentation() -> None:
    text = "```python\nif True:\n    print('ok')\n```"

    assert "    print('ok')" in clean_response_text(text)


def test_clean_response_text_removes_internal_prompt_and_images() -> None:
    text = "You are participating in a multi-model debate.\n\n![hot state](x.png)\nAnswer"

    assert clean_response_text(text) == "Answer"


def test_clean_response_html_preserves_tables_as_markdown() -> None:
    markup = """
    <div>
      <h2>Result</h2>
      <table>
        <thead><tr><th>Meaning</th><th>Answer</th></tr></thead>
        <tbody><tr><td>Highest temperature</td><td>Rajasthan</td></tr></tbody>
      </table>
      <img src="x.png" alt="remove me" />
    </div>
    """

    cleaned = clean_response_html(markup)

    assert "## Result" in cleaned
    assert "| Meaning | Answer |" in cleaned
    assert "Rajasthan" in cleaned
    assert "remove me" not in cleaned
