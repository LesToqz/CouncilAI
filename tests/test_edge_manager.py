from src.browser.edge_manager import _page_matches_target


def test_page_matches_exact_host() -> None:
    assert _page_matches_target("https://chatgpt.com/c/123", "chatgpt.com")


def test_page_does_not_match_wrong_host() -> None:
    assert not _page_matches_target("https://example.com/", "chatgpt.com")
