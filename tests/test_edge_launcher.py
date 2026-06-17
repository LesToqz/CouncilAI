from src.utils import edge_launcher
from src.utils.edge_launcher import _cdp_port, _ensure_single_app_tab, _same_app_origin


def test_cdp_port_defaults_to_9222() -> None:
    assert _cdp_port("http://127.0.0.1") == 9222


def test_cdp_port_reads_url_port() -> None:
    assert _cdp_port("http://127.0.0.1:9333") == 9333


def test_same_app_origin_matches_local_app_with_query() -> None:
    assert _same_app_origin("http://127.0.0.1:7860/?__theme=dark", "http://127.0.0.1:7860")


def test_same_app_origin_rejects_different_port() -> None:
    assert not _same_app_origin("http://127.0.0.1:7861", "http://127.0.0.1:7860")


def test_ensure_single_app_tab_opens_when_missing(monkeypatch) -> None:
    opened = []
    closed = []

    monkeypatch.setattr(edge_launcher, "_cdp_tabs", lambda cdp_url: [])
    monkeypatch.setattr(edge_launcher, "_open_cdp_tab", lambda cdp_url, url: opened.append(url) or True)
    monkeypatch.setattr(edge_launcher, "_close_cdp_tab", lambda cdp_url, target_id: closed.append(target_id))

    _ensure_single_app_tab("http://127.0.0.1:9222", "http://127.0.0.1:7860")

    assert opened == ["http://127.0.0.1:7860"]
    assert closed == []


def test_ensure_single_app_tab_closes_duplicate_app_tabs(monkeypatch) -> None:
    opened = []
    closed = []
    tabs = [
        {"id": "keep", "type": "page", "url": "http://127.0.0.1:7860"},
        {"id": "duplicate", "type": "page", "url": "http://127.0.0.1:7860/?__theme=dark"},
        {"id": "other", "type": "page", "url": "https://gemini.google.com"},
    ]

    monkeypatch.setattr(edge_launcher, "_cdp_tabs", lambda cdp_url: tabs)
    monkeypatch.setattr(edge_launcher, "_open_cdp_tab", lambda cdp_url, url: opened.append(url) or True)
    monkeypatch.setattr(edge_launcher, "_close_cdp_tab", lambda cdp_url, target_id: closed.append(target_id))

    _ensure_single_app_tab("http://127.0.0.1:9222", "http://127.0.0.1:7860")

    assert opened == []
    assert closed == ["duplicate"]
