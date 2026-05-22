from src.utils.edge_launcher import _cdp_port


def test_cdp_port_defaults_to_9222() -> None:
    assert _cdp_port("http://127.0.0.1") == 9222


def test_cdp_port_reads_url_port() -> None:
    assert _cdp_port("http://127.0.0.1:9333") == 9333
