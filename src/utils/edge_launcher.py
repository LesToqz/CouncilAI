from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class EdgeLaunchResult:
    available: bool
    launched: bool
    message: str


def is_cdp_available(cdp_url: str, timeout_seconds: float = 0.5) -> bool:
    try:
        with urlopen(f"{cdp_url.rstrip('/')}/json/version", timeout=timeout_seconds) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def launch_controlled_edge(settings: dict, app_url: str) -> EdgeLaunchResult:
    browser_settings = settings.get("browser", {})
    cdp_url = browser_settings.get("cdp_url", "http://127.0.0.1:9222")
    if is_cdp_available(cdp_url):
        _ensure_single_app_tab(cdp_url, app_url)
        return EdgeLaunchResult(True, False, f"Edge debugging is already available at {cdp_url}.")

    if not browser_settings.get("auto_launch_cdp", True):
        return EdgeLaunchResult(False, False, f"Edge debugging is not available at {cdp_url}.")

    edge_path = _find_edge_executable()
    if edge_path is None:
        return EdgeLaunchResult(False, False, "Could not find Microsoft Edge executable.")

    port = _cdp_port(cdp_url)
    profile_dir = Path(browser_settings.get("app_profile", "profiles/app_edge_profile"))
    profile_dir.mkdir(parents=True, exist_ok=True)

    args = [
        edge_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if is_cdp_available(cdp_url):
            _ensure_single_app_tab(cdp_url, app_url)
            return EdgeLaunchResult(True, True, f"Started controllable Edge at {cdp_url}.")
        time.sleep(0.25)

    return EdgeLaunchResult(
        False,
        True,
        "Started Edge, but the debugging endpoint did not become available. "
        "Close the CouncilAI Edge window and restart the app.",
    )


def _find_edge_executable() -> str | None:
    edge = which("msedge")
    if edge:
        return edge

    candidates = [
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _cdp_port(cdp_url: str) -> int:
    parsed = urlparse(cdp_url)
    if parsed.port:
        return parsed.port
    return 9222


def _ensure_single_app_tab(cdp_url: str, app_url: str) -> None:
    tabs = _cdp_tabs(cdp_url)
    app_tabs = [tab for tab in tabs if _is_app_tab(tab, app_url)]

    if app_tabs:
        for duplicate in app_tabs[1:]:
            target_id = duplicate.get("id")
            if target_id:
                _close_cdp_tab(cdp_url, str(target_id))
        return

    _open_cdp_tab(cdp_url, app_url)


def _cdp_tabs(cdp_url: str) -> list[dict]:
    try:
        with urlopen(f"{cdp_url.rstrip('/')}/json/list", timeout=1) as response:
            payload = response.read().decode("utf-8")
        tabs = json.loads(payload)
        return tabs if isinstance(tabs, list) else []
    except Exception:
        return []


def _open_cdp_tab(cdp_url: str, url: str) -> bool:
    endpoint = f"{cdp_url.rstrip('/')}/json/new?{quote(url, safe='')}"
    try:
        request = Request(endpoint, method="PUT")
        with urlopen(request, timeout=1) as response:
            return 200 <= response.status < 300
    except Exception:
        try:
            with urlopen(endpoint, timeout=1) as response:
                return 200 <= response.status < 300
        except Exception:
            return False


def _close_cdp_tab(cdp_url: str, target_id: str) -> None:
    try:
        with urlopen(f"{cdp_url.rstrip('/')}/json/close/{target_id}", timeout=1):
            return
    except Exception:
        return


def _is_app_tab(tab: dict, app_url: str) -> bool:
    if tab.get("type") != "page":
        return False
    return _same_app_origin(tab.get("url") or "", app_url)


def _same_app_origin(candidate_url: str, app_url: str) -> bool:
    candidate = urlparse(candidate_url)
    target = urlparse(app_url)
    if candidate.scheme not in {"http", "https"}:
        return False
    return candidate.scheme == target.scheme and candidate.netloc == target.netloc
