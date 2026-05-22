from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from urllib.parse import urlparse
from urllib.request import urlopen


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
        app_url,
    ]
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if is_cdp_available(cdp_url):
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
