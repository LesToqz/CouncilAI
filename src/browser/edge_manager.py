from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright
except ImportError:  # pragma: no cover - handled at runtime with a clearer error.
    Browser = Any  # type: ignore
    BrowserContext = Any  # type: ignore
    Page = Any  # type: ignore
    async_playwright = None  # type: ignore


class EdgeManager:
    def __init__(self, settings: dict) -> None:
        self.settings = settings
        self.playwright = None
        self.browser: Browser | None = None
        self.contexts: dict[str, BrowserContext] = {}
        self.connected_over_cdp = False

    async def start(self) -> None:
        if async_playwright is None:
            raise RuntimeError("Playwright is not installed. Run: pip install -r requirements.txt")
        if self.playwright is None:
            self.playwright = await async_playwright().start()

    async def launch_context(self, model_key: str, profile_path: str) -> BrowserContext:
        await self.start()
        if model_key in self.contexts:
            return self.contexts[model_key]

        Path(profile_path).mkdir(parents=True, exist_ok=True)
        browser_settings = self.settings.get("browser", {})
        context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=profile_path,
            channel=browser_settings.get("channel", "msedge"),
            headless=bool(browser_settings.get("headless", False)),
            slow_mo=int(browser_settings.get("slow_mo_ms", 100)),
            viewport={"width": 1400, "height": 900},
        )
        context.set_default_timeout(int(browser_settings.get("timeout_ms", 120000)))
        self.contexts[model_key] = context
        return context

    async def connect_existing_browser(self) -> Browser:
        await self.start()
        if self.browser is not None:
            return self.browser

        browser_settings = self.settings.get("browser", {})
        cdp_url = browser_settings.get("cdp_url", "http://127.0.0.1:9222")
        try:
            self.browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not attach to Microsoft Edge. Start Edge with "
                f"--remote-debugging-port=9222, then open the AI tabs. Details: {exc}"
            ) from exc

        self.connected_over_cdp = True
        timeout = int(browser_settings.get("timeout_ms", 120000))
        for context in self.browser.contexts:
            context.set_default_timeout(timeout)
        return self.browser

    async def attach_to_existing_page(self, model_key: str, url: str) -> tuple[BrowserContext, Page]:
        browser = await self.connect_existing_browser()
        target_host = _normalized_host(url)

        for context in browser.contexts:
            for page in context.pages:
                if _page_matches_target(page.url, target_host):
                    self.contexts[model_key] = context
                    try:
                        await page.bring_to_front()
                    except Exception:
                        pass
                    return context, page

        if self.settings.get("browser", {}).get("open_missing_tabs", False):
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            self.contexts[model_key] = context
            return context, page

        raise RuntimeError(
            f"No open Edge tab found for {model_key} ({url}). "
            "Open that site in the same Edge window, then run the debate again."
        )

    async def find_existing_ai_tabs(self) -> dict[str, str | None]:
        browser = await self.connect_existing_browser()
        sites = self.settings.get("model_sites", {})
        found: dict[str, str | None] = {}

        for model_key, site in sites.items():
            target_host = _normalized_host(site["url"])
            found[model_key] = None
            for context in browser.contexts:
                for page in context.pages:
                    if _page_matches_target(page.url, target_host):
                        found[model_key] = page.url
                        break
                if found[model_key]:
                    break

        return found

    async def open_model_page(self, model_key: str, url: str) -> Page:
        context = self.contexts[model_key]
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        return page

    async def stop(self) -> None:
        if not self.connected_over_cdp:
            for context in list(self.contexts.values()):
                try:
                    await context.close()
                except Exception:
                    pass
        self.contexts.clear()
        if self.playwright:
            try:
                await self.playwright.stop()
            finally:
                self.playwright = None
                self.browser = None
                self.connected_over_cdp = False


def _normalized_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _page_matches_target(page_url: str, target_host: str) -> bool:
    if not page_url or page_url == "about:blank":
        return False
    page_host = _normalized_host(page_url)
    return page_host == target_host or page_host.endswith(f".{target_host}")
