import asyncio

from src.browser.edge_manager import EdgeManager
from src.browser.edge_manager import _page_matches_target


def test_page_matches_exact_host() -> None:
    assert _page_matches_target("https://chatgpt.com/c/123", "chatgpt.com")


def test_page_does_not_match_wrong_host() -> None:
    assert not _page_matches_target("https://example.com/", "chatgpt.com")


def test_attach_to_existing_page_does_not_force_tab_to_front() -> None:
    class FakePage:
        url = "https://chatgpt.com/c/123"

        def __init__(self) -> None:
            self.brought_to_front = False

        async def bring_to_front(self) -> None:
            self.brought_to_front = True

    class FakeContext:
        def __init__(self, page: FakePage) -> None:
            self.pages = [page]

    class FakeBrowser:
        def __init__(self, context: FakeContext) -> None:
            self.contexts = [context]

    async def run() -> None:
        page = FakePage()
        context = FakeContext(page)
        manager = EdgeManager({"browser": {}, "model_sites": {}})

        async def connect_existing_browser() -> FakeBrowser:
            return FakeBrowser(context)

        manager.connect_existing_browser = connect_existing_browser  # type: ignore[method-assign]
        attached_context, attached_page = await manager.attach_to_existing_page("chatgpt", "https://chatgpt.com")

        assert attached_context is context
        assert attached_page is page
        assert not page.brought_to_front

    asyncio.run(run())


def test_open_url_in_background_uses_cdp_background_target_and_refocuses_app() -> None:
    class FakePage:
        def __init__(self, url: str) -> None:
            self.url = url
            self.brought_to_front = False

        async def bring_to_front(self) -> None:
            self.brought_to_front = True

    class FakeContext:
        def __init__(self, pages: list[FakePage]) -> None:
            self.pages = pages

    class FakeSession:
        def __init__(self, context: FakeContext) -> None:
            self.context = context
            self.params = None
            self.detached = False

        async def send(self, method: str, params: dict) -> None:
            self.params = (method, params)
            self.context.pages.append(FakePage(params["url"]))

        async def detach(self) -> None:
            self.detached = True

    class FakeBrowser:
        def __init__(self, context: FakeContext, session: FakeSession) -> None:
            self.contexts = [context]
            self.session = session

        async def new_browser_cdp_session(self) -> FakeSession:
            return self.session

    async def run() -> None:
        app_page = FakePage("http://127.0.0.1:7860")
        context = FakeContext([app_page])
        session = FakeSession(context)
        browser = FakeBrowser(context, session)
        manager = EdgeManager({"browser": {}, "model_sites": {}})

        async def connect_existing_browser() -> FakeBrowser:
            manager.browser = browser  # type: ignore[assignment]
            return browser

        manager.connect_existing_browser = connect_existing_browser  # type: ignore[method-assign]
        page = await manager.open_url_in_background("https://claude.ai/new", "http://127.0.0.1:7860")

        assert page is context.pages[-1]
        assert session.params == (
            "Target.createTarget",
            {"url": "https://claude.ai/new", "background": True},
        )
        assert session.detached
        assert app_page.brought_to_front

    asyncio.run(run())
