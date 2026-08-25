from collections.abc import Sequence

import httpx
import pytest

from leave_information_bubble.security import UrlSafetyPolicy
from leave_information_bubble.tools.web_search import PublicWebSearchTool, WebDocumentOutcome


class Resolver:
    def __init__(self, answers: dict[str, Sequence[str]]) -> None:
        self.answers = answers

    def __call__(self, host: str, port: int) -> Sequence[str]:
        del port
        return self.answers.get(host, ())


@pytest.mark.asyncio
async def test_fetch_rejects_unsafe_initial_target_without_transport_call() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request)

    tool = PublicWebSearchTool(
        url_policy=UrlSafetyPolicy(resolver=Resolver({})),
        transport=httpx.MockTransport(handler),
    )

    document = await tool.fetch("http://127.0.0.1/admin")

    assert document.error == "url rejected: non_public_address"
    assert calls == 0


@pytest.mark.asyncio
async def test_fetch_validates_every_redirect_before_following() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "http://127.0.0.1/admin"},
            request=request,
        )

    policy = UrlSafetyPolicy(resolver=Resolver({"public.example": ("93.184.216.34",)}))
    tool = PublicWebSearchTool(
        url_policy=policy,
        transport=httpx.MockTransport(handler),
    )

    document = await tool.fetch("https://public.example/start")

    assert document.error == "url rejected: non_public_address"
    assert calls == ["https://public.example/start"]


@pytest.mark.asyncio
async def test_fetch_extracts_safe_public_document() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><title>Public</title><body>safe text</body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    policy = UrlSafetyPolicy(resolver=Resolver({"public.example": ("93.184.216.34",)}))
    tool = PublicWebSearchTool(
        url_policy=policy,
        transport=httpx.MockTransport(handler),
    )

    document = await tool.fetch("https://public.example/read")

    assert document.error == ""
    assert document.title == "Public"
    assert document.text == "safe text"
    assert document.outcome is WebDocumentOutcome.QUALIFIED_FULL
    assert document.limitations == ()
