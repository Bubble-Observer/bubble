from __future__ import annotations

from collections.abc import Sequence

import pytest

from leave_information_bubble.security import (
    UrlPolicyConfig,
    UrlPolicyViolation,
    UrlSafetyPolicy,
    validate_url,
)


class Resolver:
    def __init__(self, answers: dict[str, Sequence[str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> Sequence[str]:
        self.calls.append((host, port))
        return self.answers.get(host, ())


def test_validate_public_url_normalizes_and_resolves() -> None:
    resolver = Resolver({"example.com": ("93.184.216.34", "93.184.216.34")})

    target = validate_url("HTTPS://Example.COM:443/path?q=1#fragment", resolver=resolver)

    assert target.normalized_url == "https://example.com/path?q=1"
    assert target.resolved_addresses == ("93.184.216.34",)
    assert resolver.calls == [("example.com", 443)]


def test_public_ip_literal_does_not_call_resolver() -> None:
    resolver = Resolver({})

    target = UrlSafetyPolicy(resolver=resolver).validate_url("https://[2606:4700:4700::1111]/")

    assert target.host == "2606:4700:4700::1111"
    assert target.resolved_addresses == ("2606:4700:4700::1111",)
    assert resolver.calls == []


@pytest.mark.parametrize(
    ("url", "answers", "code"),
    [
        ("ftp://example.com/a", {}, "scheme_not_allowed"),
        ("https://user:secret@example.com/a", {}, "credentials_not_allowed"),
        ("https://localhost/a", {}, "local_host"),
        ("https://api.localhost/a", {}, "local_host"),
        ("https://example.com:99999/a", {}, "invalid_port"),
        ("https://example.com/a\nx", {}, "control_character"),
        ("https:///missing", {}, "host_required"),
        ("https://example.com/a", {"example.com": ()}, "resolution_empty"),
    ],
)
def test_invalid_url_shapes_are_rejected(
    url: str,
    answers: dict[str, Sequence[str]],
    code: str,
) -> None:
    with pytest.raises(UrlPolicyViolation) as caught:
        UrlSafetyPolicy(resolver=Resolver(answers)).validate_url(url)

    assert caught.value.code == code


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::",
    ],
)
def test_all_non_public_resolved_addresses_are_rejected(address: str) -> None:
    resolver = Resolver({"attacker.example": ("93.184.216.34", address)})

    with pytest.raises(UrlPolicyViolation) as caught:
        UrlSafetyPolicy(resolver=resolver).validate_url("https://attacker.example/data")

    assert caught.value.code == "non_public_address"


def test_invalid_resolver_address_is_rejected() -> None:
    resolver = Resolver({"example.com": ("not-an-ip",)})

    with pytest.raises(UrlPolicyViolation) as caught:
        UrlSafetyPolicy(resolver=resolver).validate_url("https://example.com")

    assert caught.value.code == "invalid_resolved_address"


def test_redirect_target_is_resolved_and_validated_again() -> None:
    resolver = Resolver(
        {
            "public.example": ("93.184.216.34",),
            "redirect.example": ("127.0.0.1",),
        }
    )
    policy = UrlSafetyPolicy(resolver=resolver)
    original = policy.validate_url("https://public.example/start")

    with pytest.raises(UrlPolicyViolation) as caught:
        policy.validate_redirect(original, "https://redirect.example/admin")

    assert caught.value.code == "non_public_address"
    assert resolver.calls == [
        ("public.example", 443),
        ("redirect.example", 443),
    ]


def test_relative_redirect_and_https_downgrade_policy() -> None:
    resolver = Resolver({"public.example": ("93.184.216.34",)})
    policy = UrlSafetyPolicy(resolver=resolver)
    original = policy.validate_url("https://public.example/start")

    relative = policy.validate_redirect(original, "/next")
    assert relative.normalized_url == "https://public.example/next"

    with pytest.raises(UrlPolicyViolation) as caught:
        policy.validate_redirect(original, "http://public.example/plain")
    assert caught.value.code == "https_downgrade"

    permissive = UrlSafetyPolicy(
        resolver=resolver,
        config=UrlPolicyConfig(allow_https_downgrade=True),
    )
    downgraded = permissive.validate_redirect(original, "http://public.example/plain")
    assert downgraded.scheme == "http"


def test_transport_address_is_rechecked_and_pinned() -> None:
    resolver = Resolver({"public.example": ("93.184.216.34",)})
    policy = UrlSafetyPolicy(resolver=resolver)
    target = policy.validate_url("https://public.example")

    assert policy.validate_connected_address(target, "93.184.216.34") == "93.184.216.34"

    with pytest.raises(UrlPolicyViolation) as rebound:
        policy.validate_connected_address(target, "1.1.1.1")
    assert rebound.value.code == "dns_answer_changed"

    with pytest.raises(UrlPolicyViolation) as private:
        policy.validate_resolved_addresses(target, ("127.0.0.1",))
    assert private.value.code == "non_public_address"


def test_reresolution_can_be_public_without_pinning_or_strict_with_pinning() -> None:
    policy = UrlSafetyPolicy(resolver=Resolver({"public.example": ("93.184.216.34",)}))
    target = policy.validate_url("https://public.example")

    assert policy.validate_resolved_addresses(target, ("1.1.1.1",)) == ("1.1.1.1",)

    with pytest.raises(UrlPolicyViolation) as caught:
        policy.validate_resolved_addresses(
            target,
            ("1.1.1.1",),
            require_prevalidated=True,
        )
    assert caught.value.code == "dns_answer_changed"
