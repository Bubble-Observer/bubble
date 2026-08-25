"""Network target validation for public-document acquisition."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit


class AddressResolver(Protocol):
    """Resolve a host without coupling the policy to a specific HTTP client."""

    def __call__(self, host: str, port: int) -> Sequence[str]:
        """Return every address the fetch implementation may connect to."""


class UrlPolicyViolation(ValueError):
    """A stable, non-retryable URL or network-target policy rejection."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class UrlPolicyConfig:
    """Explicit URL policy limits."""

    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    max_url_length: int = 4096
    allow_https_downgrade: bool = False


@dataclass(frozen=True, slots=True)
class ValidatedUrlTarget:
    """A URL plus the public addresses approved immediately before fetching."""

    normalized_url: str
    scheme: str
    host: str
    port: int
    resolved_addresses: tuple[str, ...]


def system_resolver(host: str, port: int) -> tuple[str, ...]:
    """Resolve a target with the operating-system resolver."""
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UrlPolicyViolation("resolution_failed", "the target host could not be resolved") from exc
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


class UrlSafetyPolicy:
    """Reject non-public network targets before and during an HTTP fetch."""

    def __init__(
        self,
        *,
        resolver: AddressResolver = system_resolver,
        config: UrlPolicyConfig | None = None,
    ) -> None:
        self._resolver = resolver
        self._config = config or UrlPolicyConfig()

    def validate_url(self, url: str) -> ValidatedUrlTarget:
        """Parse, resolve, and approve one HTTP(S) target."""
        parsed = self._parse(url)
        host = self._normalize_host(parsed.hostname or "")
        port = self._port(parsed)
        self._reject_local_hostname(host)

        literal = self._parse_ip_literal(host)
        raw_addresses = (host,) if literal is not None else tuple(self._resolver(host, port))
        addresses = self._validate_addresses(raw_addresses)
        if not addresses:
            raise UrlPolicyViolation("resolution_empty", "the target host resolved to no addresses")

        return ValidatedUrlTarget(
            normalized_url=self._normalize_url(parsed, host, port),
            scheme=parsed.scheme.lower(),
            host=host,
            port=port,
            resolved_addresses=addresses,
        )

    def validate_redirect(
        self,
        previous: ValidatedUrlTarget,
        location: str,
    ) -> ValidatedUrlTarget:
        """Resolve and validate every redirect target independently."""
        target_url = urljoin(previous.normalized_url, location)
        target = self.validate_url(target_url)
        if (
            previous.scheme == "https"
            and target.scheme == "http"
            and not self._config.allow_https_downgrade
        ):
            raise UrlPolicyViolation(
                "https_downgrade",
                "an HTTPS fetch cannot redirect to plaintext HTTP",
            )
        return target

    def validate_resolved_addresses(
        self,
        target: ValidatedUrlTarget,
        addresses: Sequence[str],
        *,
        require_prevalidated: bool = False,
    ) -> tuple[str, ...]:
        """Recheck transport-resolved addresses immediately before connection.

        Fetchers that can pin an address should set ``require_prevalidated`` or
        call :meth:`validate_connected_address`. Fetchers that re-resolve after
        policy validation must at minimum pass the new set through this method.
        """
        validated = self._validate_addresses(addresses)
        if not validated:
            raise UrlPolicyViolation("resolution_empty", "the target host resolved to no addresses")
        if require_prevalidated and not set(validated).issubset(target.resolved_addresses):
            raise UrlPolicyViolation(
                "dns_answer_changed",
                "the transport selected an address that was not prevalidated",
            )
        return validated

    def validate_connected_address(self, target: ValidatedUrlTarget, address: str) -> str:
        """Approve the actual peer address and enforce DNS-answer pinning."""
        return self.validate_resolved_addresses(
            target,
            (address,),
            require_prevalidated=True,
        )[0]

    def revalidate_target(
        self,
        target: ValidatedUrlTarget,
        *,
        require_prevalidated: bool = True,
    ) -> tuple[str, ...]:
        """Resolve again immediately before transport connection."""
        addresses = tuple(self._resolver(target.host, target.port))
        return self.validate_resolved_addresses(
            target,
            addresses,
            require_prevalidated=require_prevalidated,
        )

    def _parse(self, url: str) -> SplitResult:
        if not url or len(url) > self._config.max_url_length:
            raise UrlPolicyViolation("invalid_length", "URL length is outside the configured limit")
        if any(ord(character) <= 32 or ord(character) == 127 for character in url):
            raise UrlPolicyViolation("control_character", "URL contains whitespace or control characters")
        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise UrlPolicyViolation("malformed_url", "URL parsing failed") from exc
        scheme = parsed.scheme.lower()
        if scheme not in self._config.allowed_schemes:
            raise UrlPolicyViolation("scheme_not_allowed", "only configured HTTP schemes are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise UrlPolicyViolation("credentials_not_allowed", "credentials in URLs are forbidden")
        if not parsed.hostname:
            raise UrlPolicyViolation("host_required", "URL must include a host")
        return parsed

    @staticmethod
    def _port(parsed: SplitResult) -> int:
        try:
            explicit_port = parsed.port
        except ValueError as exc:
            raise UrlPolicyViolation("invalid_port", "URL port is invalid") from exc
        return explicit_port or (443 if parsed.scheme.lower() == "https" else 80)

    @staticmethod
    def _normalize_host(host: str) -> str:
        try:
            return host.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise UrlPolicyViolation("invalid_host", "URL host cannot be normalized") from exc

    @staticmethod
    def _reject_local_hostname(host: str) -> None:
        if host == "localhost" or host.endswith(".localhost"):
            raise UrlPolicyViolation("local_host", "localhost targets are forbidden")

    @staticmethod
    def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        try:
            return ipaddress.ip_address(host)
        except ValueError:
            return None

    @classmethod
    def _validate_addresses(cls, addresses: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw_address in addresses:
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError as exc:
                raise UrlPolicyViolation(
                    "invalid_resolved_address",
                    "the resolver returned a non-IP address",
                ) from exc
            if not address.is_global or address.is_multicast:
                raise UrlPolicyViolation(
                    "non_public_address",
                    "local, private, reserved, and otherwise non-public addresses are forbidden",
                )
            normalized.append(address.compressed)
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def _normalize_url(parsed: SplitResult, host: str, port: int) -> str:
        scheme = parsed.scheme.lower()
        default_port = 443 if scheme == "https" else 80
        rendered_host = f"[{host}]" if ":" in host else host
        netloc = rendered_host if port == default_port else f"{rendered_host}:{port}"
        return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def validate_url(
    url: str,
    *,
    resolver: AddressResolver = system_resolver,
    config: UrlPolicyConfig | None = None,
) -> ValidatedUrlTarget:
    """Validate a URL through a convenience entry point for fetchers."""
    return UrlSafetyPolicy(resolver=resolver, config=config).validate_url(url)


__all__ = [
    "AddressResolver",
    "UrlPolicyConfig",
    "UrlPolicyViolation",
    "UrlSafetyPolicy",
    "ValidatedUrlTarget",
    "system_resolver",
    "validate_url",
]
