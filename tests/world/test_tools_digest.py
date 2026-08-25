# Split from tests/world/test_tools.py (audit 2026-08-18, baseline e50bce4).
# Behavior-neutral move; assertions and case names unchanged.

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from leave_information_bubble.world import (
    CognitiveDelta,
    ObservationDepth,
    ObservationInput,
    WorldStore,
)
from leave_information_bubble.world.tools import WorldTools, digest_material_id


def _digest_seed(store: WorldStore) -> None:
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-1",
                    source_uri="https://bbs.hupu.com/1",
                    source_kind="hupu",
                    title="t",
                    excerpt="A returned preview excerpt that is eligible for C2 digestion.",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-1",
    )


_DIGEST_EXCERPT = "A returned preview excerpt that is eligible for C2 digestion."


def _digest_material_context(
    observation_id: str = "hupu:obs-1",
    *,
    material_kind: str = "excerpt",
    text: str = _DIGEST_EXCERPT,
    allowed_evidence_refs: list[str] | None = None,
) -> tuple[str, dict[str, dict[str, object]]]:
    """Build one exact Digest v2 material context for facade tests."""
    material_id = digest_material_id(observation_id, material_kind, text)
    return material_id, {
        material_id: {
            "observation_id": observation_id,
            "material_kind": material_kind,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "allowed_evidence_refs": allowed_evidence_refs or [observation_id],
        }
    }


def _digest_arguments(material_id: str, **overrides: object) -> dict[str, object]:
    """Return a complete Digest v2 invocation payload for concise tests."""
    return {
        "material_id": material_id,
        "points": ["p"],
        "assertion_candidates": [],
        "evidence_refs": [],
        **overrides,
    }


async def test_digest_observation_validates_and_echoes(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a digest that fabricates ids instead of echoing the verified ones back."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()

    result = await tools.execute(
        "digest_observation",
        {
            "material_id": material_id,
            "points": ["BLG 官号 23:48 发动态被解读为暗讽 Bin"],
            "assertion_candidates": [
                {
                    "predicate": "community_reaction",
                    "literal": "社区解读为暗讽",
                    "epistemic_role": "community_view",
                    "confidence": 0.8,
                }
            ],
            "evidence_refs": ["hupu:obs-1"],
        },
        "c1",
    )

    assert result["ok"] is True
    assert result["material_id"] == material_id
    assert result["observation_id"] == "hupu:obs-1"
    assert result["material_kind"] == "excerpt"
    assert result["content_hash"] == hashlib.sha256(_DIGEST_EXCERPT.encode("utf-8")).hexdigest()
    assert result["already_digested"] is False
    assert result["digest"]["points"] == ["BLG 官号 23:48 发动态被解读为暗讽 Bin"]
    assert result["digest"]["assertion_candidates"] == [
        {
            "predicate": "community_reaction",
            "literal": "社区解读为暗讽",
            "epistemic_role": "community_view",
            "confidence": 0.8,
        }
    ]
    assert result["digest"]["evidence_refs"] == ["hupu:obs-1"]


async def test_digest_observation_rejects_unknown_ids(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a digest accepted against an observation id that never entered the store."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute("digest_observation", _digest_arguments("nope", points=["x"]), "c2")

    assert result == {
        "ok": False,
        "limitations": ["unknown_material_id"],
        "error": {"code": "unknown_material_id", "field": "material_id"},
    }


@pytest.mark.parametrize(
    ("case", "expected_code", "expected_field", "expected_ids"),
    [
        ("observation_id", "digest_material_id_required", "material_id", []),
        ("stale_hash", "stale_material_hash", "material_id", []),
        ("unknown_evidence", "unknown_evidence_id", "evidence_refs", ["missing:obs-9"]),
        ("cross_material", "evidence_ref_outside_material", "evidence_refs", ["hupu:obs-2"]),
    ],
)
async def test_digest_observation_reports_exact_bounded_failure(
    tmp_path: pytest.TempPathFactory,
    case: str,
    expected_code: str,
    expected_field: str,
    expected_ids: list[str],
) -> None:
    """Digest failures identify the rejected field without returning source text."""
    store = WorldStore(tmp_path / f"{case}.sqlite3")
    _digest_seed(store)
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-2",
                    source_uri="https://bbs.hupu.com/2",
                    source_kind="hupu",
                    title="other",
                    excerpt="A different stored observation.",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        f"seed-{case}",
    )
    tools = WorldTools(store=store, adapters={})
    material_id, context = _digest_material_context()
    arguments = _digest_arguments(material_id)
    if case == "observation_id":
        arguments["material_id"] = "hupu:obs-1"
    elif case == "stale_hash":
        context[material_id]["content_hash"] = "stale"
    elif case == "unknown_evidence":
        context[material_id]["allowed_evidence_refs"] = ["hupu:obs-1", "missing:obs-9"]
        arguments["evidence_refs"] = ["missing:obs-9"]
    else:
        arguments["evidence_refs"] = ["hupu:obs-2"]
    tools.set_digest_context(context)

    result = await tools.execute("digest_observation", arguments, f"digest-{case}")

    assert result["ok"] is False
    assert result["limitations"] == [expected_code]
    assert result["error"]["code"] == expected_code
    assert result["error"]["field"] == expected_field
    assert result["error"].get("rejected_ids", []) == expected_ids
    if expected_ids:
        assert result["error"]["rejected_count"] == len(expected_ids)
        assert result["error"]["rejected_ids_truncated"] is False
    assert _DIGEST_EXCERPT not in json.dumps(result, ensure_ascii=False)


async def test_digest_failure_bounds_rejected_evidence_ids(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A hostile evidence list exposes its size without returning an unbounded diagnostic."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, context = _digest_material_context()
    tools.set_digest_context(context)

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(material_id, evidence_refs=[f"other:{index}" for index in range(20)]),
        "many-outside-refs",
    )

    assert result["error"]["code"] == "evidence_ref_outside_material"
    assert len(result["error"]["rejected_ids"]) == 8
    assert result["error"]["rejected_count"] == 20
    assert result["error"]["rejected_ids_truncated"] is True


async def test_digest_observation_rejects_primary_observation_id_without_side_effects(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A primary id is never an alias for the returned card's material id."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    digested_ids: set[str] = set()
    tools = WorldTools(store=store, adapters={}, digested_ids=digested_ids)
    material_id, context = _digest_material_context()
    tools.set_digest_context(context)
    with store.read_connection() as connection:
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("observations", "observation_bodies", "world_audit")
        }

    result = await tools.execute(
        "digest_observation",
        _digest_arguments("hupu:obs-1", evidence_refs=["hupu:obs-1"]),
        "primary-id-is-not-material-id",
    )

    with store.read_connection() as connection:
        after = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in before}
    assert result == {
        "ok": False,
        "limitations": ["digest_material_id_required"],
        "error": {"code": "digest_material_id_required", "field": "material_id"},
    }
    assert material_id not in digested_ids
    assert before == after


async def test_digest_observation_rejects_empty_navigation_echo(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """An empty stored card is navigation metadata, not digestible returned text."""
    store = WorldStore(tmp_path / "world.sqlite3")
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:empty-card",
                    source_uri="https://bbs.hupu.com/empty",
                    source_kind="hupu",
                    title="empty",
                    depth=ObservationDepth.SEEN,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "1"},
                )
            ]
        ),
        "seed-empty-digest",
    )
    tools = WorldTools(store=store, adapters={})

    result = await tools.execute(
        "digest_observation",
        _digest_arguments("empty-material", points=["unsupported"]),
        "empty-digest",
    )

    assert result == {
        "ok": False,
        "limitations": ["unknown_material_id"],
        "error": {"code": "unknown_material_id", "field": "material_id"},
    }


async def test_digest_observation_rejects_unknown_evidence_refs(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Catch a digest citing evidence ids that do not exist in the store."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(material_id, evidence_refs=["hupu:obs-1", "missing:obs-9"]),
        "c3",
    )

    assert result["ok"] is False
    assert result["limitations"] == ["evidence_ref_outside_material"]
    assert result["error"] == {
        "code": "evidence_ref_outside_material",
        "field": "evidence_refs",
        "rejected_ids": ["missing:obs-9"],
        "rejected_count": 1,
        "rejected_ids_truncated": False,
    }


async def test_digest_observation_rejects_existing_cross_material_evidence_ref(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A real stored observation outside this material is still forbidden evidence."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    store.memory_commit(
        CognitiveDelta(
            observations=[
                ObservationInput(
                    id="hupu:obs-2",
                    source_uri="https://bbs.hupu.com/2",
                    source_kind="hupu",
                    title="other",
                    excerpt="A different stored observation.",
                    depth=ObservationDepth.CONTENT,
                    observed_at=datetime(2026, 8, 5, tzinfo=UTC),
                    metadata={"adapter_version": "v1"},
                )
            ]
        ),
        "seed-2",
    )
    tools = WorldTools(store=store, adapters={})
    material_id, context = _digest_material_context()
    tools.set_digest_context(context)

    result = await tools.execute(
        "digest_observation", _digest_arguments(material_id, evidence_refs=["hupu:obs-2"]), "cross-real"
    )

    assert result == {
        "ok": False,
        "limitations": ["evidence_ref_outside_material"],
        "error": {
            "code": "evidence_ref_outside_material",
            "field": "evidence_refs",
            "rejected_ids": ["hupu:obs-2"],
            "rejected_count": 1,
            "rejected_ids_truncated": False,
        },
    }


async def test_digest_observation_requires_returned_evidence_context(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Stored ids outside this wake cannot be reused as digest evidence."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, context = _digest_material_context()
    tools.set_digest_context(context)

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(material_id, evidence_refs=["hupu:obs-1", "missing:obs-9"]),
        "c3-context",
    )

    assert result == {
        "ok": False,
        "limitations": ["evidence_ref_outside_material"],
        "error": {
            "code": "evidence_ref_outside_material",
            "field": "evidence_refs",
            "rejected_ids": ["missing:obs-9"],
            "rejected_count": 1,
            "rejected_ids_truncated": False,
        },
    }


async def test_digest_material_context_fails_closed_for_hash_and_material_mismatches(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Digest v2 never accepts a stale hash or a material bound to another source."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, context = _digest_material_context()

    context[material_id]["content_hash"] = "wrong-hash"
    tools.set_digest_context(context)
    bad_hash = await tools.execute("digest_observation", _digest_arguments(material_id), "bad-hash")
    assert bad_hash == {
        "ok": False,
        "limitations": ["stale_material_hash"],
        "error": {"code": "stale_material_hash", "field": "material_id"},
    }

    material_id, context = _digest_material_context()
    context[material_id]["observation_id"] = "hupu:other-observation"
    tools.set_digest_context(context)
    cross_observation = await tools.execute(
        "digest_observation", _digest_arguments(material_id), "cross-observation"
    )
    assert cross_observation == {
        "ok": False,
        "limitations": ["digest_material_identity_mismatch"],
        "error": {"code": "digest_material_identity_mismatch", "field": "material_id"},
    }

    tools.set_digest_context({})
    cross_material = await tools.execute(
        "digest_observation", _digest_arguments(material_id), "cross-material"
    )
    assert cross_material == {
        "ok": False,
        "limitations": ["unknown_material_id"],
        "error": {"code": "unknown_material_id", "field": "material_id"},
    }


async def test_digest_context_none_restores_standalone_exact_material_lookup(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Clearing graph-local context cannot leak it or disable direct facade use."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, context = _digest_material_context()
    tools.set_digest_context(context)
    tools.set_digest_context(None)

    result = await tools.execute(
        "digest_observation", _digest_arguments(material_id, evidence_refs=["hupu:obs-1"]), "standalone"
    )

    assert result["ok"] is True
    assert result["material_id"] == material_id


async def test_digest_observation_idempotent(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a re-digest of an already-digested observation escaping the short-circuit."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    digested_ids: set[str] = set()
    tools = WorldTools(store=store, adapters={}, digested_ids=digested_ids)
    material_id, _ = _digest_material_context()
    args = _digest_arguments(material_id)

    first = await tools.execute("digest_observation", args, "c1")
    second = await tools.execute("digest_observation", args, "c2")

    assert first["ok"] is True
    assert first["already_digested"] is False
    assert second["ok"] is True
    assert second["already_digested"] is True
    assert second["material_id"] == material_id
    assert material_id in digested_ids


async def test_digest_observation_without_tracking_never_dedups(tmp_path: pytest.TempPathFactory) -> None:
    """Catch accidental dedup before the graph supplies its shared digested_ids set."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()
    args = _digest_arguments(material_id)

    first = await tools.execute("digest_observation", args, "c1")
    second = await tools.execute("digest_observation", args, "c2")

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["already_digested"] is False


async def test_digest_observation_rejects_non_list_points(tmp_path: pytest.TempPathFactory) -> None:
    """Catch a hostile points argument escaping the facade as a typed limitation."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(material_id, points="not-a-list"),
        "c4",
    )

    assert result == {"ok": False, "limitations": ["invalid_arguments"]}


def test_digest_schema_advertises_optional_unresolved(tmp_path: pytest.TempPathFactory) -> None:
    """The digest schema must advertise the optional unresolved field so the model can fill it."""
    store = WorldStore(tmp_path / "world.sqlite3")
    tools = WorldTools(store=store, adapters={})
    parameters = next(
        item["function"]["parameters"]
        for item in tools.schemas()
        if item["function"]["name"] == "digest_observation"
    )

    assert "unresolved" in parameters["properties"]
    assert "unresolved" not in parameters["required"]
    unresolved = parameters["properties"]["unresolved"]
    assert unresolved["type"] == "array"
    assert unresolved["items"]["type"] == "object"
    assert unresolved["items"]["required"] == ["what", "why"]
    material_id = parameters["properties"]["material_id"]
    assert "digest_material_id" in material_id["description"]
    assert "observation_id" in material_id["description"]
    description = next(
        item["function"]["description"]
        for item in tools.schemas()
        if item["function"]["name"] == "digest_observation"
    )
    assert "digest_material_id" in description
    assert "never use its observation_id" in description


async def test_digest_observation_passes_through_unresolved(tmp_path: pytest.TempPathFactory) -> None:
    """A digest carrying unresolved items must echo them back in the digest payload."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()
    unresolved = [
        {"what": "为什么官号在23:48发动态", "why": "时间点可疑"},
        {"what": "暗讽解读缺乏直接证据", "why": "只有时间巧合"},
    ]

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(material_id, evidence_refs=["hupu:obs-1"], unresolved=unresolved),
        "c5",
    )

    assert result["ok"] is True
    assert result["digest"]["unresolved"] == unresolved


async def test_digest_observation_drops_invalid_unresolved_items(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Garbage unresolved items must be dropped, not fatal to the whole digest."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(
            material_id,
            evidence_refs=["hupu:obs-1"],
            unresolved=[
                {"what": "kept", "why": "valid"},
                {"what": "", "why": "blank what dropped"},
                {"what": "missing why"},
                {"what": "blank why dropped", "why": "   "},
                "not-a-dict",
                {"what": 42, "why": "non-string what dropped"},
            ],
        ),
        "c6",
    )

    assert result["ok"] is True
    assert result["digest"]["unresolved"] == [{"what": "kept", "why": "valid"}]


async def test_digest_observation_non_list_unresolved_is_tolerated(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """A hostile non-list unresolved argument must not reject the whole digest."""
    store = WorldStore(tmp_path / "world.sqlite3")
    _digest_seed(store)
    tools = WorldTools(store=store, adapters={})
    material_id, _ = _digest_material_context()

    result = await tools.execute(
        "digest_observation",
        _digest_arguments(
            material_id,
            evidence_refs=["hupu:obs-1"],
            unresolved="not-a-list",
        ),
        "c7",
    )

    assert result["ok"] is True
    assert result["digest"]["unresolved"] == []
