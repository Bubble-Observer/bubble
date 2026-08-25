"""Single-source Graph Patch action contract.

The schemas in this module are both the model-advertised contract and the
pre-dispatch/staging validation source.  Each supported ``kind``/``action``
pair has an exact outer shape and an exact payload shape so semantic fields
cannot be accepted and then silently discarded by staging or finalize.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import jsonschema
from jsonschema.exceptions import ValidationError

from .graph_contract import QUALIFIER_KEYS
from .graph_contract_text import EVENT_REIFICATION_RULE

# The six durable object kinds, domain-neutral by design: person (a natural
# person), organization (a group with members and decisions), place (a
# location), event (a fact at a known time — a time anchor), concept (an
# abstract meaning without a time), and entity (the fallback for durable
# things that fit none of the above). type_key carries the domain refinement
# (team, player, match, ...) orthogonally; see graph_contract_text for the
# full contract wording.
OBJECT_KINDS = ("entity", "event", "organization", "person", "place", "concept")
INQUIRY_KINDS = ("factual", "semantic", "stateful")
EPISTEMIC_ROLES = (
    "fact",
    "community_view",
    "semantic_explanation",
    "agent_synthesis",
    "uncertainty",
    "meta_knowledge",
)

SUPPORTED_GRAPH_PATCH_ACTIONS: dict[str, tuple[str, ...]] = {
    "object": ("create", "update", "drop"),
    "assertion": ("create", "update", "drop", "supersede"),
    "inquiry": ("create", "resolve", "drop"),
}

_STRING = {"type": "string", "minLength": 1}
# Advertised as "(UTC ISO)" in the tool contract; the format keyword keeps
# the advertisement truthful while _event_time_violation enforces the actual
# parse (jsonschema's date-time checker is lenient and accepts naive values).
_OPTIONAL_TIME = {
    "type": ["string", "null"],
    "format": "date-time",
    "description": (
        "Occurrence time at the precision known from the material (UTC ISO). "
        "On a direct assertion this records temporal scope; it does not replace "
        "an event object when the relationship itself is a bounded occurrence."
    ),
}
_CONFIDENCE = {"type": "number", "minimum": 0.0, "maximum": 1.0}
# Qualifiers are a closed key set shared with the compile path
# (normalize_qualifiers); the write path must reject invented keys at the
# moment of staging, not at finalize, so one bad key can never fail a whole
# wake (F-B contract drift). propertyNames bounds the keys while values stay
# free-form strings; unknown keys fail with the advertised schema error.
_QUALIFIERS_SCHEMA = {
    "type": "object",
    "propertyNames": {"enum": sorted(QUALIFIER_KEYS)},
    "description": (
        "Modifiers of this one assertion edge only. Do not put an occurrence's "
        "date, result, or participant list in role, scope, or another qualifier "
        "to avoid creating the event that the occurrence denotes."
    ),
}
_DECISION = {
    "type": "object",
    "properties": {
        "action": {"const": "confirm_distinct"},
        "distinct_from": _STRING,
        "basis": _STRING,
    },
    "required": ["action", "distinct_from", "basis"],
    "additionalProperties": False,
}

_OBJECT_PAYLOAD = {
    "type": "object",
    "properties": {
        "canonical_name": _STRING,
        "kind": {
            "type": "string",
            "enum": list(OBJECT_KINDS),
            "description": (
                "person: a natural person. organization: a group with members "
                "and decisions. place: a location on the map. event: a durable "
                "bounded occurrence or state change at a known time; carry "
                "event_time_start/end and anchor every known participant or "
                "changed subject through assertions. concept: an abstract "
                "meaning without a time. "
                "entity: the fallback for durable things that fit none of the "
                "above — choose the kind that names what the referent is, "
                "entity is not the default. type_key refines the kind (team, "
                "player, match, tournament, season, version) and is never an "
                "identity key."
            ),
        },
        "type_key": {"type": ["string", "null"]},
        "domain_hints": {"type": "array", "items": _STRING},
        "provisional": {"type": "boolean"},
        "event_time_start": _OPTIONAL_TIME,
        "event_time_end": _OPTIONAL_TIME,
        "aliases": {"type": "array", "items": _STRING},
    },
    "required": ["canonical_name", "kind"],
    "additionalProperties": False,
}

_ASSERTION_CREATE_FIELDS = {
    "subject_ref": _STRING,
    "predicate": _STRING,
    "object_ref": {
        **_STRING,
        "description": (
            "The durable object at the other end of a direct relation. If the "
            "relationship is shorthand for a bounded occurrence, outcome, or "
            "state change, create or reuse an event and connect its participants "
            "instead of connecting the participants directly."
        ),
    },
    "literal": {},
    "epistemic_role": {"type": "string", "enum": list(EPISTEMIC_ROLES)},
    "confidence": _CONFIDENCE,
    "event_time_start": _OPTIONAL_TIME,
    "event_time_end": _OPTIONAL_TIME,
    "qualifiers": deepcopy(_QUALIFIERS_SCHEMA),
    "evidence": {"type": "array", "items": _STRING},
    "answers_ref": _STRING,
}


def _assertion_payload(*, supersede: bool) -> dict[str, Any]:
    properties = deepcopy(_ASSERTION_CREATE_FIELDS)
    required = ["subject_ref", "predicate"]
    if supersede:
        properties["supersedes_ref"] = deepcopy(_STRING)
        required.append("supersedes_ref")
    return {
        "type": "object",
        "description": EVENT_REIFICATION_RULE,
        "properties": properties,
        "required": required,
        "oneOf": [
            {"required": ["object_ref"], "not": {"required": ["literal"]}},
            {"required": ["literal"], "not": {"required": ["object_ref"]}},
        ],
        "additionalProperties": False,
    }


_ASSERTION_UPDATE_PAYLOAD = {
    "type": "object",
    "properties": {
        "literal": {},
        "epistemic_role": {"type": "string", "enum": list(EPISTEMIC_ROLES)},
        "confidence": _CONFIDENCE,
        "event_time_start": _OPTIONAL_TIME,
        "event_time_end": _OPTIONAL_TIME,
        "qualifiers": deepcopy(_QUALIFIERS_SCHEMA),
        "evidence": {"type": "array", "items": _STRING},
        "answers_ref": {"type": ["string", "null"], "minLength": 1},
    },
    "minProperties": 1,
    "additionalProperties": False,
}

_INQUIRY_CREATE_PAYLOAD = {
    "type": "object",
    "properties": {
        "subject_ref": _STRING,
        "prompt": _STRING,
        "rationale": _STRING,
        "kind": {"type": "string", "enum": list(INQUIRY_KINDS)},
        "deepens_ref": _STRING,
    },
    "required": ["subject_ref", "prompt", "rationale"],
    "additionalProperties": False,
}

_INQUIRY_RESOLVE_PAYLOAD = {
    "type": "object",
    "properties": {
        "expected_version": {"type": "integer", "minimum": 1},
        "answers_ref": _STRING,
    },
    "required": ["expected_version", "answers_ref"],
    "additionalProperties": False,
}


def _variant(
    kind: str,
    action: str,
    *,
    payload: Mapping[str, Any] | None = None,
    target: bool = False,
    version: bool = False,
    decision: bool = False,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "op_id": deepcopy(_STRING),
        "kind": {"const": kind},
        "action": {"const": action},
    }
    required = ["op_id", "kind", "action"]
    if target:
        properties["target_ref"] = deepcopy(_STRING)
        required.append("target_ref")
    if version:
        properties["expected_version"] = {"type": "integer", "minimum": 1}
    if payload is not None:
        properties["payload"] = deepcopy(dict(payload))
        required.append("payload")
    if decision:
        properties["decision"] = deepcopy(_DECISION)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


GRAPH_PATCH_VARIANT_SCHEMAS: dict[tuple[str, str], dict[str, Any]] = {
    ("object", "create"): _variant("object", "create", payload=_OBJECT_PAYLOAD, decision=True),
    ("object", "update"): _variant(
        "object",
        "update",
        payload=_OBJECT_PAYLOAD,
        target=True,
        version=True,
        decision=True,
    ),
    ("object", "drop"): _variant("object", "drop", target=True),
    ("assertion", "create"): _variant("assertion", "create", payload=_assertion_payload(supersede=False)),
    ("assertion", "update"): _variant(
        "assertion",
        "update",
        payload=_ASSERTION_UPDATE_PAYLOAD,
        target=True,
        version=True,
    ),
    ("assertion", "drop"): _variant("assertion", "drop", target=True),
    ("assertion", "supersede"): _variant(
        "assertion", "supersede", payload=_assertion_payload(supersede=True)
    ),
    ("inquiry", "create"): _variant("inquiry", "create", payload=_INQUIRY_CREATE_PAYLOAD),
    ("inquiry", "resolve"): _variant("inquiry", "resolve", payload=_INQUIRY_RESOLVE_PAYLOAD, target=True),
    ("inquiry", "drop"): _variant("inquiry", "drop", target=True),
}


def graph_patch_parameters_schema(*, batch_cap: int) -> dict[str, Any]:
    """Return the exact parameters schema advertised to the Agent."""
    return {
        "type": "object",
        "description": EVENT_REIFICATION_RULE,
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": batch_cap,
                "items": {
                    "oneOf": [
                        deepcopy(GRAPH_PATCH_VARIANT_SCHEMAS[key])
                        for key in (
                            ("object", "create"),
                            ("object", "update"),
                            ("object", "drop"),
                            ("assertion", "create"),
                            ("assertion", "update"),
                            ("assertion", "drop"),
                            ("assertion", "supersede"),
                            ("inquiry", "create"),
                            ("inquiry", "resolve"),
                            ("inquiry", "drop"),
                        )
                    ]
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class GraphPatchContractViolation:
    """Typed, field-addressed rejection for one invalid Graph Patch call."""

    code: str
    field: str
    message: str
    action_hint: str
    violations: tuple[dict[str, str], ...] = ()
    total_violations: int = 0
    violations_truncated: bool = False


_REQUIRED_RE = re.compile(r"^'([^']+)' is a required property$")
_UNEXPECTED_RE = re.compile(r"'([^']+)' was unexpected")


def _path(prefix: str, error: ValidationError) -> str:
    parts = [prefix, *(str(part) for part in error.absolute_path)]
    field = ".".join(part for part in parts if part)
    required = _REQUIRED_RE.match(error.message)
    if required:
        return f"{field}.{required.group(1)}" if field else required.group(1)
    if error.validator == "additionalProperties":
        unexpected = _UNEXPECTED_RE.search(error.message)
        if unexpected:
            return f"{field}.{unexpected.group(1)}" if field else unexpected.group(1)
    return field or "arguments"


_XOR_ACTIONS = ("create", "supersede")


def _one_of_payload_message(error: ValidationError) -> str | None:
    """Describe an assertion payload oneOf failure precisely (both/neither)."""
    payload = error.instance if isinstance(error.instance, Mapping) else {}
    has_object_ref = "object_ref" in payload
    has_literal = "literal" in payload
    if has_object_ref and has_literal:
        return "Graph Patch assertion payload provides both object_ref and literal; exactly one is required"
    if not has_object_ref and not has_literal:
        return (
            "Graph Patch assertion payload provides neither object_ref nor literal; exactly one is required"
        )
    return None


def _variant_hint(
    kind: str,
    action: str,
    field: str,
    error: ValidationError | None = None,
) -> str:
    if error is not None and error.validator == "oneOf" and kind == "assertion" and action in _XOR_ACTIONS:
        return (
            "The assertion payload must provide exactly one of object_ref (a "
            "ref to an existing object) or literal (a free-form value); "
            "providing both or neither violates the oneOf constraint"
        )
    if field.endswith(".qualifiers"):
        return (
            "Assertion qualifiers support only: role, language, community, "
            "scope, granularity. Move other notes (e.g. source conflicts) "
            "into the assertion literal."
        )
    if field.endswith(".deepens_id"):
        return (
            "Use payload.deepens_ref on an inquiry create; copy the formal inquiry "
            "id or host-issued staged inquiry id verbatim"
        )
    if field.endswith(".answers_ref") and kind == "inquiry" and action == "create":
        return (
            "Create the answer as kind 'assertion' with payload.answers_ref naming "
            "the inquiry, then use inquiry action 'resolve' naming that assertion"
        )
    if field.endswith(".supersedes_ref") and kind == "assertion" and action == "create":
        return "Use assertion action 'supersede' when payload.supersedes_ref is present"
    if action == "drop":
        return f"Use {kind} action 'drop' with target_ref only; omit payload and decision"
    actions = ", ".join(SUPPORTED_GRAPH_PATCH_ACTIONS.get(kind, ()))
    if not actions:
        return "Use kind object, assertion, or inquiry and one advertised action variant"
    return f"Use the exact {kind}.{action} schema; supported {kind} actions: {actions}"


def _violation_from_error(
    error: ValidationError,
    *,
    prefix: str,
    kind: str,
    action: str,
) -> GraphPatchContractViolation:
    field = _path(prefix, error)
    message = (
        _one_of_payload_message(error)
        if error.validator == "oneOf" and kind == "assertion" and action in _XOR_ACTIONS
        else None
    )
    if message is None:
        message = f"Graph Patch item violates the advertised {kind}.{action} schema: {error.message}"
    return GraphPatchContractViolation(
        code="invalid_tool_arguments",
        field=field,
        message=message,
        action_hint=_variant_hint(kind, action, field, error=error),
    )


def _additional_property_violations(
    error: ValidationError,
    *,
    prefix: str,
    kind: str,
    action: str,
) -> list[GraphPatchContractViolation]:
    """Split one jsonschema additionalProperties error into exact fields."""
    if error.validator != "additionalProperties" or not isinstance(error.instance, Mapping):
        return []
    schema = error.schema if isinstance(error.schema, Mapping) else {}
    raw_properties = schema.get("properties")
    properties = set(raw_properties) if isinstance(raw_properties, Mapping) else set()
    raw_patterns = schema.get("patternProperties")
    patterns = list(raw_patterns) if isinstance(raw_patterns, Mapping) else []
    base = ".".join([prefix, *(str(part) for part in error.absolute_path)])
    violations: list[GraphPatchContractViolation] = []
    for raw_name in sorted(error.instance, key=str):
        name = str(raw_name)
        if raw_name in properties or any(re.search(pattern, name) for pattern in patterns):
            continue
        field = f"{base}.{name}"
        violations.append(
            GraphPatchContractViolation(
                "invalid_tool_arguments",
                field,
                f"Graph Patch item contains unsupported field {name!r}",
                _variant_hint(kind, action, field, error=error),
            )
        )
    return violations


def _common_item_violations(
    item: Mapping[str, Any],
    *,
    prefix: str,
    kind: str,
    action: str,
) -> list[GraphPatchContractViolation]:
    """Validate fields whose meaning does not depend on a selected variant."""
    violations: list[GraphPatchContractViolation] = []
    for field_name in ("op_id", "kind", "action"):
        if field_name not in item:
            continue
        value = item[field_name]
        if not isinstance(value, str) or not value:
            field = f"{prefix}.{field_name}"
            violations.append(
                GraphPatchContractViolation(
                    "invalid_tool_arguments",
                    field,
                    f"Graph Patch field {field_name!r} must be a non-empty string",
                    _variant_hint(kind, action, field),
                )
            )
    allowed_fields = {
        name
        for variant_schema in GRAPH_PATCH_VARIANT_SCHEMAS.values()
        for name in variant_schema.get("properties", {})
    }
    for raw_name in sorted(item, key=str):
        if raw_name in allowed_fields:
            continue
        name = str(raw_name)
        field = f"{prefix}.{name}"
        violations.append(
            GraphPatchContractViolation(
                "invalid_tool_arguments",
                field,
                f"Graph Patch item contains unsupported field {name!r}",
                _variant_hint(kind, action, field),
            )
        )
    return violations


_EVENT_TIME_FIELDS = ("event_time_start", "event_time_end")


def _event_time_violations(item: Mapping[str, Any], prefix: str) -> list[GraphPatchContractViolation]:
    """Enforce the advertised '(UTC ISO)' event_time contract.

    jsonschema's date-time format is lenient (accepts date-only and naive
    instants), and staging stores the raw value without parsing, so a bad
    instant would otherwise sail to finalize compile and explode there with
    the model's patch context long gone. Fail it at patch time instead, with
    the exact field named.
    """
    payload = item.get("payload")
    if not isinstance(payload, Mapping):
        return []
    violations: list[GraphPatchContractViolation] = []
    parsed_times: dict[str, datetime] = {}
    for field in _EVENT_TIME_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            violations.append(
                GraphPatchContractViolation(
                    "invalid_tool_arguments",
                    f"{prefix}.payload.{field}",
                    f"Graph Patch payload {field} is not an ISO-8601 instant: {value!r}",
                    "Use a full instant such as '2026-08-22T10:30:00Z' (the contract advertises '(UTC ISO)')",
                )
            )
            continue
        if parsed.tzinfo is None:
            violations.append(
                GraphPatchContractViolation(
                    "invalid_tool_arguments",
                    f"{prefix}.payload.{field}",
                    f"Graph Patch payload {field} must be tz-aware; naive instant {value!r} is ambiguous",
                    "Append a timezone ('2026-08-22T10:30:00+08:00' or '...Z'); the "
                    "contract advertises '(UTC ISO)'",
                )
            )
        else:
            parsed_times[field] = parsed
    if (
        _EVENT_TIME_FIELDS[0] in parsed_times
        and _EVENT_TIME_FIELDS[1] in parsed_times
        and parsed_times[_EVENT_TIME_FIELDS[1]] < parsed_times[_EVENT_TIME_FIELDS[0]]
    ):
        violations.append(
            GraphPatchContractViolation(
                "invalid_tool_arguments",
                f"{prefix}.payload.event_time_end",
                "Graph Patch payload event_time_end must not precede event_time_start",
                "Keep the event interval ordered, or omit the unknown endpoint",
            )
        )
    return violations


def _violation_key(violation: GraphPatchContractViolation) -> tuple[str, str, str]:
    return violation.code, violation.field, violation.message


def _dedupe_violations(
    violations: Sequence[GraphPatchContractViolation],
) -> list[GraphPatchContractViolation]:
    seen: set[tuple[str, str, str]] = set()
    output: list[GraphPatchContractViolation] = []
    for violation in violations:
        key = _violation_key(violation)
        if key in seen:
            continue
        seen.add(key)
        output.append(violation)
    return output


def graph_patch_item_violations(
    item: object,
    *,
    index: int | None = None,
) -> list[GraphPatchContractViolation]:
    """Return every independent violation for one selected patch variant."""
    prefix = f"items[{index}]" if index is not None else "item"
    if not isinstance(item, Mapping):
        return [
            GraphPatchContractViolation(
                "invalid_tool_arguments",
                prefix,
                "Graph Patch item must be an object",
                "Use one exact object/assertion/inquiry action object",
            )
        ]
    violations: list[GraphPatchContractViolation] = []
    for required in ("op_id", "kind", "action"):
        if required not in item:
            kind = str(item.get("kind") or "")
            action = str(item.get("action") or "")
            field = f"{prefix}.{required}"
            violations.append(
                GraphPatchContractViolation(
                    "invalid_tool_arguments",
                    field,
                    f"Graph Patch item is missing required field {required!r}",
                    _variant_hint(kind, action, field),
                )
            )
    kind = str(item.get("kind") or "")
    action = str(item.get("action") or "")
    common_violations = _common_item_violations(
        item,
        prefix=prefix,
        kind=kind,
        action=action,
    )
    if not kind or not action:
        return _dedupe_violations([*violations, *common_violations])
    schema = GRAPH_PATCH_VARIANT_SCHEMAS.get((kind, action))
    if schema is None:
        field_name = "kind" if kind not in SUPPORTED_GRAPH_PATCH_ACTIONS else "action"
        field = f"{prefix}.{field_name}"
        violations.append(
            GraphPatchContractViolation(
                "invalid_tool_arguments",
                field,
                f"Unsupported Graph Patch variant {kind!r}.{action!r}",
                _variant_hint(kind, action, field),
            )
        )
        return _dedupe_violations([*violations, *common_violations])
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(item))
    for error in sorted(
        errors,
        key=lambda entry: (
            1 if entry.absolute_path and entry.absolute_path[0] == "payload" else 0,
            len(entry.absolute_path),
            _path(prefix, entry),
            entry.message,
        ),
    ):
        expanded = _additional_property_violations(
            error,
            prefix=prefix,
            kind=kind,
            action=action,
        )
        if expanded:
            violations.extend(expanded)
        else:
            violations.append(_violation_from_error(error, prefix=prefix, kind=kind, action=action))
    violations.extend(_event_time_violations(item, prefix))
    return _dedupe_violations(violations)


def graph_patch_item_violation(
    item: object,
    *,
    index: int | None = None,
) -> GraphPatchContractViolation | None:
    """Compatibility wrapper returning the first precise item violation."""
    violations = graph_patch_item_violations(item, index=index)
    return violations[0] if violations else None


def graph_patch_arguments_violation(
    arguments: object,
    *,
    schema: Mapping[str, Any],
) -> GraphPatchContractViolation | None:
    """Validate a complete graph_patch call with precise item field feedback.

    Reports EVERY invalid item in one rejection instead of only the first, so
    the model can fix the whole batch in a single follow-up call instead of
    burning one tool-error streak per remaining item (failure-chain fix).
    """
    item_violations: list[tuple[int, list[GraphPatchContractViolation]]] = []
    if isinstance(arguments, Mapping):
        items = arguments.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            for index, item in enumerate(items):
                violations = graph_patch_item_violations(item, index=index)
                if violations:
                    item_violations.append((index, violations))
    if item_violations:
        # Guarantee one diagnostic for every bad item first, then add further
        # independent fields round-robin. A large malformed batch therefore
        # never hides later items behind many errors in its first item.
        selected: list[GraphPatchContractViolation] = [violations[0] for _, violations in item_violations]
        depth = 1
        while len(selected) < 32:
            added = False
            for _, violations in item_violations:
                if depth < len(violations) and len(selected) < 32:
                    selected.append(violations[depth])
                    added = True
            if not added:
                break
            depth += 1
        total = sum(len(violations) for _, violations in item_violations)
        hints = list(dict.fromkeys(v.action_hint for v in selected))
        details = tuple(
            {"code": violation.code, "field": violation.field, "message": violation.message}
            for violation in selected
        )
        first = selected[0]
        first_per_item = [violations[0] for _, violations in item_violations]
        item_summary = "; ".join(f"{violation.field}: {violation.message}" for violation in first_per_item)
        return GraphPatchContractViolation(
            "invalid_tool_arguments",
            first.field,
            f"Graph Patch has {total} validation issue(s) across "
            f"{len(item_violations)} invalid item(s): {item_summary}",
            " ".join(hints),
            violations=details,
            total_violations=total,
            violations_truncated=total > len(details),
        )
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(arguments))
    if not errors:
        return None
    expanded: list[dict[str, str]] = []
    for error in sorted(errors, key=lambda entry: (_path("arguments", entry), entry.message)):
        if error.validator == "additionalProperties" and isinstance(error.instance, Mapping):
            raw_properties = (
                error.schema.get("properties") if isinstance(error.schema, Mapping) else None
            )
            properties = set(raw_properties) if isinstance(raw_properties, Mapping) else set()
            raw_patterns = (
                error.schema.get("patternProperties") if isinstance(error.schema, Mapping) else None
            )
            patterns = list(raw_patterns) if isinstance(raw_patterns, Mapping) else []
            base = ".".join(["arguments", *(str(part) for part in error.absolute_path)])
            for raw_name in sorted(error.instance, key=str):
                name = str(raw_name)
                if raw_name in properties or any(re.search(pattern, name) for pattern in patterns):
                    continue
                field = f"{base}.{name}"
                expanded.append(
                    {
                        "code": "invalid_tool_arguments",
                        "field": field,
                        "message": f"Graph Patch arguments contain unsupported field {name!r}",
                    }
                )
            continue
        field = _path("arguments", error)
        expanded.append(
            {
                "code": "invalid_tool_arguments",
                "field": field,
                "message": f"Graph Patch arguments violate the advertised schema: {error.message}",
            }
        )
    deduped = list(
        {
            (detail["field"], detail["message"]): detail
            for detail in expanded
        }.values()
    )
    details = tuple(deduped[:32])
    first = details[0]
    total = len(deduped)
    return GraphPatchContractViolation(
        "invalid_tool_arguments",
        first["field"],
        f"Graph Patch arguments have {total} validation issue(s): "
        + "; ".join(detail["message"] for detail in details[:8]),
        "Use one to twenty exact Graph Patch item variants in arguments.items",
        violations=details,
        total_violations=total,
        violations_truncated=total > len(details),
    )


__all__ = [
    "GRAPH_PATCH_VARIANT_SCHEMAS",
    "GraphPatchContractViolation",
    "INQUIRY_KINDS",
    "OBJECT_KINDS",
    "SUPPORTED_GRAPH_PATCH_ACTIONS",
    "graph_patch_arguments_violation",
    "graph_patch_item_violation",
    "graph_patch_item_violations",
    "graph_patch_parameters_schema",
]
