from __future__ import annotations

import pytest
from pydantic import ValidationError

from leave_information_bubble.evaluation import (
    AdversarialEvaluationReport,
    AdversarialEvaluator,
    AttackEvidence,
    AttackStatus,
    default_adversarial_scenarios,
)
from leave_information_bubble.security import ContentTrust


def test_default_attack_suite_can_pass_with_real_control_evidence() -> None:
    scenarios = default_adversarial_scenarios()
    evidence = {
        "ssrf-loopback": AttackEvidence(security_rejection_code="non_public_address"),
        "prompt-injection-tool-command": AttackEvidence(
            plan_sha256_before="a" * 64,
            plan_sha256_after="a" * 64,
            content_trust=ContentTrust.EXTERNAL_UNTRUSTED,
        ),
        "cache-content-substitution": AttackEvidence(cache_admissible=False),
        "provenance-source-substitution": AttackEvidence(provenance_valid=False),
    }

    report = AdversarialEvaluator().evaluate(scenarios, evidence)

    assert len(report.findings) == 4
    assert report.passed == 4
    assert report.failed == 0
    assert report.not_run == 0
    assert all(finding.status is AttackStatus.PASSED for finding in report.findings)


def test_missing_evidence_is_not_reported_as_success() -> None:
    report = AdversarialEvaluator().evaluate(default_adversarial_scenarios(), {})

    assert report.passed == 0
    assert report.failed == 0
    assert report.not_run == 4


def test_changed_plan_and_admitted_poison_are_failures() -> None:
    evidence = {
        "prompt-injection-tool-command": AttackEvidence(
            plan_sha256_before="a" * 64,
            plan_sha256_after="b" * 64,
            content_trust=ContentTrust.EXTERNAL_UNTRUSTED,
            notes=("model output attempted an unplanned action",),
        ),
        "cache-content-substitution": AttackEvidence(cache_admissible=True),
    }

    report = AdversarialEvaluator().evaluate(default_adversarial_scenarios(), evidence)

    assert report.failed == 2
    assert report.not_run == 2
    prompt_finding = next(
        finding
        for finding in report.findings
        if finding.scenario_id == "prompt-injection-tool-command"
    )
    assert prompt_finding.status is AttackStatus.FAILED
    assert prompt_finding.notes == ("model output attempted an unplanned action",)


def test_incomplete_invariant_evidence_is_not_run() -> None:
    report = AdversarialEvaluator().evaluate(
        default_adversarial_scenarios(),
        {
            "prompt-injection-tool-command": AttackEvidence(
                content_trust=ContentTrust.EXTERNAL_UNTRUSTED
            )
        },
    )

    prompt_finding = next(
        finding
        for finding in report.findings
        if finding.scenario_id == "prompt-injection-tool-command"
    )
    assert prompt_finding.status is AttackStatus.NOT_RUN
    assert {result.status for result in prompt_finding.invariants} == {
        AttackStatus.PASSED,
        AttackStatus.NOT_RUN,
    }


def test_report_totals_cannot_hide_failed_findings() -> None:
    valid = AdversarialEvaluator().evaluate(
        default_adversarial_scenarios(),
        {"ssrf-loopback": AttackEvidence(security_rejection_code="non_public_address")},
    )

    with pytest.raises(ValidationError, match="total does not match"):
        AdversarialEvaluationReport(
            findings=valid.findings,
            passed=4,
            failed=0,
            not_run=0,
        )
