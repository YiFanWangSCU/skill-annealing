"""Deterministic warranty-claim oracle."""

from __future__ import annotations

from typing import Any, Mapping

from skill_annealing.multiskill_prompt_robustness.core import (
    DecisionResult,
    decimal_value,
    make_result,
    require_keys,
)


REQUIRED_FIELDS = {
    "product_category",
    "days_since_purchase",
    "warranty_days",
    "failure_type",
    "failure_description_present",
    "has_purchase_proof",
    "has_diagnostic_evidence",
    "unauthorized_repair",
    "unauthorized_repair_caused_current_failure",
    "extended_coverage",
    "prior_repair_count",
    "same_failure_recurrence",
    "serial_status",
    "safety_recall",
    "claim_amount_cny",
    "evidence_conflict",
}
FAILURE_TYPES = {
    "manufacturing_defect",
    "accidental_damage",
    "liquid_damage",
    "normal_wear",
    "unknown",
}


def decide(fields: Mapping[str, Any]) -> DecisionResult:
    require_keys(fields, REQUIRED_FIELDS)
    _validate(fields)
    if fields["safety_recall"]:
        return make_result(
            "approve_with_conditions", "safety_recall", safety_recall=True
        )
    if fields["evidence_conflict"]:
        return make_result("manual_review", "evidence_conflict")
    if fields["serial_status"] == "mismatch":
        return make_result("manual_review", "serial_mismatch")
    if fields["serial_status"] == "missing":
        return make_result("request_more_evidence", "serial_missing")
    if not fields["has_purchase_proof"]:
        return make_result("request_more_evidence", "purchase_proof_missing")
    if not fields["failure_description_present"]:
        return make_result("request_more_evidence", "failure_description_missing")
    if decimal_value(fields["claim_amount_cny"], field="claim_amount_cny") >= 3000:
        return make_result("manual_review", "abnormal_claim_amount")
    if fields["unauthorized_repair_caused_current_failure"]:
        return make_result("reject", "unauthorized_repair_causal")
    in_warranty = int(fields["days_since_purchase"]) <= int(fields["warranty_days"])
    if not in_warranty:
        if int(fields["prior_repair_count"]) >= 2 and fields[
            "same_failure_recurrence"
        ]:
            return make_result("manual_review", "repeat_failure_after_repairs")
        return make_result("reject", "warranty_expired")
    failure = fields["failure_type"]
    if failure in {"accidental_damage", "liquid_damage"}:
        if fields["extended_coverage"]:
            return make_result("approve_with_conditions", "extended_damage_covered")
        return make_result("reject", "accidental_or_liquid_not_covered")
    if failure == "manufacturing_defect":
        if fields["has_diagnostic_evidence"]:
            return make_result("approve", "manufacturing_defect_supported")
        return make_result("request_more_evidence", "diagnostic_evidence_missing")
    if failure == "normal_wear":
        return make_result("reject", "normal_wear_excluded")
    return make_result("request_more_evidence", "failure_type_unclear")


def _validate(fields: Mapping[str, Any]) -> None:
    if fields["product_category"] not in {"electronics", "appliance", "tool", "other"}:
        raise ValueError("invalid product_category")
    if fields["failure_type"] not in FAILURE_TYPES:
        raise ValueError("invalid failure_type")
    if fields["serial_status"] not in {"missing", "match", "mismatch"}:
        raise ValueError("invalid serial_status")
    for name in ("days_since_purchase", "warranty_days", "prior_repair_count"):
        if not isinstance(fields[name], int) or fields[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if fields["unauthorized_repair_caused_current_failure"] and not fields[
        "unauthorized_repair"
    ]:
        raise ValueError("causal unauthorized repair requires repair history")
    if fields["same_failure_recurrence"] and fields["prior_repair_count"] < 1:
        raise ValueError("recurrence requires prior repair")
