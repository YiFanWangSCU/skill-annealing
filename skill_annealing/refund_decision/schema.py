"""Shared schema definitions for the refund_decision skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


INTENTS = {
    "refund_request",
    "return_refund_request",
    "exchange_request",
    "complaint",
    "unknown",
}

DECISIONS = {
    "allow_refund",
    "allow_return_refund",
    "allow_exchange",
    "request_more_evidence",
    "deny",
    "manual_review",
}

RISK_LEVELS = {"low", "medium", "high"}

REASON_CODES = {
    "within_7_days_unused",
    "quality_issue_with_evidence",
    "quality_issue_no_evidence",
    "fresh_goods_no_reason_denied",
    "customized_goods_no_reason_denied",
    "virtual_goods_denied",
    "used_or_not_resellable",
    "over_7_days_no_quality_issue",
    "high_value_order",
    "frequent_refund_user",
    "evidence_conflict",
    "order_info_conflict",
    "logistics_issue",
    "manual_review_required",
}

REQUIRED_OUTPUT_FIELDS = {
    "intent",
    "decision",
    "risk_level",
    "need_human",
    "reason_codes",
    "explanation",
}


@dataclass(frozen=True)
class RefundDecision:
    intent: str
    decision: str
    risk_level: str
    need_human: bool
    reason_codes: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "need_human": self.need_human,
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
        }


def validate_output_schema(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors. An empty list means the payload is valid."""

    errors: list[str] = []
    missing = REQUIRED_OUTPUT_FIELDS - set(payload)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
        return errors

    if payload.get("intent") not in INTENTS:
        errors.append(f"invalid intent: {payload.get('intent')!r}")
    if payload.get("decision") not in DECISIONS:
        errors.append(f"invalid decision: {payload.get('decision')!r}")
    if payload.get("risk_level") not in RISK_LEVELS:
        errors.append(f"invalid risk_level: {payload.get('risk_level')!r}")
    if not isinstance(payload.get("need_human"), bool):
        errors.append("need_human must be boolean")

    reason_codes = payload.get("reason_codes")
    if not isinstance(reason_codes, list) or not all(
        isinstance(code, str) for code in reason_codes
    ):
        errors.append("reason_codes must be a list of strings")
    else:
        invalid_codes = sorted(set(reason_codes) - REASON_CODES)
        if invalid_codes:
            errors.append(f"invalid reason_codes: {invalid_codes}")

    if not isinstance(payload.get("explanation"), str) or not payload.get(
        "explanation"
    ):
        errors.append("explanation must be a non-empty string")

    return errors
