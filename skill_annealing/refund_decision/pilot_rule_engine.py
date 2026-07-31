"""Frozen v2.1 refund oracle used only by the multi-Skill pilot."""

from __future__ import annotations

from typing import Any, Mapping

from skill_annealing.multiskill_prompt_robustness.core import (
    DecisionResult,
    decimal_value,
    make_result,
    require_keys,
)


REQUIRED_FIELDS = {
    "product_type",
    "days_since_delivery",
    "is_used",
    "is_resellable",
    "has_quality_issue",
    "has_evidence",
    "order_amount_cny",
    "refund_count_30d",
    "user_claim",
    "order_status",
    "seller_fault",
    "logistics_issue",
    "evidence_conflict",
    "order_info_conflict",
}
PRODUCT_TYPES = {"standard", "fresh", "customized", "virtual"}
USER_CLAIMS = {
    "no_reason",
    "size_not_fit",
    "exchange",
    "quality_issue",
    "logistics_issue",
    "complaint",
    "strong_complaint",
}
ORDER_STATUSES = {"delivered", "not_delivered", "closed", "unknown"}


def decide(fields: Mapping[str, Any]) -> DecisionResult:
    require_keys(fields, REQUIRED_FIELDS)
    _validate(fields)
    days = fields["days_since_delivery"]
    if days is None or fields["order_status"] == "unknown":
        return make_result("request_more_evidence", "missing_required_facts")
    if fields["evidence_conflict"]:
        return make_result("manual_review", "evidence_conflict")
    if fields["order_info_conflict"]:
        return make_result("manual_review", "order_info_conflict")
    if decimal_value(fields["order_amount_cny"], field="order_amount_cny") >= 1000:
        return make_result("manual_review", "high_value_order")
    if int(fields["refund_count_30d"]) >= 3:
        return make_result("manual_review", "frequent_refund_user")
    if fields["logistics_issue"]:
        if fields["seller_fault"] or fields["has_evidence"]:
            return make_result("approve", "logistics_issue_supported")
        return make_result(
            "request_more_evidence", "logistics_issue_insufficient_evidence"
        )
    returnable = (
        not fields["is_used"]
        and fields["is_resellable"]
        and fields["product_type"] != "virtual"
    )
    if fields["has_quality_issue"]:
        if not fields["has_evidence"]:
            return make_result("request_more_evidence", "quality_issue_no_evidence")
        if returnable:
            return make_result(
                "approve_with_conditions", "quality_issue_return_refund"
            )
        return make_result("approve", "quality_issue_refund_only")
    no_reason = fields["user_claim"] in {"no_reason", "size_not_fit", "exchange"}
    if fields["product_type"] in {"fresh", "customized", "virtual"} and no_reason:
        return make_result("reject", "special_goods_exclusion")
    if int(days) <= 7 and returnable:
        return make_result("approve_with_conditions", "within_7_days_unused")
    if int(days) <= 7:
        return make_result("reject", "used_or_not_resellable")
    return make_result("reject", "over_7_days_no_quality_issue")


def _validate(fields: Mapping[str, Any]) -> None:
    if fields["product_type"] not in PRODUCT_TYPES:
        raise ValueError("invalid product_type")
    if fields["user_claim"] not in USER_CLAIMS:
        raise ValueError("invalid user_claim")
    if fields["order_status"] not in ORDER_STATUSES:
        raise ValueError("invalid order_status")
    days = fields["days_since_delivery"]
    if days is not None and (not isinstance(days, int) or days < 0):
        raise ValueError("days_since_delivery must be a non-negative integer or null")
    if fields["product_type"] == "virtual" and fields["is_resellable"]:
        raise ValueError("virtual products must be non-resellable")
    if fields["order_status"] == "unknown" and days is not None:
        raise ValueError("unknown order_status requires null days")
    if fields["order_status"] in {"delivered", "closed"} and days is None:
        raise ValueError("delivered/closed requires known days")
    if fields["order_status"] == "not_delivered" and (
        days != 0 or not fields["logistics_issue"]
    ):
        raise ValueError("not_delivered requires day zero and logistics issue")
    if fields["seller_fault"] and not fields["logistics_issue"]:
        raise ValueError("seller_fault requires logistics_issue")
