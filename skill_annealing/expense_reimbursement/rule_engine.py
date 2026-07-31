"""Deterministic expense-reimbursement oracle."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from skill_annealing.multiskill_prompt_robustness.core import (
    DecisionResult,
    decimal_value,
    make_result,
    require_keys,
)


REQUIRED_FIELDS = {
    "expense_category",
    "amount",
    "currency",
    "hotel_nights",
    "days_since_expense",
    "has_receipt",
    "receipt_matches_amount",
    "business_purpose_present",
    "manager_approved",
    "project_code_valid",
    "is_duplicate_claim",
    "is_personal_expense",
    "is_weekend_or_holiday",
    "international_trip",
    "policy_exception_documented",
    "evidence_conflict",
}
RATES = {
    "CNY": Decimal("1.00"),
    "USD": Decimal("7.20"),
    "EUR": Decimal("7.80"),
    "JPY": Decimal("0.048"),
    "GBP": Decimal("9.10"),
    "HKD": Decimal("0.92"),
}
CAPS = {
    "meal": Decimal("300.00"),
    "local_transport": Decimal("500.00"),
    "office_supply": Decimal("1000.00"),
    "training": Decimal("3000.00"),
}


def decide(fields: Mapping[str, Any]) -> DecisionResult:
    require_keys(fields, REQUIRED_FIELDS)
    cny = _validate_and_convert(fields)
    if fields["is_duplicate_claim"]:
        return make_result("manual_review", "duplicate_claim")
    if fields["evidence_conflict"]:
        return make_result("manual_review", "evidence_conflict")
    if fields["has_receipt"] and not fields["receipt_matches_amount"]:
        return make_result("manual_review", "receipt_amount_mismatch")
    if not fields["has_receipt"]:
        return make_result("request_more_evidence", "receipt_missing")
    if not fields["business_purpose_present"]:
        return make_result("request_more_evidence", "business_purpose_missing")
    if not fields["project_code_valid"]:
        return make_result("request_more_evidence", "project_code_invalid")
    if fields["currency"] == "CNY" and cny >= Decimal("5000.00"):
        return make_result("manual_review", "high_value_expense")
    if fields["currency"] != "CNY" and cny >= Decimal("2000.00"):
        return make_result("manual_review", "high_value_foreign_currency")
    exception_case = fields["is_personal_expense"] or fields["days_since_expense"] > 30
    if fields["is_personal_expense"] and not fields["policy_exception_documented"]:
        return make_result("reject", "personal_expense")
    if fields["days_since_expense"] > 30 and not fields[
        "policy_exception_documented"
    ]:
        return make_result("reject", "late_submission")
    if exception_case and fields["policy_exception_documented"]:
        if not fields["manager_approved"]:
            return make_result(
                "request_more_evidence", "exception_manager_approval_missing"
            )
        return make_result(
            "approve_with_conditions", "documented_policy_exception"
        )
    cap = (
        Decimal("1200.00") * fields["hotel_nights"]
        if fields["expense_category"] == "hotel"
        else CAPS[fields["expense_category"]]
    )
    if cny > cap:
        if fields["manager_approved"]:
            return make_result("approve_with_conditions", "cap_override_approved")
        return make_result("reject", "category_cap_exceeded")
    return make_result("approve", "standard_compliant")


def _validate_and_convert(fields: Mapping[str, Any]) -> Decimal:
    if fields["expense_category"] not in {*CAPS, "hotel"}:
        raise ValueError("invalid expense_category")
    if fields["currency"] not in RATES:
        raise ValueError("invalid currency")
    if not isinstance(fields["hotel_nights"], int) or fields["hotel_nights"] < 0:
        raise ValueError("hotel_nights must be a non-negative integer")
    if fields["expense_category"] == "hotel" and fields["hotel_nights"] < 1:
        raise ValueError("hotel requires at least one night")
    if fields["expense_category"] != "hotel" and fields["hotel_nights"] != 0:
        raise ValueError("non-hotel expense requires zero hotel nights")
    if not isinstance(fields["days_since_expense"], int) or fields[
        "days_since_expense"
    ] < 0:
        raise ValueError("days_since_expense must be non-negative")
    if not fields["has_receipt"] and fields["receipt_matches_amount"]:
        raise ValueError("missing receipt cannot match amount")
    amount = decimal_value(fields["amount"], field="amount")
    if amount <= 0 or amount.as_tuple().exponent < -2:
        raise ValueError("amount must be positive with at most two decimals")
    return (amount * RATES[fields["currency"]]).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
