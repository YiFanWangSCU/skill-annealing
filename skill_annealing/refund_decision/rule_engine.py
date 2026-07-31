"""Rule engine that produces gold labels for refund_decision samples."""

from __future__ import annotations

from typing import Any, Mapping

from .schema import RefundDecision


HIGH_VALUE_ORDER_AMOUNT = 1000
FREQUENT_REFUND_COUNT_30D = 3

SPECIAL_DENY_REASON = {
    "fresh": "fresh_goods_no_reason_denied",
    "customized": "customized_goods_no_reason_denied",
    "virtual": "virtual_goods_denied",
}


def decide(fields: Mapping[str, Any]) -> RefundDecision:
    """Apply deterministic business rules to structured order facts."""

    normalized = _normalize_fields(fields)
    intent = _infer_intent(normalized)
    missing = _missing_critical_fields(normalized)
    if missing:
        return RefundDecision(
            intent=intent,
            decision="request_more_evidence",
            risk_level="medium",
            need_human=False,
            reason_codes=["manual_review_required"],
            explanation=f"订单缺少关键信息：{','.join(missing)}，需要补充材料后再判断。",
        )

    decision, reasons, explanation = _base_decision(intent, normalized)
    risk_reasons = _risk_reasons(normalized)

    if risk_reasons:
        reasons = _dedupe(reasons + risk_reasons + ["manual_review_required"])
        risk_level = _risk_level(risk_reasons)
        return RefundDecision(
            intent=intent,
            decision="manual_review",
            risk_level=risk_level,
            need_human=True,
            reason_codes=reasons,
            explanation=f"{explanation} 同时命中风险规则：{_reason_text(risk_reasons)}，需人工审核。",
        )

    risk_level = "medium" if decision == "request_more_evidence" else "low"
    return RefundDecision(
        intent=intent,
        decision=decision,
        risk_level=risk_level,
        need_human=False,
        reason_codes=_dedupe(reasons),
        explanation=explanation,
    )


def _base_decision(
    intent: str, fields: Mapping[str, Any]
) -> tuple[str, list[str], str]:
    product_label = _product_label(fields)

    if fields["logistics_issue"]:
        if fields["seller_fault"] or fields["has_evidence"]:
            return (
                "allow_refund",
                ["logistics_issue"],
                "订单存在物流异常且责任或证据明确，允许退款处理。",
            )
        return (
            "request_more_evidence",
            ["logistics_issue"],
            "用户反馈物流异常，但责任和证据尚不充分，需要补充物流凭证。",
        )

    if fields["has_quality_issue"]:
        if not fields["has_evidence"]:
            return (
                "request_more_evidence",
                ["quality_issue_no_evidence"],
                "用户反馈质量问题但没有提供有效证据，需要补充图片、视频或检测材料。",
            )

        if fields["is_virtual"] or fields["is_used"] or not fields["is_resellable"]:
            return (
                "allow_refund",
                ["quality_issue_with_evidence"],
                "商品存在质量问题且证据充分，但不适合退回二次销售，允许退款处理。",
            )

        return (
            "allow_return_refund",
            ["quality_issue_with_evidence"],
            "商品存在质量问题且证据充分，当前商品状态支持退货退款。",
        )

    special_reason = _special_no_reason_deny_reason(fields)
    if special_reason:
        return (
            "deny",
            [special_reason],
            f"{product_label}原则上不支持无理由退款或退货，且当前没有质量问题证据。",
        )

    if fields["days_since_delivery"] <= 7:
        if not fields["is_used"] and fields["is_resellable"]:
            if intent == "exchange_request":
                return (
                    "allow_exchange",
                    ["within_7_days_unused"],
                    "订单签收 7 天内，商品未使用且不影响二次销售，支持换货。",
                )
            return (
                "allow_return_refund",
                ["within_7_days_unused"],
                "订单签收 7 天内，商品未使用且不影响二次销售，符合退货退款条件。",
            )

        return (
            "deny",
            ["used_or_not_resellable"],
            "订单虽在 7 天内，但商品已使用或影响二次销售，且没有质量问题证据。",
        )

    return (
        "deny",
        ["over_7_days_no_quality_issue"],
        "订单签收已超过 7 天，且没有质量问题或商家责任证据，不支持退款退货。",
    )


def _normalize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    product_type = str(fields.get("product_type", "other"))
    return {
        "product_type": product_type,
        "days_since_delivery": _optional_int(fields.get("days_since_delivery")),
        "is_used": _bool(fields.get("is_used", False)),
        "is_resellable": _bool(fields.get("is_resellable", True)),
        "has_quality_issue": _bool(fields.get("has_quality_issue", False)),
        "has_evidence": _bool(fields.get("has_evidence", False)),
        "order_amount": _int(fields.get("order_amount", 0)),
        "refund_count_30d": _int(fields.get("refund_count_30d", 0)),
        "user_claim": str(fields.get("user_claim", "no_reason")),
        "order_status": str(fields.get("order_status", "delivered")),
        "is_customized": _bool(fields.get("is_customized", False))
        or product_type == "customized",
        "is_virtual": _bool(fields.get("is_virtual", False))
        or product_type == "virtual",
        "is_fresh": _bool(fields.get("is_fresh", False)) or product_type == "fresh",
        "seller_fault": _bool(fields.get("seller_fault", False)),
        "logistics_issue": _bool(fields.get("logistics_issue", False)),
        "evidence_conflict": _bool(fields.get("evidence_conflict", False)),
        "order_info_conflict": _bool(fields.get("order_info_conflict", False))
        or _claim_conflicts_with_order(fields),
    }


def _infer_intent(fields: Mapping[str, Any]) -> str:
    claim = fields["user_claim"]
    if claim == "exchange":
        return "exchange_request"
    if claim in {"complaint", "strong_complaint"}:
        return "complaint"
    if claim in {"size_not_fit", "no_reason"}:
        return "return_refund_request"
    if claim in {"quality_issue", "logistics_issue"}:
        return "refund_request"
    return "refund_request"


def _missing_critical_fields(fields: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if fields["days_since_delivery"] is None:
        missing.append("days_since_delivery")
    if fields["order_status"] not in {"delivered", "not_delivered", "closed"}:
        missing.append("order_status")
    return missing


def _risk_reasons(fields: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if fields["evidence_conflict"]:
        reasons.append("evidence_conflict")
    if fields["order_info_conflict"]:
        reasons.append("order_info_conflict")
    if fields["order_amount"] >= HIGH_VALUE_ORDER_AMOUNT:
        reasons.append("high_value_order")
    if fields["refund_count_30d"] >= FREQUENT_REFUND_COUNT_30D:
        reasons.append("frequent_refund_user")
    return reasons


def _risk_level(risk_reasons: list[str]) -> str:
    conflict_reasons = {"evidence_conflict", "order_info_conflict"}
    if conflict_reasons & set(risk_reasons):
        return "high"
    if len(set(risk_reasons) & {"high_value_order", "frequent_refund_user"}) >= 2:
        return "high"
    return "medium"


def _special_no_reason_deny_reason(fields: Mapping[str, Any]) -> str | None:
    if fields["is_virtual"]:
        return SPECIAL_DENY_REASON["virtual"]
    if fields["is_fresh"]:
        return SPECIAL_DENY_REASON["fresh"]
    if fields["is_customized"]:
        return SPECIAL_DENY_REASON["customized"]
    return None


def _product_label(fields: Mapping[str, Any]) -> str:
    if fields["is_virtual"]:
        return "虚拟商品"
    if fields["is_fresh"]:
        return "生鲜商品"
    if fields["is_customized"]:
        return "定制商品"
    return "普通商品"


def _claim_conflicts_with_order(fields: Mapping[str, Any]) -> bool:
    claim = str(fields.get("user_claim", ""))
    return claim == "claims_unused_but_used" and _bool(fields.get("is_used", False))


def _reason_text(reasons: list[str]) -> str:
    names = {
        "evidence_conflict": "证据冲突",
        "order_info_conflict": "用户描述与订单事实冲突",
        "high_value_order": "高价值订单",
        "frequent_refund_user": "30 天内频繁退款",
    }
    return "、".join(names.get(reason, reason) for reason in reasons)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
