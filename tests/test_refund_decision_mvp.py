from skill_annealing.refund_decision.data_generator import generate_samples
from skill_annealing.refund_decision.evaluator import evaluate_records
from skill_annealing.refund_decision.prompt_builder import (
    assign_annealed_exposures,
    build_messages,
)
from skill_annealing.refund_decision.rule_engine import decide


def _fields(**overrides):
    base = {
        "product_type": "clothing",
        "days_since_delivery": 3,
        "is_used": False,
        "is_resellable": True,
        "has_quality_issue": False,
        "has_evidence": False,
        "order_amount": 199,
        "refund_count_30d": 0,
        "user_claim": "no_reason",
        "order_status": "delivered",
        "is_customized": False,
        "is_virtual": False,
        "is_fresh": False,
        "seller_fault": False,
        "logistics_issue": False,
        "evidence_conflict": False,
        "order_info_conflict": False,
    }
    return {**base, **overrides}


def test_within_7_days_unused_allows_return_refund():
    result = decide(_fields()).to_dict()
    assert result["decision"] == "allow_return_refund"
    assert result["risk_level"] == "low"
    assert result["reason_codes"] == ["within_7_days_unused"]


def test_virtual_no_reason_is_denied():
    result = decide(
        _fields(product_type="virtual", is_virtual=True, user_claim="no_reason")
    ).to_dict()
    assert result["decision"] == "deny"
    assert result["reason_codes"] == ["virtual_goods_denied"]


def test_quality_issue_without_evidence_requests_more_evidence():
    result = decide(
        _fields(has_quality_issue=True, has_evidence=False, user_claim="quality_issue")
    ).to_dict()
    assert result["decision"] == "request_more_evidence"
    assert result["risk_level"] == "medium"
    assert result["reason_codes"] == ["quality_issue_no_evidence"]


def test_high_value_order_forces_manual_review():
    result = decide(_fields(order_amount=2499)).to_dict()
    assert result["decision"] == "manual_review"
    assert result["need_human"] is True
    assert "high_value_order" in result["reason_codes"]
    assert "manual_review_required" in result["reason_codes"]


def test_order_conflict_forces_high_risk_manual_review():
    result = decide(
        _fields(is_used=True, user_claim="claims_unused_but_used")
    ).to_dict()
    assert result["decision"] == "manual_review"
    assert result["risk_level"] == "high"
    assert "order_info_conflict" in result["reason_codes"]


def test_prompt_builder_no_skill_has_no_system_message():
    sample = generate_samples(sample_count=1, seed=7)[0]
    no_skill_messages = build_messages(sample, "no_skill")
    full_messages = build_messages(sample, "full")
    assert no_skill_messages[0]["role"] == "user"
    assert full_messages[0]["role"] == "system"
    assert "业务规则" in full_messages[0]["content"]


def test_annealed_schedule_uses_reduced_contexts():
    exposures = assign_annealed_exposures(120, seed=11)
    assert set(exposures) == {"full", "partial", "minimal"}
    assert exposures[:36].count("full") > exposures[-36:].count("full")


def test_oracle_evaluator_scores_perfect():
    sample = generate_samples(sample_count=1, seed=9)[0]
    metrics = evaluate_records([{**sample, "predicted_output": sample["target_output"]}])
    assert metrics["overall_score"] == 1.0
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 1.0


def test_missing_prediction_is_not_treated_as_oracle():
    sample = generate_samples(sample_count=1, seed=10)[0]
    metrics = evaluate_records([sample])
    assert metrics["overall_score"] == 0.0
    assert metrics["json_valid_rate"] == 0.0
    assert metrics["schema_valid_rate"] == 0.0
