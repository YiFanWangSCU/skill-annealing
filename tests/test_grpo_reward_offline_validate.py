import pytest

from scripts.validate_grpo_reward_offline import (
    core_similarity,
    group_consistency,
    score_record,
)


def _record(sample_id, exposure, prediction, target):
    return {
        "_parsed_output": prediction,
        "_schema_valid": True,
        "hard_tags": [],
        "prompt_exposure": exposure,
        "sample_id": sample_id,
        "target_output": target,
    }


def test_gold_gated_consistency_rewards_correct_pairs():
    gold = {
        "decision": "manual_review",
        "need_human": True,
        "reason_codes": ["manual_review_required"],
        "risk_level": "high",
    }
    full = _record("s1", "full", gold, gold)
    partial = _record("s1", "partial", gold, gold)

    result = group_consistency([full, partial])

    assert result["available_pairs"] == 1
    assert result["exposure_consistency_reward"] == 1.0
    assert result["wrong_high_similarity_pairs"] == 0


def test_gold_gated_consistency_penalizes_consistent_wrong_pairs():
    gold = {
        "decision": "deny",
        "need_human": False,
        "reason_codes": ["virtual_goods_denied"],
        "risk_level": "low",
    }
    wrong = {
        "decision": "manual_review",
        "need_human": True,
        "reason_codes": ["manual_review_required"],
        "risk_level": "high",
    }
    full = _record("s1", "full", wrong, gold)
    partial = _record("s1", "partial", wrong, gold)

    result = group_consistency([full, partial])

    assert result["exposure_consistency_reward"] == pytest.approx(-0.1)
    assert result["wrong_high_similarity_pairs"] == 1


def test_false_manual_review_penalty_only_when_gold_is_not_manual_review():
    gold = {
        "decision": "deny",
        "need_human": False,
        "reason_codes": ["virtual_goods_denied"],
        "risk_level": "low",
    }
    pred = {
        "decision": "manual_review",
        "need_human": True,
        "reason_codes": ["manual_review_required"],
        "risk_level": "high",
    }
    score = score_record(_record("s1", "full", pred, gold), 0.0)

    assert score["false_manual_review_penalty"] == 1.0
    assert score["total_reward"] < 0.1


def test_core_similarity_ignores_explanation_text():
    left = {
        "decision": "deny",
        "explanation": "a",
        "need_human": False,
        "reason_codes": ["virtual_goods_denied"],
        "risk_level": "low",
    }
    right = {
        "decision": "deny",
        "explanation": "b",
        "need_human": False,
        "reason_codes": ["virtual_goods_denied"],
        "risk_level": "low",
    }

    assert core_similarity(left, right) == 1.0
