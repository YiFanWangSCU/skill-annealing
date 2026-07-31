from __future__ import annotations

from collections import Counter

import pytest

from skill_annealing.multiskill_prompt_robustness.analysis import (
    case_cluster_bootstrap,
    classify_frozen_decision,
    construct_balanced_effect,
    exact_mcnemar_pvalue,
    family_cluster_bootstrap,
    grouped_effects,
    holm_adjust,
    strict_json_object,
)
from skill_annealing.multiskill_prompt_robustness.builder import (
    ENGINES,
    build_bundle,
    build_skill_cases,
    build_training_arms,
    prototype_fields,
)
from skill_annealing.multiskill_prompt_robustness.core import load_protocol
from skill_annealing.multiskill_prompt_robustness.registries import (
    build_prompt_registry,
    build_target_registry,
)
from skill_annealing.multiskill_prompt_robustness.validator import (
    reject_confirmation_path,
    validate_bundle,
)


def _code_to_decision(skill: dict) -> dict[str, str]:
    mapping = {}
    for rule in skill["priority_order"]:
        rhs = rule.split(" -> ", 1)[1]
        decision_part, code = rhs.rsplit(":", 1)
        mapping[code] = decision_part.split("/", 1)[0]
    return mapping


def test_every_registered_reason_has_reachable_oracle_prototype() -> None:
    protocol = load_protocol()
    nonce = 0
    for skill_id, skill in protocol["skills"].items():
        decisions = _code_to_decision(skill)
        for reason in skill["reason_codes"]:
            result = ENGINES[skill_id](prototype_fields(skill_id, reason, nonce))
            assert result.reason_codes == [reason]
            assert result.decision == decisions[reason]
            nonce += 1


def test_scheme_b_cases_are_exact_and_split_safe() -> None:
    protocol = load_protocol()
    target_registry = build_target_registry(protocol)
    for skill_id, skill in protocol["skills"].items():
        splits = build_skill_cases(skill_id, protocol, target_registry)
        assert {key: len(value) for key, value in splits.items()} == {
            "train": 600,
            "validation": 64,
            "discovery_eval": 64,
            "confirmation": 64,
        }
        assert Counter(
            row["target_output"]["decision"] for row in splits["train"]
        ) == Counter(
            {
                "approve": 120,
                "approve_with_conditions": 120,
                "request_more_evidence": 120,
                "reject": 120,
                "manual_review": 120,
            }
        )
        for split in ("validation", "discovery_eval", "confirmation"):
            assert Counter(row["rule_family"] for row in splits[split]) == Counter(
                {family: 8 for family in skill["rule_families"]}
            )
        for field in (
            "semantic_case_hash",
            "facts_hash",
            "lexical_template_family",
            "lexical_template_source_sha256",
        ):
            values = {
                split: {row[field] for row in rows}
                for split, rows in splits.items()
            }
            for left in values:
                for right in values:
                    if left < right:
                        assert values[left].isdisjoint(values[right])


def test_training_arms_have_same_multiset_and_different_order() -> None:
    protocol = load_protocol()
    cases = build_skill_cases("refund_decision", protocol)["train"]
    arms = build_training_arms(
        "refund_decision", cases, protocol, build_prompt_registry(protocol)
    )
    staged = [row["record_id"] for row in arms["staged_order"]]
    random_rows = [row["record_id"] for row in arms["random_mix_order"]]
    assert len(staged) == len(set(staged)) == 1800
    assert Counter(staged) == Counter(random_rows)
    assert staged != random_rows
    assert Counter(row["prompt_exposure"] for row in arms["staged_order"]) == Counter(
        {"full": 630, "partial": 540, "minimal": 420, "no_skill": 210}
    )


def test_full_bundle_passes_local_static_gates(tmp_path) -> None:
    build_bundle(tmp_path)
    report = validate_bundle(tmp_path)
    assert report["local_static_valid"], report["errors"]
    assert report["remote_training_ready"] is False
    assert sum(skill["eval_requests"] for skill in report["skills"].values()) == 3072


def test_confirmation_is_hard_rejected() -> None:
    with pytest.raises(PermissionError):
        reject_confirmation_path("data/confirmation.locked.jsonl")
    reject_confirmation_path("data/discovery_eval.jsonl")


def test_strict_json_rejects_duplicate_keys_and_non_object() -> None:
    assert strict_json_object('{"decision":"approve"}')["decision"] == "approve"
    with pytest.raises(ValueError):
        strict_json_object('{"decision":"approve","decision":"reject"}')
    with pytest.raises(ValueError):
        strict_json_object("[]")


def test_construct_balanced_metric_weights_three_constructs_equally() -> None:
    rows = []
    for skill in ("a", "b", "c"):
        for case in ("1", "2"):
            for endpoint in ("random_mix_order", "staged_order"):
                values = {
                    "legacy_no_system": 1,
                    "full_length_irrelevant": 1 if endpoint == "staged_order" else 0,
                    "partial_semantic_paraphrase_a": 0,
                    "partial_semantic_paraphrase_b": 0,
                }
                for condition, correct in values.items():
                    rows.append(
                        {
                            "skill_id": skill,
                            "semantic_case_hash": case,
                            "endpoint": endpoint,
                            "prompt_condition": condition,
                            "rule_family": f"family_{case}",
                            "exact_tuple_correct": correct,
                        }
                    )
    report = construct_balanced_effect(rows)
    assert report["macro_delta"] == pytest.approx(1 / 3)
    bootstrap = case_cluster_bootstrap(rows, samples=100, seed=1)
    assert bootstrap["mean"] == pytest.approx(1 / 3)
    family_bootstrap = family_cluster_bootstrap(rows, samples=100, seed=1)
    assert family_bootstrap["mean"] == pytest.approx(1 / 3)


def test_grouped_effects_average_conditions_within_construct() -> None:
    rows = []
    for skill in ("a", "b", "c"):
        for case in ("1", "2"):
            for endpoint in ("random_mix_order", "staged_order"):
                for condition, correct in (
                    ("paraphrase_a", endpoint == "staged_order"),
                    ("paraphrase_b", case == "1"),
                ):
                    rows.append(
                        {
                            "skill_id": skill,
                            "semantic_case_hash": case,
                            "endpoint": endpoint,
                            "prompt_condition": condition,
                            "exact_tuple_correct": correct,
                        }
                    )
    report = grouped_effects(
        rows, {"paraphrase": ("paraphrase_a", "paraphrase_b")}
    )["paraphrase"]
    assert report["random_mix_order"] == pytest.approx(0.25)
    assert report["staged_order"] == pytest.approx(0.75)
    assert report["delta"] == pytest.approx(0.5)
    assert all(
        values["delta"] == pytest.approx(0.5)
        for values in report["per_skill"].values()
    )


def test_exact_mcnemar_and_holm_boundaries() -> None:
    assert exact_mcnemar_pvalue(0, 5) == pytest.approx(0.0625)
    adjusted = holm_adjust(
        [
            {"name": "third", "p_exact_two_sided": 0.04},
            {"name": "first", "p_exact_two_sided": 0.01},
            {"name": "second", "p_exact_two_sided": 0.03},
        ]
    )
    sorted_adjusted = sorted(
        adjusted, key=lambda row: row["p_exact_two_sided"]
    )
    assert [row["p_holm"] for row in sorted_adjusted] == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def _decision_inputs(
    skill_deltas: tuple[float, float, float],
    *,
    primary_macro: float | None = None,
    minimal_delta: float = 0.0,
) -> dict:
    skills = ("a", "b", "c")
    per_skill = {
        skill: {"delta": delta}
        for skill, delta in zip(skills, skill_deltas, strict=True)
    }
    macro = (
        sum(skill_deltas) / len(skill_deltas)
        if primary_macro is None
        else primary_macro
    )
    neutral_per_skill = {
        skill: {"delta": 0.0} for skill in skills
    }
    return {
        "primary": {"macro_delta": macro, "per_skill": per_skill},
        "case_bootstrap": {"ci95_lower": 0.01},
        "family_bootstrap": {"ci95_lower": 0.01},
        "construct_effects": {
            "no_system": {"delta": 0.0},
            "irrelevant": {"delta": 0.0},
            "paraphrase": {"delta": 0.0},
        },
        "canonical": {"delta": 0.0, "per_skill": neutral_per_skill},
        "minimal": {"delta": minimal_delta},
        "no_skill": {"delta": 0.0},
        "schema_guards_pass": True,
    }


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (_decision_inputs((0.06, 0.06, 0.06)), "PASS"),
        (
            _decision_inputs(
                (0.06, 0.06, 0.06), minimal_delta=-0.05
            ),
            "FAIL",
        ),
        (
            _decision_inputs(
                (0.06, -0.01, 0.01), primary_macro=0.02
            ),
            "MIXED",
        ),
        (_decision_inputs((0.02, 0.02, 0.02)), "INCONCLUSIVE"),
    ],
)
def test_frozen_decision_states_and_strict_minus_0_05_boundary(
    inputs: dict, expected: str
) -> None:
    report = classify_frozen_decision(**inputs)
    assert report["state"] == expected
    if expected == "FAIL":
        assert report["fail_checks"]["minimal_macro_le_minus_0_05"]
        assert not report["pass_checks"]["minimal_macro_gt_minus_0_05"]
