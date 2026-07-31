"""Construct-balanced behavior analysis for completed prediction records."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping


ROBUST_CONDITIONS = (
    "legacy_no_system",
    "full_length_irrelevant",
    "partial_semantic_paraphrase_a",
    "partial_semantic_paraphrase_b",
)
CANONICAL_CONDITIONS = ("full", "partial", "minimal", "no_skill")
ENDPOINTS = ("random_mix_order", "staged_order")


def strict_json_object(text: str, *, prefix: str = "<think>\n\n</think>\n\n") -> dict[str, Any]:
    value = text
    if value.startswith(prefix):
        value = value[len(prefix) :]
    value = value.strip()
    pairs_seen: list[tuple[str, Any]] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        pairs_seen.clear()
        pairs_seen.extend(pairs)
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON key")
        return dict(pairs)

    parsed = json.loads(value, object_pairs_hook=hook)
    if not isinstance(parsed, dict):
        raise ValueError("completion must be one JSON object")
    return parsed


def exact_tuple(gold: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    return all(
        gold.get(field) == prediction.get(field)
        for field in ("decision", "risk_level", "need_human")
    )


def construct_balanced_effect(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    indexed: dict[tuple[str, str, str, str], float] = {}
    skills = set()
    cases: dict[str, set[str]] = defaultdict(set)
    for row in records:
        skill = str(row["skill_id"])
        case = str(row["semantic_case_hash"])
        condition = str(row["prompt_condition"])
        endpoint = str(row["endpoint"])
        skills.add(skill)
        cases[skill].add(case)
        indexed[(skill, case, condition, endpoint)] = float(row["exact_tuple_correct"])
    per_skill = {}
    per_case_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for skill in sorted(skills):
        endpoint_means = {}
        for endpoint in ("random_mix_order", "staged_order"):
            values = []
            for case in sorted(cases[skill]):
                no_system = indexed[(skill, case, "legacy_no_system", endpoint)]
                irrelevant = indexed[
                    (skill, case, "full_length_irrelevant", endpoint)
                ]
                paraphrase = (
                    indexed[
                        (
                            skill,
                            case,
                            "partial_semantic_paraphrase_a",
                            endpoint,
                        )
                    ]
                    + indexed[
                        (
                            skill,
                            case,
                            "partial_semantic_paraphrase_b",
                            endpoint,
                        )
                    ]
                ) / 2
                score = (no_system + irrelevant + paraphrase) / 3
                values.append(score)
                per_case_scores[skill].setdefault(case, {})[endpoint] = score
            endpoint_means[endpoint] = sum(values) / len(values)
        per_skill[skill] = {
            **endpoint_means,
            "delta": endpoint_means["staged_order"]
            - endpoint_means["random_mix_order"],
        }
    macro = sum(value["delta"] for value in per_skill.values()) / len(per_skill)
    return {
        "macro_delta": macro,
        "per_skill": per_skill,
        "per_case_scores": per_case_scores,
    }


def case_cluster_bootstrap(
    records: Iterable[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20260724,
) -> dict[str, float]:
    analysis = construct_balanced_effect(records)
    scores = analysis["per_case_scores"]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        skill_deltas = []
        for skill, cases in sorted(scores.items()):
            case_ids = sorted(cases)
            sampled = [rng.choice(case_ids) for _ in case_ids]
            staged = sum(cases[case]["staged_order"] for case in sampled) / len(sampled)
            random_mean = sum(
                cases[case]["random_mix_order"] for case in sampled
            ) / len(sampled)
            skill_deltas.append(staged - random_mean)
        draws.append(sum(skill_deltas) / len(skill_deltas))
    draws.sort()
    return {
        "mean": sum(draws) / len(draws),
        "ci95_lower": draws[int(0.025 * samples)],
        "ci95_upper": draws[min(samples - 1, int(0.975 * samples))],
    }


def family_cluster_bootstrap(
    records: Iterable[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 20260724,
) -> dict[str, float]:
    rows = list(records)
    analysis = construct_balanced_effect(rows)
    scores = analysis["per_case_scores"]
    case_family: dict[tuple[str, str], str] = {}
    for row in rows:
        case_family[(str(row["skill_id"]), str(row["semantic_case_hash"]))] = str(
            row["rule_family"]
        )
    family_cases: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for skill, cases in scores.items():
        for case in cases:
            family_cases[skill][case_family[(skill, case)]].append(case)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        skill_deltas = []
        for skill, families in sorted(family_cases.items()):
            family_ids = sorted(families)
            sampled_families = [rng.choice(family_ids) for _ in family_ids]
            staged_family_means = []
            random_family_means = []
            for family in sampled_families:
                selected = families[family]
                staged_family_means.append(
                    sum(scores[skill][case]["staged_order"] for case in selected)
                    / len(selected)
                )
                random_family_means.append(
                    sum(
                        scores[skill][case]["random_mix_order"] for case in selected
                    )
                    / len(selected)
                )
            skill_deltas.append(
                sum(staged_family_means) / len(staged_family_means)
                - sum(random_family_means) / len(random_family_means)
            )
        draws.append(sum(skill_deltas) / len(skill_deltas))
    draws.sort()
    return {
        "mean": sum(draws) / len(draws),
        "ci95_lower": draws[int(0.025 * samples)],
        "ci95_upper": draws[min(samples - 1, int(0.975 * samples))],
    }


def grouped_effects(
    records: Iterable[Mapping[str, Any]],
    groups: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """Compute equal-case, equal-skill endpoint means for condition groups."""

    rows = list(records)
    indexed: dict[tuple[str, str, str, str], float] = {}
    cases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = (
            str(row["skill_id"]),
            str(row["semantic_case_hash"]),
            str(row["prompt_condition"]),
            str(row["endpoint"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate prediction panel key: {key}")
        indexed[key] = float(row["exact_tuple_correct"])
        cases[key[0]].add(key[1])

    result = {}
    for group_name, raw_conditions in groups.items():
        conditions = tuple(raw_conditions)
        if not conditions:
            raise ValueError(f"empty condition group: {group_name}")
        per_skill = {}
        for skill, case_ids in sorted(cases.items()):
            endpoint_means = {}
            for endpoint in ENDPOINTS:
                case_scores = []
                for case_id in sorted(case_ids):
                    condition_scores = [
                        indexed[(skill, case_id, condition, endpoint)]
                        for condition in conditions
                    ]
                    case_scores.append(
                        sum(condition_scores) / len(condition_scores)
                    )
                endpoint_means[endpoint] = sum(case_scores) / len(
                    case_scores
                )
            per_skill[skill] = {
                **endpoint_means,
                "delta": (
                    endpoint_means["staged_order"]
                    - endpoint_means["random_mix_order"]
                ),
            }
        macro_random = sum(
            value["random_mix_order"] for value in per_skill.values()
        ) / len(per_skill)
        macro_staged = sum(
            value["staged_order"] for value in per_skill.values()
        ) / len(per_skill)
        result[group_name] = {
            "conditions": list(conditions),
            "random_mix_order": macro_random,
            "staged_order": macro_staged,
            "delta": macro_staged - macro_random,
            "per_skill": per_skill,
        }
    return result


def exact_mcnemar_pvalue(random_only: int, staged_only: int) -> float:
    """Two-sided exact McNemar p-value under a Binomial(n, 0.5) null."""

    discordant = random_only + staged_only
    if discordant == 0:
        return 1.0
    tail = min(random_only, staged_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def paired_mcnemar_tests(
    records: Iterable[Mapping[str, Any]],
    conditions: Iterable[str],
) -> list[dict[str, Any]]:
    """Run paired endpoint tests separately for each skill and condition."""

    condition_set = set(conditions)
    indexed: dict[tuple[str, str, str, str], bool] = {}
    cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        condition = str(row["prompt_condition"])
        if condition not in condition_set:
            continue
        skill = str(row["skill_id"])
        case = str(row["semantic_case_hash"])
        endpoint = str(row["endpoint"])
        key = (skill, case, condition, endpoint)
        if key in indexed:
            raise ValueError(f"duplicate prediction panel key: {key}")
        indexed[key] = bool(row["exact_tuple_correct"])
        cases[(skill, condition)].add(case)

    tests = []
    for (skill, condition), case_ids in sorted(cases.items()):
        random_only = 0
        staged_only = 0
        both_correct = 0
        both_incorrect = 0
        for case in sorted(case_ids):
            random_correct = indexed[
                (skill, case, condition, "random_mix_order")
            ]
            staged_correct = indexed[
                (skill, case, condition, "staged_order")
            ]
            if random_correct and staged_correct:
                both_correct += 1
            elif random_correct:
                random_only += 1
            elif staged_correct:
                staged_only += 1
            else:
                both_incorrect += 1
        count = len(case_ids)
        tests.append(
            {
                "skill_id": skill,
                "prompt_condition": condition,
                "case_count": count,
                "both_correct": both_correct,
                "random_only_correct": random_only,
                "staged_only_correct": staged_only,
                "both_incorrect": both_incorrect,
                "delta": (staged_only - random_only) / count,
                "p_exact_two_sided": exact_mcnemar_pvalue(
                    random_only, staged_only
                ),
            }
        )
    return tests


def holm_adjust(
    tests: Iterable[Mapping[str, Any]],
    *,
    p_field: str = "p_exact_two_sided",
) -> list[dict[str, Any]]:
    """Apply a monotone Holm family-wise correction."""

    rows = [dict(row) for row in tests]
    order = sorted(
        range(len(rows)),
        key=lambda index: (float(rows[index][p_field]), index),
    )
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order, start=1):
        adjusted = min(
            1.0, (total - rank + 1) * float(rows[index][p_field])
        )
        running = max(running, adjusted)
        rows[index]["p_holm"] = running
    return rows


def schema_validity_report(
    records: Iterable[Mapping[str, Any]],
    *,
    endpoint_min: float,
    condition_min: float,
) -> dict[str, Any]:
    """Evaluate the two pre-registered schema-validity guard levels."""

    endpoint_cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    condition_cells: dict[tuple[str, str, str], list[bool]] = defaultdict(
        list
    )
    for row in records:
        skill = str(row["skill_id"])
        endpoint = str(row["endpoint"])
        condition = str(row["prompt_condition"])
        value = bool(row["schema_valid"])
        endpoint_cells[(skill, endpoint)].append(value)
        condition_cells[(skill, endpoint, condition)].append(value)

    endpoint_rates = []
    for (skill, endpoint), values in sorted(endpoint_cells.items()):
        rate = sum(values) / len(values)
        endpoint_rates.append(
            {
                "skill_id": skill,
                "endpoint": endpoint,
                "record_count": len(values),
                "schema_valid_count": sum(values),
                "schema_valid_rate": rate,
                "threshold": endpoint_min,
                "pass": rate >= endpoint_min,
            }
        )
    condition_rates = []
    for (skill, endpoint, condition), values in sorted(
        condition_cells.items()
    ):
        rate = sum(values) / len(values)
        condition_rates.append(
            {
                "skill_id": skill,
                "endpoint": endpoint,
                "prompt_condition": condition,
                "record_count": len(values),
                "schema_valid_count": sum(values),
                "schema_valid_rate": rate,
                "threshold": condition_min,
                "pass": rate >= condition_min,
            }
        )
    return {
        "pass": all(row["pass"] for row in endpoint_rates)
        and all(row["pass"] for row in condition_rates),
        "per_skill_endpoint_min_threshold": endpoint_min,
        "per_skill_endpoint_condition_min_threshold": condition_min,
        "minimum_per_skill_endpoint_rate": min(
            row["schema_valid_rate"] for row in endpoint_rates
        ),
        "minimum_per_skill_endpoint_condition_rate": min(
            row["schema_valid_rate"] for row in condition_rates
        ),
        "failed_per_skill_endpoint_count": sum(
            not row["pass"] for row in endpoint_rates
        ),
        "failed_per_skill_endpoint_condition_count": sum(
            not row["pass"] for row in condition_rates
        ),
        "per_skill_endpoint": endpoint_rates,
        "per_skill_endpoint_condition": condition_rates,
    }


def classify_frozen_decision(
    *,
    primary: Mapping[str, Any],
    case_bootstrap: Mapping[str, float],
    family_bootstrap: Mapping[str, float],
    construct_effects: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[str, Any],
    minimal: Mapping[str, Any],
    no_skill: Mapping[str, Any],
    schema_guards_pass: bool,
    data_valid: bool = True,
) -> dict[str, Any]:
    """Apply the exact pre-registered decision priority and boundaries."""

    per_skill_deltas = [
        float(value["delta"]) for value in primary["per_skill"].values()
    ]
    construct_deltas = [
        float(value["delta"]) for value in construct_effects.values()
    ]
    canonical_per_skill = [
        float(value["delta"]) for value in canonical["per_skill"].values()
    ]
    pass_checks = {
        "primary_macro_ge_0_05": float(primary["macro_delta"]) >= 0.05,
        "case_bootstrap_lower_gt_0": float(
            case_bootstrap["ci95_lower"]
        )
        > 0,
        "family_bootstrap_lower_gt_0": float(
            family_bootstrap["ci95_lower"]
        )
        > 0,
        "at_least_two_skill_deltas_ge_0_05": sum(
            value >= 0.05 for value in per_skill_deltas
        )
        >= 2,
        "every_skill_delta_gt_minus_0_05": all(
            value > -0.05 for value in per_skill_deltas
        ),
        "every_construct_delta_gt_minus_0_05": all(
            value > -0.05 for value in construct_deltas
        ),
        "canonical_macro_ge_minus_0_02": float(canonical["delta"])
        >= -0.02,
        "every_skill_canonical_delta_gt_minus_0_05": all(
            value > -0.05 for value in canonical_per_skill
        ),
        "minimal_macro_gt_minus_0_05": float(minimal["delta"]) > -0.05,
        "no_skill_macro_gt_minus_0_05": float(no_skill["delta"]) > -0.05,
        "schema_validity_guards_pass": bool(schema_guards_pass),
    }
    fail_checks = {
        "primary_macro_le_0": float(primary["macro_delta"]) <= 0,
        "at_least_two_skill_deltas_lt_0": sum(
            value < 0 for value in per_skill_deltas
        )
        >= 2,
        "any_skill_delta_le_minus_0_05": any(
            value <= -0.05 for value in per_skill_deltas
        ),
        "any_construct_delta_le_minus_0_05": any(
            value <= -0.05 for value in construct_deltas
        ),
        "canonical_macro_le_minus_0_05": float(canonical["delta"])
        <= -0.05,
        "any_skill_canonical_delta_le_minus_0_05": any(
            value <= -0.05 for value in canonical_per_skill
        ),
        "minimal_macro_le_minus_0_05": float(minimal["delta"]) <= -0.05,
        "no_skill_macro_le_minus_0_05": float(no_skill["delta"]) <= -0.05,
    }
    mixed_checks = {
        "positive_and_negative_skill_effects": (
            any(value >= 0.05 for value in per_skill_deltas)
            and any(value < 0 for value in per_skill_deltas)
        ),
        "skill_effect_range_ge_0_10": (
            max(per_skill_deltas) - min(per_skill_deltas) >= 0.10
        ),
    }
    if not data_valid:
        state = "DATA_INVALID"
    elif all(pass_checks.values()):
        state = "PASS"
    elif any(fail_checks.values()):
        state = "FAIL"
    elif any(mixed_checks.values()):
        state = "MIXED"
    else:
        state = "INCONCLUSIVE"
    return {
        "state": state,
        "priority": [
            "DATA_INVALID",
            "PASS",
            "FAIL",
            "MIXED",
            "INCONCLUSIVE",
        ],
        "pass_all": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "fail_any": any(fail_checks.values()),
        "fail_checks": fail_checks,
        "mixed_any": any(mixed_checks.values()),
        "mixed_checks": mixed_checks,
    }
