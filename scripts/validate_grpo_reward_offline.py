"""Offline validation for the refund_decision GRPO reward design.

The script scores existing prediction JSONL files. It is intentionally
CPU-only so reward design can be checked before launching GRPO training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_annealing.refund_decision.evaluator import (
    parse_prediction,
    reason_code_f1,
)
from skill_annealing.refund_decision.schema import validate_output_schema


EXPOSURE_ORDER = ("full", "partial", "minimal", "no_skill")
PAIR_WEIGHTS = {
    ("full", "partial"): 0.40,
    ("partial", "minimal"): 0.30,
    ("minimal", "no_skill"): 0.20,
    ("full", "no_skill"): 0.10,
}
HARD_REASON_CODES = {
    "manual_review_required",
    "evidence_conflict",
    "order_info_conflict",
    "high_value_order",
    "frequent_refund_user",
    "quality_issue_no_evidence",
    "fresh_goods_no_reason_denied",
    "customized_goods_no_reason_denied",
    "virtual_goods_denied",
}
HARD_TAG_FRAGMENTS = (
    "hard",
    "risk",
    "conflict",
    "manual",
    "special",
    "boundary",
    "logistics",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parsed_output(record: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    parsed, json_valid = parse_prediction(record)
    schema_valid = bool(json_valid and not validate_output_schema(parsed))
    return parsed, json_valid, schema_valid


def gold_compatible(
    parsed: Mapping[str, Any], gold: Mapping[str, Any], schema_valid: bool
) -> bool:
    return bool(
        schema_valid
        and parsed.get("decision") == gold.get("decision")
        and parsed.get("need_human") == gold.get("need_human")
    )


def core_similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return (
        0.45 * float(a.get("decision") == b.get("decision"))
        + 0.20 * float(a.get("need_human") == b.get("need_human"))
        + 0.15 * float(a.get("risk_level") == b.get("risk_level"))
        + 0.20
        * reason_code_f1(
            list(a.get("reason_codes", []) or []),
            list(b.get("reason_codes", []) or []),
        )
    )


def pair_consistency_reward(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[float, bool, bool, float]:
    gold = left["target_output"]
    left_parsed = left["_parsed_output"]
    right_parsed = right["_parsed_output"]
    left_compatible = gold_compatible(left_parsed, gold, left["_schema_valid"])
    right_compatible = gold_compatible(right_parsed, gold, right["_schema_valid"])
    similarity = core_similarity(left_parsed, right_parsed)

    if left_compatible and right_compatible:
        reward = similarity
    elif left_compatible or right_compatible:
        reward = 0.25 * similarity
    else:
        reward = -0.10 if similarity > 0.90 else 0.0
    return reward, left_compatible, right_compatible, similarity


def group_consistency(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_exposure = {record.get("prompt_exposure"): record for record in records}
    weighted_total = 0.0
    weight_total = 0.0
    wrong_high_similarity_pairs = 0
    available_pairs = 0

    for pair, weight in PAIR_WEIGHTS.items():
        left = by_exposure.get(pair[0])
        right = by_exposure.get(pair[1])
        if not left or not right:
            continue
        reward, left_ok, right_ok, similarity = pair_consistency_reward(left, right)
        weighted_total += weight * reward
        weight_total += weight
        available_pairs += 1
        if not left_ok and not right_ok and similarity > 0.90:
            wrong_high_similarity_pairs += 1

    reward = weighted_total / weight_total if weight_total else 0.0
    return {
        "available_pairs": available_pairs,
        "exposure_consistency_reward": reward,
        "wrong_high_similarity_pairs": wrong_high_similarity_pairs,
    }


def is_hard_case(record: Mapping[str, Any]) -> bool:
    gold = record["target_output"]
    if gold.get("need_human") is True:
        return True
    if set(gold.get("reason_codes", []) or []) & HARD_REASON_CODES:
        return True
    tags = [str(tag).lower() for tag in record.get("hard_tags", []) or []]
    return any(fragment in tag for tag in tags for fragment in HARD_TAG_FRAGMENTS)


def score_record(record: Mapping[str, Any], consistency_reward: float) -> dict[str, float]:
    gold = record["target_output"]
    parsed = record["_parsed_output"]
    schema_valid = bool(record["_schema_valid"])
    compatible = gold_compatible(parsed, gold, schema_valid)
    false_manual_review = bool(
        schema_valid
        and parsed.get("decision") == "manual_review"
        and gold.get("decision") != "manual_review"
    )

    components = {
        "decision_reward": float(
            schema_valid and parsed.get("decision") == gold.get("decision")
        ),
        "need_human_reward": float(
            schema_valid and parsed.get("need_human") == gold.get("need_human")
        ),
        "risk_level_reward": float(
            schema_valid and parsed.get("risk_level") == gold.get("risk_level")
        ),
        "reason_code_reward": reason_code_f1(
            list(gold.get("reason_codes", []) or []),
            list(parsed.get("reason_codes", []) or []) if schema_valid else [],
        ),
        "schema_reward": float(schema_valid),
        "exposure_consistency_reward": consistency_reward,
        "hard_case_bonus": float(is_hard_case(record) and compatible),
        "false_manual_review_penalty": float(false_manual_review),
    }
    components["total_reward"] = (
        0.30 * components["decision_reward"]
        + 0.15 * components["need_human_reward"]
        + 0.10 * components["risk_level_reward"]
        + 0.15 * components["reason_code_reward"]
        + 0.05 * components["schema_reward"]
        + 0.15 * components["exposure_consistency_reward"]
        + 0.10 * components["hard_case_bonus"]
        - 0.10 * components["false_manual_review_penalty"]
    )
    return components


def summarize_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = []
    for record in records:
        parsed, json_valid, schema_valid = parsed_output(record)
        enriched.append(
            {
                **record,
                "_parsed_output": parsed,
                "_json_valid": json_valid,
                "_schema_valid": schema_valid,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in enriched:
        groups[str(record.get("sample_id", "unknown"))].append(record)

    group_scores = {
        sample_id: group_consistency(group_records)
        for sample_id, group_records in groups.items()
    }

    record_scores = []
    for record in enriched:
        sample_id = str(record.get("sample_id", "unknown"))
        record_scores.append(
            score_record(
                record,
                float(group_scores[sample_id]["exposure_consistency_reward"]),
            )
        )

    exposures: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record, score in zip(enriched, record_scores):
        exposures[str(record.get("prompt_exposure", "unknown"))].append(score)

    component_names = sorted(record_scores[0]) if record_scores else []
    component_means = {
        name: mean(score[name] for score in record_scores)
        for name in component_names
    }
    by_exposure = {
        exposure: {
            "count": len(scores),
            "total_reward": mean(score["total_reward"] for score in scores),
            "decision_reward": mean(score["decision_reward"] for score in scores),
            "schema_reward": mean(score["schema_reward"] for score in scores),
        }
        for exposure, scores in sorted(exposures.items())
    }

    return {
        "component_means": component_means,
        "count": len(record_scores),
        "exposure_consistency_reward": mean(
            item["exposure_consistency_reward"] for item in group_scores.values()
        )
        if group_scores
        else 0.0,
        "groups": len(group_scores),
        "by_exposure": by_exposure,
        "wrong_high_similarity_groups": sum(
            1
            for item in group_scores.values()
            if item["wrong_high_similarity_pairs"] > 0
        ),
    }


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def default_models() -> list[tuple[str, Path]]:
    root = Path("experiments/server_results_20260508_stage_c_v4_sft")
    return [
        (
            "stagec_v4_full_skill_sft",
            root
            / "stagec_v4_full_skill_sft_stage_c_v4_on_long_context_stage_c_contract_20260508_predictions.jsonl",
        ),
        (
            "stagec_v4_random_mix_sft",
            root
            / "stagec_v4_random_mix_sft_stage_c_v4_on_long_context_stage_c_contract_20260508_predictions.jsonl",
        ),
        (
            "stagec_v4_annealed_sft",
            root
            / "stagec_v4_annealed_sft_stage_c_v4_on_long_context_stage_c_contract_20260508_predictions.jsonl",
        ),
    ]


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Offline GRPO Reward Validation",
        "",
        "Date: 2026-05-19",
        "",
        "This CPU-only validation applies the proposed GRPO-v1 reward to already completed Stage C v4 predictions.",
        "",
        "## Ranking",
        "",
        "| rank | model | total_reward | decision_reward | consistency | false_manual_review_penalty | wrong_high_similarity_groups |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(summary["ranking"], start=1):
        model = item["model"]
        metrics = summary["models"][model]
        means = metrics["component_means"]
        lines.append(
            "| {rank} | `{model}` | {total:.4f} | {decision:.4f} | {consistency:.4f} | {penalty:.4f} | {wrong} |".format(
                rank=index,
                model=model,
                total=means["total_reward"],
                decision=means["decision_reward"],
                consistency=metrics["exposure_consistency_reward"],
                penalty=means["false_manual_review_penalty"],
                wrong=metrics["wrong_high_similarity_groups"],
            )
        )
    lines.extend(["", "## Interpretation", ""])
    if summary["phase_a_gate_passed"]:
        lines.append(
            "The reward ranking passes the Phase A gate: random-mix > annealed > full-skill."
        )
    else:
        lines.append(
            "The reward ranking does not pass the Phase A gate. Do not launch GRPO before revisiting the reward design."
        )
    lines.extend(["", "## Per-Exposure Reward", ""])
    for model, metrics in summary["models"].items():
        lines.extend(
            [
                f"### {model}",
                "",
                "| exposure | count | total_reward | decision_reward | schema_reward |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for exposure in EXPOSURE_ORDER:
            if exposure not in metrics["by_exposure"]:
                continue
            values = metrics["by_exposure"][exposure]
            lines.append(
                "| {exposure} | {count} | {total:.4f} | {decision:.4f} | {schema:.4f} |".format(
                    exposure=exposure,
                    count=values["count"],
                    total=values["total_reward"],
                    decision=values["decision_reward"],
                    schema=values["schema_reward"],
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GRPO reward design on existing refund_decision predictions."
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model input as name=predictions.jsonl. Defaults to Stage C v4 models.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/grpo_reward_validation_20260519",
        help="Directory for summary.json and summary.md.",
    )
    args = parser.parse_args()

    model_paths = [parse_model_arg(item) for item in args.model] if args.model else default_models()
    models = {}
    for name, path in model_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        models[name] = summarize_model(load_jsonl(path))

    ranking = sorted(
        (
            {
                "model": name,
                "total_reward": metrics["component_means"]["total_reward"],
            }
            for name, metrics in models.items()
        ),
        key=lambda item: item["total_reward"],
        reverse=True,
    )
    order = [item["model"] for item in ranking]
    phase_a_gate_passed = (
        order.index("stagec_v4_random_mix_sft")
        < order.index("stagec_v4_annealed_sft")
        < order.index("stagec_v4_full_skill_sft")
        if all(
            name in order
            for name in (
                "stagec_v4_random_mix_sft",
                "stagec_v4_annealed_sft",
                "stagec_v4_full_skill_sft",
            )
        )
        else False
    )

    summary = {
        "models": models,
        "phase_a_gate_passed": phase_a_gate_passed,
        "ranking": ranking,
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
