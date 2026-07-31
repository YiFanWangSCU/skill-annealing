"""Evaluation utilities for refund_decision predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .prompt_builder import EXPOSURE_ORDER
from .schema import validate_output_schema


def evaluate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [score_record(record) for record in records]
    return aggregate_scores(scores)


def evaluate_by_exposure(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        exposure = str(record.get("prompt_exposure", "unknown"))
        groups[exposure].append(record)

    metrics = {
        exposure: evaluate_records(groups[exposure])
        for exposure in EXPOSURE_ORDER
        if exposure in groups
    }
    metrics["retention"] = retention_metrics(metrics)
    return metrics


def score_record(record: Mapping[str, Any]) -> dict[str, float]:
    gold = record["target_output"]
    parsed, json_valid = parse_prediction(record)
    schema_valid = bool(json_valid and not validate_output_schema(parsed))

    return {
        "json_valid": float(json_valid),
        "schema_valid": float(schema_valid),
        "decision_correct": float(
            schema_valid and parsed.get("decision") == gold.get("decision")
        ),
        "risk_level_correct": float(
            schema_valid and parsed.get("risk_level") == gold.get("risk_level")
        ),
        "need_human_correct": float(
            schema_valid and parsed.get("need_human") == gold.get("need_human")
        ),
        "reason_code_f1": reason_code_f1(
            gold.get("reason_codes", []),
            parsed.get("reason_codes", []) if schema_valid else [],
        ),
    }


def aggregate_scores(scores: list[dict[str, float]]) -> dict[str, Any]:
    if not scores:
        return {
            "count": 0,
            "json_valid_rate": 0.0,
            "schema_valid_rate": 0.0,
            "decision_accuracy": 0.0,
            "risk_level_accuracy": 0.0,
            "need_human_accuracy": 0.0,
            "reason_code_f1": 0.0,
            "overall_score": 0.0,
        }

    count = len(scores)

    def avg(key: str) -> float:
        return sum(score[key] for score in scores) / count

    metrics = {
        "count": count,
        "json_valid_rate": avg("json_valid"),
        "schema_valid_rate": avg("schema_valid"),
        "decision_accuracy": avg("decision_correct"),
        "risk_level_accuracy": avg("risk_level_correct"),
        "need_human_accuracy": avg("need_human_correct"),
        "reason_code_f1": avg("reason_code_f1"),
    }
    metrics["overall_score"] = sum(
        metrics[key]
        for key in [
            "decision_accuracy",
            "risk_level_accuracy",
            "need_human_accuracy",
            "reason_code_f1",
            "schema_valid_rate",
        ]
    ) / 5
    return metrics


def retention_metrics(metrics_by_exposure: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    full = float(metrics_by_exposure.get("full", {}).get("overall_score", 0.0))
    no_skill = float(metrics_by_exposure.get("no_skill", {}).get("overall_score", 0.0))
    curve = {
        exposure: metrics_by_exposure[exposure]["overall_score"]
        for exposure in EXPOSURE_ORDER
        if exposure in metrics_by_exposure
    }
    return {
        "skill_retention_curve": curve,
        "skill_degradation_gap": full - no_skill,
        "skill_retention_score": no_skill / full if full else 0.0,
    }


def parse_prediction(record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if "predicted_output" in record:
        value = record["predicted_output"]
    elif "prediction" in record:
        value = record["prediction"]
    elif "target_output" in record and record.get("oracle", False):
        value = record["target_output"]
    else:
        return {}, False

    if isinstance(value, dict):
        return dict(value), True
    if not isinstance(value, str):
        return {}, False

    text = _strip_code_fence(value.strip())
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}, False
        try:
            return json.loads(match.group(0)), True
        except json.JSONDecodeError:
            return {}, False


def reason_code_f1(gold_codes: list[str], predicted_codes: list[str]) -> float:
    gold = set(gold_codes)
    pred = set(predicted_codes)
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    return (2 * overlap) / (len(gold) + len(pred))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate refund_decision predictions.")
    parser.add_argument("--predictions", required=True, help="JSONL with target_output and prediction/predicted_output.")
    parser.add_argument("--by-exposure", action="store_true")
    parser.add_argument("--oracle", action="store_true", help="Use target_output as prediction for a smoke test.")
    parser.add_argument("--output", help="Optional metrics JSON path.")
    args = parser.parse_args()

    records = load_jsonl(args.predictions)
    if args.oracle:
        records = [{**record, "oracle": True} for record in records]

    metrics = evaluate_by_exposure(records) if args.by_exposure else evaluate_records(records)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
