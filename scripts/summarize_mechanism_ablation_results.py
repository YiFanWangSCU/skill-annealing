"""Summarize mechanism ablation prediction files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_annealing.refund_decision.evaluator import (  # noqa: E402
    evaluate_records,
    parse_prediction,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def group_key(record: Mapping[str, Any]) -> str:
    return "/".join(
        [
            str(record.get("ablation_type", "original")),
            str(record.get("prompt_exposure", "unknown")),
            str(record.get("mutation_type", "none")),
        ]
    )


def _metric(groups: Mapping[str, Mapping[str, Any]], key: str, metric: str) -> float:
    return float(groups.get(key, {}).get(metric, 0.0))


def dependency_gaps(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    original = "original/full/none"
    rule_deleted = "rule_deleted_full/full/none"
    schema_deleted = "schema_deleted_full/full/none"
    no_skill = "original/no_skill/none"
    return {
        "rule_dependency_gap": _metric(groups, original, "overall_score")
        - _metric(groups, rule_deleted, "overall_score"),
        "schema_dependency_gap": _metric(groups, original, "schema_valid_rate")
        - _metric(groups, schema_deleted, "schema_valid_rate"),
        "retention_gap": _metric(groups, original, "overall_score")
        - _metric(groups, no_skill, "overall_score"),
        "full_to_no_skill_gap": _metric(groups, original, "overall_score")
        - _metric(groups, no_skill, "overall_score"),
    }


def fact_mutation_accuracy(records: list[Mapping[str, Any]]) -> float:
    fact_records = [
        record
        for record in records
        if record.get("ablation_type") == "fact_mutation"
        or record.get("mutation_type", "none") != "none"
    ]
    if not fact_records:
        return 0.0
    correct = 0
    for record in fact_records:
        parsed, json_valid = parse_prediction(record)
        if json_valid and parsed.get("decision") == record["target_output"].get("decision"):
            correct += 1
    return correct / len(fact_records)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[group_key(record)].append(record)

    groups: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(buckets.items()):
        metrics = evaluate_records(rows)
        if key.startswith("fact_mutation/"):
            metrics["fact_mutation_accuracy"] = fact_mutation_accuracy(rows)
        groups[key] = metrics

    return {
        "record_count": len(records),
        "groups": groups,
        "dependency_gaps": dependency_gaps(groups),
        "fact_mutation_accuracy": fact_mutation_accuracy(records),
    }


def summarize_prediction_files(model_paths: Mapping[str, str | Path]) -> dict[str, Any]:
    models = {}
    for model_name, path in model_paths.items():
        records = [{**record, "model_name": model_name} for record in load_jsonl(path)]
        models[model_name] = summarize_records(records)
    return {"models": models}


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model args must use name=path")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("model name must not be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize mechanism ablation predictions.",
    )
    parser.add_argument("--model", action="append", required=True, help="name=predictions.jsonl")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    model_paths = dict(parse_model_arg(item) for item in args.model)
    summary = summarize_prediction_files(model_paths)
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
