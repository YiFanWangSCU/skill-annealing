"""Build mechanism ablation eval prompts for Stage C refund_decision."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_long_context_stage_c_contract_eval import (  # noqa: E402
    build_messages,
    dedupe_source_records,
)
from skill_annealing.refund_decision.rule_engine import decide  # noqa: E402
from skill_annealing.refund_decision.skill import (  # noqa: E402
    BUSINESS_RULES,
    EDGE_CASES,
    EXAMPLES,
    OUTPUT_SCHEMA,
    PARTIAL_BUSINESS_RULES,
    TASK_DEFINITION,
)


EXPOSURES = ("full", "partial", "minimal", "no_skill")
ABLATION_TYPES = {
    "original",
    "rule_deleted_full",
    "schema_deleted_full",
    "examples_deleted_full",
    "schema_only_full",
    "rules_only_full",
    "no_skill_current_order_only",
}
FACT_MUTATIONS = {
    "days_since_delivery_6": {"days_since_delivery": 6},
    "days_since_delivery_7": {"days_since_delivery": 7},
    "days_since_delivery_8": {"days_since_delivery": 8},
    "order_amount_999": {"order_amount": 999},
    "order_amount_1000": {"order_amount": 1000},
    "order_amount_1001": {"order_amount": 1001},
    "refund_count_30d_2": {"refund_count_30d": 2},
    "refund_count_30d_3": {"refund_count_30d": 3},
    "evidence_conflict_false": {"evidence_conflict": False},
    "evidence_conflict_true": {"evidence_conflict": True},
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _strip_system(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
        if message.get("role") != "system"
    ]


def _replace_system(
    messages: Sequence[Mapping[str, str]],
    content: str,
) -> list[dict[str, str]]:
    rest = _strip_system(messages)
    return [{"role": "system", "content": content}, *rest] if content else rest


def _ablate_messages(
    messages: Sequence[Mapping[str, str]],
    ablation_type: str,
) -> list[dict[str, str]]:
    if ablation_type == "original":
        return [
            {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
            for message in messages
        ]
    if ablation_type == "rule_deleted_full":
        return _replace_system(
            messages,
            "\n\n".join([TASK_DEFINITION, OUTPUT_SCHEMA, EXAMPLES, EDGE_CASES]),
        )
    if ablation_type == "schema_deleted_full":
        return _replace_system(
            messages,
            "\n\n".join([TASK_DEFINITION, BUSINESS_RULES, EXAMPLES, EDGE_CASES]),
        )
    if ablation_type == "examples_deleted_full":
        return _replace_system(
            messages,
            "\n\n".join([TASK_DEFINITION, OUTPUT_SCHEMA, BUSINESS_RULES, EDGE_CASES]),
        )
    if ablation_type == "schema_only_full":
        return _replace_system(messages, "\n\n".join([TASK_DEFINITION, OUTPUT_SCHEMA]))
    if ablation_type == "rules_only_full":
        return _replace_system(
            messages,
            "\n\n".join([TASK_DEFINITION, BUSINESS_RULES, PARTIAL_BUSINESS_RULES]),
        )
    if ablation_type == "no_skill_current_order_only":
        return _strip_system(messages)
    raise ValueError(f"unknown ablation_type: {ablation_type}")


def _base_record(
    source: Mapping[str, Any],
    *,
    exposure: str,
    seed: int,
    noise_repeats: int,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{source['sample_id']}:{exposure}:mechanism")
    fields = dict(source["fields"])
    return {
        "sample_id": source["sample_id"],
        "skill_id": "refund_decision",
        "prompt_exposure": exposure,
        "input_variant": "mechanism_ablation_stage_c_contract",
        "split": "eval",
        "messages": build_messages(
            {**source, "fields": fields},
            exposure,
            rng=rng,
            noise_repeats=noise_repeats,
        ),
        "fields": fields,
        "target_output": dict(source["target_output"]),
        "hard_tags": list(source.get("hard_tags", [])) + ["mechanism_ablation"],
        "active_order_id": f"C-ACTIVE-{source['sample_id']}",
        "noise_repeats": noise_repeats,
        "backend_contract": "current_order_guaranteed",
    }


def build_ablation_records(
    source_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    noise_repeats: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in dedupe_source_records(source_records):
        for exposure in EXPOSURES:
            original = _base_record(
                source,
                exposure=exposure,
                seed=seed,
                noise_repeats=noise_repeats,
            )
            for ablation_type in sorted(ABLATION_TYPES):
                if ablation_type != "original" and exposure != "full":
                    continue
                record = dict(original)
                record["sample_id"] = f"{source['sample_id']}__{ablation_type}__{exposure}"
                record["ablation_type"] = ablation_type
                record["mutation_type"] = "none"
                record["messages"] = _ablate_messages(
                    original["messages"],
                    ablation_type,
                )
                records.append(record)
    return records


def build_fact_mutation_records(
    source_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    noise_repeats: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in dedupe_source_records(source_records):
        for mutation_type, overrides in FACT_MUTATIONS.items():
            mutated_fields = {**dict(source["fields"]), **overrides}
            mutated_source = {
                **source,
                "fields": mutated_fields,
                "target_output": decide(mutated_fields).to_dict(),
            }
            for exposure in ("minimal", "no_skill", "full"):
                record = _base_record(
                    mutated_source,
                    exposure=exposure,
                    seed=seed,
                    noise_repeats=noise_repeats,
                )
                record["sample_id"] = f"{source['sample_id']}__{mutation_type}__{exposure}"
                record["ablation_type"] = "fact_mutation"
                record["mutation_type"] = mutation_type
                record["hard_tags"] = list(record["hard_tags"]) + [
                    "fact_mutation",
                    mutation_type,
                ]
                records.append(record)
    return records


def build_records(
    source_records: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    seed: int,
    noise_repeats: int,
) -> list[dict[str, Any]]:
    selected = dedupe_source_records(source_records)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if sample_count:
        selected = selected[:sample_count]
    records = build_ablation_records(
        selected,
        seed=seed,
        noise_repeats=noise_repeats,
    )
    records.extend(
        build_fact_mutation_records(
            selected,
            seed=seed,
            noise_repeats=noise_repeats,
        )
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mechanism ablation eval prompts.",
    )
    parser.add_argument("--source-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--noise-repeats", type=int, default=28)
    args = parser.parse_args()

    if args.sample_count < 0:
        raise SystemExit("--sample-count must be >= 0; use 0 for all source samples")
    if args.noise_repeats < 0:
        raise SystemExit("--noise-repeats must be >= 0")

    source_records = load_jsonl(args.source_eval)
    records = build_records(
        source_records,
        sample_count=args.sample_count,
        seed=args.seed,
        noise_repeats=args.noise_repeats,
    )
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "output": args.output,
                "record_count": len(records),
                "source_count": len({record["sample_id"].split("__", 1)[0] for record in records}),
                "ablation_types": sorted(ABLATION_TYPES),
                "fact_mutations": sorted(FACT_MUTATIONS),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
