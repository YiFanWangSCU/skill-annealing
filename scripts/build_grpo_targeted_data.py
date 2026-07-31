"""Build a targeted prompt set for refund_decision GRPO experiments.

The output is not an SFT dataset: assistant answers are not appended to the
messages. Gold labels are kept as metadata for reward computation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_long_context_stage_c_contract_eval import build_messages  # noqa: E402
from skill_annealing.refund_decision.data_generator import (  # noqa: E402
    PRODUCT_PROFILES,
    render_user_request,
)
from skill_annealing.refund_decision.rule_engine import decide  # noqa: E402


EXPOSURES = ("full", "partial", "minimal", "no_skill")
CATEGORY_WEIGHTS = (
    ("conflict_manual_review", 0.40),
    ("backend_contradiction", 0.20),
    ("quality_without_evidence", 0.15),
    ("special_goods_boundary", 0.15),
    ("logistics_ambiguity", 0.10),
)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def weighted_choice(rng: random.Random, weights: Sequence[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in weights)
    point = rng.random() * total
    upto = 0.0
    for value, weight in weights:
        upto += weight
        if point <= upto:
            return value
    return weights[-1][0]


def base_fields(product_type: str, **overrides: Any) -> dict[str, Any]:
    fields = {
        "product_type": product_type,
        "days_since_delivery": 3,
        "is_used": False,
        "is_resellable": True,
        "has_quality_issue": False,
        "has_evidence": False,
        "order_amount": 199,
        "refund_count_30d": 0,
        "user_claim": "no_reason",
        "order_status": "delivered",
        "seller_fault": False,
        "logistics_issue": False,
        "evidence_conflict": False,
        "order_info_conflict": False,
        **PRODUCT_PROFILES[product_type],
    }
    fields.update(overrides)
    return fields


def sample_fields_for_category(rng: random.Random, category: str) -> dict[str, Any]:
    if category == "conflict_manual_review":
        return base_fields(
            rng.choice(["clothing", "electronics", "home", "fresh", "customized"]),
            days_since_delivery=rng.choice([1, 3, 7, 8]),
            has_quality_issue=rng.choice([True, False]),
            has_evidence=True,
            evidence_conflict=rng.choice([True, False]),
            order_info_conflict=rng.choice([False, True]),
            order_amount=rng.choice([999, 1000, 1299, 2499]),
            refund_count_30d=rng.choice([2, 3, 4]),
            user_claim=rng.choice(["strong_complaint", "quality_issue", "claims_unused_but_used"]),
            is_used=rng.choice([False, True]),
            is_resellable=rng.choice([False, True]),
        )
    if category == "backend_contradiction":
        return base_fields(
            rng.choice(["clothing", "electronics", "home"]),
            days_since_delivery=rng.choice([1, 3, 6]),
            is_used=True,
            is_resellable=False,
            user_claim="claims_unused_but_used",
            order_amount=rng.choice([99, 299, 899]),
            refund_count_30d=rng.choice([0, 1, 2]),
        )
    if category == "quality_without_evidence":
        return base_fields(
            rng.choice(["clothing", "electronics", "home", "fresh", "customized"]),
            days_since_delivery=rng.choice([0, 1, 3, 7, 8]),
            has_quality_issue=True,
            has_evidence=False,
            user_claim=rng.choice(["quality_issue", "strong_complaint"]),
            is_used=rng.choice([False, True]),
            is_resellable=rng.choice([False, True]),
            order_amount=rng.choice([99, 399, 899]),
            refund_count_30d=rng.choice([0, 1, 2]),
        )
    if category == "special_goods_boundary":
        product_type = rng.choice(["fresh", "customized", "virtual"])
        quality_exception = rng.random() < 0.45
        return base_fields(
            product_type,
            days_since_delivery=rng.choice([0, 1, 3, 8]),
            has_quality_issue=quality_exception,
            has_evidence=quality_exception and rng.random() < 0.80,
            seller_fault=quality_exception and rng.random() < 0.40,
            user_claim=rng.choice(["no_reason", "quality_issue", "strong_complaint"]),
            is_used=rng.choice([False, True]) if product_type != "virtual" else True,
            is_resellable=rng.choice([False, True]) if product_type != "virtual" else False,
            order_amount=rng.choice([99, 299, 899]),
            refund_count_30d=rng.choice([0, 1, 2]),
        )
    if category == "logistics_ambiguity":
        return base_fields(
            rng.choice(["clothing", "electronics", "home"]),
            days_since_delivery=rng.choice([0, 1, 3, 8]),
            logistics_issue=True,
            has_evidence=rng.choice([False, False, True]),
            seller_fault=rng.choice([False, False, True]),
            order_status=rng.choice(["not_delivered", "delivered"]),
            user_claim="logistics_issue",
            order_amount=rng.choice([99, 399, 899]),
            refund_count_30d=rng.choice([0, 1, 2]),
        )
    raise ValueError(f"unknown category: {category!r}")


def hard_tags_for(fields: Mapping[str, Any], category: str) -> list[str]:
    tags = [f"grpo_target_{category}"]
    if fields.get("evidence_conflict"):
        tags.append("evidence_conflict")
    if fields.get("order_info_conflict") or fields.get("user_claim") == "claims_unused_but_used":
        tags.append("order_info_conflict")
    if int(fields.get("order_amount", 0)) >= 1000:
        tags.append("high_value_order")
    if int(fields.get("refund_count_30d", 0)) >= 3:
        tags.append("frequent_refund_user")
    if fields.get("is_fresh") or fields.get("is_customized") or fields.get("is_virtual"):
        tags.append("special_goods")
    if fields.get("logistics_issue"):
        tags.append("logistics_issue")
    if fields.get("has_quality_issue") and not fields.get("has_evidence"):
        tags.append("quality_without_evidence")
    return list(dict.fromkeys(tags))


def build_source_samples(sample_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for index in range(1, sample_count + 1):
        category = weighted_choice(rng, CATEGORY_WEIGHTS)
        fields = sample_fields_for_category(rng, category)
        user_input, evidence_description = render_user_request(fields, rng)
        sample = {
            "sample_id": f"refund_grpo_targeted_{index:06d}",
            "skill_id": "refund_decision",
            "grpo_category": category,
            "fields": fields,
            "user_input": user_input,
            "target_output": decide(fields).to_dict(),
            "hard_tags": hard_tags_for(fields, category),
        }
        if evidence_description:
            sample["evidence_description"] = evidence_description
        samples.append(sample)
    return samples


def prompt_record(
    source: Mapping[str, Any],
    exposure: str,
    *,
    seed: int,
    noise_repeats: int,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{source['sample_id']}:{exposure}")
    plain_user_messages = [
        {"role": "user", "content": source.get("user_input", "")},
    ]
    if source.get("evidence_description"):
        plain_user_messages.append(
            {"role": "user", "content": str(source["evidence_description"])}
        )
    stage_c_source = {
        **source,
        "messages": plain_user_messages,
    }
    messages = build_messages(
        stage_c_source,
        exposure,
        rng=rng,
        noise_repeats=noise_repeats,
    )
    messages[-1]["content"] += (
        "\n\n## Reward Lookup Metadata\n"
        f"sample_id: {source['sample_id']}\n"
        f"prompt_exposure: {exposure}\n"
    )
    return {
        "active_order_id": f"GRPO-ACTIVE-{source['sample_id']}",
        "backend_contract": "current_order_guaranteed",
        "fields": source["fields"],
        "grpo_category": source["grpo_category"],
        "hard_tags": source.get("hard_tags", []),
        "input_variant": "grpo_targeted_stage_c_contract",
        "messages": messages,
        "prompt_exposure": exposure,
        "sample_id": source["sample_id"],
        "skill_id": "refund_decision",
        "split": "train",
        "target_output": source["target_output"],
    }


def build_prompt_records(
    sources: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    noise_repeats: int,
    shuffle_records: bool = False,
) -> list[dict[str, Any]]:
    records = [
        prompt_record(source, exposure, seed=seed, noise_repeats=noise_repeats)
        for source in sources
        for exposure in EXPOSURES
    ]
    if shuffle_records:
        random.Random(seed).shuffle(records)
    return records


def distribution(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown")) for record in records))


def decision_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(
            str(record.get("target_output", {}).get("decision", "unknown"))
            for record in records
        )
    )


def tag_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(tag for record in records for tag in record.get("hard_tags", [])))


def char_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    lengths = [
        sum(len(message.get("content", "")) for message in record["messages"])
        for record in records
    ]
    return {
        "max": max(lengths) if lengths else 0,
        "mean": round(statistics.mean(lengths), 2) if lengths else 0,
        "median": round(statistics.median(lengths), 2) if lengths else 0,
        "min": min(lengths) if lengths else 0,
    }


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# GRPO Targeted Data Summary",
        "",
        "Date: 2026-05-19",
        "",
        "This dataset is for reward-training preparation. Assistant answers are not appended to prompts; `target_output` is retained for the local rule/evaluator reward.",
        "",
        "## Counts",
        "",
        f"- source_samples: `{summary['source_sample_count']}`",
        f"- prompt_records: `{summary['prompt_record_count']}`",
        f"- records_per_sample: `{summary['records_per_sample']}`",
        "",
        "## Category Distribution",
        "",
        "| category | source samples |",
        "| --- | ---: |",
    ]
    for category, count in sorted(summary["source_category_distribution"].items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Exposure Distribution", "", "| exposure | records |", "| --- | ---: |"])
    for exposure in EXPOSURES:
        lines.append(f"| {exposure} | {summary['exposure_distribution'].get(exposure, 0)} |")
    lines.extend(["", "## Decision Distribution", "", "| decision | records |", "| --- | ---: |"])
    for decision, count in sorted(summary["decision_distribution"].items()):
        lines.append(f"| {decision} | {count} |")
    lines.extend(
        [
            "",
            "## Paths",
            "",
            f"- source_samples: `{summary['paths']['source_samples']}`",
            f"- annotated_prompts: `{summary['paths']['annotated_prompts']}`",
            f"- model_prompts: `{summary['paths']['model_prompts']}`",
            f"- summary_json: `{summary['paths']['summary_json']}`",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted GRPO prompt data.")
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--noise-repeats", type=int, default=28)
    parser.add_argument(
        "--shuffle-records",
        action="store_true",
        help="Shuffle prompt records. Default keeps full/partial/minimal/no_skill adjacent per sample_id for group-aware reward smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/refund_grpo_targeted_20260519",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sources = build_source_samples(args.sample_count, args.seed)
    prompts = build_prompt_records(
        sources,
        seed=args.seed,
        noise_repeats=args.noise_repeats,
        shuffle_records=args.shuffle_records,
    )
    model_prompts = [{"messages": record["messages"]} for record in prompts]

    source_path = output_dir / "source_samples.jsonl"
    annotated_path = output_dir / "grpo_targeted_prompts.jsonl"
    model_prompts_path = output_dir / "grpo_targeted_model_prompts.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    write_jsonl(source_path, sources)
    write_jsonl(annotated_path, prompts)
    write_jsonl(model_prompts_path, model_prompts)

    summary = {
        "char_length": char_stats(prompts),
        "decision_distribution": decision_distribution(prompts),
        "exposure_distribution": distribution(prompts, "prompt_exposure"),
        "noise_repeats": args.noise_repeats,
        "paths": {
            "annotated_prompts": str(annotated_path),
            "model_prompts": str(model_prompts_path),
            "source_samples": str(source_path),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
        "prompt_record_count": len(prompts),
        "records_per_sample": len(prompts) / len(sources) if sources else 0,
        "seed": args.seed,
        "shuffle_records": args.shuffle_records,
        "source_category_distribution": distribution(sources, "grpo_category"),
        "source_sample_count": len(sources),
        "tag_distribution": tag_distribution(prompts),
        "target_category_weights": dict(CATEGORY_WEIGHTS),
    }
    write_json(summary_json_path, summary)
    write_markdown(summary_md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
