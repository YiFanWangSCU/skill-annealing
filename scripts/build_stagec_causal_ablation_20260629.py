"""Build Stage C causal ablation data for annealing-vs-random analysis."""

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

from scripts.build_long_context_stage_c_contract_eval import (  # noqa: E402
    build_messages,
    dedupe_source_records,
)


EXPOSURES = ("full", "partial", "minimal", "no_skill")
BALANCED_STAGE_NAMES = (
    "stage1_full_partial",
    "stage2_partial_minimal",
    "stage3_minimal_no_skill",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def messages_only(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": record["messages"]} for record in records]


def balanced_shuffled_exposures(total_count: int, seed: int) -> list[str]:
    if total_count % len(EXPOSURES) != 0:
        raise ValueError("total_count must be divisible by 4")
    exposures = [
        exposure
        for exposure in EXPOSURES
        for _ in range(total_count // len(EXPOSURES))
    ]
    rng = random.Random(f"{seed}:balanced_shuffled")
    rng.shuffle(exposures)
    return exposures


def balanced_stage_exposures(stage: str, stage_sample_count: int, seed: int) -> list[str]:
    if stage_sample_count % 4 != 0:
        raise ValueError("stage_sample_count must be divisible by 4")
    half = stage_sample_count // 2
    quarter = stage_sample_count // 4
    if stage == "stage1_full_partial":
        exposures = ["full"] * (3 * quarter) + ["partial"] * quarter
    elif stage == "stage2_partial_minimal":
        exposures = ["partial"] * half + ["minimal"] * half
    elif stage == "stage3_minimal_no_skill":
        exposures = ["minimal"] * quarter + ["no_skill"] * (3 * quarter)
    else:
        raise ValueError(f"unknown balanced stage: {stage!r}")
    rng = random.Random(f"{seed}:{stage}:balanced_stage")
    rng.shuffle(exposures)
    return exposures


def sample_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    total_count: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    if not sources:
        raise ValueError("cannot sample from empty sources")
    rng = random.Random(seed)
    repeated = [source for _ in range((total_count + len(sources) - 1) // len(sources)) for source in sources]
    rng.shuffle(repeated)
    return repeated[:total_count]


def build_sft_record(
    source: Mapping[str, Any],
    *,
    exposure: str,
    method: str,
    record_index: int,
    seed: int,
    noise_repeats: int,
    stage: str | None = None,
) -> dict[str, Any]:
    item_rng = random.Random(
        f"{seed}:{method}:{stage or 'single'}:{source['sample_id']}:{record_index}:{exposure}"
    )
    messages = build_messages(source, exposure, rng=item_rng, noise_repeats=noise_repeats)
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(source["target_output"], ensure_ascii=False, sort_keys=True),
        }
    )
    return {
        "sample_id": source["sample_id"],
        "stagec_causal_record_id": f"{method}_{record_index:06d}",
        "stagec_causal_method": method,
        "stagec_causal_stage": stage,
        "skill_id": "refund_decision",
        "prompt_exposure": exposure,
        "input_variant": "stage_c_causal_ablation_20260629",
        "split": "train",
        "messages": messages,
        "fields": source["fields"],
        "target_output": source["target_output"],
        "hard_tags": list(source.get("hard_tags", []))
        + ["stage_c_causal_ablation_20260629", method],
        "active_order_id": f"C-CAUSAL-{source['sample_id']}",
        "noise_repeats": noise_repeats,
        "backend_contract": "current_order_guaranteed",
    }


def build_records(
    sources: Sequence[Mapping[str, Any]],
    *,
    method: str,
    exposures: Sequence[str],
    seed: int,
    noise_repeats: int,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    selected_sources = sample_sources(sources, total_count=len(exposures), seed=seed)
    return [
        build_sft_record(
            source,
            exposure=exposure,
            method=method,
            stage=stage,
            record_index=index,
            seed=seed,
            noise_repeats=noise_repeats,
        )
        for index, (source, exposure) in enumerate(
            zip(selected_sources, exposures, strict=True)
        )
    ]


def exposure_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("prompt_exposure", "unknown")) for record in records))


def char_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    lengths = [
        sum(len(message.get("content", "")) for message in record["messages"])
        for record in records
    ]
    return {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": round(statistics.mean(lengths), 2) if lengths else 0,
        "median": round(statistics.median(lengths), 2) if lengths else 0,
    }


def write_variant(
    *,
    name: str,
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    swift_dir: Path,
) -> dict[str, Any]:
    annotated_path = output_dir / f"{name}.jsonl"
    swift_path = swift_dir / f"{name}.jsonl"
    write_jsonl(annotated_path, records)
    write_jsonl(swift_path, messages_only(records))
    return {
        "name": name,
        "annotated_path": str(annotated_path),
        "swift_path": str(swift_path),
        "record_count": len(records),
        "exposure_distribution": exposure_distribution(records),
        "char_length": char_stats(records),
    }


def build_all(
    sources: Sequence[Mapping[str, Any]],
    *,
    records_per_method: int,
    seed: int,
    noise_repeats: int,
    output_dir: Path,
    swift_dir: Path,
) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []

    balanced_random = build_records(
        sources,
        method="stagec_causal_balanced_random_mix_sft",
        exposures=balanced_shuffled_exposures(records_per_method, seed),
        seed=seed + 10,
        noise_repeats=noise_repeats,
    )
    variants.append(
        write_variant(
            name="stagec_causal_balanced_random_mix_sft",
            records=balanced_random,
            output_dir=output_dir,
            swift_dir=swift_dir,
        )
    )

    staged_all: list[dict[str, Any]] = []
    stage_sample_count = records_per_method // len(BALANCED_STAGE_NAMES)
    for index, stage in enumerate(BALANCED_STAGE_NAMES):
        records = build_records(
            sources,
            method="stagec_causal_balanced_staged_sft",
            exposures=balanced_stage_exposures(stage, stage_sample_count, seed + index),
            seed=seed + 100 + index,
            noise_repeats=noise_repeats,
            stage=stage,
        )
        staged_all.extend(records)
        variants.append(
            write_variant(
                name=f"stagec_causal_balanced_staged_{stage}_sft",
                records=records,
                output_dir=output_dir,
                swift_dir=swift_dir,
            )
        )
    variants.append(
        write_variant(
            name="stagec_causal_balanced_staged_all_annotation_only",
            records=staged_all,
            output_dir=output_dir,
            swift_dir=swift_dir,
        )
    )

    summary = {
        "name": "stagec_causal_ablation_20260629",
        "records_per_method": records_per_method,
        "stage_sample_count": stage_sample_count,
        "seed": seed,
        "noise_repeats": noise_repeats,
        "variants": variants,
        "matrix_note": {
            "old_annealed": "annealed marginal distribution + staged order; existing baseline",
            "stagec_v4_random_mix_sft": "old annealed marginal distribution + shuffled order; existing baseline",
            "stagec_causal_balanced_random_mix_sft": "uniform marginal distribution + shuffled order",
            "stagec_causal_balanced_staged_sft": "uniform marginal distribution + staged order",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    swift_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return summary


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage C Causal Ablation Data Summary",
        "",
        f"- records_per_method: `{summary['records_per_method']}`",
        f"- stage_sample_count: `{summary['stage_sample_count']}`",
        "",
        "## Variants",
        "",
        "| variant | records | full | partial | minimal | no_skill | mean_chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in summary["variants"]:
        dist = variant["exposure_distribution"]
        lines.append(
            "| {name} | {records} | {full} | {partial} | {minimal} | {no_skill} | {mean} |".format(
                name=variant["name"],
                records=variant["record_count"],
                full=dist.get("full", 0),
                partial=dist.get("partial", 0),
                minimal=dist.get("minimal", 0),
                no_skill=dist.get("no_skill", 0),
                mean=variant["char_length"]["mean"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage C causal ablation SFT data.")
    parser.add_argument(
        "--source-file",
        default="data/refund_metadata_augmented_stage_ab/metadata_augmented_random_mix_sft.jsonl",
    )
    parser.add_argument("--output-dir", default="data/stagec_causal_ablation_20260629")
    parser.add_argument("--swift-dir", default="data/ms_swift/stagec_causal_ablation_20260629")
    parser.add_argument("--records-per-method", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--noise-repeats", type=int, default=18)
    args = parser.parse_args()

    if args.records_per_method % 12 != 0:
        raise SystemExit("--records-per-method must be divisible by 12")
    if args.noise_repeats < 0:
        raise SystemExit("--noise-repeats must be >= 0")

    sources = dedupe_source_records(load_jsonl(args.source_file))
    summary = build_all(
        sources,
        records_per_method=args.records_per_method,
        seed=args.seed,
        noise_repeats=args.noise_repeats,
        output_dir=Path(args.output_dir),
        swift_dir=Path(args.swift_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
