"""Build replay-preserved annealing v2 SFT data for Stage C.

This variant keeps all prompt exposures visible in every annealing stage while
still shifting the mixture toward lower-skill prompts over time. Each stage also
receives a fixed slice of hard-boundary replay examples.
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

from scripts.build_long_context_stage_c_contract_eval import (  # noqa: E402
    build_messages,
    dedupe_source_records,
)


EXPOSURES = ("full", "partial", "minimal", "no_skill")
STAGE_NAMES = (
    "stage1_full_replay",
    "stage2_balanced_replay",
    "stage3_low_heavy",
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


def stage_weights(stage: str) -> dict[str, float]:
    if stage == "stage1_full_replay":
        return {"full": 0.55, "partial": 0.25, "minimal": 0.15, "no_skill": 0.05}
    if stage == "stage2_balanced_replay":
        return {"full": 0.30, "partial": 0.35, "minimal": 0.25, "no_skill": 0.10}
    if stage == "stage3_low_heavy":
        return {"full": 0.15, "partial": 0.25, "minimal": 0.40, "no_skill": 0.20}
    raise ValueError(f"unknown stage: {stage!r}")


def allocate_exposures(stage: str, sample_count: int, seed: int) -> list[str]:
    """Allocate exact-ish exposure counts and shuffle them deterministically."""
    if sample_count < len(EXPOSURES):
        raise ValueError("sample_count must be at least the number of exposures")
    weights = stage_weights(stage)
    counts = {exposure: max(1, int(sample_count * weights[exposure])) for exposure in EXPOSURES}
    while sum(counts.values()) < sample_count:
        deficits = sorted(
            EXPOSURES,
            key=lambda item: (sample_count * weights[item]) - counts[item],
            reverse=True,
        )
        counts[deficits[0]] += 1
    while sum(counts.values()) > sample_count:
        surplus = sorted(
            EXPOSURES,
            key=lambda item: counts[item] - (sample_count * weights[item]),
            reverse=True,
        )
        for exposure in surplus:
            if counts[exposure] > 1:
                counts[exposure] -= 1
                break
    exposures = [exposure for exposure, count in counts.items() for _ in range(count)]
    rng = random.Random(f"{seed}:{stage}:exposures")
    rng.shuffle(exposures)
    return exposures


def sample_records(
    sources: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: str,
) -> list[Mapping[str, Any]]:
    if not sources:
        raise ValueError("cannot sample from empty sources")
    rng = random.Random(seed)
    if count <= len(sources):
        return rng.sample(list(sources), k=count)
    return [rng.choice(sources) for _ in range(count)]


def build_sft_record(
    source: Mapping[str, Any],
    *,
    exposure: str,
    stage: str,
    source_kind: str,
    record_index: int,
    seed: int,
    noise_repeats: int,
) -> dict[str, Any]:
    item_rng = random.Random(
        f"{seed}:stagec_v5:{stage}:{source_kind}:{source['sample_id']}:{record_index}:{exposure}"
    )
    messages = build_messages(
        source,
        exposure,
        rng=item_rng,
        noise_repeats=noise_repeats,
    )
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(source["target_output"], ensure_ascii=False, sort_keys=True),
        }
    )
    hard_tags = list(source.get("hard_tags", []) or [])
    return {
        "sample_id": source["sample_id"],
        "stagec_v5_record_id": f"stagec_v5_replay_preserved_annealed_{record_index:06d}",
        "stagec_v5_method": "stagec_v5_replay_preserved_annealed_sft",
        "stagec_v5_stage": stage,
        "stagec_v5_source": source_kind,
        "skill_id": "refund_decision",
        "prompt_exposure": exposure,
        "input_variant": "stage_c_v5_replay_preserved_annealing_sft",
        "split": "train",
        "messages": messages,
        "fields": source["fields"],
        "target_output": source["target_output"],
        "hard_tags": hard_tags
        + ["stage_c_v5_sft", "replay_preserved_annealing", source_kind],
        "noise_repeats": noise_repeats,
        "backend_contract": "current_order_guaranteed",
    }


def build_stage_records(
    *,
    stage: str,
    base_sources: Sequence[Mapping[str, Any]],
    hard_sources: Sequence[Mapping[str, Any]],
    stage_sample_count: int,
    hard_replay_fraction: float,
    seed: int,
    noise_repeats: int,
    record_offset: int,
) -> list[dict[str, Any]]:
    if not 0 <= hard_replay_fraction < 1:
        raise ValueError("hard_replay_fraction must be in [0, 1)")
    hard_count = round(stage_sample_count * hard_replay_fraction)
    base_count = stage_sample_count - hard_count
    selected = [
        ("base_replay", record)
        for record in sample_records(
            base_sources,
            count=base_count,
            seed=f"{seed}:{stage}:base",
        )
    ]
    selected.extend(
        [
            ("hard_replay", record)
            for record in sample_records(
                hard_sources,
                count=hard_count,
                seed=f"{seed}:{stage}:hard",
            )
        ]
    )
    rng = random.Random(f"{seed}:{stage}:selected")
    rng.shuffle(selected)
    exposures = allocate_exposures(stage, stage_sample_count, seed)
    return [
        build_sft_record(
            source,
            exposure=exposure,
            stage=stage,
            source_kind=source_kind,
            record_index=record_offset + index,
            seed=seed,
            noise_repeats=noise_repeats,
        )
        for index, ((source_kind, source), exposure) in enumerate(
            zip(selected, exposures, strict=True)
        )
    ]


def build_records_by_stage(
    base_sources: Sequence[Mapping[str, Any]],
    hard_sources: Sequence[Mapping[str, Any]],
    *,
    stage_sample_count: int,
    hard_replay_fraction: float,
    seed: int,
    noise_repeats: int,
) -> dict[str, list[dict[str, Any]]]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for stage_index, stage in enumerate(STAGE_NAMES):
        records = build_stage_records(
            stage=stage,
            base_sources=base_sources,
            hard_sources=hard_sources,
            stage_sample_count=stage_sample_count,
            hard_replay_fraction=hard_replay_fraction,
            seed=seed + stage_index,
            noise_repeats=noise_repeats,
            record_offset=offset,
        )
        by_stage[stage] = records
        offset += len(records)
    return by_stage


def exposure_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("prompt_exposure", "unknown")) for record in records))


def source_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("stagec_v5_source", "unknown")) for record in records))


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


def write_outputs(
    by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_dir: Path,
    swift_dir: Path,
    source_file: str,
    hard_replay_source_file: str,
    seed: int,
    noise_repeats: int,
    hard_replay_fraction: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    swift_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    all_records = [record for stage in STAGE_NAMES for record in by_stage[stage]]
    for stage in STAGE_NAMES:
        name = f"stagec_v5_replay_preserved_annealed_{stage}_sft"
        records = list(by_stage[stage])
        annotated_path = output_dir / f"{name}.jsonl"
        swift_path = swift_dir / f"{name}.jsonl"
        write_jsonl(annotated_path, records)
        write_jsonl(swift_path, messages_only(records))
        variants.append(
            {
                "name": name,
                "stage": stage,
                "annotated_path": str(annotated_path),
                "swift_path": str(swift_path),
                "record_count": len(records),
                "exposure_distribution": exposure_distribution(records),
                "source_distribution": source_distribution(records),
                "char_length": char_stats(records),
            }
        )
    all_annotated = output_dir / "stagec_v5_replay_preserved_annealed_all_annotation_only.jsonl"
    all_swift = swift_dir / "stagec_v5_replay_preserved_annealed_all_annotation_only.jsonl"
    write_jsonl(all_annotated, all_records)
    write_jsonl(all_swift, messages_only(all_records))
    summary = {
        "name": "stage_c_v5_replay_preserved_annealing_v2",
        "source_file": source_file,
        "hard_replay_source_file": hard_replay_source_file,
        "stage_names": list(STAGE_NAMES),
        "records_per_method": len(all_records),
        "records_per_stage": {stage: len(by_stage[stage]) for stage in STAGE_NAMES},
        "hard_replay_fraction": hard_replay_fraction,
        "seed": seed,
        "noise_repeats": noise_repeats,
        "exposure_schedule": {stage: stage_weights(stage) for stage in STAGE_NAMES},
        "variants": variants,
        "all_annotation_only": str(all_annotated),
        "note": "Replay-preserved annealing v2: every stage keeps full/partial/minimal/no_skill exposure and hard-boundary replay.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    return summary


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Replay-Preserved Annealing V2 Data Summary",
        "",
        f"- source_file: `{summary['source_file']}`",
        f"- hard_replay_source_file: `{summary['hard_replay_source_file']}`",
        f"- records_per_method: `{summary['records_per_method']}`",
        f"- hard_replay_fraction: `{summary['hard_replay_fraction']}`",
        "",
        "## Variants",
        "",
        "| variant | records | full | partial | minimal | no_skill | hard_replay | mean_chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in summary["variants"]:
        exposures = variant["exposure_distribution"]
        sources = variant["source_distribution"]
        chars = variant["char_length"]
        lines.append(
            "| {name} | {records} | {full} | {partial} | {minimal} | {no_skill} | {hard} | {mean} |".format(
                name=variant["name"],
                records=variant["record_count"],
                full=exposures.get("full", 0),
                partial=exposures.get("partial", 0),
                minimal=exposures.get("minimal", 0),
                no_skill=exposures.get("no_skill", 0),
                hard=sources.get("hard_replay", 0),
                mean=chars["mean"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay-preserved annealing v2 SFT data.")
    parser.add_argument(
        "--source-file",
        default="data/refund_metadata_augmented_stage_ab/metadata_augmented_random_mix_sft.jsonl",
    )
    parser.add_argument(
        "--hard-replay-source-file",
        default="data/refund_hard_boundary_lite_20260627/eval_prompts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/refund_metadata_stage_c_v5_replay_preserved_annealing_20260627",
    )
    parser.add_argument(
        "--swift-dir",
        default="data/ms_swift/refund_metadata_stage_c_v5_replay_preserved_annealing_20260627",
    )
    parser.add_argument("--stage-sample-count", type=int, default=600)
    parser.add_argument("--hard-replay-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--noise-repeats", type=int, default=18)
    args = parser.parse_args()

    if args.stage_sample_count < len(EXPOSURES):
        raise SystemExit("--stage-sample-count must be at least 4")
    if args.noise_repeats < 0:
        raise SystemExit("--noise-repeats must be >= 0")
    base_sources = dedupe_source_records(load_jsonl(args.source_file))
    hard_sources = dedupe_source_records(load_jsonl(args.hard_replay_source_file))
    by_stage = build_records_by_stage(
        base_sources,
        hard_sources,
        stage_sample_count=args.stage_sample_count,
        hard_replay_fraction=args.hard_replay_fraction,
        seed=args.seed,
        noise_repeats=args.noise_repeats,
    )
    summary = write_outputs(
        by_stage,
        output_dir=Path(args.output_dir),
        swift_dir=Path(args.swift_dir),
        source_file=args.source_file,
        hard_replay_source_file=args.hard_replay_source_file,
        seed=args.seed,
        noise_repeats=args.noise_repeats,
        hard_replay_fraction=args.hard_replay_fraction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
