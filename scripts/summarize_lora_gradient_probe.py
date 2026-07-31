"""Summarize four-state paired LoRA gradient smoke outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lora_gradient_probe import EXPOSURES
from scripts.run_remote_lora_gradient_smoke import load_manifest, validate_manifest


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _positive_fraction(values: Sequence[float]) -> float | None:
    return sum(value > 0 for value in values) / len(values) if values else None


def collect_runs(
    root: str | Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    expected = int(manifest["expected_records_per_checkpoint"])
    for checkpoint in manifest["checkpoints"]:
        checkpoint_id = str(checkpoint["id"])
        run_dir = root_path / checkpoint_id
        aggregate = json.loads(
            (run_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        if aggregate.get("status") != "pass":
            raise ValueError(f"checkpoint {checkpoint_id} did not pass")
        if int(aggregate.get("record_count", 0)) != expected:
            raise ValueError(f"checkpoint {checkpoint_id} has incomplete records")
        checkpoint_rows = load_jsonl(run_dir / "per_case_metrics.jsonl")
        if len(checkpoint_rows) * len(EXPOSURES) != expected:
            raise ValueError(f"checkpoint {checkpoint_id} has incomplete paired rows")
        for row in checkpoint_rows:
            found_id = row.get("checkpoint_id")
            if found_id not in (None, checkpoint_id):
                raise ValueError(
                    f"checkpoint ID mismatch in {checkpoint_id}: {found_id}"
                )
            rows.append({**row, "checkpoint_id": checkpoint_id})
    return rows


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        for exposure in EXPOSURES:
            metrics = row["exposure_metrics"][exposure]
            flattened.append(
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "case_id": row["case_id"],
                    "semantic_case_hash": row.get("semantic_case_hash"),
                    "rule_family": row.get("rule_family"),
                    "context_load": row["context_load"],
                    "prompt_exposure": exposure,
                    "loss": float(metrics["loss"]),
                    "global_gradient_norm": float(metrics["global_gradient_norm"]),
                    "full_dot": float(row["raw_pairwise"]["dot"]["full"][exposure]),
                    "full_cosine": float(
                        row["raw_pairwise"]["cosine"]["full"][exposure]
                    ),
                    "centered_residual_norm": float(
                        row["centered_residual_pairwise"]["norm"][exposure]
                    ),
                    "residual_full_dot": float(
                        row["centered_residual_pairwise"]["dot"]["full"][exposure]
                    ),
                    "residual_full_cosine": float(
                        row["centered_residual_pairwise"]["cosine"]["full"][exposure]
                    ),
                    "runtime_seconds": float(metrics["runtime_seconds"]),
                    "gpu_peak_allocated_bytes": int(
                        metrics["gpu_peak_allocated_bytes"]
                    ),
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in flattened:
        grouped[
            (
                str(item["checkpoint_id"]),
                str(item["context_load"]),
                str(item["prompt_exposure"]),
            )
        ].append(item)
    aggregate_rows: list[dict[str, Any]] = []
    for (checkpoint_id, context_load, exposure), items in sorted(grouped.items()):
        dots = [float(item["full_dot"]) for item in items]
        cosines = [float(item["full_cosine"]) for item in items]
        aggregate_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "context_load": context_load,
                "prompt_exposure": exposure,
                "unique_case_count": len({item["case_id"] for item in items}),
                "loss": _stats([float(item["loss"]) for item in items]),
                "global_gradient_norm": _stats(
                    [float(item["global_gradient_norm"]) for item in items]
                ),
                "full_dot": _stats(dots),
                "full_dot_positive_fraction": _positive_fraction(dots),
                "full_cosine": _stats(cosines),
                "centered_residual_norm": _stats(
                    [float(item["centered_residual_norm"]) for item in items]
                ),
                "residual_full_dot": _stats(
                    [float(item["residual_full_dot"]) for item in items]
                ),
                "residual_full_cosine": _stats(
                    [float(item["residual_full_cosine"]) for item in items]
                ),
                "runtime_seconds": _stats(
                    [float(item["runtime_seconds"]) for item in items]
                ),
            }
        )

    matched: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in flattened:
        matched[
            (
                str(item["checkpoint_id"]),
                str(item["case_id"]),
                str(item["prompt_exposure"]),
            )
        ][str(item["context_load"])] = item
    delta_groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (checkpoint_id, _case_id, exposure), contexts in matched.items():
        if set(contexts) != {"short", "long"}:
            raise ValueError("short/long context pairing is incomplete")
        short = contexts["short"]
        long = contexts["long"]
        delta_groups[(checkpoint_id, exposure)].append(
            {
                "full_cosine": float(long["full_cosine"])
                - float(short["full_cosine"]),
                "full_dot": float(long["full_dot"]) - float(short["full_dot"]),
                "loss": float(long["loss"]) - float(short["loss"]),
                "global_gradient_norm": float(long["global_gradient_norm"])
                - float(short["global_gradient_norm"]),
                "residual_full_dot": float(long["residual_full_dot"])
                - float(short["residual_full_dot"]),
                "residual_full_cosine": float(long["residual_full_cosine"])
                - float(short["residual_full_cosine"]),
            }
        )
    context_delta_rows: list[dict[str, Any]] = []
    for (checkpoint_id, exposure), items in sorted(delta_groups.items()):
        context_delta_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "prompt_exposure": exposure,
                "unique_case_count": len(items),
                "long_minus_short_full_cosine": _stats(
                    [item["full_cosine"] for item in items]
                ),
                "long_minus_short_full_dot": _stats(
                    [item["full_dot"] for item in items]
                ),
                "long_minus_short_loss": _stats([item["loss"] for item in items]),
                "long_minus_short_gradient_norm": _stats(
                    [item["global_gradient_norm"] for item in items]
                ),
                "long_minus_short_residual_full_dot": _stats(
                    [item["residual_full_dot"] for item in items]
                ),
                "long_minus_short_residual_full_cosine": _stats(
                    [item["residual_full_cosine"] for item in items]
                ),
            }
        )

    by_rule: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in flattened:
        if item["prompt_exposure"] == "full":
            continue
        by_rule[
            (
                str(item["checkpoint_id"]),
                str(item["rule_family"]),
                str(item["prompt_exposure"]),
            )
        ].append(item)
    rule_rows = [
        {
            "checkpoint_id": checkpoint_id,
            "rule_family": rule_family,
            "prompt_exposure": exposure,
            "record_count": len(items),
            "full_cosine": _stats([float(item["full_cosine"]) for item in items]),
            "full_dot": _stats([float(item["full_dot"]) for item in items]),
        }
        for (checkpoint_id, rule_family, exposure), items in sorted(by_rule.items())
    ]

    return {
        "protocol_version": manifest["protocol_version"],
        "manifest_version": manifest["manifest_version"],
        "status": "pass",
        "checkpoint_count": len(manifest["checkpoints"]),
        "paired_group_count": len(rows),
        "record_count": len(flattened),
        "unique_semantic_case_count": len(
            {item["semantic_case_hash"] for item in flattened}
        ),
        "checkpoint_ids": [item["id"] for item in manifest["checkpoints"]],
        "aggregate_by_checkpoint_context_exposure": aggregate_rows,
        "context_load_deltas": context_delta_rows,
        "by_rule_family": rule_rows,
        "per_record": flattened,
        "interpretation_status": "exploratory_smoke_only",
    }


def write_markdown(path: str | Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Four-State LoRA Gradient Smoke Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- records: `{summary['record_count']}`",
        f"- paired groups: `{summary['paired_group_count']}`",
        f"- unique semantic cases: `{summary['unique_semantic_case_count']}`",
        f"- checkpoints: `{', '.join(summary['checkpoint_ids'])}`",
        "- interpretation: `exploratory_smoke_only`",
        "",
        "## Full-to-Exposure Raw Gradient Geometry",
        "",
        "| checkpoint | context | exposure | median dot | positive dot | median cosine | median norm | mean loss |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_by_checkpoint_context_exposure"]:
        lines.append(
            f"| `{row['checkpoint_id']}` | {row['context_load']} | {row['prompt_exposure']} | "
            f"{row['full_dot']['median']:.6f} | {row['full_dot_positive_fraction']:.2f} | "
            f"{row['full_cosine']['median']:.6f} | "
            f"{row['global_gradient_norm']['median']:.6f} | {row['loss']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Long Minus Short Context Delta",
            "",
            "| checkpoint | exposure | median Δdot | median Δcosine | median Δnorm | median Δloss |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["context_load_deltas"]:
        lines.append(
            f"| `{row['checkpoint_id']}` | {row['prompt_exposure']} | "
            f"{row['long_minus_short_full_dot']['median']:.6f} | "
            f"{row['long_minus_short_full_cosine']['median']:.6f} | "
            f"{row['long_minus_short_gradient_norm']['median']:.6f} | "
            f"{row['long_minus_short_loss']['median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "> Dot product is the primary first-order quantity. Cosine is descriptive only. "
            "This smoke summary does not establish transfer, interference, internalization, or causality.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    rows = collect_runs(args.run_root, manifest)
    summary = summarize_rows(rows, manifest)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md, summary)
    print(
        json.dumps(
            {key: summary[key] for key in (
                "status",
                "checkpoint_count",
                "paired_group_count",
                "record_count",
                "unique_semantic_case_count",
                "interpretation_status",
            )},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
