"""Run the frozen multi-skill discovery analysis without confirmation access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multiskill_behavior_eval import (
    canonical_json,
    exact_tuple,
    strict_json_object,
    validate_schema,
)
from skill_annealing.multiskill_prompt_robustness.analysis import (
    CANONICAL_CONDITIONS,
    ENDPOINTS,
    ROBUST_CONDITIONS,
    case_cluster_bootstrap,
    classify_frozen_decision,
    construct_balanced_effect,
    family_cluster_bootstrap,
    grouped_effects,
    holm_adjust,
    paired_mcnemar_tests,
    schema_validity_report,
)
from skill_annealing.multiskill_prompt_robustness.core import load_protocol
from skill_annealing.multiskill_prompt_robustness.validator import (
    load_jsonl,
    reject_confirmation_path,
)


SKILLS = (
    "refund_decision",
    "warranty_claim_decision",
    "expense_reimbursement_decision",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_engineering_gates(
    *,
    predictions_path: Path,
    rows: list[dict[str, Any]],
    completion_path: Path,
    recovery_manifest_path: Path,
    behavior_manifest_path: Path,
    schema_config_path: Path,
    formal_summary_path: Path,
    formal_recovery_manifest_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    for path in (
        predictions_path,
        completion_path,
        recovery_manifest_path,
        behavior_manifest_path,
        schema_config_path,
        formal_summary_path,
        formal_recovery_manifest_path,
    ):
        reject_confirmation_path(path)
        if not path.is_file():
            errors.append(f"missing required artifact: {path}")
    if errors:
        return {"pass": False, "error_count": len(errors), "errors": errors}

    completion = load_json(completion_path)
    recovery = load_json(recovery_manifest_path)
    behavior_manifest = load_json(behavior_manifest_path)
    schema_config = load_json(schema_config_path)
    formal_summary = load_json(formal_summary_path)
    formal_recovery = load_json(formal_recovery_manifest_path)
    prediction_hash = sha256_file(predictions_path)

    required_completion = {
        "status": "pass",
        "expected_records": 3072,
        "actual_records": 3072,
        "unique_pairing_key_count": 3072,
        "duplicate_pairing_key_count": 0,
        "missing_pairing_key_count": 0,
        "unexpected_pairing_key_count": 0,
        "endpoint_completion_count": 6,
        "exit_code": 0,
        "confirmation_used": False,
    }
    for field, expected in required_completion.items():
        if completion.get(field) != expected:
            errors.append(
                f"completion {field}={completion.get(field)!r} != {expected!r}"
            )
    if completion.get("result_file_sha256") != prediction_hash:
        errors.append("completion result hash differs from predictions")

    required_recovery = {
        "status": "pass",
        "prediction_count": 3072,
        "prediction_sha256": prediction_hash,
        "discovery_endpoint_count": 6,
        "confirmation_used": False,
    }
    for field, expected in required_recovery.items():
        if recovery.get(field) != expected:
            errors.append(
                f"recovery {field}={recovery.get(field)!r} != {expected!r}"
            )

    required_behavior = {
        "status": "frozen_approved",
        "discovery_expected_records": 3072,
        "confirmation_used": False,
    }
    for field, expected in required_behavior.items():
        if behavior_manifest.get(field) != expected:
            errors.append(
                f"behavior manifest {field}="
                f"{behavior_manifest.get(field)!r} != {expected!r}"
            )
    for field in (
        "tokenizer_files_sha256",
        "chat_template_sha256",
        "generation_config_sha256",
        "discovery_ordered_eval_manifest_sha256",
    ):
        completion_field = (
            "ordered_eval_manifest_sha256"
            if field == "discovery_ordered_eval_manifest_sha256"
            else field
        )
        if behavior_manifest.get(field) != completion.get(completion_field):
            errors.append(f"runtime hash mismatch: {field}")

    if schema_config.get("confirmation_used") is not False:
        errors.append("schema config confirmation gate failed")
    if set(schema_config.get("skills", {})) != set(SKILLS):
        errors.append("schema config skill set mismatch")

    required_formal = {
        "status": "pass",
        "run_count": 6,
        "optimizer_steps_per_run": 225,
        "total_optimizer_steps": 1350,
        "confirmation_used": False,
    }
    for field, expected in required_formal.items():
        if formal_summary.get(field) != expected:
            errors.append(
                f"formal summary {field}="
                f"{formal_summary.get(field)!r} != {expected!r}"
            )
    initial_hashes = {
        run.get("initial_trainable_parameter_sha256")
        for run in formal_summary.get("runs", [])
    }
    if len(initial_hashes) != 1:
        errors.append("formal arms do not share one initial trainable hash")
    for run in formal_summary.get("runs", []):
        if (
            run.get("optimizer_steps") != 225
            or run.get("record_count") != 1800
            or run.get("dataloader_trace_exact") is not True
        ):
            errors.append(
                f"formal run gate failed: "
                f"{run.get('skill_id')}/{run.get('arm')}"
            )
    if (
        formal_recovery.get("status") != "pass"
        or formal_recovery.get("checkpoint_adapter_count") != 6
        or formal_recovery.get("formal_total_optimizer_steps") != 1350
    ):
        errors.append("formal recovery gate failed")

    confirmation_contract = protocol["confirmation_protection"]
    if (
        confirmation_contract.get("locked") is not True
        or confirmation_contract.get("model_evaluation_permitted") is not False
        or confirmation_contract.get("training_permitted") is not False
    ):
        errors.append("protocol confirmation lock gate failed")

    expected_conditions = tuple(protocol["behavior_eval"]["conditions"])
    expected_condition_set = set(expected_conditions)
    expected_endpoint_set = set(protocol["behavior_eval"]["endpoints"])
    indexed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    case_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        skill = str(row.get("skill_id"))
        case = str(row.get("semantic_case_hash"))
        condition = str(row.get("prompt_condition"))
        endpoint = str(row.get("endpoint"))
        key = (skill, case, condition, endpoint)
        if key in indexed:
            errors.append(f"duplicate prediction panel key: {key}")
            continue
        indexed[key] = row
        case_rows[(skill, case)].append(row)
        if skill not in SKILLS:
            errors.append(f"unexpected skill: {skill}")
        if condition not in expected_condition_set:
            errors.append(f"unexpected condition: {condition}")
        if endpoint not in expected_endpoint_set:
            errors.append(f"unexpected endpoint: {endpoint}")
        expected_pairing = f"{skill}|{case}|{condition}|{endpoint}"
        if row.get("pairing_key") != expected_pairing:
            errors.append(f"pairing key mismatch: {expected_pairing}")
        if "confirmation" in str(row.get("pairing_key", "")).lower():
            errors.append("confirmation marker found in pairing key")

        schema = schema_config.get("skills", {}).get(skill, {})
        parsed = None
        parse_failed = False
        try:
            parsed = strict_json_object(str(row.get("completion", "")))
        except (ValueError, json.JSONDecodeError):
            parse_failed = True
        if parse_failed != (row.get("parse_error") is not None):
            errors.append(f"parse flag mismatch: {expected_pairing}")
        if parsed != row.get("parsed_output"):
            errors.append(f"parsed output mismatch: {expected_pairing}")
        schema_errors = []
        if parsed is not None:
            schema_errors = validate_schema(
                parsed,
                allowed_decisions=set(schema.get("decisions", [])),
                allowed_reason_codes=set(schema.get("reason_codes", [])),
            )
        if schema_errors != row.get("schema_errors"):
            errors.append(f"schema error mismatch: {expected_pairing}")
        schema_valid = parsed is not None and not schema_errors
        if schema_valid != bool(row.get("schema_valid")):
            errors.append(f"schema-valid flag mismatch: {expected_pairing}")
        gold = row.get("target_output", {})
        exact = bool(schema_valid and exact_tuple(gold, parsed or {}))
        if exact != bool(row.get("exact_tuple_correct")):
            errors.append(f"exact tuple mismatch: {expected_pairing}")
        for field, result_field in (
            ("decision", "decision_correct"),
            ("risk_level", "risk_level_correct"),
            ("need_human", "need_human_correct"),
        ):
            field_correct = bool(
                schema_valid and parsed.get(field) == gold.get(field)
            )
            if field_correct != bool(row.get(result_field)):
                errors.append(
                    f"{result_field} mismatch: {expected_pairing}"
                )

    if len(rows) != 3072 or len(indexed) != 3072:
        errors.append(
            f"prediction panel count invalid: rows={len(rows)}, "
            f"unique={len(indexed)}"
        )
    ordered_hash = sha256_text(
        canonical_json([row.get("pairing_key") for row in rows])
    )
    if ordered_hash != completion.get("ordered_eval_manifest_sha256"):
        errors.append("ordered prediction panel hash mismatch")

    skill_cases = Counter(skill for skill, _ in case_rows)
    if skill_cases != Counter({skill: 64 for skill in SKILLS}):
        errors.append(f"per-skill case counts differ: {dict(skill_cases)}")
    expected_cell_set = {
        (condition, endpoint)
        for condition in expected_conditions
        for endpoint in ENDPOINTS
    }
    for (skill, case), selected in case_rows.items():
        cells = {
            (str(row["prompt_condition"]), str(row["endpoint"]))
            for row in selected
        }
        if cells != expected_cell_set or len(selected) != 16:
            errors.append(f"incomplete case panel: {skill}/{case}")
        target_values = {
            canonical_json(row["target_output"]) for row in selected
        }
        families = {str(row["rule_family"]) for row in selected}
        if len(target_values) != 1 or len(families) != 1:
            errors.append(f"case target/family mismatch: {skill}/{case}")

    if sum(row.get("parse_error") is not None for row in rows) != completion.get(
        "model_completion_parse_failure_count"
    ):
        errors.append("global parse failure count mismatch")
    if sum(not bool(row.get("schema_valid")) for row in rows) != completion.get(
        "model_completion_schema_failure_count"
    ):
        errors.append("global schema failure count mismatch")
    if sum(bool(row.get("exact_tuple_correct")) for row in rows) != completion.get(
        "exact_tuple_correct_count"
    ):
        errors.append("global exact tuple count mismatch")

    return {
        "pass": not errors,
        "error_count": len(errors),
        "errors": errors,
        "prediction_count": len(rows),
        "unique_panel_key_count": len(indexed),
        "prediction_sha256": prediction_hash,
        "ordered_eval_manifest_sha256": ordered_hash,
        "formal_initial_trainable_parameter_sha256": next(
            iter(initial_hashes), None
        ),
        "confirmation_used": False,
    }


def analyze(
    rows: list[dict[str, Any]],
    *,
    protocol: Mapping[str, Any],
    bootstrap_samples: int,
    engineering: Mapping[str, Any],
) -> dict[str, Any]:
    primary = construct_balanced_effect(rows)
    primary.pop("per_case_scores", None)
    case_bootstrap = case_cluster_bootstrap(
        rows,
        samples=bootstrap_samples,
        seed=protocol["statistics"]["bootstrap_seed"],
    )
    family_bootstrap = family_cluster_bootstrap(
        rows,
        samples=bootstrap_samples,
        seed=protocol["statistics"]["bootstrap_seed"],
    )
    constructs = grouped_effects(
        rows, protocol["statistics"]["constructs"]
    )
    condition_effects = grouped_effects(
        rows,
        {
            condition: (condition,)
            for condition in protocol["behavior_eval"]["conditions"]
        },
    )
    canonical = grouped_effects(
        rows, {"canonical": CANONICAL_CONDITIONS}
    )["canonical"]
    minimal = condition_effects["minimal"]
    no_skill = condition_effects["no_skill"]
    guards = protocol["quality_guards"]
    schema_guards = schema_validity_report(
        rows,
        endpoint_min=guards[
            "schema_valid_rate_per_skill_endpoint_min_inclusive"
        ],
        condition_min=guards[
            "schema_valid_rate_per_skill_endpoint_condition_min_inclusive"
        ],
    )
    robust_tests = holm_adjust(
        paired_mcnemar_tests(rows, ROBUST_CONDITIONS)
    )
    canonical_tests = paired_mcnemar_tests(rows, CANONICAL_CONDITIONS)
    decision = classify_frozen_decision(
        primary=primary,
        case_bootstrap=case_bootstrap,
        family_bootstrap=family_bootstrap,
        construct_effects=constructs,
        canonical=canonical,
        minimal=minimal,
        no_skill=no_skill,
        schema_guards_pass=schema_guards["pass"],
        data_valid=bool(engineering["pass"]),
    )
    raw = {
        "record_count": len(rows),
        "exact_tuple_correct_count": sum(
            bool(row["exact_tuple_correct"]) for row in rows
        ),
        "exact_tuple_accuracy": sum(
            bool(row["exact_tuple_correct"]) for row in rows
        )
        / len(rows),
        "parse_failure_count": sum(
            row["parse_error"] is not None for row in rows
        ),
        "schema_failure_count": sum(
            not bool(row["schema_valid"]) for row in rows
        ),
    }
    return {
        "analysis_version": "multiskill_prompt_robustness_frozen_v2_1",
        "decision": decision,
        "engineering_gates": dict(engineering),
        "raw_counts": raw,
        "primary_construct_balanced_effect": primary,
        "case_cluster_bootstrap": case_bootstrap,
        "family_cluster_bootstrap": family_bootstrap,
        "construct_effects": constructs,
        "canonical_effect": canonical,
        "condition_effects": condition_effects,
        "schema_validity_guards": schema_guards,
        "mcnemar_robust_holm_family": robust_tests,
        "mcnemar_canonical_descriptive": canonical_tests,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": protocol["statistics"]["bootstrap_seed"],
        "confirmation_used": False,
    }


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    primary = report["primary_construct_balanced_effect"]
    case_ci = report["case_cluster_bootstrap"]
    family_ci = report["family_cluster_bootstrap"]
    lines = [
        "# Multi-skill Prompt Robustness Frozen Analysis",
        "",
        "## Frozen decision",
        "",
        f"**{decision['state']}**",
        "",
        (
            "- Primary construct-balanced staged-minus-random delta: "
            f"`{format_percent(primary['macro_delta'])}`"
        ),
        (
            "- Case-cluster bootstrap CI95: "
            f"`[{format_percent(case_ci['ci95_lower'])}, "
            f"{format_percent(case_ci['ci95_upper'])}]`"
        ),
        (
            "- Family-cluster bootstrap CI95: "
            f"`[{format_percent(family_ci['ci95_lower'])}, "
            f"{format_percent(family_ci['ci95_upper'])}]`"
        ),
        (
            "- Engineering gates: "
            f"`{'PASS' if report['engineering_gates']['pass'] else 'FAIL'}`"
        ),
        (
            "- Schema validity guards: "
            f"`{'PASS' if report['schema_validity_guards']['pass'] else 'FAIL'}`"
        ),
        "- Confirmation used: `false`",
        "",
        "## Primary effect by skill",
        "",
        "| Skill | Random | Staged | Delta |",
        "|---|---:|---:|---:|",
    ]
    for skill, values in sorted(primary["per_skill"].items()):
        lines.append(
            f"| `{skill}` | {format_percent(values['random_mix_order'])} | "
            f"{format_percent(values['staged_order'])} | "
            f"{format_percent(values['delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Robust constructs",
            "",
            "| Construct | Random | Staged | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in report["construct_effects"].items():
        lines.append(
            f"| `{name}` | {format_percent(values['random_mix_order'])} | "
            f"{format_percent(values['staged_order'])} | "
            f"{format_percent(values['delta'])} |"
        )
    canonical = report["canonical_effect"]
    minimal = report["condition_effects"]["minimal"]
    no_skill = report["condition_effects"]["no_skill"]
    lines.extend(
        [
            "",
            "## Non-degradation controls",
            "",
            "| Metric | Delta |",
            "|---|---:|",
            f"| Canonical macro | {format_percent(canonical['delta'])} |",
            f"| Minimal macro | {format_percent(minimal['delta'])} |",
            f"| No-skill macro | {format_percent(no_skill['delta'])} |",
            "",
            "## Schema validity",
            "",
            (
                "- Minimum per skill/endpoint rate: "
                f"`{format_percent(report['schema_validity_guards']['minimum_per_skill_endpoint_rate'])}`"
            ),
            (
                "- Minimum per skill/endpoint/condition rate: "
                f"`{format_percent(report['schema_validity_guards']['minimum_per_skill_endpoint_condition_rate'])}`"
            ),
            (
                "- Failed skill/endpoint cells: "
                f"`{report['schema_validity_guards']['failed_per_skill_endpoint_count']}`"
            ),
            (
                "- Failed skill/endpoint/condition cells: "
                f"`{report['schema_validity_guards']['failed_per_skill_endpoint_condition_count']}`"
            ),
            "",
            "## Decision checks",
            "",
            "### Triggered FAIL checks",
            "",
        ]
    )
    triggered_fail = [
        name
        for name, passed in decision["fail_checks"].items()
        if passed
    ]
    lines.extend(
        [f"- `{name}`" for name in triggered_fail]
        or ["- None"]
    )
    lines.extend(["", "### Unmet PASS checks", ""])
    unmet_pass = [
        name
        for name, passed in decision["pass_checks"].items()
        if not passed
    ]
    lines.extend([f"- `{name}`" for name in unmet_pass] or ["- None"])
    lines.extend(
        [
            "",
            "## Raw completion counts",
            "",
            f"- Records: `{report['raw_counts']['record_count']}`",
            (
                "- Exact tuple correct: "
                f"`{report['raw_counts']['exact_tuple_correct_count']}` "
                f"(`{format_percent(report['raw_counts']['exact_tuple_accuracy'])}`)"
            ),
            (
                "- Parse failures: "
                f"`{report['raw_counts']['parse_failure_count']}`"
            ),
            (
                "- Schema failures: "
                f"`{report['raw_counts']['schema_failure_count']}`"
            ),
            "",
            "Malformed completions remain incorrect scientific observations. "
            "No retry, repair, confirmation data, or post-result threshold "
            "change was used.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = (
        "experiments/multiskill_prompt_robustness_pilot_20260724/"
        "behavior_eval_20260724/recovered"
    )
    formal = (
        "experiments/multiskill_prompt_robustness_pilot_20260724/"
        "formal_sft_20260724_retry1/recovered"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions", nargs="?", default=f"{base}/predictions.jsonl"
    )
    parser.add_argument(
        "--completion-manifest",
        default=f"{base}/behavior_eval_completion_manifest.json",
    )
    parser.add_argument(
        "--recovery-manifest", default=f"{base}/recovery_manifest.json"
    )
    parser.add_argument(
        "--behavior-manifest", default=f"{base}/behavior_eval_manifest.json"
    )
    parser.add_argument(
        "--schema-config", default=f"{base}/behavior_schema_config.json"
    )
    parser.add_argument(
        "--formal-summary", default=f"{formal}/formal_sft_summary.json"
    )
    parser.add_argument(
        "--formal-recovery-manifest",
        default=f"{formal}/recovery_manifest.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--output-json",
        default=(
            "experiments/multiskill_prompt_robustness_pilot_20260724/"
            "frozen_discovery_analysis_20260725.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=(
            "experiments/multiskill_prompt_robustness_pilot_20260724/"
            "frozen_discovery_analysis_20260725.md"
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    expected_samples = protocol["statistics"]["bootstrap_samples"]
    if args.bootstrap_samples != expected_samples:
        raise ValueError(
            f"formal analysis requires {expected_samples} bootstrap samples"
        )
    predictions_path = Path(args.predictions)
    reject_confirmation_path(predictions_path)
    rows = load_jsonl(predictions_path)
    engineering = validate_engineering_gates(
        predictions_path=predictions_path,
        rows=rows,
        completion_path=Path(args.completion_manifest),
        recovery_manifest_path=Path(args.recovery_manifest),
        behavior_manifest_path=Path(args.behavior_manifest),
        schema_config_path=Path(args.schema_config),
        formal_summary_path=Path(args.formal_summary),
        formal_recovery_manifest_path=Path(
            args.formal_recovery_manifest
        ),
        protocol=protocol,
    )
    if engineering["pass"]:
        report = analyze(
            rows,
            protocol=protocol,
            bootstrap_samples=args.bootstrap_samples,
            engineering=engineering,
        )
    else:
        report = {
            "analysis_version": (
                "multiskill_prompt_robustness_frozen_v2_1"
            ),
            "decision": {
                "state": "DATA_INVALID",
                "priority": [
                    "DATA_INVALID",
                    "PASS",
                    "FAIL",
                    "MIXED",
                    "INCONCLUSIVE",
                ],
            },
            "engineering_gates": engineering,
            "confirmation_used": False,
        }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if engineering["pass"]:
        output_md.write_text(render_markdown(report), encoding="utf-8")
    else:
        output_md.write_text(
            "# Multi-skill Prompt Robustness Frozen Analysis\n\n"
            "**DATA_INVALID**\n\n"
            + "\n".join(
                f"- {error}" for error in engineering["errors"]
            )
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "decision": report["decision"]["state"],
        "engineering_gates_pass": engineering["pass"],
        "output_json": str(output_json),
        "output_md": str(output_md),
        "confirmation_used": False,
    }
    if engineering["pass"]:
        summary.update(
            {
                "primary_macro_delta": report[
                    "primary_construct_balanced_effect"
                ]["macro_delta"],
                "case_ci95": [
                    report["case_cluster_bootstrap"]["ci95_lower"],
                    report["case_cluster_bootstrap"]["ci95_upper"],
                ],
                "family_ci95": [
                    report["family_cluster_bootstrap"]["ci95_lower"],
                    report["family_cluster_bootstrap"]["ci95_upper"],
                ],
                "schema_guards_pass": report[
                    "schema_validity_guards"
                ]["pass"],
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if engineering["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
