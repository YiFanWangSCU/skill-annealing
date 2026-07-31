"""Static local gates for a built v2.1 data bundle."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from .builder import ENGINES
from .core import PROTOCOL_PATH, canonical_json, load_protocol, sha256_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_bundle(data_dir: Path) -> dict[str, Any]:
    protocol = load_protocol()
    errors: list[str] = []
    skill_reports = {}
    expected_decisions = Counter(
        protocol["split_gates"]["train_decision_counts_per_skill"]
    )
    for skill_id, skill in protocol["skills"].items():
        skill_dir = data_dir / skill_id
        splits = {
            split: load_jsonl(
                skill_dir
                / (
                    "confirmation.locked.jsonl"
                    if split == "confirmation"
                    else f"{split}.jsonl"
                )
            )
            for split in ("train", "validation", "discovery_eval", "confirmation")
        }
        expected_counts = protocol["splits_per_skill"]
        for split, rows in splits.items():
            if len(rows) != expected_counts[split]:
                errors.append(
                    f"{skill_id}/{split}: {len(rows)} != {expected_counts[split]}"
                )
            _validate_oracles(skill_id, rows, errors)
        for left, right in combinations(splits, 2):
            for field in protocol["split_gates"]["zero_overlap_required_for_each_pair"]:
                overlap = {row[field] for row in splits[left]} & {
                    row[field] for row in splits[right]
                }
                if overlap:
                    errors.append(
                        f"{skill_id}/{left}:{right}: {field} overlap={len(overlap)}"
                    )
        train_decisions = Counter(
            row["target_output"]["decision"] for row in splits["train"]
        )
        if train_decisions != expected_decisions:
            errors.append(
                f"{skill_id}/train decision quota mismatch: {dict(train_decisions)}"
            )
        expected_matrix = protocol["split_gates"]["train_family_decision_counts"][
            skill_id
        ]
        actual_matrix = Counter(
            (row["rule_family"], row["target_output"]["decision"])
            for row in splits["train"]
        )
        for family, decisions in expected_matrix.items():
            for decision, count in decisions.items():
                if actual_matrix[(family, decision)] != count:
                    errors.append(
                        f"{skill_id}/{family}/{decision}: "
                        f"{actual_matrix[(family, decision)]} != {count}"
                    )
        for split in ("validation", "discovery_eval", "confirmation"):
            families = Counter(row["rule_family"] for row in splits[split])
            expected = Counter({family: 8 for family in skill["rule_families"]})
            if families != expected:
                errors.append(f"{skill_id}/{split}: family quotas differ")
        discovery_decisions = Counter(
            row["target_output"]["decision"] for row in splits["discovery_eval"]
        )
        if any(discovery_decisions[decision] < 7 for decision in expected_decisions):
            errors.append(
                f"{skill_id}/discovery: decision floor failed {dict(discovery_decisions)}"
            )
        staged = load_jsonl(skill_dir / "train_staged_order.jsonl")
        random_rows = load_jsonl(skill_dir / "train_random_mix_order.jsonl")
        staged_ids = [row["record_id"] for row in staged]
        random_ids = [row["record_id"] for row in random_rows]
        if len(staged_ids) != 1800 or len(set(staged_ids)) != 1800:
            errors.append(f"{skill_id}/staged: invalid record IDs")
        if Counter(staged_ids) != Counter(random_ids):
            errors.append(f"{skill_id}: arm multisets differ")
        if staged_ids == random_ids:
            errors.append(f"{skill_id}: arm orders are identical")
        eval_rows = load_jsonl(skill_dir / "discovery_eval_requests.jsonl")
        pairing = [row["pairing_key"] for row in eval_rows]
        if len(pairing) != 1024 or len(set(pairing)) != 1024:
            errors.append(f"{skill_id}: eval request/pairing count invalid")
        reason_counts = Counter(
            row["primary_reason_code"] for row in splits["train"]
        )
        skill_reports[skill_id] = {
            "split_counts": {split: len(rows) for split, rows in splits.items()},
            "train_decisions": dict(train_decisions),
            "discovery_decisions": dict(discovery_decisions),
            "train_records_per_arm": len(staged),
            "eval_requests": len(eval_rows),
            "primary_reason_code_counts": {
                code: reason_counts[code] for code in skill["reason_codes"]
            },
            "zero_train_reason_codes": [
                code for code in skill["reason_codes"] if reason_counts[code] == 0
            ],
        }
    lock = json.loads(
        (data_dir / "confirmation_lock_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not lock.get("locked")
        or lock.get("model_evaluation_permitted") is not False
        or lock.get("training_permitted") is not False
    ):
        errors.append("confirmation lock contract failed")
    prompt_registry = json.loads(
        (data_dir / "prompt_registry.json").read_text(encoding="utf-8")
    )
    remote_blockers = []
    if (
        prompt_registry.get("token_exact_irrelevant_status")
        == "passed_remote_qwen35_tokenizer_audit"
    ):
        required_hashes = (
            "tokenizer_files_sha256",
            "chat_template_sha256",
            "prompt_registry_sha256",
        )
        missing_hashes = [
            field for field in required_hashes if not prompt_registry.get(field)
        ]
        if missing_hashes:
            errors.append(
                "resolved prompt registry missing hashes: "
                + ", ".join(missing_hashes)
            )
        claimed_hash = prompt_registry.get("prompt_registry_sha256")
        actual_hash = sha256_text(
            canonical_json(
                {
                    key: value
                    for key, value in prompt_registry.items()
                    if key != "prompt_registry_sha256"
                }
            )
        )
        if claimed_hash and claimed_hash != actual_hash:
            errors.append("resolved prompt registry self-hash mismatch")
    else:
        remote_blockers.extend(
            [
                "Qwen35 tokenizer/chat-template hash audit pending",
                "full_length_irrelevant exact-token audit pending",
            ]
        )
    pipeline_evidence = protocol.get("pipeline_smoke_evidence", {})
    pipeline_ready = False
    if pipeline_evidence.get("status") == "pass":
        summary_relative = pipeline_evidence.get("summary")
        summary_path = (
            PROTOCOL_PATH.parent / summary_relative
            if summary_relative
            else None
        )
        if summary_path is None or not summary_path.is_file():
            errors.append("pipeline smoke summary is missing")
        else:
            actual_summary_hash = __import__("hashlib").sha256(
                summary_path.read_bytes()
            ).hexdigest()
            if actual_summary_hash != pipeline_evidence.get(
                "summary_sha256"
            ):
                errors.append("pipeline smoke summary hash mismatch")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            required_pipeline_fields = {
                "status": "pass",
                "run_count": 6,
                "total_optimizer_steps": 12,
                "validation_scratch_only": True,
                "confirmation_used": False,
                "formal_training_started": False,
                "error_count": 0,
            }
            for field, expected in required_pipeline_fields.items():
                if summary.get(field) != expected:
                    errors.append(
                        f"pipeline smoke {field}={summary.get(field)!r}"
                    )
            if not errors:
                pipeline_ready = True
    if not pipeline_ready:
        remote_blockers.append(
            "initialized LoRA/environment/dataloader runtime gates pending"
        )
    return {
        "local_static_valid": not errors,
        "remote_training_ready": not errors and not remote_blockers,
        "remote_training_permitted": bool(
            protocol.get("remote_training_permitted")
        ),
        "remote_blockers": remote_blockers,
        "error_count": len(errors),
        "errors": errors,
        "skills": skill_reports,
    }


def reject_confirmation_path(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if "confirmation" in normalized:
        raise PermissionError("confirmation split is locked for model use")


def _validate_oracles(
    skill_id: str,
    rows: list[Mapping[str, Any]],
    errors: list[str],
) -> None:
    for row in rows:
        expected = ENGINES[skill_id](row["fields"]).to_dict()
        target = row["target_output"]
        for field in ("decision", "risk_level", "need_human", "reason_codes"):
            if expected[field] != target[field]:
                errors.append(f"{row['case_id']}: oracle field mismatch {field}")
        if row["facts_hash"] != sha256_text(canonical_json(row["fields"])):
            errors.append(f"{row['case_id']}: facts hash mismatch")
        if row["primary_reason_code"] != target["reason_codes"][0]:
            errors.append(f"{row['case_id']}: primary reason mismatch")
