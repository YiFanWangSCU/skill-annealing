"""Deterministic local data builder for protocol v2.1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from skill_annealing.expense_reimbursement.rule_engine import decide as decide_expense
from skill_annealing.refund_decision.pilot_rule_engine import decide as decide_refund
from skill_annealing.warranty_claim.rule_engine import decide as decide_warranty

from .core import canonical_json, load_protocol, sha256_text
from .registries import (
    LEXICAL_TEMPLATES,
    build_prompt_registry,
    build_target_registry,
    lexical_template_id,
)


ENGINES: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "refund_decision": decide_refund,
    "warranty_claim_decision": decide_warranty,
    "expense_reimbursement_decision": decide_expense,
}
SEGMENT_COUNTS = {
    "stage1": {"full": 510, "partial": 90},
    "stage2": {"full": 120, "partial": 330, "minimal": 150},
    "stage3": {"partial": 120, "minimal": 270, "no_skill": 210},
}
EVAL_REASON_CYCLES = {
    ("refund_decision", "R3_logistics_issue"): [
        "logistics_issue_supported",
    ] * 5
    + ["logistics_issue_insufficient_evidence"] * 3,
    ("refund_decision", "R4_quality_issue"): [
        "quality_issue_refund_only",
    ] * 2
    + ["quality_issue_return_refund"] * 3
    + ["quality_issue_no_evidence"] * 3,
    ("warranty_claim_decision", "W2_in_warranty_manufacturing_defect"): [
        "manufacturing_defect_supported",
    ] * 7
    + ["diagnostic_evidence_missing"],
    ("expense_reimbursement_decision", "E6_manager_approval_override"): [
        "documented_policy_exception",
    ] * 4
    + ["cap_override_approved"] * 3
    + ["exception_manager_approval_missing"],
}


def build_bundle(output_dir: Path) -> dict[str, Any]:
    protocol = load_protocol()
    if not protocol.get("implementation_permitted"):
        raise RuntimeError("protocol does not permit local implementation")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_registry = build_prompt_registry(protocol)
    target_registry = build_target_registry(protocol)
    _write_json(output_dir / "prompt_registry.json", prompt_registry)
    _write_json(output_dir / "target_template_registry.json", target_registry)

    all_cases: dict[str, dict[str, list[dict[str, Any]]]] = {}
    summary: dict[str, Any] = {
        "protocol_version": protocol["protocol_version"],
        "skills": {},
    }
    for skill_id in protocol["skills"]:
        skill_dir = output_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        cases = build_skill_cases(skill_id, protocol, target_registry)
        all_cases[skill_id] = cases
        for split, rows in cases.items():
            name = "confirmation.locked.jsonl" if split == "confirmation" else f"{split}.jsonl"
            _write_jsonl(skill_dir / name, rows)
        arms = build_training_arms(
            skill_id,
            cases["train"],
            protocol,
            prompt_registry,
        )
        _write_jsonl(skill_dir / "train_staged_order.jsonl", arms["staged_order"])
        _write_jsonl(skill_dir / "train_random_mix_order.jsonl", arms["random_mix_order"])
        eval_rows = build_eval_requests(
            skill_id, cases["discovery_eval"], prompt_registry
        )
        _write_jsonl(skill_dir / "discovery_eval_requests.jsonl", eval_rows)
        summary["skills"][skill_id] = {
            "split_counts": {key: len(value) for key, value in cases.items()},
            "train_decisions": dict(
                Counter(row["target_output"]["decision"] for row in cases["train"])
            ),
            "train_families": dict(Counter(row["rule_family"] for row in cases["train"])),
            "train_record_count_per_arm": len(arms["staged_order"]),
            "train_multiset_sha256": _multiset_hash(arms["staged_order"]),
            "staged_order_sha256": _ordered_hash(arms["staged_order"]),
            "random_order_sha256": _ordered_hash(arms["random_mix_order"]),
            "eval_request_count": len(eval_rows),
        }
    lock = {
        "locked": True,
        "model_evaluation_permitted": False,
        "training_permitted": False,
        "splits": {
            skill: {
                "case_manifest_sha256": sha256_text(
                    canonical_json(
                        [row["semantic_case_hash"] for row in splits["confirmation"]]
                    )
                ),
                "facts_multiset_sha256": sha256_text(
                    canonical_json(
                        sorted(row["facts_hash"] for row in splits["confirmation"])
                    )
                ),
                "rendered_text_sha256": sha256_text(
                    canonical_json(
                        [row["user_input"] for row in splits["confirmation"]]
                    )
                ),
                "token_ids_sha256": None,
            }
            for skill, splits in all_cases.items()
        },
        "token_hash_status": "pending_remote_qwen35_tokenizer_audit",
    }
    _write_json(output_dir / "confirmation_lock_manifest.json", lock)
    _write_json(output_dir / "build_summary.json", summary)
    return summary


def build_skill_cases(
    skill_id: str,
    protocol: Mapping[str, Any],
    target_registry: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    target_registry = target_registry or build_target_registry(protocol)
    skill = protocol["skills"][skill_id]
    code_to_family = skill["reason_code_to_rule_family"]
    code_to_decision = _code_to_decision(skill)
    family_codes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for code in skill["reason_codes"]:
        family_codes[(code_to_family[code], code_to_decision[code])].append(code)
    specs: dict[str, list[tuple[str, str, str]]] = {}
    train_specs: list[tuple[str, str, str]] = []
    matrix = protocol["split_gates"]["train_family_decision_counts"][skill_id]
    for family, decision_counts in matrix.items():
        for decision, count in decision_counts.items():
            codes = family_codes[(family, decision)]
            for index in range(count):
                train_specs.append((family, decision, codes[index % len(codes)]))
    specs["train"] = train_specs
    for split in ("validation", "discovery_eval", "confirmation"):
        rows: list[tuple[str, str, str]] = []
        for family in skill["rule_families"]:
            cycle = EVAL_REASON_CYCLES.get((skill_id, family))
            if cycle is None:
                codes = [
                    code
                    for code in skill["reason_codes"]
                    if code_to_family[code] == family
                ]
                cycle = [codes[index % len(codes)] for index in range(8)]
            for code in cycle:
                rows.append((family, code_to_decision[code], code))
        specs[split] = rows

    output: dict[str, list[dict[str, Any]]] = {}
    global_index = 0
    for split in ("train", "validation", "discovery_eval", "confirmation"):
        records = []
        templates = LEXICAL_TEMPLATES[split]
        for local_index, (family, decision, reason) in enumerate(specs[split]):
            fields = prototype_fields(skill_id, reason, global_index)
            result = ENGINES[skill_id](fields).to_dict()
            if result["decision"] != decision or result["reason_codes"] != [reason]:
                raise AssertionError(
                    f"oracle mismatch {skill_id}/{reason}: {result}"
                )
            result["explanation"] = target_registry["templates"][skill_id][reason]
            source = templates[local_index % len(templates)]
            user_input = source.format(facts_json=canonical_json(fields))
            semantic_hash = sha256_text(f"{skill_id}|{canonical_json(fields)}")
            facts_hash = sha256_text(canonical_json(fields))
            record = {
                "case_id": f"{skill_id}_{split}_{local_index:04d}",
                "skill_id": skill_id,
                "split": split,
                "fields": fields,
                "user_input": user_input,
                "target_output": result,
                "rule_family": family,
                "primary_reason_code": reason,
                "semantic_case_hash": semantic_hash,
                "facts_hash": facts_hash,
                "target_hash": sha256_text(canonical_json(result)),
                "lexical_template_family": lexical_template_id(source),
                "lexical_template_source_sha256": lexical_template_id(source),
            }
            records.append(record)
            global_index += 1
        output[split] = records
    return output


def build_training_arms(
    skill_id: str,
    train_cases: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    prompt_registry: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    records = []
    seed = protocol["prompt_freeze_registry"]["freeze_seeds"]["exposure_assignment"]
    order_seed = protocol["prompt_freeze_registry"]["freeze_seeds"]["within_segment_order"]
    for segment, counts in SEGMENT_COUNTS.items():
        ranked = sorted(
            train_cases,
            key=lambda row: (
                sha256_text(
                    f"{seed}|{skill_id}|{segment}|{row['semantic_case_hash']}"
                ),
                row["semantic_case_hash"],
            ),
        )
        exposures = [
            exposure for exposure, count in counts.items() for _ in range(count)
        ]
        segment_rows = []
        for case, exposure in zip(ranked, exposures, strict=True):
            record_id = sha256_text(
                f"{protocol['protocol_version']}|{skill_id}|"
                f"{case['semantic_case_hash']}|{segment}|{exposure}"
            )
            segment_rows.append(
                {
                    **case,
                    "segment_id": segment,
                    "prompt_exposure": exposure,
                    "record_id": record_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt_registry["prompts"][skill_id][exposure],
                        },
                        {"role": "user", "content": case["user_input"]},
                        {
                            "role": "assistant",
                            "content": canonical_json(case["target_output"]),
                        },
                    ],
                }
            )
        segment_rows.sort(
            key=lambda row: (
                sha256_text(
                    f"{order_seed}|{skill_id}|{segment}|{row['record_id']}"
                ),
                row["record_id"],
            )
        )
        records.extend(segment_rows)
    random_seed = protocol["prompt_freeze_registry"]["freeze_seeds"]["random_mix_order"]
    random_rows = sorted(
        records,
        key=lambda row: (
            sha256_text(f"{random_seed}|{skill_id}|{row['record_id']}"),
            row["record_id"],
        ),
    )
    return {"staged_order": records, "random_mix_order": random_rows}


def build_eval_requests(
    skill_id: str,
    discovery_cases: list[dict[str, Any]],
    prompt_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    conditions = [
        "full",
        "partial",
        "minimal",
        "no_skill",
        "legacy_no_system",
        "full_length_irrelevant",
        "partial_semantic_paraphrase_a",
        "partial_semantic_paraphrase_b",
    ]
    rows = []
    for endpoint in ("random_mix_order", "staged_order"):
        for case in discovery_cases:
            for condition in conditions:
                messages = [{"role": "user", "content": case["user_input"]}]
                if condition != "legacy_no_system":
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": prompt_registry["prompts"][skill_id][condition],
                        },
                    )
                rows.append(
                    {
                        "pairing_key": (
                            f"{skill_id}|{case['semantic_case_hash']}|"
                            f"{condition}|{endpoint}"
                        ),
                        "skill_id": skill_id,
                        "endpoint": endpoint,
                        "prompt_condition": condition,
                        "semantic_case_hash": case["semantic_case_hash"],
                        "rule_family": case["rule_family"],
                        "target_output": case["target_output"],
                        "messages": messages,
                    }
                )
    return rows


def prototype_fields(skill_id: str, reason: str, nonce: int) -> dict[str, Any]:
    if skill_id == "refund_decision":
        return _refund_fields(reason, nonce)
    if skill_id == "warranty_claim_decision":
        return _warranty_fields(reason, nonce)
    if skill_id == "expense_reimbursement_decision":
        return _expense_fields(reason, nonce)
    raise KeyError(skill_id)


def _refund_fields(reason: str, nonce: int) -> dict[str, Any]:
    base = {
        "product_type": "standard",
        "days_since_delivery": 3,
        "is_used": False,
        "is_resellable": True,
        "has_quality_issue": False,
        "has_evidence": False,
        "order_amount_cny": f"{100 + nonce / 100:.2f}",
        "refund_count_30d": 0,
        "user_claim": "no_reason",
        "order_status": "delivered",
        "seller_fault": False,
        "logistics_issue": False,
        "evidence_conflict": False,
        "order_info_conflict": False,
    }
    overrides = {
        "missing_required_facts": {"days_since_delivery": None, "order_status": "unknown"},
        "evidence_conflict": {"evidence_conflict": True},
        "order_info_conflict": {"order_info_conflict": True},
        "high_value_order": {"order_amount_cny": f"{1000 + nonce / 100:.2f}"},
        "frequent_refund_user": {"refund_count_30d": 3},
        "logistics_issue_supported": {
            "days_since_delivery": 0,
            "order_status": "not_delivered",
            "logistics_issue": True,
            "seller_fault": True,
            "user_claim": "logistics_issue",
        },
        "logistics_issue_insufficient_evidence": {
            "days_since_delivery": 0,
            "order_status": "not_delivered",
            "logistics_issue": True,
            "user_claim": "logistics_issue",
        },
        "quality_issue_no_evidence": {
            "has_quality_issue": True,
            "user_claim": "quality_issue",
        },
        "quality_issue_return_refund": {
            "has_quality_issue": True,
            "has_evidence": True,
            "user_claim": "quality_issue",
        },
        "quality_issue_refund_only": {
            "product_type": "virtual",
            "is_resellable": False,
            "has_quality_issue": True,
            "has_evidence": True,
            "user_claim": "quality_issue",
        },
        "special_goods_exclusion": {"product_type": "fresh"},
        "within_7_days_unused": {},
        "used_or_not_resellable": {"is_used": True, "is_resellable": False},
        "over_7_days_no_quality_issue": {"days_since_delivery": 8},
    }
    base.update(overrides[reason])
    return base


def _warranty_fields(reason: str, nonce: int) -> dict[str, Any]:
    base = {
        "product_category": "electronics",
        "days_since_purchase": 10,
        "warranty_days": 365,
        "failure_type": "manufacturing_defect",
        "failure_description_present": True,
        "has_purchase_proof": True,
        "has_diagnostic_evidence": True,
        "unauthorized_repair": False,
        "unauthorized_repair_caused_current_failure": False,
        "extended_coverage": False,
        "prior_repair_count": 0,
        "same_failure_recurrence": False,
        "serial_status": "match",
        "safety_recall": False,
        "claim_amount_cny": f"{100 + nonce / 100:.2f}",
        "evidence_conflict": False,
    }
    overrides = {
        "safety_recall": {"safety_recall": True},
        "evidence_conflict": {"evidence_conflict": True},
        "serial_mismatch": {"serial_status": "mismatch"},
        "serial_missing": {"serial_status": "missing"},
        "purchase_proof_missing": {"has_purchase_proof": False},
        "failure_description_missing": {"failure_description_present": False},
        "abnormal_claim_amount": {"claim_amount_cny": f"{3000 + nonce / 100:.2f}"},
        "unauthorized_repair_causal": {
            "unauthorized_repair": True,
            "unauthorized_repair_caused_current_failure": True,
        },
        "repeat_failure_after_repairs": {
            "days_since_purchase": 400,
            "prior_repair_count": 2,
            "same_failure_recurrence": True,
        },
        "warranty_expired": {"days_since_purchase": 400},
        "extended_damage_covered": {
            "failure_type": "accidental_damage",
            "extended_coverage": True,
        },
        "accidental_or_liquid_not_covered": {
            "failure_type": "liquid_damage",
        },
        "manufacturing_defect_supported": {},
        "diagnostic_evidence_missing": {"has_diagnostic_evidence": False},
        "normal_wear_excluded": {"failure_type": "normal_wear"},
        "failure_type_unclear": {"failure_type": "unknown"},
    }
    base.update(overrides[reason])
    return base


def _expense_fields(reason: str, nonce: int) -> dict[str, Any]:
    base = {
        "expense_category": "meal",
        "amount": f"{100 + nonce / 100:.2f}",
        "currency": "CNY",
        "hotel_nights": 0,
        "days_since_expense": 10,
        "has_receipt": True,
        "receipt_matches_amount": True,
        "business_purpose_present": True,
        "manager_approved": False,
        "project_code_valid": True,
        "is_duplicate_claim": False,
        "is_personal_expense": False,
        "is_weekend_or_holiday": False,
        "international_trip": False,
        "policy_exception_documented": False,
        "evidence_conflict": False,
    }
    overrides = {
        "duplicate_claim": {"is_duplicate_claim": True},
        "evidence_conflict": {"evidence_conflict": True},
        "receipt_amount_mismatch": {"receipt_matches_amount": False},
        "receipt_missing": {"has_receipt": False, "receipt_matches_amount": False},
        "business_purpose_missing": {"business_purpose_present": False},
        "project_code_invalid": {"project_code_valid": False},
        "high_value_expense": {"amount": f"{5000 + nonce / 100:.2f}"},
        "high_value_foreign_currency": {
            "amount": f"{300 + nonce / 100:.2f}",
            "currency": "USD",
        },
        "personal_expense": {"is_personal_expense": True},
        "late_submission": {"days_since_expense": 31},
        "exception_manager_approval_missing": {
            "is_personal_expense": True,
            "policy_exception_documented": True,
        },
        "documented_policy_exception": {
            "is_personal_expense": True,
            "policy_exception_documented": True,
            "manager_approved": True,
        },
        "cap_override_approved": {
            "amount": f"{400 + nonce / 100:.2f}",
            "manager_approved": True,
        },
        "category_cap_exceeded": {"amount": f"{400 + nonce / 100:.2f}"},
        "standard_compliant": {},
    }
    base.update(overrides[reason])
    return base


def _code_to_decision(skill: Mapping[str, Any]) -> dict[str, str]:
    mapping = {}
    for rule in skill["priority_order"]:
        rhs = rule.split(" -> ", 1)[1]
        decision_part, code = rhs.rsplit(":", 1)
        mapping[code] = decision_part.split("/", 1)[0]
    return mapping


def _ordered_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json([row["record_id"] for row in rows]))


def _multiset_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(sorted(row["record_id"] for row in rows)))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
