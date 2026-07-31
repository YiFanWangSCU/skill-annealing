"""CPU-only contracts for the prompt-component factorial boundary study."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .core import canonical_json, sha256_text


SKILLS = (
    "refund_decision",
    "warranty_claim_decision",
    "expense_reimbursement_decision",
)
ENDPOINTS = ("random_mix_order", "staged_order")
FACTOR_ORDER = ("identity", "schema", "business_rules")
FACTORIAL_CONDITIONS: dict[str, tuple[int, int, int]] = {
    "isb_111_full": (1, 1, 1),
    "isb_011_no_identity": (0, 1, 1),
    "isb_101_no_schema": (1, 0, 1),
    "isb_110_no_rules": (1, 1, 0),
    "isb_001_rules_only": (0, 0, 1),
    "isb_010_schema_only": (0, 1, 0),
    "isb_100_identity_only": (1, 0, 0),
    "isb_000_empty_system": (0, 0, 0),
}
PLACEMENT_CONDITION = "isb_111_user_placement"
CONDITIONS = (*FACTORIAL_CONDITIONS, PLACEMENT_CONDITION)
HISTORICAL_BRIDGES = {
    "isb_111_full": "full",
    "isb_110_no_rules": "minimal",
}
NEW_CONDITIONS = tuple(
    condition for condition in CONDITIONS if condition not in HISTORICAL_BRIDGES
)


def component_texts(
    protocol: Mapping[str, Any], skill_id: str
) -> dict[str, str]:
    skill = protocol["skills"][skill_id]
    rules = "\n".join(
        f"{index + 1}. {rule}"
        for index, rule in enumerate(skill["priority_order"])
    )
    return {
        "identity": f"你正在执行 {skill_id}。",
        "schema": (
            "只输出 JSON，字段必须为 decision、risk_level、need_human、"
            "reason_codes、explanation。"
        ),
        "business_rules": f"按以下 first-match 规则执行：\n{rules}",
    }


def compose_prompt(
    components: Mapping[str, str], levels: Sequence[int]
) -> str:
    if len(levels) != len(FACTOR_ORDER) or any(level not in (0, 1) for level in levels):
        raise ValueError(f"invalid factor levels: {levels}")
    return "\n".join(
        components[name]
        for name, enabled in zip(FACTOR_ORDER, levels, strict=True)
        if enabled
    )


def build_component_registry(
    protocol: Mapping[str, Any],
    historical_registry: Mapping[str, Any],
) -> dict[str, Any]:
    skills: dict[str, Any] = {}
    for skill_id in SKILLS:
        components = component_texts(protocol, skill_id)
        prompts = {
            condition: compose_prompt(components, levels)
            for condition, levels in FACTORIAL_CONDITIONS.items()
        }
        prompts[PLACEMENT_CONDITION] = prompts["isb_111_full"]
        old_prompts = historical_registry["prompts"][skill_id]
        if prompts["isb_111_full"] != old_prompts["full"]:
            raise ValueError(f"{skill_id}: Full bridge is not byte-identical")
        if prompts["isb_110_no_rules"] != old_prompts["minimal"]:
            raise ValueError(f"{skill_id}: Minimal bridge is not byte-identical")
        skills[skill_id] = {
            "components": components,
            "component_sha256": {
                name: sha256_text(text) for name, text in components.items()
            },
            "prompts": prompts,
            "prompt_sha256": {
                condition: sha256_text(text)
                for condition, text in prompts.items()
            },
        }
    registry = {
        "study_version": "prompt_component_factorial_boundary_v0",
        "source_protocol_version": protocol["protocol_version"],
        "factor_order": list(FACTOR_ORDER),
        "factorial_conditions": {
            name: dict(zip(FACTOR_ORDER, levels, strict=True))
            for name, levels in FACTORIAL_CONDITIONS.items()
        },
        "placement_condition": PLACEMENT_CONDITION,
        "condition_order": list(CONDITIONS),
        "historical_bridges": HISTORICAL_BRIDGES,
        "skills": skills,
        "examples_in_scope": False,
        "edge_cases_in_scope": False,
        "token_counts_status": "PENDING_EXACT_QWEN35_CPU_TOKENIZER_AUDIT",
    }
    registry["registry_sha256"] = sha256_text(canonical_json(registry))
    return registry


def build_messages(
    registry: Mapping[str, Any],
    *,
    skill_id: str,
    condition: str,
    user_input: str,
) -> list[dict[str, str]]:
    if condition not in CONDITIONS:
        raise KeyError(condition)
    prompt = registry["skills"][skill_id]["prompts"][condition]
    if condition == PLACEMENT_CONDITION:
        return [{"role": "user", "content": f"{prompt}\n\n{user_input}"}]
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ]


def build_factorial_requests(
    skill_id: str,
    discovery_cases: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases = list(discovery_cases)
    if len(cases) != 64:
        raise ValueError(f"{skill_id}: expected 64 Discovery cases")
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        for case in cases:
            if case.get("split") != "discovery_eval":
                raise PermissionError(f"{skill_id}: non-Discovery case is forbidden")
            if case.get("skill_id") != skill_id:
                raise ValueError(f"{skill_id}: case Skill mismatch")
            for condition in CONDITIONS:
                key = (
                    f"{skill_id}|{case['semantic_case_hash']}|"
                    f"{condition}|{endpoint}"
                )
                messages = build_messages(
                    registry,
                    skill_id=skill_id,
                    condition=condition,
                    user_input=str(case["user_input"]),
                )
                rows.append(
                    {
                        "record_id": sha256_text(key),
                        "pairing_key": key,
                        "skill_id": skill_id,
                        "endpoint": endpoint,
                        "prompt_condition": condition,
                        "factor_levels": (
                            dict(
                                zip(
                                    FACTOR_ORDER,
                                    FACTORIAL_CONDITIONS[condition],
                                    strict=True,
                                )
                            )
                            if condition in FACTORIAL_CONDITIONS
                            else None
                        ),
                        "placement": (
                            "user"
                            if condition == PLACEMENT_CONDITION
                            else "system"
                        ),
                        "semantic_case_hash": case["semantic_case_hash"],
                        "rule_family": case["rule_family"],
                        "target_output": case["target_output"],
                        "messages": messages,
                        "messages_sha256": sha256_text(canonical_json(messages)),
                        "confirmation_used": False,
                    }
                )
    expected = len(ENDPOINTS) * 64 * len(CONDITIONS)
    if len(rows) != expected:
        raise AssertionError(f"{skill_id}: expected {expected} requests")
    return rows


def build_validation_smoke_requests(
    skill_id: str,
    validation_cases: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases = list(validation_cases)
    if len(cases) != 2:
        raise ValueError(f"{skill_id}: smoke requires exactly two validation cases")
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        for case in cases:
            if case.get("split") != "validation":
                raise PermissionError(f"{skill_id}: smoke requires validation cases")
            if case.get("skill_id") != skill_id:
                raise ValueError(f"{skill_id}: smoke case Skill mismatch")
            for condition in NEW_CONDITIONS:
                key = (
                    f"{skill_id}|{case['semantic_case_hash']}|"
                    f"{condition}|{endpoint}"
                )
                messages = build_messages(
                    registry,
                    skill_id=skill_id,
                    condition=condition,
                    user_input=str(case["user_input"]),
                )
                rows.append(
                    {
                        "record_id": sha256_text(key),
                        "pairing_key": key,
                        "skill_id": skill_id,
                        "endpoint": endpoint,
                        "prompt_condition": condition,
                        "factor_levels": (
                            dict(
                                zip(
                                    FACTOR_ORDER,
                                    FACTORIAL_CONDITIONS[condition],
                                    strict=True,
                                )
                            )
                            if condition in FACTORIAL_CONDITIONS
                            else None
                        ),
                        "placement": (
                            "user"
                            if condition == PLACEMENT_CONDITION
                            else "system"
                        ),
                        "semantic_case_hash": case["semantic_case_hash"],
                        "rule_family": case["rule_family"],
                        "target_output": case["target_output"],
                        "messages": messages,
                        "messages_sha256": sha256_text(canonical_json(messages)),
                        "split": "validation_smoke",
                        "confirmation_used": False,
                    }
                )
    expected = len(ENDPOINTS) * 2 * len(NEW_CONDITIONS)
    if len(rows) != expected:
        raise AssertionError(f"{skill_id}: expected {expected} smoke requests")
    return rows
