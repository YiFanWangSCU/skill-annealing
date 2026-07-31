"""Shared deterministic contracts for the multi-Skill pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "multiskill_protocol_manifest.json"


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    risk_level: str
    need_human: bool
    reason_codes: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def decimal_value(value: Any, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a decimal string") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def require_keys(fields: Mapping[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(fields))
    extra = sorted(set(fields) - required)
    if missing or extra:
        raise ValueError(f"invalid fact keys: missing={missing}, extra={extra}")


def make_result(
    decision: str,
    reason_code: str,
    *,
    explanation: str | None = None,
    safety_recall: bool = False,
) -> DecisionResult:
    contract = {
        "approve": ("low", False),
        "approve_with_conditions": ("medium", False),
        "request_more_evidence": ("medium", False),
        "reject": ("low", False),
        "manual_review": ("high", True),
    }
    risk_level, need_human = contract[decision]
    if safety_recall:
        risk_level, need_human = "high", True
    return DecisionResult(
        decision=decision,
        risk_level=risk_level,
        need_human=need_human,
        reason_codes=[reason_code],
        explanation=explanation or f"命中冻结规则：{reason_code}。",
    )
