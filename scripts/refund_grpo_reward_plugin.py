"""Custom MS-Swift GRPO reward plugin for refund_decision.

Use with:

    --external_plugins scripts/refund_grpo_reward_plugin.py
    --reward_funcs refund_decision
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from swift.rewards import ORM, orms
except Exception:  # pragma: no cover - local tests may not have MS-Swift.
    class ORM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    orms = {}

from skill_annealing.refund_decision.evaluator import reason_code_f1
from skill_annealing.refund_decision.schema import validate_output_schema


def _disable_broken_vllm_import_path() -> None:
    """Prevent TRL from importing an incompatible installed vLLM package.

    The server environment currently has a vLLM build whose C extension does
    not match the active torch ABI. TRL imports vLLM during GRPOTrainer import
    whenever the package is discoverable, even when Swift is not using vLLM.
    This process-local patch leaves the conda environment untouched.
    """

    try:
        import trl.import_utils as trl_import_utils
    except Exception:
        trl_import_utils = None
    if trl_import_utils is not None:
        trl_import_utils.is_vllm_available = lambda: False

    try:
        import swift.utils as swift_utils
    except Exception:
        swift_utils = None
    if swift_utils is not None:
        swift_utils.is_vllm_available = lambda: False


def _stub_unused_swift_grpo_vllm_engine() -> None:
    """Stub Swift's GRPO vLLM engine import when `use_vllm=False`.

    Swift imports `GRPOVllmEngine` while defining rollout scheduler types,
    before it checks `args.use_vllm`. In this environment that import reaches
    the broken vLLM extension even though this smoke uses transformers
    generation. The stub is deliberately minimal so a real use of vLLM fails
    loudly instead of silently doing the wrong thing.
    """

    module_name = "swift.infer_engine.grpo_vllm_engine"
    if module_name in sys.modules:
        return

    class GRPOVllmEngine:  # noqa: N801 - match Swift class name
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "GRPOVllmEngine is stubbed because this run disables vLLM and "
                "the installed vLLM binary is incompatible with torch."
            )

    module = types.ModuleType(module_name)
    module.GRPOVllmEngine = GRPOVllmEngine
    sys.modules[module_name] = module


def _patch_generation_metadata_kwargs() -> None:
    """Drop Swift request metadata before it reaches `model.generate`.

    In this Swift/TRL version, GRPO rollout carries `prompt_id` and
    `request_id` through to the Transformers engine. They are useful for
    request tracking, but Transformers rejects them as unknown generation
    kwargs. Removing only these two metadata keys preserves model behavior.
    """

    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:
        return
    if getattr(GenerationMixin.generate, "_refund_metadata_patch", False):
        return
    original_generate = GenerationMixin.generate

    def generate_without_swift_metadata(self, *args, **kwargs):
        kwargs.pop("prompt_id", None)
        kwargs.pop("request_id", None)
        return original_generate(self, *args, **kwargs)

    generate_without_swift_metadata._refund_metadata_patch = True
    GenerationMixin.generate = generate_without_swift_metadata


_disable_broken_vllm_import_path()
_stub_unused_swift_grpo_vllm_engine()
_patch_generation_metadata_kwargs()


PAIR_WEIGHTS = {
    ("full", "partial"): 0.40,
    ("partial", "minimal"): 0.30,
    ("minimal", "no_skill"): 0.20,
    ("full", "no_skill"): 0.10,
}
HARD_REASON_CODES = {
    "manual_review_required",
    "evidence_conflict",
    "order_info_conflict",
    "high_value_order",
    "frequent_refund_user",
    "quality_issue_no_evidence",
    "fresh_goods_no_reason_denied",
    "customized_goods_no_reason_denied",
    "virtual_goods_denied",
}
DEFAULT_LOOKUP_PATH = "data/refund_grpo_targeted_20260519/source_samples.jsonl"
_LOOKUP_CACHE: dict[str, dict[str, Any]] | None = None


def parse_completion(text: Any) -> tuple[dict[str, Any], bool, bool]:
    if isinstance(text, Mapping):
        payload = dict(text)
        return payload, True, not validate_output_schema(payload)
    if not isinstance(text, str):
        return {}, False, False
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    candidate = match.group(0) if match else stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}, False, False
    return payload, True, not validate_output_schema(payload)


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load_lookup() -> dict[str, dict[str, Any]]:
    global _LOOKUP_CACHE
    if _LOOKUP_CACHE is not None:
        return _LOOKUP_CACHE
    path = Path(os.environ.get("REFUND_GRPO_REWARD_LOOKUP", DEFAULT_LOOKUP_PATH))
    lookup: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                lookup[str(row["sample_id"])] = row
    _LOOKUP_CACHE = lookup
    return lookup


def prompt_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content", "")) if isinstance(message, Mapping) else str(message)
        for message in messages
    )


def parse_lookup_metadata(messages: Any) -> tuple[str | None, str | None]:
    text = prompt_text(messages)
    sample_match = re.search(r"sample_id:\s*([A-Za-z0-9_\-]+)", text)
    exposure_match = re.search(r"prompt_exposure:\s*(full|partial|minimal|no_skill)", text)
    if sample_match:
        sample_id = sample_match.group(1)
    else:
        active_match = re.search(r"C-ACTIVE-([A-Za-z0-9_\-]+)", text)
        sample_id = active_match.group(1) if active_match else None
    exposure = exposure_match.group(1) if exposure_match else None
    return sample_id, exposure


def gold_compatible(parsed: Mapping[str, Any], gold: Mapping[str, Any], schema_valid: bool) -> bool:
    return bool(
        schema_valid
        and parsed.get("decision") == gold.get("decision")
        and parsed.get("need_human") == gold.get("need_human")
    )


def core_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return (
        0.45 * float(left.get("decision") == right.get("decision"))
        + 0.20 * float(left.get("need_human") == right.get("need_human"))
        + 0.15 * float(left.get("risk_level") == right.get("risk_level"))
        + 0.20
        * reason_code_f1(
            list(left.get("reason_codes", []) or []),
            list(right.get("reason_codes", []) or []),
        )
    )


def is_hard_case(gold: Mapping[str, Any], hard_tags: list[Any]) -> bool:
    if gold.get("need_human") is True:
        return True
    if set(gold.get("reason_codes", []) or []) & HARD_REASON_CODES:
        return True
    return bool(hard_tags)


class RefundDecisionReward(ORM):
    """Rule/evaluator reward for refund_decision GRPO.

    If a mini-batch contains multiple exposures for the same sample_id, this
    reward also adds the gold-gated exposure consistency term. Otherwise that
    term is zero, which is acceptable for smoke tests.
    """

    def __call__(self, completions, **kwargs) -> list[float]:
        target_outputs = as_list(kwargs.get("target_output"))
        sample_ids = as_list(kwargs.get("sample_id"))
        exposures = as_list(kwargs.get("prompt_exposure"))
        hard_tags_list = as_list(kwargs.get("hard_tags"))
        messages_list = as_list(kwargs.get("messages"))
        prompt_list = as_list(kwargs.get("prompts"))
        lookup = load_lookup()

        rows: list[dict[str, Any]] = []
        for index, completion in enumerate(completions):
            parsed, json_valid, schema_valid = parse_completion(completion)
            messages = messages_list[index] if index < len(messages_list) else []
            prompt = prompt_list[index] if index < len(prompt_list) else []
            parsed_sample_id, parsed_exposure = parse_lookup_metadata(messages or prompt)
            sample_id = (
                sample_ids[index]
                if index < len(sample_ids)
                else parsed_sample_id or f"row_{index}"
            )
            lookup_row = lookup.get(str(sample_id), {})
            gold = as_dict(
                target_outputs[index]
                if index < len(target_outputs)
                else lookup_row.get("target_output", {})
            )
            rows.append(
                {
                    "parsed": parsed,
                    "json_valid": json_valid,
                    "schema_valid": schema_valid,
                    "gold": gold,
                    "sample_id": sample_id,
                    "prompt_exposure": exposures[index] if index < len(exposures) else parsed_exposure or "unknown",
                    "hard_tags": hard_tags_list[index] if index < len(hard_tags_list) else lookup_row.get("hard_tags", []),
                }
            )

        consistency_by_row = [0.0 for _ in rows]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row["sample_id"])].append(index)
        for indices in groups.values():
            exposure_to_index = {
                str(rows[index]["prompt_exposure"]): index for index in indices
            }
            pair_rewards = []
            for pair, weight in PAIR_WEIGHTS.items():
                if pair[0] not in exposure_to_index or pair[1] not in exposure_to_index:
                    continue
                left = rows[exposure_to_index[pair[0]]]
                right = rows[exposure_to_index[pair[1]]]
                left_ok = gold_compatible(left["parsed"], left["gold"], left["schema_valid"])
                right_ok = gold_compatible(right["parsed"], right["gold"], right["schema_valid"])
                similarity = core_similarity(left["parsed"], right["parsed"])
                if left_ok and right_ok:
                    reward = similarity
                elif left_ok or right_ok:
                    reward = 0.25 * similarity
                else:
                    reward = -0.10 if similarity > 0.90 else 0.0
                pair_rewards.append((weight, reward))
            if pair_rewards:
                weighted = sum(weight * reward for weight, reward in pair_rewards) / sum(
                    weight for weight, _ in pair_rewards
                )
                for index in indices:
                    consistency_by_row[index] = weighted

        rewards = []
        for index, row in enumerate(rows):
            parsed = row["parsed"]
            gold = row["gold"]
            schema_valid = bool(row["schema_valid"])
            compatible = gold_compatible(parsed, gold, schema_valid)
            hard_tags = row["hard_tags"] if isinstance(row["hard_tags"], list) else []
            false_manual_review = bool(
                schema_valid
                and parsed.get("decision") == "manual_review"
                and gold.get("decision") != "manual_review"
            )
            reward = (
                0.30 * float(schema_valid and parsed.get("decision") == gold.get("decision"))
                + 0.15 * float(schema_valid and parsed.get("need_human") == gold.get("need_human"))
                + 0.10 * float(schema_valid and parsed.get("risk_level") == gold.get("risk_level"))
                + 0.15
                * reason_code_f1(
                    list(gold.get("reason_codes", []) or []),
                    list(parsed.get("reason_codes", []) or []) if schema_valid else [],
                )
                + 0.05 * float(schema_valid)
                + 0.15 * consistency_by_row[index]
                + 0.10 * float(is_hard_case(gold, hard_tags) and compatible)
                - 0.10 * float(false_manual_review)
            )
            rewards.append(float(reward))
        return rewards


orms["refund_decision"] = RefundDecisionReward
