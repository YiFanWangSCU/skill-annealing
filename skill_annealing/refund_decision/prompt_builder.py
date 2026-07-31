"""Build SFT and evaluation prompts for different skill exposure levels."""

from __future__ import annotations

import json
import random
from typing import Any, Iterable, Mapping, Sequence

from .skill import EXPOSURE_LEVELS, render_skill_prompt


EXPOSURE_ORDER = ("full", "partial", "minimal", "no_skill")


def build_user_prompt(sample: Mapping[str, Any]) -> str:
    fields = sample["fields"]
    order_facts = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    evidence = sample.get("evidence_description")
    lines = [
        "请处理以下电商售后请求。",
        "",
        f"用户请求：{sample['user_input']}",
        f"订单事实：{order_facts}",
    ]
    if evidence:
        lines.append(f"证据描述：{evidence}")
    return "\n".join(lines)


def build_messages(
    sample: Mapping[str, Any],
    exposure: str,
    include_answer: bool = True,
) -> list[dict[str, str]]:
    if exposure not in EXPOSURE_LEVELS:
        raise ValueError(f"unknown exposure level: {exposure!r}")

    messages: list[dict[str, str]] = []
    system_prompt = render_skill_prompt(exposure)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": build_user_prompt(sample)})

    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    sample["target_output"], ensure_ascii=False, sort_keys=True
                ),
            }
        )

    return messages


def build_sft_record(sample: Mapping[str, Any], exposure: str) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "skill_id": sample["skill_id"],
        "prompt_exposure": exposure,
        "messages": build_messages(sample, exposure, include_answer=True),
        "fields": sample["fields"],
        "target_output": sample["target_output"],
    }


def build_eval_record(sample: Mapping[str, Any], exposure: str) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "skill_id": sample["skill_id"],
        "prompt_exposure": exposure,
        "messages": build_messages(sample, exposure, include_answer=False),
        "fields": sample["fields"],
        "target_output": sample["target_output"],
    }


def assign_annealed_exposures(
    sample_count: int, seed: int = 42
) -> list[str]:
    """Assign exposure levels using the MVP annealing schedule."""

    rng = random.Random(seed)
    exposures: list[str] = []
    first_cut = int(sample_count * 0.30)
    second_cut = int(sample_count * 0.70)

    for idx in range(sample_count):
        if idx < first_cut:
            weights = [("full", 0.80), ("partial", 0.20), ("minimal", 0.00)]
        elif idx < second_cut:
            weights = [("full", 0.30), ("partial", 0.50), ("minimal", 0.20)]
        else:
            weights = [("full", 0.10), ("partial", 0.40), ("minimal", 0.50)]
        exposures.append(_weighted_choice(rng, weights))

    return exposures


def build_full_skill_sft(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_sft_record(sample, "full") for sample in samples]


def build_no_skill_sft(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_sft_record(sample, "no_skill") for sample in samples]


def build_annealed_sft(
    samples: Sequence[Mapping[str, Any]], seed: int = 42
) -> list[dict[str, Any]]:
    exposures = assign_annealed_exposures(len(samples), seed=seed)
    return [
        build_sft_record(sample, exposure)
        for sample, exposure in zip(samples, exposures, strict=True)
    ]


def build_eval_records(
    samples: Iterable[Mapping[str, Any]],
    exposures: Sequence[str] = EXPOSURE_ORDER,
) -> list[dict[str, Any]]:
    return [
        build_eval_record(sample, exposure)
        for sample in samples
        for exposure in exposures
    ]


def _weighted_choice(rng: random.Random, weights: Sequence[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in weights)
    point = rng.random() * total
    upto = 0.0
    for value, weight in weights:
        upto += weight
        if point <= upto:
            return value
    return weights[-1][0]
