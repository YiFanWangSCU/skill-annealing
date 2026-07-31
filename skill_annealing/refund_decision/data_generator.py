"""Synthetic sample generation for refund_decision."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .prompt_builder import (
    build_annealed_sft,
    build_eval_records,
    build_full_skill_sft,
    build_no_skill_sft,
)
from .rule_engine import decide


PRODUCT_PROFILES = {
    "clothing": {"is_fresh": False, "is_customized": False, "is_virtual": False},
    "electronics": {"is_fresh": False, "is_customized": False, "is_virtual": False},
    "home": {"is_fresh": False, "is_customized": False, "is_virtual": False},
    "fresh": {"is_fresh": True, "is_customized": False, "is_virtual": False},
    "customized": {"is_fresh": False, "is_customized": True, "is_virtual": False},
    "virtual": {"is_fresh": False, "is_customized": False, "is_virtual": True},
}

PRODUCT_NAMES = {
    "clothing": "衣服",
    "electronics": "耳机",
    "home": "收纳架",
    "fresh": "水果礼盒",
    "customized": "定制刻字杯",
    "virtual": "会员权益",
}


def generate_samples(sample_count: int = 200, seed: int = 42) -> list[dict[str, Any]]:
    """Generate labeled samples. Labels are always produced by the rule engine."""

    rng = random.Random(seed)
    fields_list = _canonical_difficult_cases()
    while len(fields_list) < sample_count:
        fields_list.append(_random_fields(rng))

    samples: list[dict[str, Any]] = []
    for idx, fields in enumerate(fields_list[:sample_count], start=1):
        user_input, evidence_description = render_user_request(fields, rng)
        target_output = decide(fields).to_dict()
        sample = {
            "sample_id": f"refund_{idx:06d}",
            "skill_id": "refund_decision",
            "fields": fields,
            "user_input": user_input,
            "target_output": target_output,
        }
        if evidence_description:
            sample["evidence_description"] = evidence_description
        samples.append(sample)
    return samples


def export_mvp_datasets(
    output_dir: str | Path = "data/refund_decision",
    sample_count: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(sample_count=sample_count, seed=seed)
    datasets = {
        "samples": samples,
        "full_skill_sft": build_full_skill_sft(samples),
        "no_skill_sft": build_no_skill_sft(samples),
        "annealed_sft": build_annealed_sft(samples, seed=seed),
        "eval_prompts": build_eval_records(samples),
    }

    paths: dict[str, str] = {}
    for name, records in datasets.items():
        path = output_path / f"{name}.jsonl"
        write_jsonl(path, records)
        paths[name] = str(path)

    return {
        "sample_count": len(samples),
        "paths": paths,
        "label_distribution": _target_distribution(samples),
        "annealed_exposure_distribution": dict(
            Counter(row["prompt_exposure"] for row in datasets["annealed_sft"])
        ),
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_user_request(
    fields: dict[str, Any], rng: random.Random | None = None
) -> tuple[str, str | None]:
    rng = rng or random.Random(42)
    product_name = PRODUCT_NAMES.get(fields["product_type"], "商品")
    days = fields.get("days_since_delivery")
    day_text = "刚收到" if days in {0, 1} else f"收到 {days} 天了"
    evidence_description: str | None = None
    claim = fields.get("user_claim", "no_reason")

    if claim == "exchange":
        request = f"我买的{product_name}{day_text}，想换一个合适的，可以处理吗？"
    elif claim == "size_not_fit":
        request = f"这个{product_name}{day_text}，不太合适，我想退货退款。"
    elif claim == "quality_issue":
        request = f"{product_name}{day_text}，发现有质量问题，我想申请售后。"
        evidence_description = (
            "用户已上传清晰照片或视频。"
            if fields.get("has_evidence")
            else "用户暂未上传有效图片或视频。"
        )
    elif claim == "logistics_issue":
        request = f"{product_name}物流状态异常，我没有正常收到，想申请退款。"
        evidence_description = (
            "物流轨迹显示异常或商家承认发错/漏发。"
            if fields.get("has_evidence") or fields.get("seller_fault")
            else "物流责任暂不明确。"
        )
    elif claim == "claims_unused_but_used":
        request = f"这个{product_name}我没有用过，不影响销售，想退。"
    elif claim == "strong_complaint":
        request = f"这个{product_name}问题很严重，我已经很不满意了，必须马上退款。"
        evidence_description = (
            "用户语气强烈，但未提供可以验证质量问题的证据。"
            if not fields.get("has_evidence")
            else "用户提供了证据，但描述和证据存在冲突。"
        )
    else:
        templates = [
            f"{product_name}{day_text}，我不想要了，可以退吗？",
            f"我买错了{product_name}，想申请退款。",
            f"{product_name}还没达到我的预期，麻烦帮我处理退货退款。",
        ]
        request = rng.choice(templates)

    if fields.get("is_used") and claim != "claims_unused_but_used":
        request += " 商品我已经试用过。"
    if not fields.get("is_resellable", True):
        request += " 包装或状态可能已经影响二次销售。"
    return request, evidence_description


def _random_fields(rng: random.Random) -> dict[str, Any]:
    product_type = rng.choices(
        list(PRODUCT_PROFILES),
        weights=[28, 20, 18, 12, 11, 11],
        k=1,
    )[0]
    profile = PRODUCT_PROFILES[product_type]
    claim = rng.choices(
        [
            "no_reason",
            "size_not_fit",
            "quality_issue",
            "exchange",
            "logistics_issue",
            "strong_complaint",
        ],
        weights=[26, 20, 24, 10, 8, 12],
        k=1,
    )[0]

    has_quality_issue = claim in {"quality_issue", "strong_complaint"} and rng.random() < 0.75
    has_evidence = has_quality_issue and rng.random() < 0.58
    logistics_issue = claim == "logistics_issue"
    seller_fault = (has_quality_issue and has_evidence and rng.random() < 0.35) or (
        logistics_issue and rng.random() < 0.45
    )
    is_used = rng.random() < 0.28
    is_resellable = not is_used and rng.random() > 0.14

    fields = {
        "product_type": product_type,
        "days_since_delivery": rng.randint(0, 18),
        "is_used": is_used,
        "is_resellable": is_resellable,
        "has_quality_issue": has_quality_issue,
        "has_evidence": has_evidence,
        "order_amount": rng.choice([59, 99, 199, 299, 499, 899, 1299, 2499]),
        "refund_count_30d": rng.choices([0, 1, 2, 3, 4, 5], [45, 24, 14, 8, 6, 3])[0],
        "user_claim": claim,
        "order_status": "not_delivered" if logistics_issue and rng.random() < 0.35 else "delivered",
        "seller_fault": seller_fault,
        "logistics_issue": logistics_issue,
        "evidence_conflict": has_evidence and rng.random() < 0.06,
        "order_info_conflict": False,
        **profile,
    }
    return fields


def _canonical_difficult_cases() -> list[dict[str, Any]]:
    base_common = {
        "days_since_delivery": 3,
        "is_used": False,
        "is_resellable": True,
        "has_quality_issue": False,
        "has_evidence": False,
        "order_amount": 199,
        "refund_count_30d": 0,
        "user_claim": "no_reason",
        "order_status": "delivered",
        "seller_fault": False,
        "logistics_issue": False,
        "evidence_conflict": False,
        "order_info_conflict": False,
    }

    cases = []
    for product_type in ["clothing", "fresh", "customized", "virtual"]:
        cases.append({**base_common, "product_type": product_type, **PRODUCT_PROFILES[product_type]})

    cases.extend(
        [
            {
                **base_common,
                "product_type": "clothing",
                **PRODUCT_PROFILES["clothing"],
                "days_since_delivery": 9,
                "user_claim": "size_not_fit",
            },
            {
                **base_common,
                "product_type": "electronics",
                **PRODUCT_PROFILES["electronics"],
                "has_quality_issue": True,
                "has_evidence": True,
                "user_claim": "quality_issue",
            },
            {
                **base_common,
                "product_type": "electronics",
                **PRODUCT_PROFILES["electronics"],
                "has_quality_issue": True,
                "has_evidence": False,
                "user_claim": "quality_issue",
            },
            {
                **base_common,
                "product_type": "clothing",
                **PRODUCT_PROFILES["clothing"],
                "order_amount": 2499,
            },
            {
                **base_common,
                "product_type": "home",
                **PRODUCT_PROFILES["home"],
                "refund_count_30d": 4,
            },
            {
                **base_common,
                "product_type": "clothing",
                **PRODUCT_PROFILES["clothing"],
                "is_used": True,
                "is_resellable": False,
                "user_claim": "claims_unused_but_used",
            },
            {
                **base_common,
                "product_type": "fresh",
                **PRODUCT_PROFILES["fresh"],
                "has_quality_issue": True,
                "has_evidence": True,
                "seller_fault": True,
                "user_claim": "quality_issue",
            },
            {
                **base_common,
                "product_type": "customized",
                **PRODUCT_PROFILES["customized"],
                "has_quality_issue": True,
                "has_evidence": True,
                "seller_fault": True,
                "user_claim": "quality_issue",
            },
            {
                **base_common,
                "product_type": "virtual",
                **PRODUCT_PROFILES["virtual"],
                "has_quality_issue": True,
                "has_evidence": True,
                "seller_fault": True,
                "user_claim": "quality_issue",
            },
            {
                **base_common,
                "product_type": "electronics",
                **PRODUCT_PROFILES["electronics"],
                "has_quality_issue": True,
                "has_evidence": False,
                "user_claim": "strong_complaint",
            },
            {
                **base_common,
                "product_type": "electronics",
                **PRODUCT_PROFILES["electronics"],
                "has_quality_issue": True,
                "has_evidence": True,
                "evidence_conflict": True,
                "user_claim": "strong_complaint",
            },
            {
                **base_common,
                "product_type": "clothing",
                **PRODUCT_PROFILES["clothing"],
                "user_claim": "exchange",
            },
            {
                **base_common,
                "product_type": "home",
                **PRODUCT_PROFILES["home"],
                "logistics_issue": True,
                "seller_fault": True,
                "has_evidence": True,
                "user_claim": "logistics_issue",
                "order_status": "not_delivered",
            },
        ]
    )
    return cases


def _target_distribution(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "decision": dict(Counter(row["target_output"]["decision"] for row in samples)),
        "risk_level": dict(Counter(row["target_output"]["risk_level"] for row in samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refund_decision MVP data.")
    parser.add_argument("--output-dir", default="data/refund_decision")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = export_mvp_datasets(
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
