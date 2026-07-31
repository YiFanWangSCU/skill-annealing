"""Build Stage C strong-backend-contract long-context eval prompts."""

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

from skill_annealing.refund_decision.skill import EXPOSURE_LEVELS, render_skill_prompt


EXPOSURES = ("full", "partial", "minimal", "no_skill")
BACKEND_KEYS = (
    "product_type",
    "days_since_delivery",
    "order_amount",
    "refund_count_30d",
    "order_status",
    "is_customized",
    "is_virtual",
    "is_fresh",
    "has_quality_issue",
    "has_evidence",
    "evidence_conflict",
    "order_info_conflict",
    "logistics_issue",
    "seller_fault",
    "is_used",
    "is_resellable",
    "user_claim",
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


def write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def dedupe_source_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in by_sample:
            by_sample[sample_id] = dict(record)
    return list(by_sample.values())


def compact_backend(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: fields.get(key) for key in BACKEND_KEYS if key in fields}


def user_text_from_record(record: Mapping[str, Any]) -> str:
    messages = record.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return json.dumps(compact_backend(record.get("fields", {})), ensure_ascii=False)


def render_conversation(record: Mapping[str, Any], rng: random.Random) -> str:
    claim = record["fields"].get("user_claim", "refund")
    lines = [
        "用户：我想处理这笔售后，请按当前工单里的订单来判断。",
        "客服：系统已经绑定 active_order_id，并拉取了 current_order 结构化字段。",
        f"用户：我的诉求类型大概是 {claim}，具体以 current_order 和证据摘要为准。",
        "客服：历史订单和用户画像只用于风险背景，不会覆盖 current_order。",
        "用户：请尽快给出能进入工作流的结构化结论。",
    ]
    rng.shuffle(lines)
    return "\n".join(lines)


def render_backend_contract(record: Mapping[str, Any], rng: random.Random) -> str:
    fields = compact_backend(record["fields"])
    payload = {
        "task": "refund_decision",
        "contract_version": "strong_backend_contract_v1",
        "active_order_id": f"C-ACTIVE-{record['sample_id']}",
        "current_order": fields,
        "evidence_summary": {
            "has_evidence": fields.get("has_evidence"),
            "evidence_conflict": fields.get("evidence_conflict"),
            "quality_issue_claimed": fields.get("has_quality_issue"),
            "logistics_issue_claimed": fields.get("logistics_issue"),
            "source": "backend_aggregated_current_order_evidence",
        },
        "user_risk_summary": {
            "refund_count_30d": fields.get("refund_count_30d"),
            "risk_tags": [
                tag
                for tag, enabled in [
                    ("high_value_order", (fields.get("order_amount") or 0) >= 1000),
                    ("frequent_refund_user", (fields.get("refund_count_30d") or 0) >= 3),
                    ("conflicting_evidence", bool(fields.get("evidence_conflict"))),
                ]
                if enabled
            ],
            "source": "backend_aggregated_user_profile",
        },
        "routing": {
            "selected_skill": "refund_decision",
            "skill_routing_confidence": 1.0,
            "model_must_not_route_to_other_skill": True,
        },
        "output_contract": {
            "format": "json_only",
            "schema": [
                "intent",
                "decision",
                "risk_level",
                "need_human",
                "reason_codes",
                "explanation",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def render_history_summary(record: Mapping[str, Any], rng: random.Random) -> str:
    fields = record["fields"]
    summary = {
        "note": "These are backend aggregates only. They are not current_order fields.",
        "same_user_history_30d": {
            "completed_orders": rng.choice([1, 2, 4, 7]),
            "refund_count": fields.get("refund_count_30d"),
            "complaint_count": rng.choice([0, 0, 1]),
        },
        "historical_order_summary": {
            "has_similar_category_order": rng.choice([False, True]),
            "has_old_after_sale_ticket": rng.choice([False, True]),
            "fields_hidden_by_backend_contract": True,
        },
    }
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)


def render_policy_noise(repeats: int) -> str:
    snippets = [
        "优惠券、发票、赠品结算规则不能覆盖 refund_decision。",
        "会员等级只影响客服优先级，不改变当前订单售后规则。",
        "历史订单只用于用户风险摘要，不能替代 current_order。",
        "如果 current_order 缺少关键字段，应请求补充证据或进入人工审核。",
        "模型只需要输出当前工单的 refund_decision JSON。",
        "后端已经完成 active_order_id 绑定，模型不得重新猜测订单。",
    ]
    return "\n".join(snippets[index % len(snippets)] for index in range(repeats))


def build_user_prompt(
    record: Mapping[str, Any],
    *,
    rng: random.Random,
    noise_repeats: int,
) -> str:
    return "\n".join(
        [
            "你是电商售后 agent 的决策模块。后端已经完成强约束：当前订单已经绑定，结构化字段已经聚合。",
            "你不需要也不应该从历史订单中猜测当前订单；只能对 backend_contract.current_order 应用 refund_decision skill。",
            "",
            "## Backend Contract",
            render_backend_contract(record, rng),
            "",
            "## 用户最新请求原文",
            user_text_from_record(record),
            "",
            "## 多轮对话摘要",
            render_conversation(record, rng),
            "",
            "## 用户历史与风险聚合摘要",
            render_history_summary(record, rng),
            "",
            "## 无关工作台政策噪声",
            render_policy_noise(noise_repeats),
            "",
            "## 最终任务",
            "只基于 Backend Contract 中的 current_order、evidence_summary、user_risk_summary 和已选中的 refund_decision skill 输出最终售后决策 JSON。",
            "必须只输出 JSON，不要输出 Markdown，不要解释你如何阅读上下文。",
        ]
    )


def build_messages(
    record: Mapping[str, Any],
    exposure: str,
    *,
    rng: random.Random,
    noise_repeats: int,
) -> list[dict[str, str]]:
    if exposure not in EXPOSURE_LEVELS:
        raise ValueError(f"unknown exposure: {exposure}")
    messages: list[dict[str, str]] = []
    system_prompt = render_skill_prompt(exposure)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": build_user_prompt(record, rng=rng, noise_repeats=noise_repeats),
        }
    )
    return messages


def build_eval_records(
    source_records: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    seed: int,
    noise_repeats: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base_records = dedupe_source_records(source_records)
    rng.shuffle(base_records)
    selected = base_records[:sample_count] if sample_count else base_records
    eval_records: list[dict[str, Any]] = []
    for record in selected:
        for exposure in EXPOSURES:
            item_rng = random.Random(f"{seed}:{record['sample_id']}:{exposure}:stage_c")
            eval_records.append(
                {
                    "sample_id": record["sample_id"],
                    "skill_id": "refund_decision",
                    "prompt_exposure": exposure,
                    "input_variant": "long_context_stage_c_strong_backend_contract",
                    "split": "eval",
                    "messages": build_messages(
                        record,
                        exposure,
                        rng=item_rng,
                        noise_repeats=noise_repeats,
                    ),
                    "fields": record["fields"],
                    "target_output": record["target_output"],
                    "hard_tags": list(record.get("hard_tags", []))
                    + ["long_context_stage_c", "strong_backend_contract"],
                    "active_order_id": f"C-ACTIVE-{record['sample_id']}",
                    "noise_repeats": noise_repeats,
                    "backend_contract": "current_order_guaranteed",
                }
            )
    return eval_records


def summarize(records: Sequence[Mapping[str, Any]], source_path: str) -> dict[str, Any]:
    char_lengths = [
        sum(len(message.get("content", "")) for message in record["messages"])
        for record in records
    ]
    exposures = Counter(str(record["prompt_exposure"]) for record in records)
    decisions = Counter(str(record["target_output"]["decision"]) for record in records)
    return {
        "name": "long_context_stage_c_strong_backend_contract_eval",
        "source_path": source_path,
        "prompt_count": len(records),
        "base_sample_count": len({record["sample_id"] for record in records}),
        "exposure_distribution": dict(sorted(exposures.items())),
        "decision_distribution": dict(sorted(decisions.items())),
        "char_length": {
            "min": min(char_lengths) if char_lengths else 0,
            "max": max(char_lengths) if char_lengths else 0,
            "mean": round(statistics.mean(char_lengths), 2) if char_lengths else 0,
            "median": round(statistics.median(char_lengths), 2) if char_lengths else 0,
        },
    }


def summary_markdown(summary: Mapping[str, Any], output_jsonl: str) -> str:
    lines = [
        "# Stage C Strong Backend Contract Eval Summary",
        "",
        f"- output_jsonl: `{output_jsonl}`",
        f"- source_path: `{summary['source_path']}`",
        f"- base_sample_count: `{summary['base_sample_count']}`",
        f"- prompt_count: `{summary['prompt_count']}`",
        "",
        "## Context Length",
        "",
        "| metric | chars |",
        "| --- | ---: |",
    ]
    for key, value in summary["char_length"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Exposure Distribution", "", "| exposure | count |", "| --- | ---: |"])
    for key, value in summary["exposure_distribution"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Decision Distribution", "", "| decision | count |", "| --- | ---: |"])
    for key, value in summary["decision_distribution"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="long_context_stage_c_contract")
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--noise-repeats", type=int, default=28)
    args = parser.parse_args()

    if args.sample_count < 0:
        raise SystemExit("--sample-count must be >= 0; use 0 for all source samples")
    if args.noise_repeats < 0:
        raise SystemExit("--noise-repeats must be >= 0")

    source_records = load_jsonl(args.source_eval)
    eval_records = build_eval_records(
        source_records,
        sample_count=args.sample_count,
        seed=args.seed,
        noise_repeats=args.noise_repeats,
    )

    output_dir = Path(args.output_dir)
    output_jsonl = output_dir / f"{args.name}_prompts.jsonl"
    summary_json = output_dir / f"{args.name}_summary.json"
    summary_md = output_dir / f"{args.name}_summary.md"

    write_jsonl(output_jsonl, eval_records)
    summary = summarize(eval_records, args.source_eval)
    write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    write_text(summary_md, summary_markdown(summary, str(output_jsonl)))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
