"""Frozen local prompt, target, and lexical-template registries."""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping

from .core import sha256_text


LEXICAL_TEMPLATES = {
    "train": [
        "请依据业务规则处理下列事实，并只输出约定 JSON：{facts_json}",
        "阅读下面的结构化事实后给出合规决定。仅返回 JSON。\n{facts_json}",
        "请完成本次业务判断；输入事实如下：\n{facts_json}",
        "根据这些已核实事实作出决定，不要补充未给出的信息：{facts_json}",
    ],
    "validation": ["请检查以下业务事实并返回标准 JSON：{facts_json}"],
    "discovery_eval": [
        "现在需要你判断这组业务事实。请仅返回目标 JSON：{facts_json}",
        "对下列事实执行既定业务判断，输出不得包含 JSON 之外的文本：\n{facts_json}",
    ],
    "confirmation": [
        "请独立处理以下已确认事实，并按指定结构作答：{facts_json}",
        "下面是一组新的业务事实；请返回唯一的 JSON 决定。\n{facts_json}",
    ],
}


def normalize_template_source(text: str) -> str:
    lines = [line.rstrip() for line in unicodedata.normalize("NFC", text).splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def lexical_template_id(text: str) -> str:
    return sha256_text(normalize_template_source(text))


def build_target_registry(protocol: Mapping[str, Any]) -> dict[str, Any]:
    templates: dict[str, dict[str, str]] = {}
    for skill_id, skill in protocol["skills"].items():
        templates[skill_id] = {
            code: f"命中 {skill_id} 的冻结规则 {code}。"
            for code in skill["reason_codes"]
        }
    return {
        "protocol_version": protocol["protocol_version"],
        "templates": templates,
    }


def build_prompt_registry(protocol: Mapping[str, Any]) -> dict[str, Any]:
    prompts: dict[str, dict[str, str]] = {}
    for skill_id, skill in protocol["skills"].items():
        rules = "\n".join(
            f"{index + 1}. {rule}" for index, rule in enumerate(skill["priority_order"])
        )
        families = "、".join(skill["rule_families"])
        task = f"你正在执行 {skill_id}。"
        schema = (
            "只输出 JSON，字段必须为 decision、risk_level、need_human、"
            "reason_codes、explanation。"
        )
        prompts[skill_id] = {
            "full": f"{task}\n{schema}\n按以下 first-match 规则执行：\n{rules}",
            "partial": (
                f"{task}\n{schema}\n必须覆盖这些规则族并遵守优先级：{families}。"
            ),
            "minimal": f"{task}\n{schema}",
            "no_skill": "请遵循用户指令并只返回一个 JSON 对象。",
            "full_length_irrelevant": (
                "你是格式化助手。请保持信息完整、检查字段类型并返回结构清晰的 JSON。"
                "不要解释内部过程，也不要添加 Markdown。"
            ),
            "partial_semantic_paraphrase_a": (
                f"任务是 {skill_id}。返回固定五字段 JSON；逐项考虑 {families}，"
                "冲突时采用预先规定的优先顺序。"
            ),
            "partial_semantic_paraphrase_b": (
                f"完成 {skill_id} 判断。答案只能是约定 JSON。需要兼顾的规则类别包括："
                f"{families}；若多条同时命中，选最高优先级。"
            ),
        }
    return {
        "protocol_version": protocol["protocol_version"],
        "token_exact_irrelevant_status": "pending_remote_qwen35_tokenizer_audit",
        "prompts": prompts,
    }
