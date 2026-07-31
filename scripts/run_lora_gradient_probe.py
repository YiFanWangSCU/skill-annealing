"""Measure completion-only LoRA gradients for strictly paired exposures.

The runner is intentionally optimizer-free. Full per-record gradients live only in
CPU memory until the selected exposures for one paired group have been reduced to
norm/dot/cosine/sketch summaries; they are never serialized.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import inspect
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPOSURES = ("full", "partial", "minimal", "no_skill")
CONTEXT_LOADS = ("short", "long")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def token_hash(token_ids: Sequence[int]) -> str:
    payload = canonical_json([int(token_id) for token_id in token_ids])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, values: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _extract_input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"unsupported tokenizer output: {type(value).__name__}")
    return [int(item) for item in value]


def build_completion_mask(
    tokenizer: Any,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Render a record and prove that only its assistant span is supervised."""

    messages = [dict(message) for message in record["messages"]]
    roles = [message.get("role") for message in messages]
    if roles not in (
        ["system", "user", "assistant"],
        ["user", "assistant"],
    ):
        raise ValueError(f"unexpected role structure for {record.get('record_id')}")

    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    sequence_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    common_prefix_count = 0
    for prompt_token, sequence_token in zip(prompt_ids, sequence_ids):
        if prompt_token != sequence_token:
            break
        common_prefix_count += 1
    # Qwen3.5's generation template ends in ``<think>\n`` while a supplied
    # assistant answer renders ``<think>\n\n</think>\n\n``. The last token can
    # therefore merge differently even though all earlier context is identical.
    # The longest common token prefix is the only exact causal masking boundary.
    divergent_prompt_tail = len(prompt_ids) - common_prefix_count
    if divergent_prompt_tail > 8:
        raise ValueError(
            f"chat-template prompt diverges by {divergent_prompt_tail} tokens for "
            f"{record.get('record_id')}"
        )
    if len(sequence_ids) <= common_prefix_count:
        raise ValueError(f"empty assistant span for {record.get('record_id')}")

    expected_prompt = record.get("prompt_token_count")
    expected_sequence = record.get("full_sequence_token_count")
    if expected_prompt is not None and len(prompt_ids) != int(expected_prompt):
        raise ValueError(
            f"prompt token drift for {record.get('record_id')}: "
            f"{len(prompt_ids)} != {expected_prompt}"
        )
    if expected_sequence is not None and len(sequence_ids) != int(expected_sequence):
        raise ValueError(
            f"sequence token drift for {record.get('record_id')}: "
            f"{len(sequence_ids)} != {expected_sequence}"
        )

    target_text = str(messages[-1]["content"])
    target_ids = _extract_input_ids(
        tokenizer.encode(target_text, add_special_tokens=False)
    )
    expected_target_hash = record.get("supervised_token_hash")
    actual_target_hash = token_hash(target_ids)
    if expected_target_hash and actual_target_hash != expected_target_hash:
        raise ValueError(f"assistant target token drift for {record.get('record_id')}")

    decoded_assistant_span = tokenizer.decode(
        sequence_ids[common_prefix_count:],
        skip_special_tokens=False,
    )
    assistant_span_contains_target = target_text in decoded_assistant_span
    if not assistant_span_contains_target:
        raise ValueError(
            f"assistant span does not contain exact target text for "
            f"{record.get('record_id')}"
        )

    labels = [-100] * common_prefix_count + sequence_ids[common_prefix_count:]
    prompt_mask_exact = all(
        label == -100 for label in labels[:common_prefix_count]
    )
    assistant_mask_exact = all(
        label != -100 for label in labels[common_prefix_count:]
    )
    if not prompt_mask_exact or not assistant_mask_exact:
        raise AssertionError(f"invalid completion mask for {record.get('record_id')}")

    return {
        "input_ids": sequence_ids,
        "labels": labels,
        "generation_prompt_token_count": len(prompt_ids),
        "ignored_prefix_token_count": common_prefix_count,
        "generation_prompt_divergent_tail_token_count": divergent_prompt_tail,
        "assistant_span_token_count": len(sequence_ids) - common_prefix_count,
        "target_content_token_count": len(target_ids),
        "target_content_token_hash": actual_target_hash,
        "prompt_labels_all_ignore": prompt_mask_exact,
        "assistant_labels_all_supervised": assistant_mask_exact,
        "assistant_span_contains_target": assistant_span_contains_target,
    }


def select_probe_records(
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] | None,
    max_cases: int | None,
    preflight: bool,
    context_load: str | None = None,
    expected_exposures: Sequence[str] = EXPOSURES,
) -> list[dict[str, Any]]:
    expected_exposures = tuple(str(item) for item in expected_exposures)
    if not expected_exposures or len(set(expected_exposures)) != len(
        expected_exposures
    ):
        raise ValueError("expected exposures must be non-empty and unique")
    available_cases = sorted({str(record["case_id"]) for record in records})
    selected_cases = list(case_ids or available_cases)
    missing = sorted(set(selected_cases) - set(available_cases))
    if missing:
        raise ValueError(f"case IDs not present in input: {missing}")
    if preflight and not case_ids:
        selected_cases = available_cases[:1]
    if max_cases is not None:
        selected_cases = selected_cases[:max_cases]

    selected = [
        dict(record)
        for record in records
        if str(record["case_id"]) in set(selected_cases)
        and (
            context_load is None
            or str(record["context_load"]) == str(context_load)
        )
    ]
    selected_contexts = (
        list(CONTEXT_LOADS) if context_load is None else [str(context_load)]
    )
    if context_load is not None and str(context_load) not in CONTEXT_LOADS:
        raise ValueError(
            f"context load must be one of {list(CONTEXT_LOADS)}, got {context_load!r}"
        )
    order = {
        (context, exposure): context_index * len(expected_exposures)
        + exposure_index
        for context_index, context in enumerate(CONTEXT_LOADS)
        for exposure_index, exposure in enumerate(expected_exposures)
    }
    selected.sort(
        key=lambda record: (
            selected_cases.index(str(record["case_id"])),
            order.get(
                (str(record["context_load"]), str(record["prompt_exposure"])),
                999,
            ),
        )
    )

    group_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in selected:
        group_counts[(str(record["case_id"]), str(record["context_load"]))].add(
            str(record["prompt_exposure"])
        )
    expected_groups = len(selected_cases) * len(selected_contexts)
    if len(group_counts) != expected_groups:
        raise ValueError(
            f"selected records have {len(group_counts)} paired groups; "
            f"expected {expected_groups}"
        )
    for key, exposures in group_counts.items():
        if exposures != set(expected_exposures):
            raise ValueError(f"paired group {key} has exposures {sorted(exposures)}")
    expected_records = expected_groups * len(expected_exposures)
    if len(selected) != expected_records:
        raise ValueError(
            f"selected {len(selected)} records; expected exactly {expected_records}"
        )
    preflight_records = len(selected_contexts) * len(expected_exposures)
    if preflight and len(selected) != preflight_records:
        raise ValueError(
            f"preflight must contain exactly {preflight_records} records, "
            f"got {len(selected)}"
        )
    return selected


def lora_module_name(parameter_name: str) -> str:
    for marker in (".lora_A.", ".lora_B."):
        if marker in parameter_name:
            return parameter_name.split(marker, 1)[0]
    return parameter_name.rsplit(".", 1)[0]


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _pairwise_statistics(vectors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = list(vectors)
    dot_matrix: dict[str, dict[str, float]] = {name: {} for name in names}
    cosine_matrix: dict[str, dict[str, float | None]] = {name: {} for name in names}
    norm_sq: dict[str, float] = {}
    for name in names:
        norm_sq[name] = sum(
            float((tensor.double() * tensor.double()).sum().item())
            for tensor in vectors[name].values()
        )
    for left in names:
        for right in names:
            dot = sum(
                float(
                    (
                        vectors[left][parameter].double()
                        * vectors[right][parameter].double()
                    ).sum().item()
                )
                for parameter in vectors[left]
            )
            denominator = math.sqrt(norm_sq[left] * norm_sq[right])
            dot_matrix[left][right] = dot
            cosine_matrix[left][right] = dot / denominator if denominator else None
    return {
        "dot": dot_matrix,
        "cosine": cosine_matrix,
        "norm": {name: math.sqrt(value) for name, value in norm_sq.items()},
    }


def _center_vectors(vectors: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    names = list(vectors)
    if not names:
        raise ValueError("cannot center an empty vector mapping")
    centered: dict[str, dict[str, Any]] = {name: {} for name in names}
    for parameter in vectors[names[0]]:
        mean_tensor = sum(vectors[name][parameter] for name in names) / len(names)
        for name in names:
            centered[name][parameter] = vectors[name][parameter] - mean_tensor
    return centered


def _module_norms(vector: Mapping[str, Any]) -> dict[str, float]:
    norm_sq: dict[str, float] = defaultdict(float)
    for parameter, tensor in vector.items():
        norm_sq[lora_module_name(parameter)] += float(
            (tensor.double() * tensor.double()).sum().item()
        )
    return {module: math.sqrt(value) for module, value in sorted(norm_sq.items())}


def _count_sketch(vector: Mapping[str, Any], *, dimension: int, seed: int) -> list[float]:
    import torch

    sketch = torch.zeros(dimension, dtype=torch.float64)
    for parameter, tensor in sorted(vector.items()):
        flat = tensor.double().reshape(-1)
        salt = int.from_bytes(
            hashlib.sha256(f"{seed}:{parameter}".encode("utf-8")).digest()[:8],
            "big",
        )
        positions = torch.arange(flat.numel(), dtype=torch.int64)
        buckets = (positions * 1103515245 + salt) % dimension
        signs = (((positions * 214013 + salt // 7) & 1) * 2 - 1).double()
        sketch.scatter_add_(0, buckets, flat * signs)
    return [float(value) for value in sketch.tolist()]


def _gradient_snapshot(
    named_trainable: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, parameter in named_trainable:
        if parameter.grad is None:
            raise RuntimeError(f"trainable parameter has no gradient: {name}")
        snapshot[name] = parameter.grad.detach().float().cpu().clone()
    return snapshot


def _weights_unchanged(
    named_trainable: Sequence[tuple[str, Any]],
    initial_weights: Mapping[str, Any],
) -> bool:
    import torch

    return all(
        torch.equal(parameter.detach().cpu(), initial_weights[name])
        for name, parameter in named_trainable
    )


def _load_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    # The historical ms-swift checkpoint targets ``model.language_model.*``.
    # Qwen3.5's image-text wrapper preserves that hierarchy; the causal-LM Auto
    # class resolves to Qwen3_5ForCausalLM and silently removes it.
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    offloaded_modules: list[str] = []
    if args.offload_vision_tower:
        multimodal_model = getattr(base_model, "model", None)
        visual = getattr(multimodal_model, "visual", None)
        if visual is None:
            raise RuntimeError("could not locate Qwen3.5 visual tower for CPU offload")
        visual.to("cpu")
        offloaded_modules.append("model.visual")
        torch.cuda.empty_cache()
    if args.init_lora:
        # Re-seed immediately before adapter injection so the frozen LoRA init
        # does not depend on RNG consumed while constructing/loading the base.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            inference_mode=False,
            init_lora_weights=True,
        )
        model = get_peft_model(base_model, lora_config)
        checkpoint_source = {
            "kind": "initialized_lora",
            "seed": args.seed,
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        }
    else:
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=True,
        )
        checkpoint_source = {
            "kind": "adapter_checkpoint",
            "path": args.adapter_path,
        }
    gradient_checkpointing_metadata = {
        "enabled": bool(args.gradient_checkpointing),
        "implementation": None,
        "use_reentrant": None,
        "dropout_policy": "historical_model_eval",
    }
    if args.gradient_checkpointing:
        # Require an explicit non-reentrant implementation rather than falling
        # back silently on older Transformers versions with different semantics.
        enable_signature = inspect.signature(model.gradient_checkpointing_enable)
        if "gradient_checkpointing_kwargs" not in enable_signature.parameters:
            raise RuntimeError(
                "gradient checkpointing requested, but this Transformers version "
                "does not expose gradient_checkpointing_kwargs; refusing to "
                "silently change checkpointing semantics"
            )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        gradient_checkpointing_metadata.update(
            {
                "implementation": "transformers.gradient_checkpointing_enable",
                "use_reentrant": False,
                "dropout_policy": (
                    "temporary_model_train_with_all_torch_dropout_eval"
                ),
            }
        )
    model.eval()
    model.config.use_cache = False
    return model, tokenizer, torch, {
        "offloaded_modules": offloaded_modules,
        "checkpoint_source": checkpoint_source,
        "gradient_checkpointing": gradient_checkpointing_metadata,
    }


def _enter_gradient_checkpointing_forward(model: Any, torch: Any) -> None:
    """Enable checkpointing's training gate without re-enabling dropout."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def _leave_gradient_checkpointing_forward(model: Any) -> None:
    model.eval()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    records = select_probe_records(
        load_jsonl(args.input),
        case_ids=args.case_id,
        max_cases=args.max_cases,
        preflight=args.preflight,
        context_load=args.context_load,
        expected_exposures=args.exposures,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "per_case_metrics.jsonl"
    modules_path = output_dir / "module_summary.jsonl"
    for path in (metrics_path, modules_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    random.seed(args.seed)
    model, tokenizer, torch, load_metadata = _load_model_and_tokenizer(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_names = [name for name, _ in named_trainable]
    non_lora_trainable = [name for name in trainable_names if "lora_" not in name]
    if not named_trainable:
        raise RuntimeError("adapter has no trainable parameters")
    if non_lora_trainable:
        raise RuntimeError(f"non-LoRA trainable parameters: {non_lora_trainable[:10]}")
    initial_weights = {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_trainable
    }

    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    run_config = {
        "protocol_version": "paired_exposure_gradient_v0",
        "run_started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "preflight" if args.preflight else "probe",
        "input": str(args.input),
        "model_path": args.model_path,
        "checkpoint_id": args.checkpoint_id,
        "adapter_path": args.adapter_path,
        "checkpoint_source": load_metadata["checkpoint_source"],
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "sketch_dimension": args.sketch_dim,
        "activation_cpu_offload": args.activation_offload,
        "offloaded_modules": load_metadata["offloaded_modules"],
        "gradient_checkpointing": load_metadata["gradient_checkpointing"],
        "loss_logits_scope": "assistant_span_plus_preceding_prediction_token",
        "loss_equivalence_atol": args.loss_equivalence_atol,
        "record_count": len(records),
        "case_ids": sorted({str(record["case_id"]) for record in records}),
        "context_loads": sorted({str(record["context_load"]) for record in records}),
        "exposures": list(args.exposures),
        "model_training": bool(model.training),
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "full_gradient_serialization": False,
        "trainable_parameter_tensor_count": len(named_trainable),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in named_trainable
        ),
        "trainable_parameter_names": trainable_names,
        "non_lora_trainable_parameter_names": non_lora_trainable,
        "chat_template_hash": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "torch_version": torch.__version__,
    }
    write_json(output_dir / "run_config.json", run_config)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["case_id"]), str(record["context_load"]))].append(record)

    all_mask_checks: list[bool] = []
    all_finite_checks: list[bool] = []
    loss_equivalence_checks: list[bool] = []
    processed_records = 0
    for group_index, ((case_id, context_load), group_records) in enumerate(
        grouped.items(), start=1
    ):
        by_exposure = {str(record["prompt_exposure"]): record for record in group_records}
        gradients: dict[str, dict[str, Any]] = {}
        exposure_metrics: dict[str, dict[str, Any]] = {}

        for exposure in args.exposures:
            record = by_exposure[exposure]
            mask = build_completion_mask(tokenizer, record)
            if len(mask["input_ids"]) > args.max_length:
                raise ValueError(
                    f"sequence length {len(mask['input_ids'])} exceeds "
                    f"--max-length {args.max_length} for {record['record_id']}"
                )
            all_mask_checks.extend(
                [
                    bool(mask["prompt_labels_all_ignore"]),
                    bool(mask["assistant_labels_all_supervised"]),
                    bool(mask["assistant_span_contains_target"]),
                ]
            )

            input_ids = torch.tensor(
                [mask["input_ids"]], dtype=torch.long, device=args.device
            )
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor(
                [mask["labels"]], dtype=torch.long, device=args.device
            )
            loss_slice_start = max(int(mask["ignored_prefix_token_count"]) - 1, 0)
            loss_labels = labels[:, loss_slice_start:]
            logits_to_keep = int(input_ids.shape[-1]) - loss_slice_start
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()
            memory_before = int(torch.cuda.memory_allocated(args.device))
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            if args.gradient_checkpointing:
                _enter_gradient_checkpointing_forward(model, torch)
            activation_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=False)
                if args.activation_offload
                else contextlib.nullcontext()
            )
            try:
                with activation_context:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=loss_labels,
                        logits_to_keep=logits_to_keep,
                        use_cache=False,
                    )
                    loss = outputs.loss
                    loss.backward()
            finally:
                if args.gradient_checkpointing:
                    _leave_gradient_checkpointing_forward(model)
            torch.cuda.synchronize(args.device)
            runtime_seconds = time.perf_counter() - started
            peak_allocated = int(torch.cuda.max_memory_allocated(args.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(args.device))

            gradient = _gradient_snapshot(named_trainable)
            gradients[exposure] = gradient
            global_norm = math.sqrt(
                sum(
                    float((tensor.double() * tensor.double()).sum().item())
                    for tensor in gradient.values()
                )
            )
            loss_value = float(loss.detach().float().item())
            reference_full_label_loss: float | None = None
            optimized_loss_absolute_difference: float | None = None
            loss_equivalence_passed: bool | None = None
            if args.preflight and processed_records == 0:
                # One no-grad reference proves that slicing logits/labels to the
                # completion window is numerically equivalent to the canonical
                # full-label masked loss, while avoiding its memory cost in all
                # backward passes.
                with torch.no_grad():
                    reference_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                reference_full_label_loss = float(
                    reference_outputs.loss.detach().float().item()
                )
                optimized_loss_absolute_difference = abs(
                    loss_value - reference_full_label_loss
                )
                loss_equivalence_passed = (
                    optimized_loss_absolute_difference
                    <= args.loss_equivalence_atol
                )
                loss_equivalence_checks.append(loss_equivalence_passed)
                del reference_outputs
            finite = _finite(loss_value) and _finite(global_norm)
            all_finite_checks.append(finite)
            exposure_metrics[exposure] = {
                "record_id": record["record_id"],
                "loss": loss_value,
                "global_gradient_norm": global_norm,
                "sequence_token_count": len(mask["input_ids"]),
                "generation_prompt_token_count": mask[
                    "generation_prompt_token_count"
                ],
                "ignored_prefix_token_count": mask["ignored_prefix_token_count"],
                "generation_prompt_divergent_tail_token_count": mask[
                    "generation_prompt_divergent_tail_token_count"
                ],
                "assistant_span_token_count": mask["assistant_span_token_count"],
                "loss_label_token_count": int(loss_labels.shape[-1]),
                "logits_token_count": logits_to_keep,
                "target_content_token_count": mask["target_content_token_count"],
                "target_content_token_hash": mask["target_content_token_hash"],
                "prompt_labels_all_ignore": mask["prompt_labels_all_ignore"],
                "assistant_labels_all_supervised": mask[
                    "assistant_labels_all_supervised"
                ],
                "assistant_span_contains_target": mask[
                    "assistant_span_contains_target"
                ],
                "reference_full_label_loss": reference_full_label_loss,
                "optimized_loss_absolute_difference": (
                    optimized_loss_absolute_difference
                ),
                "loss_equivalence_passed": loss_equivalence_passed,
                "runtime_seconds": runtime_seconds,
                "gpu_memory_before_bytes": memory_before,
                "gpu_peak_allocated_bytes": peak_allocated,
                "gpu_peak_increment_bytes": max(0, peak_allocated - memory_before),
                "gpu_peak_reserved_bytes": peak_reserved,
                "finite": finite,
            }
            processed_records += 1
            print(
                f"[{processed_records}/{len(records)}] {record['record_id']} "
                f"loss={loss_value:.6f} norm={global_norm:.6f} "
                f"time={runtime_seconds:.2f}s peak={peak_allocated / 2**30:.2f}GiB",
                flush=True,
            )
            del outputs, loss, input_ids, attention_mask, labels, loss_labels
            model.zero_grad(set_to_none=True)

        raw_stats = _pairwise_statistics(gradients)
        centered = _center_vectors(gradients)
        residual_stats = _pairwise_statistics(centered)
        for exposure in args.exposures:
            exposure_metrics[exposure]["gradient_sketch"] = _count_sketch(
                gradients[exposure],
                dimension=args.sketch_dim,
                seed=args.seed,
            )

        module_rows: list[dict[str, Any]] = []
        for exposure in args.exposures:
            raw_module_norm = _module_norms(gradients[exposure])
            residual_module_norm = _module_norms(centered[exposure])
            for module in raw_module_norm:
                module_rows.append(
                    {
                        "case_id": case_id,
                        "context_load": context_load,
                        "prompt_exposure": exposure,
                        "module": module,
                        "raw_gradient_norm": raw_module_norm[module],
                        "centered_residual_norm": residual_module_norm[module],
                    }
                )
        append_jsonl(modules_path, module_rows)
        append_jsonl(
            metrics_path,
            [
                {
                    "case_id": case_id,
                    "context_load": context_load,
                    "checkpoint_id": args.checkpoint_id,
                    "checkpoint": args.adapter_path or "initialized_lora",
                    "rule_family": group_records[0].get("rule_family"),
                    "semantic_case_hash": group_records[0].get("semantic_case_hash"),
                    "exposure_metrics": exposure_metrics,
                    "raw_pairwise": raw_stats,
                    "centered_residual_pairwise": residual_stats,
                    "optimizer_step_count": 0,
                    "full_gradient_serialized": False,
                }
            ],
        )

        del gradients, centered, exposure_metrics, raw_stats, residual_stats
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"completed paired group {group_index}/{len(grouped)}: "
            f"{case_id}/{context_load}",
            flush=True,
        )

    weights_unchanged = _weights_unchanged(named_trainable, initial_weights)
    output_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    forbidden_gradient_files = [
        name
        for name in output_files
        if "gradient" in name.lower()
        and name not in {"per_case_metrics.jsonl"}
    ]
    passed = (
        processed_records == len(records)
        and all(all_mask_checks)
        and all(all_finite_checks)
        and weights_unchanged
        and all(loss_equivalence_checks)
        and not forbidden_gradient_files
        and not non_lora_trainable
        and not model.training
    )
    summary = {
        "status": "pass" if passed else "fail",
        "run_finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "record_count": processed_records,
        "paired_group_count": len(grouped),
        "completion_mask_checks_passed": all(all_mask_checks),
        "all_loss_and_gradient_metrics_finite": all(all_finite_checks),
        "loss_equivalence_check_passed": (
            all(loss_equivalence_checks) if loss_equivalence_checks else None
        ),
        "only_lora_parameters_trainable": not non_lora_trainable,
        "trainable_parameter_tensor_count": len(named_trainable),
        "trainable_parameter_count": run_config["trainable_parameter_count"],
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "weights_unchanged": weights_unchanged,
        "full_gradient_serialization": False,
        "forbidden_gradient_files": forbidden_gradient_files,
        "output_files": output_files + ["aggregate_summary.json"],
    }
    write_json(output_dir / "aggregate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("gradient probe preflight failed one or more gates")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimizer-free completion-only LoRA gradient discovery."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter-path")
    source.add_argument("--init-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=(
            r"^(model\.language_model(?=\.).*\."
            r"(out_proj|down_proj|v_proj|in_proj_b|gate_proj|k_proj|in_proj_a|"
            r"in_proj_z|up_proj|in_proj_qkv|q_proj|o_proj))$"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--context-load", choices=list(CONTEXT_LOADS))
    parser.add_argument(
        "--exposures",
        nargs="+",
        default=list(EXPOSURES),
        help="Expected prompt_exposure values in each paired group.",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--sketch-dim", type=int, default=128)
    parser.add_argument(
        "--loss-equivalence-atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance for the one-record optimized/full loss audit.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--offload-vision-tower", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable non-reentrant gradient checkpointing for the audit run. "
            "Dropout modules remain in eval mode; disabled by default."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.sketch_dim < 1:
        parser.error("--sketch-dim must be positive")
    if args.loss_equivalence_atol < 0:
        parser.error("--loss-equivalence-atol must be non-negative")
    if not args.exposures or len(set(args.exposures)) != len(args.exposures):
        parser.error("--exposures must be non-empty and unique")
    return args


def main() -> int:
    args = parse_args()
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
