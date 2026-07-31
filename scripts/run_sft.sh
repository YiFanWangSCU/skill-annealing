#!/usr/bin/env bash
set -euo pipefail

# Portable ms-swift entry point.  Set MODEL_PATH and DATASET from the shell;
# no server path, token, or checkpoint is embedded in this repository.
METHOD="${METHOD:-random_mix}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a locally licensed Qwen3.5-4B checkpoint}"
DATASET="${DATASET:-data/examples/ms_swift/refund_decision_smoke/annealed_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${METHOD}}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

"${SWIFT_BIN}" sft \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --target_modules all-linear \
  --save_steps "${SAVE_STEPS:-50}" \
  --output_dir "${OUTPUT_DIR}"
