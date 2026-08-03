#!/usr/bin/env bash
cd /mnt/aigc/users/zuojing/codes/Demystifying_Video_Reasoning
export HF_HOME=/mnt/aigc/shared_env/huggingface
export HF_HUB_CACHE=/mnt/aigc/shared_env/huggingface/hub
set -euo pipefail

# Required choices. MODEL must be an absolute local directory or a Hugging Face
# repository id (namespace/repository).
MODEL="/mnt/aigc/shared_env/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3"
DATASET="language-table"               # custom | rmbench | language-table
DATA_DIR="/mnt/aigc/users/zuojing/data/language-table/0.0.1"
OUTPUT_DIR="./output"          # A timestamped run folder is created inside.
RESUME_DIR=""                   # Existing run folder; empty starts a new run.

# Dataset and parallelism.
SELECTION=":1000"
SPLIT=""                         # Empty: Language-Table=train, RMBench=seen.
NUM_NODES="1"
NODE_RANK="${NODE_RANK:-0}"
GPUS="0,1,2,3,4,5,6,7"
BATCH_SIZE="1"                  # Concurrent inference workers per GPU.

# Generation.
NUM_FRAMES="49"
NUM_INFERENCE_STEPS="30"
MAX_DENOISING_STEPS="10"
SEED="1"
FPS="16"
MAX_SIZE="832"
OVERVIEW_COLUMNS="6"

# Runtime and optional LoRA checkpoints.
PYTHON="/mnt/aigc/users/qianjianheng/miniconda3/envs/dvr/bin/python"
HIGH_NOISE_ADAPTER_PATH="/mnt/umm/users/zhoutongxi/diffsynth-studio/output/260725_mix_train_skip_lora_linear/high_noise/step-12217.safetensors"
LOW_NOISE_ADAPTER_PATH=""
ADAPTER_SCALE="1.0"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=(
  "${SCRIPT_DIR}/run.py" --model "${MODEL}" --dataset "${DATASET}"
  --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --selection "${SELECTION}"
  --num-nodes "${NUM_NODES}" --node-rank "${NODE_RANK}"
  --gpus "${GPUS}" --batch-size "${BATCH_SIZE}" --num-frames "${NUM_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}" --seed "${SEED}" --fps "${FPS}"
  --max-size "${MAX_SIZE}" --overview-columns "${OVERVIEW_COLUMNS}"
  --python "${PYTHON}"
)
[[ -n "${SPLIT}" ]] && args+=(--split "${SPLIT}")
[[ -n "${RESUME_DIR}" ]] && args+=(--resume-dir "${RESUME_DIR}")
[[ -n "${MAX_DENOISING_STEPS}" ]] && args+=(--max-denoising-steps "${MAX_DENOISING_STEPS}")
[[ -n "${HIGH_NOISE_ADAPTER_PATH}" ]] && args+=(--high-noise-lora-path "${HIGH_NOISE_ADAPTER_PATH}")
[[ -n "${LOW_NOISE_ADAPTER_PATH}" ]] && args+=(--low-noise-lora-path "${LOW_NOISE_ADAPTER_PATH}")
args+=(--lora-alpha "${ADAPTER_SCALE}")
exec "${PYTHON}" "${args[@]}" "$@"
