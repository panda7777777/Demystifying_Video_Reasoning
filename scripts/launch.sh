#!/usr/bin/env bash
cd /mnt/umm/users/zuojing/code/Demystifying_Video_Reasoning
# export HF_HOME=/mnt/aigc/shared_env/huggingface
# export HF_HUB_CACHE=/mnt/aigc/shared_env/huggingface/hub
set -euo pipefail

# The system CUDA 13.1 compatibility package registers its older libcuda
# before the host-driver library.  Prefer the library matching nvidia-smi and
# the loaded kernel driver to avoid cudaGetDeviceCount error 803.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Required choices. MODEL must be an absolute local directory or a Hugging Face
# repository id (namespace/repository).
MODEL="/mnt/umm/shared_model/Wan2.2-I2V-A14B"
DATASET="rmbench"               # custom | rmbench | language-table
DATA_DIR="/mnt/umm/users/zuojing/code/RMBench/data/data"
OUTPUT_DIR="./output"          # A timestamped run folder is created inside.
RESUME_DIR=""                   # Existing run folder; empty starts a new run.

# Dataset and parallelism.
SELECTION="all"
SPLIT=""                         # Empty: Language-Table=train, RMBench=seen.
NUM_NODES="1"
NODE_RANK="${NODE_RANK:-0}"
GPUS="0,1,2,3,4,5,6,7"
BATCH_SIZE="2"                  # Concurrent inference workers per GPU.

# Generation.
NUM_FRAMES="49"
NUM_INFERENCE_STEPS="30"
MAX_DENOISING_STEPS="10"
SEED="1"
FPS="16"
MAX_SIZE="832"
OVERVIEW_COLUMNS="6"

# Runtime and optional LoRA checkpoints.
PYTHON="/mnt/umm/users/zuojing/env/miniforge3/envs/dvr/bin/python"
HIGH_NOISE_ADAPTER_PATH="/mnt/umm/users/wangruisi/01-project/DiffSynth-Studio-Step/results/DiffSynth-Studio/wan2.2-I2V-14B_260715_vbvr_pro/high_noise/checkpoints/checkpoint-9693/trainable_model.safetensors"
LOW_NOISE_ADAPTER_PATH="/mnt/umm/users/wangruisi/01-project/DiffSynth-Studio-Step/results/DiffSynth-Studio/wan2.2-I2V-14B_260715_vbvr_pro/low_noise/checkpoints/checkpoint-9693/trainable_model.safetensors"
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
