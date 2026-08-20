#!/usr/bin/env bash

set -euo pipefail
cd /mnt/aigc/users/zuojing/codes/Demystifying_Video_Reasoning

export HF_HOME=/mnt/aigc/shared_env/huggingface
export HF_HUB_CACHE=/mnt/aigc/shared_env/huggingface/hub

MODEL="/mnt/aigc/shared_env/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3"
DATA_ROOT="/mnt/umm/users/zuojing/codes/Embodied_Visual_Reasoning/outputs/gpt_image_2"
OUTPUT_DIR="./output"
PYTHON="/mnt/aigc/users/qianjianheng/miniconda3/envs/dvr/bin/python"
GPUS="0,1,2,3,4,5,6,7"
BATCH_SIZE="2"
HIGH_NOISE_ADAPTER_PATH="/mnt/umm/users/zhoutongxi/diffsynth-studio/output/260802_mix_train_skip_lora_linear/high_noise/step-12698.safetensors"

exec "${PYTHON}" scripts/multinode_runner.py \
  --model "${MODEL}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume-dates "${RESUME_DATES:-20260820}" \
  --num-nodes "${WORLD_SIZE:-4}" \
  --node-rank "${RANK:-${NODE_RANK:-0}}" \
  --gpus "${GPUS}" \
  --batch-size "${BATCH_SIZE}" \
  --python "${PYTHON}" \
  --high-noise-lora-path "${HIGH_NOISE_ADAPTER_PATH}" \
  "$@"
