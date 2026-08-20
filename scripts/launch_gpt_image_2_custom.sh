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

# Match the cluster launcher convention used by SenseNova-Nav. The scheduler
# injects these values identically into the command executed on every node.
num_nodes="${WORLD_SIZE:-${SLURM_NNODES:-4}}"
node_rank="${RANK:-${SLURM_PROCID:-0}}"
world_size="${WORLD_SIZE:-${num_nodes}}"
master_addr="${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}}"
master_port="${MASTER_PORT:-29500}"
session_id="${BARRIER_ID:-${TORCHELASTIC_RUN_ID:-${SLURM_JOB_ID:-${JOB_ID:-${master_addr}_${master_port}_${world_size}}}}}"

exec "${PYTHON}" scripts/multinode_runner.py \
  --model "${MODEL}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume-dates "${RESUME_DATES:-20260820}" \
  --num-nodes "${world_size}" \
  --node-rank "${node_rank}" \
  --session-id "${session_id}" \
  --gpus "${GPUS}" \
  --batch-size "${BATCH_SIZE}" \
  --python "${PYTHON}" \
  --high-noise-lora-path "${HIGH_NOISE_ADAPTER_PATH}" \
  "$@"
