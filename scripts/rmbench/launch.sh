#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# User configuration
# ==============================================================================

# Dataset and output paths (required).
DATA_DIR="/mnt/umm/users/zuojing/code/RMBench/data/data"
OUTPUT_ROOT="./output/step_visualization/rmbench_all"

# Dataset selection.
TASKS="all"                   # "all" or comma-separated task names.
EPISODES="all"               # "all", an index, or slices such as "0:10,20".
INSTRUCTION_SPLIT="seen"     # "seen" or "unseen".

# Model.
MODEL="wan2.2"               # wan2.2, wan2.1, vbvr-wan2.2, or ltx2.3.
VBVR_MODEL_PATH=""           # Required only for MODEL="vbvr-wan2.2".
LORA_PATH=""                 # Wan2.1/LTX LoRA path.
HIGH_NOISE_LORA_PATH=""      # Wan2.2 high-noise expert LoRA path.
LOW_NOISE_LORA_PATH=""       # Wan2.2 low-noise expert LoRA path.
LORA_ALPHA="1.0"
MODEL_BASE_PATH="./models"

# Generation.
NUM_FRAMES="49"
NUM_INFERENCE_STEPS="30"
MAX_DENOISING_STEPS="10"     # Set to "" to disable; LTX requires "".
SEED="1"
FPS="16"
NEGATIVE_PROMPT=""           # Empty uses the model-specific default.

# Parallel execution. Every node must use the same NUM_NODES and GPU count.
GPUS="0,1,2,3,4,5,6,7"                     # Comma-separated local GPU ids, e.g. "0,1,2,3".
NUM_NODES="1"
NODE_RANK="0"                # Unique integer in [0, NUM_NODES).

# Python interpreters.
INFERENCE_PYTHON="/mnt/umm/users/zuojing/env/miniforge3/envs/dvr/bin/python"
EXTRACTION_PYTHON=""         # Empty uses INFERENCE_PYTHON.

# Run modes: use "true" or "false".
EXTRACT_ONLY="false"
OVERWRITE="false"
DRY_RUN="false"

# ==============================================================================
# Command construction — no user configuration is needed below this line.
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${DATA_DIR}" ]]; then
  echo "error: set DATA_DIR in the user configuration section" >&2
  exit 2
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "error: set OUTPUT_ROOT in the user configuration section" >&2
  exit 2
fi
if [[ -z "${GPUS}" ]]; then
  echo "error: set at least one GPU id in GPUS" >&2
  exit 2
fi
if [[ "${MODEL}" == "vbvr-wan2.2" && -z "${VBVR_MODEL_PATH}" ]]; then
  echo "error: MODEL=vbvr-wan2.2 requires VBVR_MODEL_PATH" >&2
  exit 2
fi
if [[ "${MODEL}" == "ltx2.3" && -n "${MAX_DENOISING_STEPS}" ]]; then
  echo "error: set MAX_DENOISING_STEPS=\"\" when MODEL=ltx2.3" >&2
  exit 2
fi

arguments=(
  "${REPO_ROOT}/scripts/rmbench/visualize.py"
  --data-dir "${DATA_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --tasks "${TASKS}"
  --episodes "${EPISODES}"
  --instruction-split "${INSTRUCTION_SPLIT}"
  --model "${MODEL}"
  --num-frames "${NUM_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
  --fps "${FPS}"
  --lora-alpha "${LORA_ALPHA}"
  --model-base-path "${MODEL_BASE_PATH}"
  --gpus "${GPUS}"
  --num-nodes "${NUM_NODES}"
  --node-rank "${NODE_RANK}"
)

optional_arguments=(
  "--vbvr-model-path:${VBVR_MODEL_PATH}"
  "--lora-path:${LORA_PATH}"
  "--high-noise-lora-path:${HIGH_NOISE_LORA_PATH}"
  "--low-noise-lora-path:${LOW_NOISE_LORA_PATH}"
  "--max-denoising-steps:${MAX_DENOISING_STEPS}"
  "--negative-prompt:${NEGATIVE_PROMPT}"
  "--extraction-python:${EXTRACTION_PYTHON}"
)
for entry in "${optional_arguments[@]}"; do
  option="${entry%%:*}"
  value="${entry#*:}"
  if [[ -n "${value}" ]]; then
    arguments+=("${option}" "${value}")
  fi
done

if [[ "${EXTRACT_ONLY}" == "true" ]]; then
  arguments+=(--extract-only)
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  arguments+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  arguments+=(--dry-run)
fi

cd "${REPO_ROOT}"
exec "${INFERENCE_PYTHON}" "${arguments[@]}" "$@"
