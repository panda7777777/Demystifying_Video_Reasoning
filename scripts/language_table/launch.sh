#!/usr/bin/env bash
set -euo pipefail

# Generic multi-node launcher. The scheduler may provide NODE_RANK and
# NUM_NODES; the workflow itself does not require inter-node communication.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${DATA_DIR:?Set DATA_DIR to a TFDS Language-Table builder directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a shared result directory}"

NODE_RANK="${NODE_RANK:-${RANK:-0}}"
NUM_NODES="${NUM_NODES:-${WORLD_SIZE:-1}}"
MODEL="${MODEL:-wan2.2}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
TFDS_PYTHON="${TFDS_PYTHON:-}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-python}"

arguments=(
  "${REPO_ROOT}/scripts/language_table/visualize.py"
  --data-dir "${DATA_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --model "${MODEL}"
  --node-rank "${NODE_RANK}"
  --num-nodes "${NUM_NODES}"
)

if [[ -n "${GPUS}" ]]; then
  arguments+=(--gpus "${GPUS}")
fi
if [[ -n "${TFDS_PYTHON}" ]]; then
  arguments+=(--tfds-python "${TFDS_PYTHON}")
fi
if [[ -n "${VBVR_MODEL_PATH:-}" ]]; then
  arguments+=(--vbvr-model-path "${VBVR_MODEL_PATH}")
fi

exec "${INFERENCE_PYTHON}" "${arguments[@]}" "$@"
