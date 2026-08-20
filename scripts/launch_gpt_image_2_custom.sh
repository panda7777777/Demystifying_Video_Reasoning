#!/usr/bin/env bash

cd /mnt/aigc/users/zuojing/codes/Demystifying_Video_Reasoning
export HF_HOME=/mnt/aigc/shared_env/huggingface
export HF_HUB_CACHE=/mnt/aigc/shared_env/huggingface/hub
set -uo pipefail

MODEL="/mnt/aigc/shared_env/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B/snapshots/206a9ee1b7bfaaf8f7e4d81335650533490646a3"
DATA_ROOT="/mnt/umm/users/zuojing/codes/Embodied_Visual_Reasoning/outputs/gpt_image_2"
OUTPUT_DIR="./output"

SELECTION="all"
SPLIT=""
NUM_NODES="${WORLD_SIZE:-4}"
# Cluster launchers inject RANK for the current node (as in the reference
# multi-node launcher). Keep NODE_RANK as a manual fallback for local runs.
NODE_RANK="${RANK:-${NODE_RANK:-0}}"
GPUS="0,1,2,3,4,5,6,7"
BATCH_SIZE="1"

NUM_FRAMES="49"
NUM_INFERENCE_STEPS="30"
MAX_DENOISING_STEPS="10"
SEED="1"
FPS="16"
MAX_SIZE="832"
OVERVIEW_COLUMNS="6"

# Resume the newest run for each task whose directory starts with one of these
# dates. Override with RESUME_DATES=20260820,20260821 or --resume-dates.
RESUME_DATES="${RESUME_DATES:-20260820}"

# All nodes in one scheduler job must share this value. SLURM_JOB_ID/JOB_ID
# are stable across nodes; BARRIER_ID can be supplied by other launchers.
BARRIER_ID="${BARRIER_ID:-${SLURM_JOB_ID:-${JOB_ID:-custom_${NUM_NODES}_$(date +%Y%m%d_%H%M%S)}}}"
BARRIER_ROOT="${OUTPUT_DIR}/.barriers/${BARRIER_ID}"

if ! [[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "error: invalid node configuration rank=${NODE_RANK}, world_size=${NUM_NODES}" >&2
  exit 2
fi

wait_for_task_barrier() {
  local task_name="$1" phase="$2" status="$3" marker_dir marker rank marker_status all_present
  marker_dir="${BARRIER_ROOT}/${task_name}/${phase}"
  mkdir -p "${marker_dir}"
  marker="${marker_dir}/node_${NODE_RANK}"
  printf '%s\n' "${status}" > "${marker}.tmp"
  mv -f "${marker}.tmp" "${marker}"
  echo "[barrier] ${task_name}/${phase}: node ${NODE_RANK} reported ${status}; waiting for ${NUM_NODES} node(s)"
  while true; do
    all_present=1
    for (( rank=0; rank<NUM_NODES; rank++ )); do
      if [[ ! -s "${marker_dir}/node_${rank}" ]]; then
        all_present=0
        break
      fi
    done
    (( all_present )) && break
    sleep 5
  done
  marker_status="done"
  for (( rank=0; rank<NUM_NODES; rank++ )); do
    if [[ "$(<"${marker_dir}/node_${rank}")" != "done" ]]; then
      marker_status="failed"
    fi
  done
  if [[ "${marker_status}" != "done" ]]; then
    echo "[barrier] ${task_name}/${phase}: at least one node failed" >&2
    return 1
  fi
  echo "[barrier] ${task_name}/${phase}: all nodes complete"
}

PYTHON="/mnt/aigc/users/qianjianheng/miniconda3/envs/dvr/bin/python"
HIGH_NOISE_ADAPTER_PATH="/mnt/umm/users/zhoutongxi/diffsynth-studio/output/260802_mix_train_skip_lora_linear/high_noise/step-12698.safetensors"
LOW_NOISE_ADAPTER_PATH=""
ADAPTER_SCALE="1.0"

# These options belong to this launcher and must not be forwarded to run.py.
forwarded_args=()
while (( $# > 0 )); do
  case "$1" in
    --resume-dates)
      (( $# >= 2 )) || { echo "error: --resume-dates requires a comma-separated date list" >&2; exit 2; }
      RESUME_DATES="$2"
      shift 2
      ;;
    --resume-dates=*)
      RESUME_DATES="${1#*=}"
      shift
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

resume_dates=()
IFS=',' read -r -a resume_dates <<< "${RESUME_DATES}"
for index in "${!resume_dates[@]}"; do
  date="${resume_dates[${index}]}"
  date="${date//[[:space:]]/}"
  resume_dates[${index}]="${date}"
  if [[ ! "${date}" =~ ^[0-9]{8}$ ]]; then
    echo "error: invalid resume date '${date}'; expected YYYYMMDD" >&2
    exit 2
  fi
done

find_resume_dir() {
  local task_name="$1"
  local date candidate best=""
  [[ -d "${OUTPUT_DIR}" ]] || return 0
  for date in "${resume_dates[@]}"; do
    while IFS= read -r candidate; do
      [[ -f "${candidate}/run.json" ]] || continue
      # Directory names are timestamped, so lexical order selects the newest.
      if [[ -z "${best}" || "${candidate}" > "${best}" ]]; then
        best="${candidate}"
      fi
    done < <(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d \
      -name "${date}_*_${task_name}_wan22_custom" -print)
  done
  [[ -n "${best}" ]] && printf '%s\n' "${best}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dirs=()
while IFS= read -r -d '' data_dir; do
  data_dirs+=("${data_dir}")
done < <("${PYTHON}" "${SCRIPT_DIR}/check_custom_data.py" "${DATA_ROOT}" --list-roots --null)

if (( ${#data_dirs[@]} == 0 )); then
  echo "error: no custom datasets found under ${DATA_ROOT}" >&2
  exit 2
fi

declare -A seen_tasks=()
failures=()
for data_dir in "${data_dirs[@]}"; do
  base_name="${data_dir##*/}"
  if [[ "${base_name}" =~ ^(T[0-9]+|[GO]-[0-9]+)_ ]]; then
    task_name="${BASH_REMATCH[1]}"
  else
    task_name="${base_name}"
  fi
  if [[ -n "${seen_tasks[${task_name}]:-}" ]]; then
    echo "error: duplicate task name ${task_name}: ${seen_tasks[${task_name}]} and ${data_dir}" >&2
    exit 2
  fi
  seen_tasks["${task_name}"]="${data_dir}"

  args=(
    "${SCRIPT_DIR}/run.py" --model "${MODEL}" --dataset custom
    --data-dir "${data_dir}" --task-name "${task_name}" --output-dir "${OUTPUT_DIR}"
    --selection "${SELECTION}" --num-nodes "${NUM_NODES}" --node-rank "${NODE_RANK}"
    --gpus "${GPUS}" --batch-size "${BATCH_SIZE}" --num-frames "${NUM_FRAMES}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}" --seed "${SEED}" --fps "${FPS}"
    --max-size "${MAX_SIZE}" --overview-columns "${OVERVIEW_COLUMNS}" --python "${PYTHON}"
  )
  [[ -n "${SPLIT}" ]] && args+=(--split "${SPLIT}")
  [[ -n "${MAX_DENOISING_STEPS}" ]] && args+=(--max-denoising-steps "${MAX_DENOISING_STEPS}")
  [[ -n "${HIGH_NOISE_ADAPTER_PATH}" ]] && args+=(--high-noise-lora-path "${HIGH_NOISE_ADAPTER_PATH}")
  [[ -n "${LOW_NOISE_ADAPTER_PATH}" ]] && args+=(--low-noise-lora-path "${LOW_NOISE_ADAPTER_PATH}")
  args+=(--lora-alpha "${ADAPTER_SCALE}")

  resume_dir="$(find_resume_dir "${task_name}")"
  if [[ -n "${resume_dir}" ]]; then
    args+=(--resume-dir "${resume_dir}")
    echo "[resume] ${task_name}: ${resume_dir}"
  fi

  echo "[dataset] ${task_name}: ${data_dir}"
  if ! wait_for_task_barrier "${task_name}" "start" "ready"; then
    failures+=("${task_name}: start barrier")
    break
  fi
  task_status="done"
  if ! "${PYTHON}" "${args[@]}" "${forwarded_args[@]}"; then
    failures+=("${task_name}: ${data_dir}")
    task_status="failed"
  fi
  if ! wait_for_task_barrier "${task_name}" "done" "${task_status}"; then
    failures+=("${task_name}: node barrier")
    break
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "[summary] ${#failures[@]} of ${#data_dirs[@]} dataset(s) failed:" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi
echo "[summary] all ${#data_dirs[@]} dataset(s) completed"
