#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/mnt/aigc/users/zuojing/codes/Demystifying_Video_Reasoning"
DATA_ROOT="/mnt/umm/users/zuojing/codes/Embodied_Visual_Reasoning/outputs/gpt_image_2"
PYTHON="/mnt/aigc/users/qianjianheng/miniconda3/envs/dvr/bin/python"

exec "${PYTHON}" "${REPO_ROOT}/scripts/check_custom_data.py" "${DATA_ROOT}" --recursive "$@"
