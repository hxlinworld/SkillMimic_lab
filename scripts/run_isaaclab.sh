#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${PROJECT_ROOT}/.external/isaac-sim-4.1.0}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
    if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
        export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    elif [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
        export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    fi
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

LOG_FILE="${PROJECT_ROOT}/logs/isaaclab/latest.log"
mkdir -p -- "$(dirname -- "${LOG_FILE}")"
exec > >(tee "${LOG_FILE}") 2>&1

if [[ -f "${ISAAC_SIM_ROOT}/setup_conda_env.sh" ]]; then
    set +u
    source "${ISAAC_SIM_ROOT}/setup_conda_env.sh"
    set -u
fi

TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${PROJECT_ROOT}/logs/rl_games}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
mkdir -p -- "${TENSORBOARD_LOG_DIR}"
if ! pgrep -f "tensorboard.main.*--logdir ${TENSORBOARD_LOG_DIR}" >/dev/null; then
    nohup "${PYTHON_BIN}" -m tensorboard.main \
        --logdir "${TENSORBOARD_LOG_DIR}" \
        --port "${TENSORBOARD_PORT}" \
        >/dev/null 2>&1 &
fi

headless_args=()
if [[ "${HEADLESS:-1}" == "1" ]]; then
    headless_args+=(--headless)
fi

run_inference() {
    local task="$1"
    local run_mode="$2"
    shift 2

    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/skillmimic_lab/run.py" \
        --task "${task}" \
        --mode "${run_mode}" \
        --num_envs "${NUM_ENVS:-4}" \
        --steps "${STEPS:-600}" \
        --device_id "${DEVICE_ID:-0}" \
        "${headless_args[@]}" \
        "$@"
}

run_training() {
    local task="$1"
    local num_envs="${NUM_ENVS:-2048}"
    local num_gpus="${NUM_GPUS:-1}"
    shift

    if [[ "${num_gpus}" -gt 1 ]]; then
        exec "${PYTHON_BIN}" -m torch.distributed.run \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="${num_gpus}" \
            "${PROJECT_ROOT}/skillmimic_lab/learning/train.py" \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --distributed \
            "${headless_args[@]}" \
            "$@"
    fi

    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/skillmimic_lab/learning/train.py" \
        --task "${task}" \
        --num_envs "${num_envs}" \
        --device_id "${DEVICE_ID:-0}" \
        "${headless_args[@]}" \
        "$@"
}

case "${MODE}" in
    smoke) run_inference ballplay smoke "$@" ;;
    play) run_inference ballplay play "$@" ;;
    circling|heading|throwing|scoring) run_inference "${MODE}" play "$@" ;;
    train) run_training ballplay "$@" ;;
    train-circling) run_training circling "$@" ;;
    train-heading) run_training heading "$@" ;;
    train-throwing) run_training throwing "$@" ;;
    train-scoring) run_training scoring "$@" ;;
    *)
        printf 'ERROR: unknown mode: %s\n' "${MODE}" >&2
        exit 1
        ;;
esac
