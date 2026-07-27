#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_IMAGE="${ISAACLAB_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}"
REQUESTED_MODE="${1:-smoke}"

# Keep the public command identical on the host and in the container. On the
# host this script enters NVIDIA's pre-built Isaac Lab image; the second call
# below runs the actual Python entry point inside that container.
if [[ "${SKILLMIMIC_IN_CONTAINER:-0}" != "1" ]]; then
    container_command=("$@")
    CACHE_ROOT="${SKILLMIMIC_CACHE_ROOT:-${PROJECT_ROOT}/.docker/isaaclab}"
    mkdir -p -- \
        "${CACHE_ROOT}/cache/kit" \
        "${CACHE_ROOT}/cache/ov" \
        "${CACHE_ROOT}/cache/pip" \
        "${CACHE_ROOT}/cache/glcache" \
        "${CACHE_ROOT}/cache/computecache" \
        "${CACHE_ROOT}/logs" \
        "${CACHE_ROOT}/data" \
        "${CACHE_ROOT}/documents"

    if [[ "${REQUESTED_MODE}" == "train-webrtc" ]]; then
        WEBRTC_PUBLIC_IP_FILE="${PROJECT_ROOT}/.docker/webrtc_public_ip"
        if [[ -z "${PUBLIC_IP:-}" && -f "${WEBRTC_PUBLIC_IP_FILE}" ]]; then
            IFS= read -r PUBLIC_IP < "${WEBRTC_PUBLIC_IP_FILE}" || true
        fi
        if [[ -z "${PUBLIC_IP:-}" ]]; then
            printf 'ERROR: train-webrtc requires PUBLIC_IP or %s\n' \
                "${WEBRTC_PUBLIC_IP_FILE}" >&2
            exit 1
        fi
        export PUBLIC_IP
        export LIVESTREAM="${LIVESTREAM:-1}"
        export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
    fi

    docker_cmd=(docker)
    if ! docker info >/dev/null 2>&1; then
        docker_cmd=(sudo -n docker)
    fi

    # Give each training environment/seed pair a stable Docker name. Docker
    # reserves container names atomically, so the name also acts as a lock if
    # the same experiment is accidentally launched twice.
    training_container_args=()
    training_environment=""
    case "${REQUESTED_MODE}" in
        train|train-webrtc) training_environment="ballplay" ;;
        train-circling) training_environment="circling" ;;
        train-heading) training_environment="heading" ;;
        train-throwing) training_environment="throwing" ;;
        train-scoring) training_environment="scoring" ;;
    esac

    if [[ -n "${training_environment}" ]]; then
        training_seed=""
        expect_seed_value=0
        for argument in "${@:2}"; do
            if [[ "${expect_seed_value}" -eq 1 ]]; then
                training_seed="${argument}"
                expect_seed_value=0
                continue
            fi
            case "${argument}" in
                --seed) expect_seed_value=1 ;;
                --seed=*) training_seed="${argument#--seed=}" ;;
            esac
        done
        if [[ "${expect_seed_value}" -eq 1 || (
            -n "${training_seed}" && ! "${training_seed}" =~ ^-?[0-9]+$
        ) ]]; then
            printf 'ERROR: --seed requires an integer value.\n' >&2
            exit 1
        fi
        if [[ -z "${training_seed}" ]]; then
            training_seed=$(( ((RANDOM << 15) | RANDOM) % 2147483646 + 1 ))
            container_command+=(--seed "${training_seed}")
        fi

        if [[ "${ALLOW_PARALLEL_TRAINING:-0}" != "1" ]]; then
            active_training_containers="$("${docker_cmd[@]}" ps \
                --filter 'label=com.skillmimic.training=true' \
                --filter "label=com.skillmimic.environment=${training_environment}" \
                --format '{{.Names}}')"
            if [[ -n "${active_training_containers}" ]]; then
                printf 'ERROR: training environment %s is already running in: %s\n' \
                    "${training_environment}" "${active_training_containers}" >&2
                printf 'Set ALLOW_PARALLEL_TRAINING=1 only for intentional parallel seeds.\n' >&2
                exit 1
            fi
        fi

        training_container_name="${SKILLMIMIC_CONTAINER_NAME:-skillmimic-${training_environment}-seed-${training_seed}}"
        if "${docker_cmd[@]}" container inspect "${training_container_name}" >/dev/null 2>&1; then
            container_status="$("${docker_cmd[@]}" inspect \
                --format '{{.State.Status}}' "${training_container_name}")"
            printf 'ERROR: training environment %s with seed %s already has container %s (%s).\n' \
                "${training_environment}" "${training_seed}" \
                "${training_container_name}" "${container_status}" >&2
            exit 1
        fi
        training_container_args=(
            --name "${training_container_name}"
            --label "com.skillmimic.training=true"
            --label "com.skillmimic.environment=${training_environment}"
            --label "com.skillmimic.seed=${training_seed}"
        )
        printf '[SkillMimic Lab] Training container: %s (environment=%s, seed=%s).\n' \
            "${training_container_name}" "${training_environment}" "${training_seed}"
    fi

    docker_env=()
    for name in \
        CUDA_VISIBLE_DEVICES DEVICE DEVICE_ID ENABLE_CAMERAS HEADLESS LIVESTREAM \
        NUM_ENVS NUM_GPUS PUBLIC_IP STEPS TENSORBOARD_PORT TENSORBOARD_LOG_DIR \
        SKILLMIMIC_LOG_FILE SKILLMIMIC_RUN_NAME; do
        if [[ -v "${name}" ]]; then
            docker_env+=(-e "${name}=${!name}")
        fi
    done

    exec "${docker_cmd[@]}" run --rm \
        "${training_container_args[@]}" \
        --gpus all \
        --network host \
        --ipc host \
        --ulimit nofile=65535:65535 \
        --entrypoint bash \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        -e SKILLMIMIC_IN_CONTAINER=1 \
        -e SKILLMIMIC_KIT_RUNTIME_ROOT=/tmp/skillmimic-kit \
        "${docker_env[@]}" \
        -v "${PROJECT_ROOT}:/workspace/skillmimic-lab" \
        -v "${CACHE_ROOT}/cache/kit:/isaac-sim/kit/cache" \
        -v "${CACHE_ROOT}/cache/ov:/root/.cache/ov" \
        -v "${CACHE_ROOT}/cache/pip:/root/.cache/pip" \
        -v "${CACHE_ROOT}/cache/glcache:/root/.cache/nvidia/GLCache" \
        -v "${CACHE_ROOT}/cache/computecache:/root/.nv/ComputeCache" \
        -v "${CACHE_ROOT}/logs:/root/.nvidia-omniverse/logs" \
        -v "${CACHE_ROOT}/data:/root/.local/share/ov/data" \
        -v "${CACHE_ROOT}/documents:/root/Documents" \
        -w /workspace/skillmimic-lab \
        "${ISAACLAB_IMAGE}" \
        scripts/run_isaaclab.sh "${container_command[@]}"
fi

PROJECT_ROOT=/workspace/skillmimic-lab
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_cmd=("${PYTHON_BIN}")
else
    python_cmd=(/workspace/isaaclab/isaaclab.sh -p)
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export SKILLMIMIC_RUN_STATUS_FILE="${SKILLMIMIC_RUN_STATUS_FILE:-/tmp/skillmimic-status-$$}"
rm -f -- "${SKILLMIMIC_RUN_STATUS_FILE}"

LOG_FILE="${SKILLMIMIC_LOG_FILE:-${PROJECT_ROOT}/logs/isaaclab/latest.log}"
mkdir -p -- "$(dirname -- "${LOG_FILE}")"
exec > >(tee "${LOG_FILE}") 2>&1

case "${MODE}" in
    train|train-webrtc|train-circling|train-heading|train-throwing|train-scoring)
        TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${PROJECT_ROOT}/logs/rl_games}"
        TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
        mkdir -p -- "${TENSORBOARD_LOG_DIR}"
        if ! pgrep -f "tensorboard.main.*--logdir ${TENSORBOARD_LOG_DIR}" >/dev/null; then
            nohup "${python_cmd[@]}" -m tensorboard.main \
                --logdir "${TENSORBOARD_LOG_DIR}" \
                --port "${TENSORBOARD_PORT}" \
                >/dev/null 2>&1 &
        fi
        ;;
esac

if [[ "${MODE}" == "train-webrtc" ]]; then
    if [[ "${NUM_GPUS:-1}" -ne 1 ]]; then
        printf 'ERROR: train-webrtc supports one GPU because one process owns the WebRTC port.\n' >&2
        exit 1
    fi
    export LIVESTREAM="${LIVESTREAM:-1}"
    export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
    printf '[SkillMimic Lab] WebRTC training enabled at %s (TCP 49100, UDP 47998).\n' \
        "${PUBLIC_IP}"
fi

headless_args=()
if [[ "${HEADLESS:-1}" == "1" ]]; then
    headless_args+=(--headless)
fi

run_with_status() {
    local process_status=0
    if "$@"; then
        process_status=0
    else
        process_status=$?
    fi
    if [[ -f "${SKILLMIMIC_RUN_STATUS_FILE}" ]]; then
        return "$(<"${SKILLMIMIC_RUN_STATUS_FILE}")"
    fi
    return "${process_status}"
}

run_inference() {
    local task="$1"
    local run_mode="$2"
    shift 2

    run_with_status "${python_cmd[@]}" "${PROJECT_ROOT}/skillmimic_lab/run.py" \
        --task "${task}" \
        --mode "${run_mode}" \
        --num_envs "${NUM_ENVS:-4}" \
        --steps "${STEPS:-600}" \
        --device "${DEVICE:-cuda:${DEVICE_ID:-0}}" \
        "${headless_args[@]}" \
        "$@"
}

run_training() {
    local task="$1"
    local num_envs="${NUM_ENVS:-2048}"
    local num_gpus="${NUM_GPUS:-1}"
    shift

    if [[ "${num_gpus}" -gt 1 ]]; then
        run_with_status "${python_cmd[@]}" -m torch.distributed.run \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="${num_gpus}" \
            "${PROJECT_ROOT}/skillmimic_lab/learning/train.py" \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --distributed \
            "${headless_args[@]}" \
            "$@"
        return
    fi

    run_with_status "${python_cmd[@]}" "${PROJECT_ROOT}/skillmimic_lab/learning/train.py" \
        --task "${task}" \
        --num_envs "${num_envs}" \
        --device "${DEVICE:-cuda:${DEVICE_ID:-0}}" \
        "${headless_args[@]}" \
        "$@"
}

case "${MODE}" in
    smoke) run_inference ballplay smoke "$@" ;;
    reference) run_inference ballplay reference "$@" ;;
    play) run_inference ballplay play "$@" ;;
    circling|heading|throwing|scoring) run_inference "${MODE}" play "$@" ;;
    train) run_training ballplay "$@" ;;
    train-webrtc) run_training ballplay "$@" ;;
    train-circling) run_training circling "$@" ;;
    train-heading) run_training heading "$@" ;;
    train-throwing) run_training throwing "$@" ;;
    train-scoring) run_training scoring "$@" ;;
    *)
        printf 'ERROR: unknown mode: %s\n' "${MODE}" >&2
        exit 1
        ;;
esac
