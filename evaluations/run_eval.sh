#! /bin/bash

set -euo pipefail

export EVALUATIONS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$EVALUATIONS_PATH")"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export SRC_FILE="${EVALUATIONS_PATH}/eval_embodied_agent.py"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
export RLINF_CODE_WORKING_DIR=/workspace/RLinf
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_JOB_CONFIG_JSON_ENV_VAR='{
  "runtime_env": {
    "working_dir": "/workspace/RLinf",
      "env_vars": {
        "TOKENIZERS_PARALLELISM": "false",
        "RLINF_WEIGHT_SYNC_TRACE": "1",
        "PYTHONUNBUFFERED": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDART_DUMMY_REGISTER": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "BKCL_PCIE_16CARD_TP8": "1",
        "BKCL_NONUNIFORM_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES":"1",
        "VLLM_USE_V1": "1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN":"1",
        "BKCL_USE_AR": "1",
        "BKCL_RING_OPT": "1",
        "BKCL_FLAT_RING": "1",
        "BKCL_CCIX_RING": "1",
        "BKCL_TREE_THRESHOLD": "1",
        "BKCL_CCIX_BUFFER_GM": "1",
        "BKCL_FORCE_L3_RDMA": "0",
        "BKCL_RING_BUFFER_GM": "1",
        "BKCL_ENABLE_XDR": "1",
        "BKCL_RDMA_FORCE_TREE": "1",
        "BKCL_TREE_THRESHOLD": "1",
        "BKCL_XLINK_D2D": "0",
        "BKCL_XLINK_ETH": "0",
        "BKCL_XLINK_C2C": "1",
        "ALLREDUCE_ASYNC": "false",
        "ALLGATHER_ASYNC": "false",
        "ALLREDUCE_FUSION": "0",
        "XPU_FORCE_SHARED_DEVICE_CONTEXT": "1",
        "BKCL_RDMA_PROXY_DISABLE": "1",
        "BKCL_TRANS_UNSUPPORTED_DATATYPE": "1",
        "BKCL_KL3_TURBO_MODE": "1",
        "BKCL_RING_BUFFER_SIZE": "2097152",
        "BKCL_TIMEOUT": "400000",
        "CUDA_DISABLE_PRINTF": "1",
        "BKCL_RDMA_VERBS": "1",
        "XMLIR_FA_GEMM_TYPE": "float",
        "XBLAS_FC_HBM_VERSION": "40",
        "XMLIR_PARALLEL_SAVE_MEMORY": "false",
        "XMLIR_DISABLE_CUDA_ALLOCATOR": "false",
        "XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL": "0",
        "XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL": "true",
        "XMLIR_DUMP_FALLBACK_OP_LIST_BOOL": "true",
        "XMLIR_DIST_ASYNC_ISEND_IRECV": "false",
        "XMLIR_BATCH_PARALLEL": "false",
        "BKCL_PRELAUNCH_BARRIER": "1"
    }
  }
}'



infer_benchmark() {
    local config_name="$1"
    case "${config_name}" in
        behavior_*|behavior-* ) echo "behavior" ;;
        libero_*|libero-* ) echo "libero" ;;
        robotwin_*|robotwin-* ) echo "robotwin" ;;
        realworld_*|realworld-* ) echo "realworld" ;;
        maniskill_*|maniskill-* ) echo "maniskill" ;;
        metaworld_*|metaworld-* ) echo "metaworld" ;;
        calvin_*|calvin-* ) echo "calvin" ;;
        robocasa365_*|robocasa365-* ) echo "robocasa365" ;;
        robocasa_*|robocasa-* ) echo "robocasa" ;;
        roboverse_*|roboverse-* ) echo "roboverse" ;;
        polaris_*|polaris-* ) echo "polaris" ;;
        * )
            echo "unknown"
            ;;
    esac
}

run_eval_cmd() {
    local config_path="$1"
    local config_name="$2"
    local log_dir="$3"
    local log_file="$4"
    shift 4
    local extra_args=("$@")

    mkdir -p "${log_dir}"
    local cmd=(
        python "${SRC_FILE}"
        --config-path "${config_path}/"
        --config-name "${config_name}"
        "runner.logger.log_path=${log_dir}"
    )
    if [ ${#extra_args[@]} -gt 0 ]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "${cmd[*]}" | tee "${log_file}"
    "${cmd[@]}" 2>&1 | tee -a "${log_file}"
}

run_mani_ood_eval() {
    local config_name="$1"

    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    setup_sim_env

    local eval_name="${EVAL_NAME:-}"
    local ckpt_path="${CKPT_PATH:-}"
    local total_num_envs="${TOTAL_NUM_ENVS:-}"
    local eval_rollout_epoch="${EVAL_ROLLOUT_EPOCH:-}"

    if [ -z "${eval_name}" ] || [ -z "${ckpt_path}" ] || [ -z "${total_num_envs}" ] || [ -z "${eval_rollout_epoch}" ]; then
        echo "ManiSkill OOD eval requires env vars: EVAL_NAME, CKPT_PATH, TOTAL_NUM_ENVS, EVAL_ROLLOUT_EPOCH" >&2
        exit 1
    fi

    local config_path="${EVALUATIONS_PATH}/maniskill"
    if [ ! -f "${config_path}/${config_name}.yaml" ]; then
        config_path="${EMBODIED_PATH}/config"
        echo "Config not found under evaluations/maniskill, fallback to ${config_path}"
    fi
    echo "Using ManiSkill OOD eval: config=${config_name}, eval_name=${eval_name}"

    local common_args=(
        "env.eval.rollout_epoch=${eval_rollout_epoch}"
        "env.eval.total_num_envs=${total_num_envs}"
        "runner.ckpt_path=${ckpt_path}"
    )

    for env_id in \
        "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
        "PutOnPlateInScene25VisionWhole03-v1" "PutOnPlateInScene25VisionWhole05-v1" \
        "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
        "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
        "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1"; do
        local obj_set="test"
        local log_dir="${REPO_PATH}/logs/eval/${eval_name}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
        local log_file="${log_dir}/run_ppo.log"
        run_eval_cmd "${config_path}" "${config_name}" "${log_dir}" "${log_file}" \
            "${common_args[@]}" \
            "env.eval.init_params.id=${env_id}" \
            "env.eval.init_params.obj_set=${obj_set}"
    done

    for env_id in \
        "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25MultiCarrot-v1" \
        "PutOnPlateInScene25MultiPlate-v1"; do
        local obj_set="train"
        local log_dir="${REPO_PATH}/logs/eval/${eval_name}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
        local log_file="${log_dir}/run_ppo.log"
        run_eval_cmd "${config_path}" "${config_name}" "${log_dir}" "${log_file}" \
            "${common_args[@]}" \
            "env.eval.init_params.id=${env_id}" \
            "env.eval.init_params.obj_set=${obj_set}"
    done
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <benchmark> <config_name> [hydra_overrides...]" >&2
    echo "   or: $0 <config_name> [hydra_overrides...]" >&2
    echo "   or: $0 mani-ood [config_name]  (requires EVAL_NAME, CKPT_PATH, TOTAL_NUM_ENVS, EVAL_ROLLOUT_EPOCH)" >&2
    exit 1
fi

if [ "$1" = "mani-ood" ]; then
    shift
    MANI_OOD_CONFIG_NAME="${1:-maniskill_ood_openvlaoft_eval}"
    run_mani_ood_eval "${MANI_OOD_CONFIG_NAME}"
    exit 0
fi

if [ $# -ge 2 ] && [ -f "${EVALUATIONS_PATH}/$1/$2.yaml" ]; then
    BENCHMARK="$1"
    CONFIG_NAME="$2"
    shift 2
else
    CONFIG_NAME="$1"
    BENCHMARK="$(infer_benchmark "${CONFIG_NAME}")"
    shift
    if [ "${BENCHMARK}" = "unknown" ]; then
        echo "Cannot infer benchmark for config '${CONFIG_NAME}'. Pass benchmark explicitly." >&2
        exit 1
    fi
fi

EXTRA_ARGS=("$@")
CONFIG_PATH="${EVALUATIONS_PATH}/${BENCHMARK}"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"

MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"

run_eval_cmd "${CONFIG_PATH}" "${CONFIG_NAME}" "${LOG_DIR}" "${MEGA_LOG_FILE}" "${EXTRA_ARGS[@]}"
