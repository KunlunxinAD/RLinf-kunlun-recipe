#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_JOB_CONFIG_JSON_ENV_VAR='{
  "runtime_env": {
    "working_dir": "/workspace/RLinf",
      "env_vars": {
        "PYTHONPATH": "/workspace/Megatron-LM/:/workspace/mbridge/:$PYTHONPATH",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDART_DUMMY_REGISTER": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
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
        "XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL": "False",
        "XMLIR_DUMP_FALLBACK_OP_LIST_BOOL": "true",
        "XMLIR_DIST_ASYNC_ISEND_IRECV": "false",
        "XMLIR_BATCH_PARALLEL": "false"
    }
  }
}'

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
# MEGATRON_PATH=/opt/Megatron-LM
# export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-vl-3b-grpo-fsdp-pipeline"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path ${CONFIG_PATH}/config/vqa/  --config-name $CONFIG_NAME