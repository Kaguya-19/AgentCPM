#!/bin/bash
set -euo pipefail

# --- 用户配置区 ---
NUM_PROCESSES=<YOUR_NUM_PROCESSES>
if [ $# -lt 1 ]; then
  echo "用法: $0 <RUN_NAME>"
  exit 1
fi
RUN_NAME="$1"

FILE_DIR=$(cd "$(dirname "$0")" && pwd)
# ---------------------
# 创建日志目录
mkdir -p logs/as

cleanup() {
  echo -e "\n>>> Cleanup: killing training processes..."
  pkill -f 'accelerate launch' || true
}
trap cleanup EXIT SIGINT SIGTERM

echo "Training directory: $FILE_DIR"
echo "Launching evaluation locally..."

export TOKENIZERS_PARALLELISM=false
export TOKENIZERS_PARALLELISM=false

#多机多卡配置
export MACHINE_RANK=0             # 节点编号
export NUM_MACHINES=1             # 总机器数
export MASTER_ADDR="<YOUR_MASTER_ADDR>" 
export MASTER_PORT="<YOUR_MASTER_PORT>"

# 通信配置
export NCCL_SOCKET_IFNAME="<YOUR_NCCL_SOCKET_IFNAME>"
export GLOO_SOCKET_IFNAME="<YOUR_GLOO_SOCKET_IFNAME>"
export TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_TIMEOUT=14400000
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL="<YOUR_NCCL_NET_GDR_LEVEL>"
export NCCL_IB_HCA="<YOUR_NCCL_IB_HCA>"
unset NCCL_IB_GID_INDEX
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export MONITOR_INTERVAL=5
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export LOG_LEVEL=debug

# swanlab配置
export SWANLAB_API_KEY="<YOUR_SWANLAB_API_KEY>"
export SWANLAB_PROJECT="<YOUR_SWANLAB_PROJECT>"
export SWANLAB_WORKSPACE="<YOUR_SWANLAB_WORKSPACE>"


accelerate launch \
    --config_file assets/fsdp2_dst.yml \
    --num_processes=$NUM_PROCESSES \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$NUM_MACHINES \
    \
    src/sampling.py \
    --model_name_or_path "qwen3-4b-thinking-2507" \
    --trainer "QwenTrainer" \
    --db_connection_string "mongodb://11.11.22.3:27016/$RUN_NAME?replicaSet=rs0" \
    --seed 42 \
    --run_name $RUN_NAME \
    --report_to swanlab \
    \
    --agent_config_path "assets/agent_config.yml" \
    --mcp_manager_url "http://11.11.23.2:9700/mcpapi" \
    --enable_sampling true \
    --max_new_tokens 32768 \
    --max_prompt_tokens 128768 \
    --num_generations 4 \
    --tool_call_parser "qwen" \
    --preferred_sampling_params '{"temperature": 1, "top_p": 1, "min_p": 0, "top_k": -1}' \
    --presence_penalty 1.0 \
    --frequency_penalty 0 \
    --inf_tp_size 1 \
    --inf_mem_ratio 0.8 \
    --max_concurrent_samples_per_process 4 \
    --target_concurrency 3 \
    --max_turns 200 \
    --repetition_early_stop_times 4 \
    --enable_repetition_compress false \
    --new_context "discard_all" \
    \
    --output_dir output/$RUN_NAME \
    > logs/as/${RUN_NAME}_eval_${MACHINE_RANK}.log 2>&1

