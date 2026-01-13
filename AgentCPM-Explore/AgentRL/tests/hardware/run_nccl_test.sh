#!/bin/bash
set -euo pipefail

# NCCL P2P通信检查脚本
# 检查8张GPU之间的intraNodeP2pSupport和directMode

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== NCCL P2P通信检查工具 ===${NC}"
echo "检查8张GPU之间的intraNodeP2pSupport和directMode"
echo ""

# 获取脚本目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/nccl_p2p_check.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到nccl_p2p_check.py${NC}"
    exit 1
fi

# 设置NCCL环境变量
# export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_ALGO=Ring

# 检查CUDA是否可用
if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo -e "${RED}错误: PyTorch或CUDA不可用${NC}"
    exit 1
fi

# 获取GPU数量
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$GPU_COUNT" -eq 0 ]; then
    echo -e "${RED}错误: 没有检测到GPU${NC}"
    exit 1
fi

echo -e "${GREEN}检测到 $GPU_COUNT 张GPU${NC}"

# 函数：运行单GPU检查
run_single_check() {
    echo -e "\n${YELLOW}=== 阶段1: 单GPU检查 ===${NC}"
    echo "检查GPU P2P访问能力..."
    
    python3 "$PYTHON_SCRIPT" --mode check
    
    echo -e "\n${GREEN}单GPU检查完成${NC}"
}

# 函数：运行多GPU通信测试
run_multi_test() {
    local num_gpus=${1:-8}
    
    if [ "$GPU_COUNT" -lt "$num_gpus" ]; then
        echo -e "${YELLOW}警告: 只有 $GPU_COUNT 张GPU，将使用所有可用GPU进行测试${NC}"
        num_gpus=$GPU_COUNT
    fi
    
    echo -e "\n${YELLOW}=== 阶段2: 多GPU通信测试 ===${NC}"
    echo "使用 $num_gpus 张GPU进行NCCL通信测试..."
    
    # 使用torchrun运行多GPU测试
    echo "启动torchrun..."
    torchrun \
        --nproc_per_node=$num_gpus \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$PYTHON_SCRIPT" --mode test --gpus $num_gpus
    
    echo -e "\n${GREEN}多GPU通信测试完成${NC}"
}

# 函数：检查NCCL版本和配置
check_nccl_info() {
    echo -e "\n${YELLOW}=== NCCL环境信息 ===${NC}"
    
    # 检查NCCL版本
    python3 -c "
import torch
if hasattr(torch.cuda.nccl, 'version'):
    print(f'NCCL版本: {torch.cuda.nccl.version()}')
else:
    print('NCCL版本: 未知')

print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
" 2>/dev/null || echo "无法获取版本信息"
    
    echo ""
    echo "当前NCCL环境变量:"
    env | grep NCCL | sort
}

# 函数：检查GPU拓扑
check_gpu_topology() {
    echo -e "\n${YELLOW}=== GPU拓扑信息 ===${NC}"
    
    # 使用nvidia-smi检查GPU拓扑
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU拓扑矩阵:"
        nvidia-smi topo -m 2>/dev/null || echo "无法获取GPU拓扑信息"
    else
        echo "nvidia-smi不可用，跳过拓扑检查"
    fi
}

# 主程序
main() {
    case "${1:-all}" in
        "check")
            check_nccl_info
            check_gpu_topology
            run_single_check
            ;;
        "test")
            local num_gpus=${2:-8}
            run_multi_test "$num_gpus"
            ;;
        "topo")
            check_gpu_topology
            ;;
        "info")
            check_nccl_info
            ;;
        "all"|*)
            check_nccl_info
            check_gpu_topology
            run_single_check
            run_multi_test 8
            ;;
    esac
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项] [GPU数量]"
    echo ""
    echo "选项:"
    echo "  all          运行完整检查（默认）"
    echo "  check        只运行单GPU检查"
    echo "  test [N]     只运行多GPU测试（N张GPU，默认8）"
    echo "  topo         只显示GPU拓扑"
    echo "  info         只显示NCCL信息"
    echo "  help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0              # 运行完整检查"
    echo "  $0 check        # 只检查P2P支持"
    echo "  $0 test 4       # 用4张GPU测试通信"
    echo "  $0 topo         # 显示GPU拓扑"
}

# 处理命令行参数
if [ $# -eq 0 ] || [ "$1" = "all" ]; then
    main "all"
elif [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
else
    main "$@"
fi

echo -e "\n${BLUE}=== 检查完成 ===${NC}"
