#!/bin/bash

# PyTorch分布式Tensor操作性能测试启动脚本

set -e

echo "=== PyTorch分布式Tensor操作性能测试 ==="
echo

# 检查Python和PyTorch
echo "检查环境..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
echo

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# 默认参数
WORLD_SIZE=8
LARGE_TENSOR=""
WARMUP=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --large-tensor)
            LARGE_TENSOR="--large-tensor"
            shift
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --world-size N     使用N个GPU (默认: 8)"
            echo "  --large-tensor     包含大型tensor测试"
            echo "  --warmup N         预热轮数 (默认: 3)"
            echo "  -h, --help         显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "运行参数:"
echo "  GPU数量: $WORLD_SIZE"
echo "  大型tensor测试: $([ -n "$LARGE_TENSOR" ] && echo "启用" || echo "禁用")"
echo "  预热轮数: $WARMUP"
echo

# 运行测试
echo "开始运行分布式测试..."
python3 dtensor_bench.py \
    --world-size $WORLD_SIZE \
    $LARGE_TENSOR \
    --warmup $WARMUP

echo
echo "测试完成!"
