import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Shard, Replicate
import torch.distributed.fsdp

sys.path.append("./")

from src.patches.offloading import OffloadWrapper

def setup():
    """初始化分布式进程组。"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    return rank, dist.get_world_size(), device

def cleanup():
    """清理分布式进程组。"""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """一个简单的模型用于测试。"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4096, 4096,bias=False)
        self.layer2 = nn.Linear(4096, 4096,bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x

def run_test(rank, world_size, device):
    """运行 DTensor offloading 测试。"""
    print(f"在 rank {rank} 上运行测试...")

    # 1. 创建 DeviceMesh
    mesh = DeviceMesh("cuda", torch.arange(world_size))

    # 2. 创建模型并将其参数转换为 DTensor
    model = SimpleModel().to(device)

    # 将模型的参数转换为 Replicated DTensor
    torch.distributed.fsdp.fully_shard(model, mesh=mesh)
    assert all(isinstance(p, DTensor) for p in model.parameters()), "模型参数未全部转换为 DTensor"
    print(f"Rank {rank}: 模型参数已转换为 DTensor")

    # 3. 创建输入
    input_tenosr = torch.randn(128, 4096, device=device)
    print(f"Rank {rank}: 输入 Tensor 已创建")

    # 4. 前向传播
    try:
        model = OffloadWrapper(model)
        output = model(input_tenosr)
        print(f"Rank {rank}: 前向传播成功")
    except Exception as e:
        print(f"Rank {rank}: 前向传播失败: {e}")
        raise

    # 5. 计算损失
    loss = output.pow(2).mean()

    print(f"Rank {rank}: 损失计算成功. Loss: {loss.item()}")

    # 6. 后向传播
    try:
        loss.backward()
        print(f"Rank {rank}: 后向传播成功")
    except Exception as e:
        print(f"Rank {rank}: 后向传播失败: {e}")
        raise

    # 7. 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Rank {rank}: 参数 {name} 的梯度存在。")
        else:
            print(f"Rank {rank}: 警告: 参数 {name} 的梯度不存在。")

    print(f"Rank {rank}: 测试完成。")


if __name__ == "__main__":
    rank, world_size, device = setup()
    
    # 仅在 rank 0 上打印一次 PyTorch 和 CUDA 信息
    if rank == 0:
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("-" * 30)

    run_test(rank, world_size, device)
    cleanup()
    print(torch.cuda.memory_summary())
    
    if rank == 0:
        print("\nDTensor offloading 测试成功完成！")
        print("要运行此脚本，请使用 torchrun，例如：")
        print("torchrun --nproc_per_node=2 tests/parallel/offloading.py")

