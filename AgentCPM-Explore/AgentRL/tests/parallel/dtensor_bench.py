#!/usr/bin/env python3
"""
PyTorch分布式Tensor操作性能测试脚本
支持8卡GPU测试各种tensor操作的耗时
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import numpy as np
from contextlib import contextmanager


@contextmanager
def timer(operation_name, rank):
    """计时器上下文管理器"""
    torch.cuda.synchronize()
    start_time = time.time()
    yield
    torch.cuda.synchronize()
    end_time = time.time()
    if rank == 0:
        print(f"{operation_name}: {(end_time - start_time) * 1000:.3f} ms")


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def test_basic_operations(rank, world_size, tensor_size):
    """测试基础tensor操作"""
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n=== 基础Tensor操作测试 (大小: {tensor_size}) ===")
    
    # 创建测试tensor
    tensor_a = torch.randn(tensor_size, device=device)
    tensor_b = torch.randn(tensor_size, device=device)
    
    # 矩阵乘法
    with timer("矩阵乘法 (matmul)", rank):
        result = torch.matmul(tensor_a, tensor_b.T)
    
    # 元素级操作
    with timer("元素级加法", rank):
        result = tensor_a + tensor_b
    
    with timer("元素级乘法", rank):
        result = tensor_a * tensor_b
    
    # 归约操作
    with timer("求和操作", rank):
        result = torch.sum(tensor_a)
    
    with timer("求平均值", rank):
        result = torch.mean(tensor_a)
    
    # 激活函数
    with timer("ReLU激活", rank):
        result = torch.relu(tensor_a)
    
    with timer("Softmax激活", rank):
        result = torch.softmax(tensor_a, dim=-1)


def test_collective_operations(rank, world_size, tensor_size):
    """测试集合通信操作"""
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n=== 集合通信操作测试 (大小: {tensor_size}) ===")
    
    # 创建测试tensor
    tensor = torch.randn(tensor_size, device=device) * (rank + 1)
    
    # All-Reduce操作
    tensor_allreduce = tensor.clone()
    with timer("All-Reduce操作", rank):
        dist.all_reduce(tensor_allreduce, op=dist.ReduceOp.SUM)
    
    # All-Gather操作
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    with timer("All-Gather操作", rank):
        dist.all_gather(tensor_list, tensor)
    
    # Broadcast操作
    tensor_broadcast = tensor.clone() if rank == 0 else torch.zeros_like(tensor)
    with timer("Broadcast操作", rank):
        dist.broadcast(tensor_broadcast, src=0)
    
    # Reduce操作
    tensor_reduce = tensor.clone()
    with timer("Reduce操作", rank):
        dist.reduce(tensor_reduce, dst=0, op=dist.ReduceOp.SUM)
    
    # All-to-All操作
    input_list = [torch.randn(tensor_size[0] // world_size, *tensor_size[1:], device=device) 
                  for _ in range(world_size)]
    output_list = [torch.zeros_like(input_list[0]) for _ in range(world_size)]
    with timer("All-to-All操作", rank):
        dist.all_to_all(output_list, input_list)


def test_memory_operations(rank, world_size, tensor_size):
    """测试内存相关操作"""
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n=== 内存操作测试 (大小: {tensor_size}) ===")
    
    # 内存分配
    with timer("Tensor内存分配", rank):
        tensor = torch.randn(tensor_size, device=device)
    
    # CPU到GPU传输
    cpu_tensor = torch.randn(tensor_size)
    with timer("CPU到GPU传输", rank):
        gpu_tensor = cpu_tensor.to(device)
    
    # GPU到CPU传输
    with timer("GPU到CPU传输", rank):
        cpu_result = gpu_tensor.cpu()
    
    # 内存拷贝
    source_tensor = torch.randn(tensor_size, device=device)
    with timer("GPU内存拷贝", rank):
        dest_tensor = source_tensor.clone()
    
    # 视图操作
    with timer("Tensor重塑操作", rank):
        reshaped = source_tensor.view(-1)
        reshaped = reshaped.view(tensor_size)


def test_gradient_operations(rank, world_size, tensor_size):
    """测试梯度相关操作"""
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n=== 梯度操作测试 (大小: {tensor_size}) ===")
    
    # 创建需要梯度的tensor
    x = torch.randn(tensor_size, device=device, requires_grad=True)
    y = torch.randn(tensor_size, device=device, requires_grad=True)
    
    # 前向传播
    with timer("前向传播计算", rank):
        z = torch.sum(x * y + torch.sin(x))
    
    # 反向传播
    with timer("反向传播计算", rank):
        z.backward()
    
    # 梯度清零
    with timer("梯度清零", rank):
        x.grad.zero_()
        y.grad.zero_()


def test_ddp_operations(rank, world_size):
    """测试DDP相关操作"""
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"\n=== DDP操作测试 ===")
    
    # 创建简单模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1024, 512)
            self.linear2 = torch.nn.Linear(512, 256)
            self.linear3 = torch.nn.Linear(256, 10)
            
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            return self.linear3(x)
    
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 创建测试数据
    batch_size = 64
    input_data = torch.randn(batch_size, 1024, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # DDP前向传播
    with timer("DDP前向传播", rank):
        output = ddp_model(input_data)
        loss = criterion(output, target)
    
    # DDP反向传播
    with timer("DDP反向传播", rank):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def print_gpu_info(rank):
    """打印GPU信息"""
    if rank == 0:
        print("=== GPU信息 ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  内存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  多处理器数量: {props.multi_processor_count}")
        print()


def run_benchmark(rank, world_size, args):
    """运行基准测试"""
    setup(rank, world_size)
    
    if rank == 0:
        print(f"开始分布式Tensor操作性能测试")
        print(f"使用 {world_size} 个GPU")
        
    print_gpu_info(rank)
    
    # 不同大小的tensor测试
    test_sizes = [
        (1024, 1024),      # 1K x 1K
        (2048, 2048),      # 2K x 2K
        (4096, 4096),      # 4K x 4K
    ]
    
    if args.large_tensor:
        test_sizes.append((8192, 8192))  # 8K x 8K
    
    for tensor_size in test_sizes:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"测试Tensor大小: {tensor_size}")
            print(f"{'='*60}")
        
        # 同步所有进程
        dist.barrier()
        
        # 运行各种测试
        test_basic_operations(rank, world_size, tensor_size)
        test_collective_operations(rank, world_size, tensor_size)
        test_memory_operations(rank, world_size, tensor_size)
        test_gradient_operations(rank, world_size, tensor_size)
    
    # 测试DDP操作
    test_ddp_operations(rank, world_size)
    
    # 最终同步
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("所有测试完成!")
        print(f"{'='*60}")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='PyTorch分布式Tensor操作性能测试')
    parser.add_argument('--world-size', type=int, default=8, 
                       help='总GPU数量 (默认: 8)')
    parser.add_argument('--large-tensor', action='store_true',
                       help='包含大型tensor测试 (8192x8192)')
    parser.add_argument('--warmup', type=int, default=3,
                       help='预热轮数 (默认: 3)')
    
    args = parser.parse_args()
    
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA支持")
        return
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.world_size:
        print(f"错误: 需要 {args.world_size} 个GPU，但只检测到 {available_gpus} 个")
        return
    
    print(f"检测到 {available_gpus} 个GPU，将使用 {args.world_size} 个进行测试")
    
    # 启动多进程
    mp.spawn(
        run_benchmark,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
