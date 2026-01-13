#!/usr/bin/env python3
"""
最小化的NCCL P2P通信检查脚本
检查8张GPU之间的intraNodeP2pSupport和directMode
"""

import os
import torch
import torch.distributed as dist
import argparse
import time

def setup_nccl_env():
    """设置NCCL环境变量"""
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_P2P_DIRECT_DISABLE'] = '0'
    os.environ['NCCL_P2P_LEVEL'] = 'SYS'
    # os.environ['NCCL_IB_DISABLE'] = '1' 
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'
    os.environ['NCCL_ALGO'] = 'Ring'

def check_gpu_availability():
    """检查GPU可用性"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 张GPU")
    
    if gpu_count < 8:
        print(f"警告: 只有 {gpu_count} 张GPU，建议使用8张GPU进行完整测试")
    
    return gpu_count

def check_p2p_access():
    """检查GPU之间的P2P访问能力"""
    gpu_count = torch.cuda.device_count()
    print("\n=== GPU P2P访问矩阵 ===")
    print("    ", end="")
    for j in range(gpu_count):
        print(f"GPU{j:2d}", end=" ")
    print()
    
    p2p_matrix = []
    for i in range(gpu_count):
        row = []
        print(f"GPU{i:2d}", end=" ")
        for j in range(gpu_count):
            if i == j:
                print("  -  ", end="")
                row.append(True)
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print("  ✓  " if can_access else "  ✗  ", end="")
                row.append(can_access)
        print()
        p2p_matrix.append(row)
    
    # 统计P2P支持情况
    total_pairs = gpu_count * (gpu_count - 1)
    supported_pairs = sum(sum(row) - 1 for row in p2p_matrix)  # 减去对角线
    
    print(f"\nP2P支持情况: {supported_pairs}/{total_pairs} ({supported_pairs/total_pairs*100:.1f}%)")
    
    return p2p_matrix

def init_process_group(rank, world_size):
    """初始化进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.default_pg_timeout
    )

def test_allreduce(rank, world_size, tensor_size=1024*1024):
    """测试AllReduce通信"""
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # 创建测试张量
    tensor = torch.ones(tensor_size, device=device, dtype=torch.float32) * rank
    
    print(f"Rank {rank}: 初始张量和 = {tensor.sum().item()}")
    
    # 执行AllReduce
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end_time = time.time()
    
    expected_sum = sum(range(world_size)) * tensor_size
    actual_sum = tensor.sum().item()
    
    print(f"Rank {rank}: AllReduce完成, 耗时 {(end_time-start_time)*1000:.2f}ms")
    print(f"Rank {rank}: 期望和 = {expected_sum}, 实际和 = {actual_sum}")
    
    return abs(actual_sum - expected_sum) < 1e-6

def test_p2p_send_recv(rank, world_size):
    """测试点对点发送接收"""
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    tensor_size = 1024
    success_count = 0
    total_tests = 0
    
    for peer in range(world_size):
        if peer == rank:
            continue
            
        total_tests += 1
        
        try:
            if rank < peer:
                # 发送
                send_tensor = torch.full((tensor_size,), float(rank), device=device)
                print(f"Rank {rank}: 向 Rank {peer} 发送张量")
                dist.send(send_tensor, dst=peer)
                
                # 接收
                recv_tensor = torch.zeros(tensor_size, device=device)
                dist.recv(recv_tensor, src=peer)
                print(f"Rank {rank}: 从 Rank {peer} 接收张量，值 = {recv_tensor[0].item()}")
                
                if abs(recv_tensor[0].item() - peer) < 1e-6:
                    success_count += 1
                    print(f"Rank {rank}: P2P测试 {rank}<->{peer} 成功")
                else:
                    print(f"Rank {rank}: P2P测试 {rank}<->{peer} 失败")
            else:
                # 接收
                recv_tensor = torch.zeros(tensor_size, device=device)
                dist.recv(recv_tensor, src=peer)
                print(f"Rank {rank}: 从 Rank {peer} 接收张量，值 = {recv_tensor[0].item()}")
                
                # 发送
                send_tensor = torch.full((tensor_size,), float(rank), device=device)
                print(f"Rank {rank}: 向 Rank {peer} 发送张量")
                dist.send(send_tensor, dst=peer)
                
                if abs(recv_tensor[0].item() - peer) < 1e-6:
                    success_count += 1
                    print(f"Rank {rank}: P2P测试 {rank}<->{peer} 成功")
                else:
                    print(f"Rank {rank}: P2P测试 {rank}<->{peer} 失败")
                    
        except Exception as e:
            print(f"Rank {rank}: P2P测试 {rank}<->{peer} 异常: {e}")
    
    return success_count, total_tests

def main_worker(rank, world_size):
    """工作进程主函数"""
    try:
        print(f"启动Rank {rank}/{world_size}")
        
        # 初始化进程组
        init_process_group(rank, world_size)
        
        # 等待所有进程同步
        dist.barrier()
        
        if rank == 0:
            print("\n=== 开始NCCL通信测试 ===")
        
        # 测试AllReduce
        print(f"\nRank {rank}: 开始AllReduce测试...")
        allreduce_success = test_allreduce(rank, world_size)
        
        dist.barrier()
        
        # 测试P2P通信
        print(f"\nRank {rank}: 开始P2P通信测试...")
        p2p_success, p2p_total = test_p2p_send_recv(rank, world_size)
        
        dist.barrier()
        
        if rank == 0:
            print(f"\n=== 测试完成 ===")
            print(f"AllReduce测试: {'成功' if allreduce_success else '失败'}")
            print(f"P2P测试: {p2p_success}/{p2p_total} 成功")
        
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"Rank {rank}: 发生异常: {e}")
        import traceback
        traceback.print_exc()

def single_gpu_check():
    """单GPU检查模式"""
    print("=== 单GPU检查模式 ===")
    
    # 检查GPU
    gpu_count = check_gpu_availability()
    
    # 检查P2P访问
    check_p2p_access()
    
    # 检查CUDA能力
    print("\n=== GPU详细信息 ===")
    for i in range(min(gpu_count, 8)):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  内存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  多处理器: {props.multi_processor_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCCL P2P通信检查")
    parser.add_argument("--mode", choices=["check", "test"], default="check",
                       help="模式: check=单GPU检查, test=多GPU通信测试")
    parser.add_argument("--gpus", type=int, default=8,
                       help="使用的GPU数量")
    args = parser.parse_args()
    
    # 设置NCCL环境
    setup_nccl_env()
    
    if args.mode == "check":
        single_gpu_check()
    elif args.mode == "test":
        print("多GPU通信测试模式需要使用torchrun启动")
        print(f"使用命令: torchrun --nproc_per_node={args.gpus} {__file__} --mode test")
        
        # 如果在torchrun环境中
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            main_worker(rank, world_size)
