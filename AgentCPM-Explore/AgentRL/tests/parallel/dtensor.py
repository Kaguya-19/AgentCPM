import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor, Replicate, Shard
import os

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    
    # 1. 创建设备网格 (2个GPU)
    device_mesh = DeviceMesh("cuda", list(range(world_size)))
    
    # 2. 创建原始张量并分片为 DTensor
    global_tensor = torch.randn(4, 4).cuda(rank)
    dtensor = distribute_tensor(
        global_tensor, 
        device_mesh, 
        placements=[Shard(0)]  # 按第0维度分片
    )
    print("Rank {} dtenosr(shard) norm: {}".format(rank, torch.linalg.norm(dtensor,ord=2)))
    print("Rank {} tensor(replicate) norm: {}".format(rank, torch.linalg.norm(dtensor.redistribute(placements=[Replicate()]),ord=2)))
    print("Rank {} tensor norm: {}".format(rank, torch.linalg.norm(dtensor.full_tensor(),ord=2)))

    if rank == 0:
        print(f"Original DTensor (sharded on GPU):\n{dtensor._local_tensor}")

    # 3. Offload 到 CPU --------------------------------------------------------
    # 步骤 1: 将本地部分移动到 CPU
    cpu_local = dtensor._local_tensor.cpu()
    
    # 步骤 2: 收集所有分片到全局张量 (需要同步)
    global_shape = dtensor.size()
    gathered_tensors = [torch.zeros_like(cpu_local) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, cpu_local)
    
    # 只在 rank0 重建全局张量 (实际应用中可根据需求调整)
    if rank == 0:
        global_cpu = torch.cat(gathered_tensors, dim=0)
        print(f"\nGlobal tensor on CPU:\n{global_cpu}")
    else:
        global_cpu = None

    # 4. Reload 到 DTensor ----------------------------------------------------
    # 步骤 1: 广播全局CPU张量 (实际应用中可用文件加载替代)
    if rank == 0:
        reload_tensor = global_cpu
    else:
        reload_tensor = torch.empty(global_shape, dtype=torch.float32)
    dist.broadcast(reload_tensor, src=0)
    
    # 步骤 2: 重新分发为 DTensor
    reload_dtensor = distribute_tensor(
        reload_tensor.to(f"cuda:{rank}"),  # 移动到当前GPU
        device_mesh,
        placements=[Shard(0)]
    )
    
    print(f"Rank {rank} reloaded local shard:\n{reload_dtensor._local_tensor}")
    
    cleanup()

if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)