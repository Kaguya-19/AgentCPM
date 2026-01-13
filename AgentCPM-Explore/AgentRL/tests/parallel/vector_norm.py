# test_dtensor_norm.py

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed._tensor.placement_types import Shard

def init_dist():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def build_dtensor():
    """
    构造一个简单的 1D DTensor:
    全局张量: [1, 2, 3, ..., N]
    每个 rank 拿一段 shard。
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    N = 16  # global tensor size
    shard_size = N // world_size

    # local shard range
    start = rank * shard_size
    end = start + shard_size
    local = torch.arange(start + 1, end + 1, dtype=torch.float32, device=torch.cuda.current_device())

    # Build mesh
    mesh = DeviceMesh("cuda", list(range(world_size)))

    # shard along dim 0
    dt = DTensor.from_local(local, mesh, placements=[Shard(0)])

    return dt


def manual_global_norm(dt, p=2.0):
    """
    使用手动方法 global sum-of-squares → sqrt 来计算全局 norm。
    """
    local = dt.to_local()  # local shard
    local_sq = torch.sum(local ** 2)

    # full reduce
    global_sq = local_sq.clone()
    dist.all_reduce(global_sq, op=dist.ReduceOp.SUM)

    return global_sq.sqrt()


def test_norm():
    rank = dist.get_rank()

    # 1. 构造 DTensor
    dt = build_dtensor()

    # 2. 使用 linalg.vector_norm 得到 partial DTensor
    partial = torch.linalg.vector_norm(dt, ord=2.0)

    # 3. full_tensor() 得到全局 norm
    full_result = partial.full_tensor()

    # 4. 手动 reduce (golden result)
    manual_result = manual_global_norm(dt)

    # 打印结果(每个 rank 输出一次)
    print(f"[rank {rank}] dt.local = {dt.to_local().tolist()}")
    print(f"[rank {rank}] vector_norm().full_tensor() = {full_result.item()}")
    print(f"[rank {rank}] manual global norm       = {manual_result.item()}")
    print()

    # 验证一致性
    if not torch.allclose(full_result, manual_result, atol=1e-6, rtol=1e-6):
        print(f"[rank {rank}] ❌ MISMATCH!")
    else:
        print(f"[rank {rank}] ✅ MATCH")


if __name__ == "__main__":
    init_dist()
    torch.cuda.synchronize()
    test_norm()
