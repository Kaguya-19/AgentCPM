import sys
import os
sys.path.append("./")
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from src.patches.norm import _get_total_norm as get_total_norm


# ---------------------------------------------------------
# 构造各种测试 tensor
# ---------------------------------------------------------
def build_test_tensors(mesh):
    device = f"cuda:{dist.get_rank()}"

    # 1) 普通张量
    vanilla = torch.randn(6, device=device)

    # 2) replicate tensor
    base_rep = torch.arange(6, dtype=torch.float32)
    rep = DTensor.from_local(base_rep.to(device), mesh, placements=[Replicate()])

    # 3) shard tensor
    base_shard = torch.arange(12, dtype=torch.float32)
    shard = DTensor.from_local(
        base_shard,
        mesh,
    ).redistribute(placements=[Shard(0)])

    return vanilla, rep, shard


# ---------------------------------------------------------
# ground truth：用 rank0 直接计算全局 norm
# ---------------------------------------------------------
def ground_truth(tensors):
    full = []

    for t in tensors:
        if isinstance(t, DTensor):
            # gather shards/replicates
            gathered = t.redistribute(placements=[Replicate()]).to_local()
            full.append(gathered.cpu())
        else:
            full.append(t.cpu())

    # 拼接成一个大向量
    big = torch.cat([x.reshape(-1) for x in full])
    return torch.linalg.vector_norm(big, 2)


# ---------------------------------------------------------
# 主测试逻辑
# ---------------------------------------------------------
def test_all():
    mesh = init_device_mesh("cuda", [dist.get_world_size(),])

    vanilla, rep, shard = build_test_tensors(mesh)

    cases = {
        "vanilla only": [vanilla],
        "rep only": [rep],
        "shard only": [shard],
        "vanilla + rep": [vanilla, rep],
        "vanilla + shard": [vanilla, shard],
        "rep + shard": [rep, shard],
        "vanilla + rep + shard": [vanilla, rep, shard],
    }

    for name, tensors in cases.items():
        dist.barrier()

        val = get_total_norm(tensors, norm_type=2).cpu()

        if dist.get_rank() == 0:
            gt = ground_truth(tensors)
            ok = torch.allclose(val, gt, atol=1e-6)
            print(f"[{name}] result={val.item():.6f}, gt={gt.item():.6f}, PASS={ok}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365'
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    torch.cuda.set_device(dist.get_rank())

    if dist.get_rank() == 0:
        print("------ START TEST ------")

    test_all()

    dist.barrier()
    if dist.get_rank() == 0:
        print("------ DONE ------")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)