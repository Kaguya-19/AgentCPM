import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from collections.abc import Mapping

def auto_broadcast(data, device, group, group_src=0):
    is_src_rank = dist.get_group_rank(group, dist.get_rank()) == group_src
    data_type = type(data) if is_src_rank else None
    data_type_list = [data_type]
    dist.broadcast_object_list(data_type_list, group=group, group_src=group_src)
    data_type = data_type_list[0]

    if issubclass(data_type, torch.Tensor):
        # broadcast tensor metadata
        tensor_meta = [data.size(), data.dtype, data.layout, data.device] if is_src_rank else [None, None, None, None]
        dist.broadcast_object_list(tensor_meta, group=group, group_src=group_src)
        size, dtype, layout, _ = tensor_meta
        if not is_src_rank:
            data = torch.empty(size, dtype=dtype, layout=layout, device=device)
        else:
            data = data.to(device)
        dist.broadcast(data, group_src=group_src, group=group)
        return data

    elif issubclass(data_type, (int, float, bool)) or data_type is type(None):
        obj = [data] if is_src_rank else [None]
        dist.broadcast_object_list(obj, group=group, group_src=group_src)
        return obj[0]

    elif issubclass(data_type, (list, tuple)):
        length = [len(data)] if is_src_rank else [0]
        dist.broadcast_object_list(length, group=group, group_src=group_src)
        length = length[0]
        return [auto_broadcast(data[i] if is_src_rank else None, device, group, group_src)
                for i in range(length)]

    elif issubclass(data_type, Mapping):
        keys = list(data.keys()) if is_src_rank else []
        length = [len(keys)] if is_src_rank else [0]
        dist.broadcast_object_list(length, group=group, group_src=group_src)
        n_keys = length[0]
        new_dict = {}
        for i in range(n_keys):
            key = [keys[i]] if is_src_rank else [None]
            dist.broadcast_object_list(key, group=group, group_src=group_src)
            key = key[0]
            new_dict[key] = auto_broadcast(data[key] if is_src_rank else None, device, group, group_src)
        return new_dict

    else:
        raise TypeError(f"Unsupported type: {type(data)}")

def run(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建一个嵌套的数据结构
    if rank == 0:
        data = {
            "tensor": torch.tensor([1.0, 2.0, 3.0]),
            "list": [1, 2, 3],
            "dict": {"a": True, "b": None},
            "float": 3.14,
            "int": 42
        }
    else:
        data = None  # 非 src rank 初始为空

    group = dist.new_group(ranks=list(range(world_size)))
    result = auto_broadcast(data, device='cpu', group=group)

    print(f"Rank {rank} received: {result}")

    dist.destroy_process_group()

# ---- 主程序 ----
if __name__ == "__main__":
    world_size = 2  # 测试 2 个进程
    spawn(run, args=(world_size,), nprocs=world_size)
