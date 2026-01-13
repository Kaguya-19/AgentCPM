import torch
import time

# 先预分配 pinned tensor
cpu_tensor = torch.empty(1024 * 1024 * 100, dtype=torch.float32).pin_memory()

# 模拟已有数据
cpu_tensor.normal_()

torch.cuda.synchronize()
start = time.time()
gpu_tensor = cpu_tensor.to('cuda', non_blocking=True)
torch.cuda.synchronize()
print(f"H2D time: {time.time() - start:.6f} s")
