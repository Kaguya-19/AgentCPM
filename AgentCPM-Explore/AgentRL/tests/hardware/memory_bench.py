import torch
import numpy as np
import time
import pandas as pd

def benchmark_memory(size_bytes=256*1024*1024):
    # Prepare host arrays and GPU tensor
    num_elements = size_bytes // 8  # using float64 for host copy
    host_arr = np.random.rand(num_elements).astype(np.float64)
    pinned_tensor = torch.from_numpy(host_arr).pin_memory()
    gpu_tensor = torch.empty_like(pinned_tensor, device='cuda')

    # Host-to-Host copy
    start = time.time()
    _ = host_arr.copy()
    h2h_time = time.time() - start
    h2h_bw = size_bytes / h2h_time / 1e9

    # Host-to-Device copy
    start = time.time()
    _ = gpu_tensor.copy_(pinned_tensor, non_blocking=True)
    torch.cuda.synchronize()
    h2d_time = time.time() - start
    h2d_bw = size_bytes / h2d_time / 1e9

    # Device-to-Host copy
    start = time.time()
    _ = gpu_tensor.cpu()
    torch.cuda.synchronize()
    d2h_time = time.time() - start
    d2h_bw = size_bytes / d2h_time / 1e9

    df = pd.DataFrame({
        'Operation': ['Host→Host', 'Host→Device', 'Device→Host'],
        'Bandwidth (GB/s)': [h2h_bw, h2d_bw, d2h_bw]
    })
    print(df)

benchmark_memory()
