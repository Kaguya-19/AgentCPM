import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper
import time
from collections import defaultdict

# 用于存储计时信息的全局字典
timings = defaultdict(float)
call_counts = defaultdict(int)

def print_timings():
    """打印计时统计信息。"""
    print("\n--- Offloading 性能统计 ---")
    # 按总耗时降序排序
    sorted_timings = sorted(timings.items(), key=lambda item: item[1], reverse=True)
    for key, total_time in sorted_timings:
        count = call_counts[key]
        avg_time = total_time / count if count > 0 else 0
        print(f"{key:<30}: 总耗时: {total_time:.4f}s, 调用次数: {count}, 平均耗时: {avg_time:.6f}s")
    print("---------------------------------\n")

# --- Offloading 性能统计 ---
# pack_dtensor                  : 总耗时: 69.0925s, 调用次数: 9000, 平均耗时: 0.007677s
# pack_dtensor_pin_memory_copy  : 总耗时: 69.0717s, 调用次数: 9000, 平均耗时: 0.007675s
# pack_tensor                   : 总耗时: 34.7159s, 调用次数: 9539, 平均耗时: 0.003639s
# pack_tensor_pin_memory_copy   : 总耗时: 34.6969s, 调用次数: 9539, 平均耗时: 0.003637s
# unpack_dtensor                : 总耗时: 0.6757s, 调用次数: 8496, 平均耗时: 0.000080s
# unpack_dtensor_from_local     : 总耗时: 0.4270s, 调用次数: 8496, 平均耗时: 0.000050s
# unpack_dtensor_to_device      : 总耗时: 0.2308s, 调用次数: 8496, 平均耗时: 0.000027s
# unpack_tensor                 : 总耗时: 0.1815s, 调用次数: 9036, 平均耗时: 0.000020s
# unpack_tensor_to_device       : 总耗时: 0.1715s, 调用次数: 9036, 平均耗时: 0.000019s
# pack_dtensor_to_local         : 总耗时: 0.0088s, 调用次数: 9000, 平均耗时: 0.000001s
# ---------------------------------


# --- Offloading 性能统计 ---
# pack_dtensor                  : 总耗时: 69.0925s, 调用次数: 9000, 平均耗时: 0.007677s
# pack_dtensor_pin_memory_copy  : 总耗时: 69.0717s, 调用次数: 9000, 平均耗时: 0.007675s
# pack_tensor                   : 总耗时: 34.7165s, 调用次数: 9540, 平均耗时: 0.003639s
# pack_tensor_pin_memory_copy   : 总耗时: 34.6974s, 调用次数: 9540, 平均耗时: 0.003637s
# unpack_dtensor                : 总耗时: 0.6757s, 调用次数: 8496, 平均耗时: 0.000080s
# unpack_dtensor_from_local     : 总耗时: 0.4270s, 调用次数: 8496, 平均耗时: 0.000050s
# unpack_dtensor_to_device      : 总耗时: 0.2308s, 调用次数: 8496, 平均耗时: 0.000027s
# unpack_tensor                 : 总耗时: 0.1815s, 调用次数: 9036, 平均耗时: 0.000020s
# unpack_tensor_to_device       : 总耗时: 0.1715s, 调用次数: 9036, 平均耗时: 0.000019s
# pack_dtensor_to_local         : 总耗时: 0.0088s, 调用次数: 9000, 平均耗时: 0.000001s
# ---------------------------------


# --- Offloading 性能统计 ---
# pack_dtensor                  : 总耗时: 69.0925s, 调用次数: 9000, 平均耗时: 0.007677s
# pack_dtensor_pin_memory_copy  : 总耗时: 69.0717s, 调用次数: 9000, 平均耗时: 0.007675s
# pack_tensor                   : 总耗时: 34.7170s, 调用次数: 9541, 平均耗时: 0.003639s
# pack_tensor_pin_memory_copy   : 总耗时: 34.6979s, 调用次数: 9541, 平均耗时: 0.003637s
# unpack_dtensor                : 总耗时: 0.6757s, 调用次数: 8496, 平均耗时: 0.000080s
# unpack_dtensor_from_local     : 总耗时: 0.4270s, 调用次数: 8496, 平均耗时: 0.000050s
# unpack_dtensor_to_device      : 总耗时: 0.2308s, 调用次数: 8496, 平均耗时: 0.000027s
# unpack_tensor                 : 总耗时: 0.1815s, 调用次数: 9036, 平均耗时: 0.000020s
# unpack_tensor_to_device       : 总耗时: 0.1715s, 调用次数: 9036, 平均耗时: 0.000019s
# pack_dtensor_to_local         : 总耗时: 0.0088s, 调用次数: 9000, 平均耗时: 0.000001s

class save_dtensor_on_cpu(saved_tensors_hooks):
    """Context manager under which dtensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn't fit in GPU memory during training).

    Args:
        pin_memory (bool): If ``True`` dtensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_dtensor_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU dtensors are moved back to GPU, for backward
        >>> # all intermediary dtensors are released (deleted) after the call to backward
    """

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor|DTensor):
            # if call_counts['pack_dtensor_to_local'] % 1000 == 0 and call_counts['pack_dtensor_to_local'] > 0:
            #     print_timings()
            # start_time = time.time()
            # DTensor 需要特殊处理：保存本地 shard + 元数据
            if isinstance(tensor, DTensor):
                # to_local_start = time.time()
                local = tensor.to_local()
                # timings['pack_dtensor_to_local'] += time.time() - to_local_start
                # call_counts['pack_dtensor_to_local'] += 1

                if not pin_memory:
                    # to_cpu_start = time.time()
                    local_cpu = local.cpu()
                    # timings['pack_dtensor_to_cpu'] += time.time() - to_cpu_start
                    # call_counts['pack_dtensor_to_cpu'] += 1
                else:
                    # copy_start = time.time()
                    local_cpu = torch.empty(
                        local.size(),
                        dtype=local.dtype,
                        layout=local.layout,
                        pin_memory=(device_module.is_available() and not local.is_sparse),
                    )
                    local_cpu.copy_(local, non_blocking=True)
                    # timings['pack_dtensor_pin_memory_copy'] += time.time() - copy_start
                    # call_counts['pack_dtensor_pin_memory_copy'] += 1
                
                # timings['pack_dtensor'] += time.time() - start_time
                # call_counts['pack_dtensor'] += 1
                # 打包为标记的对象，附带重建所需的元数据
                return (
                    "DTENSOR",
                    (local_cpu, local.device, tensor.device_mesh, tensor.placements, tensor.shape, tensor.stride()),
                )
            else:
                # 普通 Tensor 仍保持原逻辑
                if not pin_memory:
                    # to_cpu_start = time.time()
                    cpu_tensor = tensor.cpu()
                    # timings['pack_tensor_to_cpu'] += time.time() - to_cpu_start
                    # call_counts['pack_tensor_to_cpu'] += 1
                    result = ("TENSOR", (tensor.device, cpu_tensor))
                else:
                    # copy_start = time.time()
                    packed = torch.empty(
                        tensor.size(),
                        dtype=tensor.dtype,
                        layout=tensor.layout,
                        pin_memory=(device_module.is_available() and not tensor.is_sparse),
                    )
                    packed.copy_(tensor, non_blocking=True)
                    # timings['pack_tensor_pin_memory_copy'] += time.time() - copy_start
                    # call_counts['pack_tensor_pin_memory_copy'] += 1
                    result = ("TENSOR", (tensor.device, packed))

                # timings['pack_tensor'] += time.time() - start_time
                # call_counts['pack_tensor'] += 1
                return result

        def unpack_from_cpu(packed):
            # start_time = time.time()
            tag, payload = packed
            if tag == "DTENSOR":
                local_cpu, local_device, mesh, placements, shape, stride = payload
                
                # to_device_start = time.time()
                local = local_cpu.to(local_device, non_blocking=pin_memory)
                # timings['unpack_dtensor_to_device'] += time.time() - to_device_start
                # call_counts['unpack_dtensor_to_device'] += 1

                # from_local_start = time.time()
                # 用原 mesh/placements 重建全局 DTensor
                dtensor = DTensor.from_local(local, mesh, placements, shape=shape, stride=stride)
                # timings['unpack_dtensor_from_local'] += time.time() - from_local_start
                # call_counts['unpack_dtensor_from_local'] += 1

                # timings['unpack_dtensor'] += time.time() - start_time
                # call_counts['unpack_dtensor'] += 1
                return dtensor
            else:
                device, tensor = payload
                # to_device_start = time.time()
                gpu_tensor = tensor.to(device, non_blocking=pin_memory)
                # timings['unpack_tensor_to_device'] += time.time() - to_device_start
                # call_counts['unpack_tensor_to_device'] += 1

                # timings['unpack_tensor'] += time.time() - start_time
                # call_counts['unpack_tensor'] += 1
                return gpu_tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)
        

class OffloadWrapper(ActivationWrapper):
    def __init__(self, mod):
        super().__init__(mod)

    def forward(self, *args, **kwargs):
        with save_dtensor_on_cpu(pin_memory=True):
            return self._checkpoint_wrapped_module(*args, **kwargs)

def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap a module for activation offloading to CPU.

    Offloads intermediate activations to the CPU for modules wrapped with this function.
    Wrappers with activation offload can be composed with ones that do recomputation-based
    checkpoint to trade off increased compute versus increased CPU
    memory usage and additional H2D transfers.

    Usage::
        offloaded_module = offload_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
    Returns:
        (nn.Module):
            Wrapped module
    """
    return OffloadWrapper(module)
