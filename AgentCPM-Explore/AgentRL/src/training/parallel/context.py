import weakref
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial, wraps
from typing import Any, Callable, Optional, Protocol, Union

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.overrides import TorchFunctionMode
from torch.distributed.tensor.experimental._attention import _enable_cp_dispatcher

torch.nn.functional._original_scaled_dot_product_attention = (
    torch.nn.functional.scaled_dot_product_attention
)

class AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a simliar signature.

    This expects the `forward` method consumes either:

    * a single tensor for self attention
    * one argument for each of: query, key, value

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """

    # use a weakref dictionary to store context managers for each nn.Module
    _CONTEXT_MANAGERS: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
        weakref.WeakKeyDictionary()
    )

    def __init__(self, use_local_output: bool = False , enable: bool = True) -> None:
        super().__init__()
        self.use_local_output = use_local_output
        self.enable = enable


    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                f"{type(device_mesh)} is not supported by {type(self)} yet."
            )

        if not device_mesh.ndim == 1:
            raise ValueError

        module.register_forward_pre_hook(
            lambda mod, inputs: self._input_fn(mod, inputs, device_mesh, enable=self.enable)
        )
        module.register_forward_hook(
            lambda mod, inputs, outputs: self._output_fn(mod, outputs, device_mesh, use_local_output=self.use_local_output, enable=self.enable)
        )
        return module

    @classmethod
    def _input_fn(
        cls,
        module: nn.Module,
        inputs: tuple[Union[torch.Tensor, int, float], ...],
        device_mesh: DeviceMesh,
        enable: bool = True,
    ) -> tuple[Union[torch.Tensor, int, float], ...]:
        def backward_hook(grad: torch.Tensor) -> None:
            if module in cls._CONTEXT_MANAGERS:
                cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
                del cls._CONTEXT_MANAGERS[module]
                # unpatch scaled_dot_product_attention
                torch.nn.functional.scaled_dot_product_attention = (
                    torch.nn.functional._original_scaled_dot_product_attention
                )

        inp = []
        for input in inputs:
            if isinstance(input, torch.Tensor) and input.requires_grad:
                input.register_hook(backward_hook)
            inp.append(input)
        
        if enable:
            manager = _enable_cp_dispatcher()
            manager.__enter__()
            cls._CONTEXT_MANAGERS[module] = manager
            # monkey path torch.nn.functional.scaled_dot_product_attention to support context parallel
            torch.nn.functional.scaled_dot_product_attention = ensure_context_parallel_qkv(
                torch.nn.functional._original_scaled_dot_product_attention,
                mesh=device_mesh
            )
        else:
            torch.nn.functional.scaled_dot_product_attention = torch.nn.functional._original_scaled_dot_product_attention
        
        return inputs

    @classmethod
    def _output_fn(
        cls,
        module: nn.Module,
        outputs: Union[torch.Tensor, tuple[Union[torch.Tensor, int, float], ...]],
        device_mesh: DeviceMesh,
        use_local_output: bool = False,
        enable: bool = True,
    ) -> Union[
        Union[torch.Tensor, int, float], tuple[Union[torch.Tensor, int, float], ...]
    ]:
        if module in cls._CONTEXT_MANAGERS:
            cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
            del cls._CONTEXT_MANAGERS[module]
        
        # unpatch scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = (
            torch.nn.functional._original_scaled_dot_product_attention
        )

        def backward_hook(grad: torch.Tensor) -> None:
            if module not in cls._CONTEXT_MANAGERS and enable:
                manager = _enable_cp_dispatcher()
                manager.__enter__()
                cls._CONTEXT_MANAGERS[module] = manager
                # monkey path torch.nn.functional.scaled_dot_product_attention to support context parallel
                torch.nn.functional.scaled_dot_product_attention = ensure_context_parallel_qkv(
                    torch.nn.functional._original_scaled_dot_product_attention,
                    mesh=device_mesh
                )
            else:
                torch.nn.functional.scaled_dot_product_attention = (
                    torch.nn.functional._original_scaled_dot_product_attention
                )
            
        # back to local tensor
        out = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) and use_local_output else output

            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(backward_hook)

            out.append(output)

        if isinstance(outputs, torch.Tensor):
            return out[0]

        return tuple(out)

def ensure_context_parallel_qkv(fn: Callable, mesh: Optional[DeviceMesh] = None) -> Callable:
    """Decorator to wrap HF `sdpa_attention_forward`-like functions.

    Ensures Q, K, V are DTensors (sequence sharded) when the module has a
    `_cp_mesh` (attached by `AttentionContextParallel`). Falls back to
    original function behavior if context parallel not enabled.
    """

    @wraps(fn)
    def _wrapped(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: Optional[bool] = None,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:

        # mesh: Optional[DeviceMesh] = getattr(module, "_cp_mesh", mesh)
        if mesh is not None:
            # TODO: Figure out why the replicate placement is necessary here.
            def _ensure(x: torch.Tensor) -> torch.Tensor:
                if isinstance(x, DTensor):
                    x = x.redistribute(
                        mesh,
                        placements=[Replicate()]
                    )
                    return x
                return DTensor.from_local(
                    x.contiguous(), 
                    mesh,
                    placements=[Shard(2)],
                    run_check=False
                ).redistribute(
                    placements=[Replicate()]
                )
            query, key, value = _ensure(query), _ensure(key), _ensure(value)

        attn_output = fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
            **kwargs,
        )

        # we always return local tensor from attention
        if mesh is not None:
            if isinstance(attn_output, DTensor):
                attn_output = attn_output.redistribute(
                    mesh,
                    placements=[Shard(2)]
                )
                attn_output = attn_output.to_local()
        return attn_output

    return _wrapped
