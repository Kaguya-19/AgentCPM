import torch
from torch.distributed.tensor.parallel import PrepareModuleInputOutput, ParallelStyle

from typing import Optional, Union
from torch.distributed.tensor import Replicate,Shard,DTensor, distribute_module, Placement
from torch.distributed.device_mesh import DeviceMesh


class NonParallel(PrepareModuleInputOutput):
    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        module = super()._apply(module, device_mesh)
        module = distribute_module(
            module,
            device_mesh,
        )
        return module

class DataParallel(ParallelStyle):
    def __init__(self,):
        super().__init__()
        
    def _prepare_input_fn(self, inputs: tuple|list, device_mesh: DeviceMesh):
        prepared_inputs = []
        if any(map(lambda inp: inp.size(0) % device_mesh.size() != 0, inputs)):
            return inputs
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                inp_local = DTensor.from_local(
                    inp,
                    device_mesh=device_mesh,
                    placements=(Replicate(),)
                )
                local_chunk_inp = inp_local.redistribute(
                    placements=(Shard(0),)
                ).to_local()
                prepared_inputs.append(local_chunk_inp)
        
            else:
                prepared_inputs.append(inp)
        return tuple(prepared_inputs)
    
    def _prepare_output_fn(self, outputs: tuple[torch.Tensor], inputs: tuple[torch.Tensor], device_mesh: DeviceMesh):
        prepared_outputs = []
        if any(map(lambda inp: inp.size(0) % device_mesh.size() != 0, inputs)):
            return outputs
        for out in zip(outputs):
            if isinstance(out, torch.Tensor):
                # assumpt the first dimension is batch dimension
                out_local = DTensor.from_local(
                    out,
                    device_mesh=device_mesh,
                    placements=(Shard(0),)
                )
                out = out_local.redistribute(
                    placements=(Replicate(),)
                ).to_local()
                prepared_outputs.append(out)
            else:
                prepared_outputs.append(out)
        return tuple(prepared_outputs)
    
    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        module.register_forward_pre_hook(
            lambda mod, inp: self._prepare_input_fn(inp, device_mesh)
        )
        module.register_forward_hook(
            lambda mod, inp, out: self._prepare_output_fn(out, inp, device_mesh)
        )
        return module