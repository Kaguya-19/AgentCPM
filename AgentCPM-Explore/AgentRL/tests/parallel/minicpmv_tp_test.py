from transformers import AutoModelForCausalLM, AutoProcessor,  AutoConfig
import os
import torch
from torch.distributed.tensor.parallel import ColwiseParallel,RowwiseParallel,parallelize_module,SequenceParallel,PrepareModuleInput,PrepareModuleOutput
from torch.distributed.tensor import  Replicate,Shard
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard
import torch.distributed.fsdp
import pdb
import os
from PIL import Image
from torch.distributed.tensor import DTensor
import sys
sys.path.append(".")
sys.path.append("./src")
from src.training.utils import _process_inputs,fsdp2_load_full_state_dict
from collections.abc import Mapping
from typing import Union, Any
from accelerate import init_empty_weights

def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": "cuda"}
        return data.to(**kwargs)
    return data


def main():
    # Ensure environment variables are set before initializing process group
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        raise EnvironmentError("WORLD_SIZE and RANK environment variables must be set for distributed training.")
    torch.cuda.set_device(torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )
    model_path = "/share_data/data1/models/MiniCPM-V-4_5-64"
    processing_class = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True
        )
    
    tp_size = 4
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(local_world_size//tp_size, 1,tp_size),
        mesh_dim_names=("dp", "pp", "tp"),
    )
    print("RANK ",dist.get_rank(), "MESH:", mesh)
    # if os.environ.get("RANK", "0") == "0":
    #     pdb.set_trace()
    # dist.barrier()
    
        
    parallel_dim = 1
    layer_tp_plan = {
        "llm.model.norm": SequenceParallel(),
        "llm.lm_head": ColwiseParallel(
            input_layouts=Shard(parallel_dim),
            # output_layouts=Replicate(),
            use_local_output=False
        ),
        # "llm.model.embed_tokens": 
        #     RowwiseParallel(
        #         input_layouts=Replicate(),
        #         output_layouts=Shard(parallel_dim),
        #         use_local_output=False
        # ),
        "llm.model.layers.*.mlp.up_proj": ColwiseParallel(),
        "llm.model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "llm.model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(parallel_dim),use_local_output=False),
        "llm.model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "llm.model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "llm.model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "llm.model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(parallel_dim),use_local_output=False),
        "llm.model.layers.*.input_layernorm": SequenceParallel(),
        "llm.model.layers.*.post_attention_layernorm": SequenceParallel(),
        "llm": PrepareModuleInput(
            input_kwarg_layouts={
                "inputs_embeds": Replicate(),
                "attention_mask": None,
                "position_ids": None,
                "use_cache": None,
            },
            desired_input_kwarg_layouts={
                "inputs_embeds": Shard(parallel_dim),
                "attention_mask": None,
                "position_ids": None,
                "use_cache": None,
            },
        ),
        # "vpm.encoder.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
        # "vpm.encoder.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
        # "vpm.encoder.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
        # "vpm.encoder.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(parallel_dim),use_local_output=False),
        # "vpm.encoder.layers.*.mlp.fc1": ColwiseParallel(use_local_output=False),
        # "vpm.encoder.layers.*.mlp.fc2": ColwiseParallel(use_local_output=False),
        # "vpm.post_layernorm": PrepareModuleOutput(
        #     output_layouts=Replicate(),
        #     use_local_output=True
        # )
        # "resampler.kv_proj": ColwiseParallel(output_layouts=Replicate(),use_local_output=True),
        # "resampler.ln_q": PrepareModuleOutput(
        #     output_layouts=Replicate(),
        #     desired_output_layouts=Replicate(),
        #     use_local_output=True,
        # ),
        # "resampler.ln_post": SequenceParallel(),
        # "resampler.attn.in_proj_weight": ColwiseParallel(),
        # # out projection of the attention: shard rows (output features)
        # "resampler.attn.out_proj": RowwiseParallel(output_layouts=Shard(parallel_dim), use_local_output=False),
        # # proj is a learnable matrix multiplied on the right: shard its columns
        # "resampler.proj": ColwiseParallel(),
        # # Ensure the resampler module returns a local Tensor (not a DTensor) with the same full shape
        # "resampler": PrepareModuleOutput(
        #     output_layouts=Shard(parallel_dim),
        #     desired_output_layouts=Replicate(),
        #     use_local_output=True,
        # ),
    }
    
    model_tp = parallelize_module(
        model,
        mesh["tp"],
        layer_tp_plan,
    )
    
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    import functools
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    apply_activation_checkpointing(
        model_tp,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                model_tp.llm.model.layers[0].self_attn.__class__,
                model_tp.llm.model.layers[0].mlp.__class__,
                model_tp.llm.model.layers[0].input_layernorm.__class__,
                model_tp.llm.model.layers[0].post_attention_layernorm.__class__,
                model_tp.vpm.encoder.layers[0].__class__},
        ),
    )
    
    
    for tf_block in model_tp.llm.model.layers:
        fully_shard(
            tf_block,
            mesh = mesh["dp"],
            mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )

    for tf_block in model_tp.vpm.encoder.layers:
        fully_shard(
            tf_block,
            mesh = mesh["dp"],
            mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )
    
    model_2d = fully_shard(
        model_tp,
        mesh=mesh["dp"],
        mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(reduce_dtype=torch.float32),
        reshard_after_forward=True,
    )
    
    if os.environ.get("RANK", "0") == "0":
        state_dict = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).state_dict()
    else:
        state_dict = None
    
    fsdp2_load_full_state_dict(model_2d,state_dict)

    optimizer = torch.optim.AdamW(model_2d.parameters(), lr=1e-5)
    torch.cuda.empty_cache()
    for _ in range(10):
        data = [{
            "prompt": [
                {"role":"user", "content": ["What is the capital?"]*2000+[Image.new(mode="RGB",size=(224,224))]*0},
            ],
            "completion": "The capital of France is Paris.",
            "advantage": 1.0,
            "reward": 1.0,
            
        }]
        data = _process_inputs(
            data,
            processing_class,
            max_prompt_length=4096000,
            pad_to_multiple_of=8
        )
        
        inputs = data["data"]
        print(inputs["input_ids"].shape, inputs["attention_mask"].shape)
        
        # if os.environ.get("RANK", "0") == "0":
        #     pdb.set_trace()
        # dist.barrier()
        inputs =_prepare_input(inputs)
        # out = model_tp(data=inputs, use_cache=False)
        out = model_2d(data=inputs,use_cache=False)
        
        print("Output shape:", out["logits"].shape)
        from torch.distributed.tensor.parallel import loss_parallel
        import torch.nn.functional as F
        advantages = torch.tensor(data["labels"]["advantages"], device=model_2d.device)
        prompt_len = data["labels"]["prompt_len"]
        completion_mask = data["labels"]["completion_mask"].to(device=model_2d.device)


        # with loss_parallel():
        logits = out["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        if isinstance(log_probs, DTensor):
            log_probs = log_probs.full_tensor()
        log_probs = log_probs[:, :-1,:]
        input_ids = inputs["input_ids"][:, 1:]
        per_token_logps = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        per_token_logps = per_token_logps[:, prompt_len -1 :]
        old_per_token_logps = per_token_logps.detach()
        
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - 1e-3, 1 + 1e-3)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = - torch.min(per_token_loss1, per_token_loss2)
        
        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        loss.backward()
        # torch.cuda.empty_cache()
        
        
        if os.environ.get("RANK", "0") == "0":
            pdb.set_trace()
        dist.barrier()
        
        if optimizer.state:
            for gid,param_group in enumerate(optimizer.param_groups):
                device_map = device_maps[gid]
                for param in param_group["params"]:
                    state = optimizer.state[param]
                    for k, v in state.items():
                        if isinstance(v, (torch.Tensor, DTensor)):
                            device = device_map[k]
                            state[k] = v.to(device=device, non_blocking=True)

        optimizer.step()
        
        
        optimizer.zero_grad()
        device_maps = []
        # offload optimizers onto cpu
        for param_group in optimizer.param_groups:
            device_map = {}
            for param in param_group["params"]:
                state = optimizer.state[param]
                for k, v in state.items():
                    if isinstance(v, (torch.Tensor, DTensor)):
                        device_map[k] = v.device
                        state[k] = v.to(device="cpu", non_blocking=True)
            device_maps.append(device_map)
        torch.cuda.empty_cache()
        
        
    if os.environ.get("RANK", "0") == "0":
        pdb.set_trace()
    dist.barrier()
    
if __name__ == "__main__":
    try:
        main()
    finally:
        dist.destroy_process_group()
    
'''
accelerate launch \
    --config_file assets/fsdp2_dst.yml \
    --num_processes=8 \
    --machine_rank=0 \
    --main_process_ip=localhost \
    --main_process_port=25600 \
    tests/minicpmv_tp_test.py

'''

