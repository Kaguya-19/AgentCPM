from transformers import AutoModelForCausalLM, AutoProcessor,  AutoConfig
from transformers.models.qwen3 import Qwen3ForCausalLM
import os
import torch
from torch.distributed.tensor.parallel import ColwiseParallel,RowwiseParallel,parallelize_module,SequenceParallel,PrepareModuleInput,PrepareModuleOutput
from torch.distributed.tensor import  Replicate,Shard
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard
import torch.distributed.fsdp
import pdb
import sys
import copy
import os
import base64
from PIL import Image
from torch.distributed.tensor import DTensor
import pdb

def process_mm_messages(
    messages:list[dict[str,str|list[dict]]],
    tokenizer,
) -> tuple[list[dict[str,str]], list[Image.Image]]:
    messages = copy.deepcopy(messages)
    images = []

    for i, msg in enumerate(messages):
        role, content = msg["role"], msg["content"]
        
        if isinstance(content,str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
            elif isinstance(c, dict):
                assert "type" in c, "Dict content must have a 'type' key."
                match c["type"]:
                    case "image_url":
                        images.append(Image.open(base64.b64decode(c["image_url"].split(",")[1])))
                        cur_msgs.append("(<image>./</image>)")
                    case "text":
                        cur_msgs.append(c["text"])
                    case _:
                        raise ValueError(f"Unsupported content type: {c['type']}")
            else:
                raise ValueError(f"Unsupported content type: {type(c)}")
                    
        msg['content'] = "\n".join(cur_msgs)
    
    return tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True), images

def _prepare_messages(
    prompts,
    processing_class,
    max_prompt_length
):
    prompts_lists = []
    input_images_lists = []

    for msgs in prompts:
        prompt,images = process_mm_messages(msgs, processing_class.tokenizer)
        prompts_lists.append(prompt)
        input_images_lists.append(images)
    try:
        ret = processing_class(
            prompts_lists,
            input_images_lists,
            return_tensors="pt",
            max_length=max_prompt_length
        )
    except Exception as e:
        if sum(map(len, input_images_lists)) == 0:
            ret = processing_class(
                prompts_lists,
                return_tensors="pt",
                max_length=max_prompt_length
            )
        else:
            raise e
    
    
    return {
        **ret
    }

def _create_inputs(
    processing_class,
    prompt_inputs,
    completions,
    old_per_token_logps=None,
    pad_to_multiple_of: int = 0 
):
    # now handle completion_ids and completion_mask
    pad_token_id = getattr(processing_class,"pad_token_id", getattr(processing_class.tokenizer,"pad_token_id",None))
    if pad_token_id is None:
        pad_token_id = 0
    completion_ids = torch.full((len(prompt_inputs["input_ids"]),max(map(len,completions))), pad_token_id , dtype=prompt_inputs["input_ids"].dtype,device=prompt_inputs["input_ids"].device)
    for idx,completion in enumerate(completions):
        completion_ids[idx,:len(completion)] = completion

    if old_per_token_logps is not None:
        old_per_token_ids_logps = torch.full((len(prompt_inputs["input_ids"]),max(map(len,completions))), float("-inf"), dtype=torch.float,device=prompt_inputs["input_ids"].device) # B,L
        for idx,old_logps in enumerate(old_per_token_logps):
            old_per_token_ids_logps[idx,:len(old_logps)] = old_logps
        old_per_token_logps = old_per_token_ids_logps

    # Mask everything after the first EOS token
    im_eos = completion_ids == processing_class.tokenizer.convert_tokens_to_ids('<|im_end|>')
    s_eos = completion_ids == processing_class.tokenizer.convert_tokens_to_ids('</s>')
    is_eos = im_eos | s_eos
    
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long,device=completion_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1).to(device=eos_idx.device)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    
    prompt_inputs["input_ids"] = torch.cat([prompt_inputs["input_ids"],completion_ids],dim=-1).to(dtype=torch.int64)
    prompt_inputs["attention_mask"] = torch.cat([prompt_inputs["attention_mask"], completion_mask], dim=1)  # (B, P+C)
    # Pad to multiple of pad_to_multiple_of
    if pad_to_multiple_of > 0:
        pad_length = (pad_to_multiple_of - prompt_inputs["input_ids"].size(1) % pad_to_multiple_of) % pad_to_multiple_of
        if pad_length > 0:
            padding = torch.full(
                (prompt_inputs["input_ids"].size(0), pad_length),
                pad_token_id,
                dtype=prompt_inputs["input_ids"].dtype,
                device=prompt_inputs["input_ids"].device
            )
            prompt_inputs["input_ids"] = torch.cat(
                [prompt_inputs["input_ids"],padding], dim=-1
            )
            prompt_inputs["attention_mask"] = torch.cat(
                [
                    prompt_inputs["attention_mask"],
                    torch.ones(
                        (prompt_inputs["attention_mask"].size(0), pad_length),
                        dtype=prompt_inputs["attention_mask"].dtype,
                        device=prompt_inputs["attention_mask"].device,
                    ),
                ],
                dim=-1,
            )
            completion_mask = torch.cat(
                [
                    completion_mask,
                    torch.zeros(
                        (completion_mask.size(0), pad_length),
                        dtype=completion_mask.dtype,
                        device=completion_mask.device,
                    ),
                ],
                dim=-1,
            )
            
    return prompt_inputs,completion_mask,old_per_token_logps

def _process_inputs(
    inputs, 
    processing_class,
    max_prompt_length,
    pad_to_multiple_of: int = 0
):
    prompts = []
    completions = []
    advantages = []
    rewards = []
    step_ids = []
    old_per_token_logps = []
    
    if not hasattr(processing_class, "tokenizer"):
        setattr(processing_class, "tokenizer", processing_class)
    
    for inp in inputs:
        prompts.append(inp["prompt"])
        if "completion_ids" in inp:
            completions.append(inp["completion_ids"])
        else:
            assert "completion" in inp, "Either 'completion_ids' or 'completion' must be present in the input."
            if isinstance(inp["completion"], str):
                completion = processing_class.tokenizer.encode(inp["completion"], add_special_tokens=True)
                
                completions.append(torch.tensor(completion + [processing_class.tokenizer.convert_tokens_to_ids('<|im_end|>')]))
            else:
                assert isinstance(inp["completion"], list), "Completion must be a string or a list of integers/str."
                if isinstance(inp["completion"][0], str):
                    completions.append(torch.tensor([
                        processing_class.tokenizer.convert_tokens_to_ids(token)
                        for token in inp["completion"]
                    ]))
                elif isinstance(inp["completion"][0], int):
                    completions.append(torch.tensor(inp["completion"]))
                else:
                    raise ValueError("Completion must be a string or a list of integers or str.")
        advantages.append(inp["advantage"])
        rewards.append(inp["reward"])
        step_ids.append(inp.get("step_id",0))
        if "old_per_token_logps" in inp:
            assert len(inp["old_per_token_logps"]) == len(completions[-1]), "Length of 'old_per_token_logps' must match the length of 'completion_ids'. Got {} and {}.\nold_per_token_logps: {}\ncompletion:{}".format(len(inp["old_per_token_logps"]) ,len(completions[-1]), old_per_token_logps, completions[-1])
            old_per_token_logps.append(inp.get("old_per_token_logps"))
    if len(prompts) == 0:
        return None

    prompt_inputs = _prepare_messages(prompts,processing_class,max_prompt_length)
    prompt_len = prompt_inputs["input_ids"].size(1)
    
    ret = {
        "advantages": torch.tensor(advantages),
        "rewards": torch.tensor(rewards),
        "prompt_len": prompt_len,
        "step_ids": torch.tensor(step_ids)
    }
    
    
    prompt_inputs,completion_mask,old_per_token_logps = _create_inputs(processing_class,prompt_inputs,completions,old_per_token_logps,pad_to_multiple_of=pad_to_multiple_of)
    
    ret["prompt_inputs"] = prompt_inputs
    ret["completion_mask"] = completion_mask
    if old_per_token_logps is not None:
        ret["old_per_token_logps"] = old_per_token_logps
    
    return ret

from collections.abc import Mapping
from typing import Union, Any

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
    model_path = "/share_data/data1/models/Qwen/Qwen3-8B"
    processing_class = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    from accelerate import init_empty_weights
    with init_empty_weights():
        model: Qwen3ForCausalLM = AutoModelForCausalLM.from_config(
            config,
        )
        
    tp_size = 2
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(local_world_size//tp_size, 1,tp_size),
        mesh_dim_names=("dp", "pp", "tp"),
    )
    
    model.resize_token_embeddings(pad_to_multiple_of=tp_size,mean_resizing=False)
    
    parallel_dim = 1
    layer_tp_plan = {
            "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(parallel_dim),
                # output_layouts=Replicate(),
                use_local_output=False
            ),
            "model.embed_tokens": 
                RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(parallel_dim),
                    use_local_output=False
            ),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(parallel_dim),use_local_output=False),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(parallel_dim),use_local_output=False),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
            # "llm.model.layers.0": PrepareModuleInput(
            #     input_layouts=(Replicate(),),
            #     desired_input_layouts=(Shard(1),)
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
            transformer_layer_cls={model_tp.model.layers[0].__class__},
        ),
    )
    
    
    for tf_block in model_tp.model.layers:
        fully_shard(
            tf_block,
            mesh = mesh["dp"],
            mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )
    
    model_2d = fully_shard(
        model_tp,
        mesh=mesh["dp"],
        mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32),
        reshard_after_forward=True,
    )
        
        
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    
    if os.environ.get("RANK", "0") == "0":
        # load model weights
        tmp_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    dist.barrier()
    
    
    for key in model_2d.state_dict().keys():
        to_init_tensor = model_2d.state_dict()[key]
    
        if os.environ.get("RANK", "0") == "0":
            tensor = tmp_model.state_dict()[key].to(device=device,dtype=to_init_tensor.dtype)
        else:
            tensor = torch.zeros(to_init_tensor.shape, dtype=to_init_tensor.dtype, device=device)


        from torch.distributed.tensor import distribute_tensor
        dtensor = distribute_tensor(
            tensor,
            device_mesh=to_init_tensor.device_mesh,
            placements=to_init_tensor.placements,
        )
        # load the tensor
        model_2d.load_state_dict({key: dtensor},strict=False,assign=True)
        dist.barrier()
        del tensor
    
    # check model weight on all ranks
    # loading models onto cpu
    from transformers import Qwen3PreTrainedModel
    gt_sd = AutoModelForCausalLM.from_pretrained(config.name_or_path,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).state_dict()
    for name, dtensor in model_2d.state_dict().items():
        full_tensor = dtensor.full_tensor()
        tensor = gt_sd[name].to(device)
        assert torch.all(full_tensor == tensor), "Full tensor does not match the original model's state_dict for key: {}".format(key)
    
    torch.cuda.empty_cache()
    if os.environ.get("RANK", "0") == "1":
        pdb.set_trace()
    dist.barrier()

    optimizer = torch.optim.Adam(model_2d.parameters(), lr=1e-5, foreach=True)

    for _ in range(10):
        data = [{
            "prompt": [
                {"role":"user", "content": ["What is the capital of"*1000]},
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
        
        inputs = data["prompt_inputs"]
        print(inputs["input_ids"].shape, inputs["attention_mask"].shape)
        
        # if os.environ.get("RANK", "0") == "0":
        #     pdb.set_trace()
        # dist.barrier()
        inputs =_prepare_input(inputs)
        # out = model_tp(data=inputs, use_cache=False)
        out = model_2d(**inputs,use_cache=False)
        
        print("Output shape:", out["logits"].shape)
        from torch.distributed.tensor.parallel import loss_parallel
        import torch.nn.functional as F
        advantages = torch.tensor(data["advantages"], device=model_2d.device)
        prompt_len = data["prompt_len"]
        completion_mask = data["completion_mask"].to(device=model_2d.device)
        
        
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
        torch.cuda.empty_cache()
        
        
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
    --num_processes=4 \
    --machine_rank=0 \
    --main_process_ip=localhost \
    --main_process_port=25600 \
    tests/efficient_loading.py

'''

