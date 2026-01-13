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
from collections.abc import Mapping
from typing import Union, Any

sys.path.append("./src")
sys.path.append("../")
sys.path.append("./")


from src.training.utils import _process_inputs

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
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK",)))
    torch.cuda.set_device(device)    
    model_path = "/share_data/data1/models/Qwen/Qwen3-8B"
    processing_class = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    # config = AutoConfig.from_pretrained(
    #     model_path,
    #     attn_implementation='flash_attention_2',
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    # )
    # model: Qwen3ForCausalLM = AutoModelForCausalLM.from_config(
    #     config,
    # )
    
    
    model:  Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()
    tp_size = 4
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(local_world_size//tp_size, 1,tp_size),
        mesh_dim_names=("dp", "pp", "tp"),
    )
    
    # if os.environ.get("RANK", "0") == "0":
    #     pdb.set_trace()
    # dist.barrier()
    model.resize_token_embeddings(pad_to_multiple_of=tp_size,mean_resizing=False)
    
    parallel_dim = 1
    layer_tp_plan = {
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(parallel_dim),
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

    optimizer = torch.optim.AdamW(model_2d.parameters(), lr=1e-5, fused=True)
    dist.barrier()
    for _ in range(10):
        data = [
            {
                "prompt": [
                    {"role":"system","content":r"Answer the MATH problem, give your final answers inside the \boxed{}."},
                    {"role":"user", "content": ["What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?"]},
                ],
                "completion": """<think>
Okay, so I need to find the smallest positive perfect cube that can be written as the sum of three consecutive integers. Let me think about how to approach this.

First, let me recall what a perfect cube is. A perfect cube is a number that can be expressed as $ n^3 $ where $ n $ is an integer. So, examples are 1, 8, 27, 64, 125, etc. The problem is asking for the smallest such number that is also the sum of three consecutive integers.

Alright, so I need to find three consecutive integers whose sum is a perfect cube. Let me denote those three consecutive integers. Let me think: if I take three consecutive integers, they can be represented as $ x-1 $, $ x $, and $ x+1 $, where $ x $ is the middle number. Alternatively, they could also be represented as $ n $, $ n+1 $, $ n+2 $, where $ n $ is the first integer. Either way should work. Let me pick one representation and see which is easier.

Let me go with $ n $, $ n+1 $, $ n+2 $. The sum of these three numbers would be $ n + (n+1) + (n+2) $. Let me compute that:

$ n + (n+1) + (n+2) = 3n + 3 $. So, that simplifies to $ 3(n + 1) $. Therefore, the sum of three consecutive integers is always a multiple of 3. Therefore, if this sum is a perfect cube, then that perfect cube must be divisible by 3. So the cube must be divisible by 3, which implies that the cube root must be divisible by 3 as well. Because if a prime divides a cube, it must divide the base. Since 3 is prime, then if 3 divides $ n^3 $, 3 divides $ n $.

Therefore, the perfect cube must be $ (3k)^3 $ for some positive integer $ k $. Let me check that logic. If the sum is a perfect cube, say $ m^3 $, and since the sum is equal to $ 3(n + 1) $, then $ m^3 = 3(n + 1) $. Therefore, 3 divides $ m^3 $, so 3 divides $ m $, hence $ m = 3k $, so $ m^3 = 27k^3 $. Therefore, the perfect cube must be divisible by 27. Therefore, the smallest possible such cube would be when $ k = 1 $, which is $ 27 \times 1^3 = 27 $. But wait, is 27 actually achievable as the sum of three consecutive integers?

Let me check. If the sum is 27, then $ 3(n + 1) = 27 $, so $ n + 1 = 9 $, so $ n = 8 $. Therefore, the three consecutive integers would be 8, 9, 10. Let me add them up: 8 + 9 + 10 = 27. Yes! That works. Wait, so 27 is already a perfect cube and can be written as the sum of three consecutive integers. Then why is this problem asking for the smallest positive perfect cube? Is there a smaller one?

Wait, hold on. Let me check the smaller perfect cubes. The perfect cubes are 1, 8, 27, 64, etc. Let me check if 1 or 8 can be written as the sum of three consecutive integers.

Starting with 1. Let's suppose that 1 is the sum of three consecutive integers. Let me use the formula again: sum = 3(n + 1). So, 3(n + 1) = 1 => n + 1 = 1/3 => n = -2/3. Which is not an integer. So, no solution here.

How about 8? Let me do the same. 3(n + 1) = 8 => n + 1 = 8/3 => n = 5/3. Again, not an integer. Therefore, 8 cannot be expressed as the sum of three consecutive integers. Therefore, the next one is 27, which works. Therefore, the answer is 27?

But wait, the problem says "smallest positive perfect cube". But maybe I made a mistake here? Let me verify again.

Alternatively, maybe there's another way to represent three consecutive integers? Wait, I assumed that the three consecutive integers are $ n $, $ n + 1 $, $ n + 2 $, but maybe there's another way? For example, sometimes people consider negative integers as well. Let me check if there are any negative integers that could lead to a smaller positive cube? Wait, but the question specifies "positive perfect cube", so even if the sum is positive, the integers themselves can be negative. But since we're looking for the smallest positive perfect cube, which is 1, 8, 27... So maybe even though there are negative integers involved, the cube itself has to be positive. So 1 and 8 don't work, but 27 does. So maybe 27 is the answer.

But let me just check if there's another way. Maybe my initial assumption that the sum is 3(n + 1) is correct. Let me verify with another approach. Suppose the three consecutive integers are $ x-1 $, $ x $, and $ x+1 $. Then their sum is $ (x - 1) + x + (x + 1) = 3x $. So, the sum is 3x. Therefore, regardless of how you represent the three consecutive integers, the sum is three times the middle number. Therefore, if the sum is a perfect cube, it must be 3 times some integer, which is the middle number. Therefore, 3x is a perfect cube. Therefore, x must be a multiple of 3? Wait, similar to before. Let me check. If 3x is a perfect cube, then 3 divides that cube, so 3 divides x? No. Wait, if 3 divides the cube, then 3 divides x. Let me think again. Let me write it as:

Let $ 3x = m^3 $, so $ x = \frac{m^3}{3} $. Therefore, for x to be integer, m^3 must be divisible by 3, hence m must be divisible by 3, so let m = 3k. Then, x = $ \frac{(3k)^3}{3} = 9k^3 $. Therefore, the sum is $ 3x = 3*(9k^3) = 27k^3 $. Therefore, the sum is 27k^3, which is a perfect cube. Therefore, the smallest such sum is when k = 1, which gives 27. Therefore, that seems to confirm the earlier result.

So, the answer is 27. But wait, the problem says "three consecutive integers". But in this case, the integers would be x - 1, x, x + 1. If x = 9k^3, then the integers are 9k^3 - 1, 9k^3, 9k^3 + 1. For k = 1, that's 8, 9, 10, which adds up to 27. So yes, that works. Therefore, 27 is the smallest positive perfect cube which can be expressed as the sum of three consecutive integers.

But let me check if there is another way to have a smaller cube. For example, if there's a cube that isn't divisible by 3, but somehow still can be expressed as the sum of three consecutive integers. However, from our previous reasoning, since the sum is always a multiple of 3, the cube must be a multiple of 3, so it must be divisible by 3. Therefore, the cube must be divisible by 3, so the cube root must be divisible by 3, leading to the cube being 27, 216, etc. Hence, the smallest is 27.

But maybe there is a mistake here? Let me check with another approach. Suppose someone thinks that maybe three consecutive integers could be something else? Wait, for example, if the numbers are not centered around an integer. Wait, but three consecutive integers will always have a middle number. For example, if you have three consecutive integers, they can be expressed as $ a $, $ a + 1 $, $ a + 2 $, which sum to $ 3a + 3 $, as before. So, the sum is always three times an integer plus three, which is three times (a + 1). So that's the same as 3 times the middle number. Therefore, regardless of how you choose the three consecutive integers, their sum is three times an integer. Therefore, if that sum is a perfect cube, then that cube must be divisible by 3, and hence must be 27 times some cube. Therefore, the smallest possible is 27.

Therefore, I think that 27 is the correct answer. However, maybe I'm missing something? Let me check with the actual numbers. Let me check for example if there's a smaller cube.

For example, take cube 1: sum of three integers is 1. As we saw before, impossible because 1 divided by 3 is not integer.

Cube 8: 8 divided by 3 is not integer. So, no solution.

Cube 27: 27 divided by 3 is 9. So, the middle number is 9, so numbers are 8,9,10. Sum is 27. Correct.

Therefore, yes, seems like 27 is the answer.

But wait, hold on. Let me check some other possibilities. Suppose the three consecutive integers are negative. For instance, maybe a negative number and two positive numbers? For example, could there be three consecutive integers, including negatives, that add up to a smaller cube? For example, maybe -1, 0, 1. Their sum is 0. But 0 is a perfect cube (0³), but the problem says "positive perfect cube", so 0 is excluded. Similarly, if you take -2, -1, 0: sum is -3. Not positive. How about -1, 0, 1: sum is 0. Still not positive. How about 0, 1, 2: sum is 3. Which is not a perfect cube. 1, 2, 3: sum is 6. Not a cube. 2, 3, 4: sum is 9. Not a cube. 3, 4, 5: 12. Not cube. 4,5,6: 15. 5,6,7: 18. 6,7,8: 21. 7,8,9: 24. 8,9,10: 27. Which is the first one that works. So yeah, seems like 27 is indeed the first positive cube that can be expressed as sum of three consecutive integers.

But wait, but what if the three consecutive integers are not starting from a positive integer? For example, maybe negative numbers? Let's see. For example, let's take -3, -2, -1. Their sum is -6. Not positive. -2, -1, 0: sum is -3. Still negative. -1, 0, 1: 0. 0,1,2: 3. 1,2,3: 6. 2,3,4: 9. 3,4,5: 12. 4,5,6: 15. 5,6,7: 18. 6,7,8: 21. 7,8,9: 24. 8,9,10: 27. So yeah, same as before. There is no positive cube between 1 and 27 that can be expressed as the sum. So 27 is the first one.

But let me check whether there are other representations. For example, suppose the three consecutive integers are not in the form n, n+1, n+2. Wait, but they have to be consecutive integers, so they must differ by 1. So the only way to represent three consecutive integers is as such. So, no alternative representations. Therefore, I think the conclusion holds.

Therefore, the answer is 27. So I think that's it. But let me check once more. Let me check the sum of three consecutive integers equals the cube. So, if we take numbers 8,9,10, their sum is 27, which is 3³. Correct. And since there are no smaller cubes that satisfy the condition, then 27 is indeed the answer.

Wait, but maybe there's a different set of three consecutive integers that sum to a smaller cube? For example, maybe not starting from n? Wait, but how else? If you take three consecutive integers, they must be in an arithmetic progression with difference 1. So, the sum is fixed as three times the middle term. Therefore, if we can write the sum as a cube, then the cube must be thrice the middle term. Therefore, unless there's some other way where the sum is a cube, but not necessarily thrice an integer? But no, because the sum is always thrice an integer. Therefore, the cube must be divisible by 3, hence the cube is 27k³. Therefore, the minimal is 27. Therefore, 27 is the answer.

Alternatively, maybe if we consider that the three consecutive integers are not integers? But the problem says "three consecutive integers", so they have to be integers. So, no. So, conclusion is 27. Therefore, I think that's the answer.

**Final Answer**
The smallest positive perfect cube that can be written as the sum of three consecutive integers is \boxed{27}.
</think>

To solve the problem of finding the **smallest positive perfect cube** that can be written as the **sum of three consecutive integers**, we begin by analyzing the structure of three consecutive integers and the properties of perfect cubes.

---

### Step 1: Understanding the Sum of Three Consecutive Integers

Let the three consecutive integers be:
$$
n, \quad n+1, \quad n+2
$$
Their sum is:
$$
n + (n+1) + (n+2) = 3n + 3 = 3(n + 1)
$$
This shows that the **sum of any three consecutive integers is always divisible by 3**.

---

### Step 2: Condition for Being a Perfect Cube

We want the **sum** (i.e., $ 3(n + 1) $) to be a **perfect cube**. Let the sum be:
$$
m^3 = 3(n + 1)
$$
This implies that $ m^3 $ must be divisible by 3, so $ m $ must be divisible by 3. Let:
$$
m = 3k
$$
Then:
$$
m^3 = (3k)^3 = 27k^3
$$
Thus, the smallest value occurs when $ k = 1 $, giving:
$$
m^3 = 27
$$

---

### Step 3: Verifying That 27 Can Be Expressed as a Sum of Three Consecutive Integers

Let’s check if 27 can be written as the sum of three consecutive integers. From the earlier formula:
$$
3(n + 1) = 27 \Rightarrow n + 1 = 9 \Rightarrow n = 8
$$
So the integers are:
$$
8,\ 9,\ 10
$$
Their sum is:
$$
8 + 9 + 10 = 27
$$
And $ 27 = 3^3 $, which confirms it is a **perfect cube**.

---

### Step 4: Checking for Smaller Candidates

We now check the smaller perfect cubes to ensure no smaller positive cube can be written as such a sum:

- $ 1^3 = 1 $: Solving $ 3(n + 1) = 1 $ gives $ n = -\tfrac{2}{3} $, not an integer.
- $ 2^3 = 8 $: $ 3(n + 1) = 8 $ → $ n = \tfrac{5}{3} $, again not an integer.
  
Thus, **no smaller positive perfect cube** satisfies the condition.

---

### Final Conclusion

The smallest positive perfect cube that can be written as the sum of three consecutive integers is:
$$
\boxed{27}
$$""",
                "advantage": 1.0,
                "reward": 1.0,
            },
        ]
        data = _process_inputs(
            data,
            processing_class,
            max_prompt_length=4096000,
            pad_to_multiple_of=8
        )
        # inputs = data["prompt_inputs"]
        labels = data.pop("labels")
        inputs =_prepare_input(data)
        if rank := int(os.environ.get("RANK", "0")) == 0:
            print(data)
            print(inputs["input_ids"].shape, inputs["attention_mask"].shape)
            print(inputs["attention_mask"])
        
        # if os.environ.get("RANK", "0") == "0":
        #     pdb.set_trace()
        # dist.barrier()
        # out = model_tp(data=inputs, use_cache=False)
        out = model_2d(**inputs)
        
        # print("Output shape:", out["logits"].shape)
        print("logits:", out["logits"])
        from torch.distributed.tensor.parallel import loss_parallel
        import torch.nn.functional as F
        advantages = torch.tensor(labels["advantages"], device=model_2d.device)
        prompt_len = labels["prompt_len"]
        completion_mask = labels["completion_mask"].to(device=model_2d.device)
        
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
        per_token_loss = per_token_loss[:, :completion_mask.size(1)]
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        loss.backward()
        
        if os.environ.get("RANK", "0") == "1":
            pdb.set_trace()
        dist.barrier()
        
                
        from src.patches import norm
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model_2d.parameters(), 1.0)
        print(f"Rank {os.environ.get('RANK', '0')} Loss: {loss.item()}, Grad Norm: {grad_norm}")
        
        torch.cuda.empty_cache()
        
        

        
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
    tests/tp_test.py

'''

