from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_path = "/share_data/data1/models/Qwen/Qwen3-8B/"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    
    # padd inputs to length 128
    
    
    inputs["input_ids"] = torch.nn.functional.pad(inputs["input_ids"], (0, 128 - inputs["input_ids"].shape[1]), value=tokenizer.pad_token_id)
    inputs["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, 128 - inputs["attention_mask"].shape[1]), value=0)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()