from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("/data3/workhome/luyaxi/ARL/output/resume-sft",trust_remote_code=True)

ret = processor.tokenizer.encode("你好，我是一个AI助手。", add_special_tokens=True) + [processor.tokenizer.convert_tokens_to_ids('<|im_end|>')]
print(ret)
for i in ret:
    print(processor.tokenizer.decode(i, skip_special_tokens=False))