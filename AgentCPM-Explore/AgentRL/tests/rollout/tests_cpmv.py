import beanie
import sys
sys.path.append("./")
sys.path.append("./src")

from src.databases import DATAMODELS_TO_INIT, DispatchedSamplingTask
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
import requests
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


async def main():
    await beanie.init_beanie(
        connection_string="mongodb://admin:2025AgentRL@172.16.1.37:27021/gui_trajectory_qwen_scale_3?authSource=admin",
        document_models=DATAMODELS_TO_INIT,
    )

    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    model = AutoModel.from_pretrained("/nfsdata/models/MiniCPM-V-4_5", trust_remote_code=True,attn_implementation="flash_attention_2",dtype=torch.bfloat16).cuda()
    # model.eval()
    tokenizer = AutoTokenizer.from_pretrained("/nfsdata/models/MiniCPM-V-4_5", trust_remote_code=True,)
    processor = AutoProcessor.from_pretrained("/nfsdata/models/MiniCPM-V-4_5", trust_remote_code=True,)

    for sample in await DispatchedSamplingTask.find_many({"request.messages":{"$size": 10}}).limit(1000).to_list():
        # sample = await DispatchedSamplingTask.get("69428bd682538381dccff6dd")
        messages = sample.request["messages"]
        print(sample.id,messages)
        for msg in messages:
            content = []
            for c in msg["content"]:
                if c["type"] == "text":
                    content.append(c["text"])
                elif c["type"] == "image_url":
                    # download image
                    image_url = c["image_url"]["url"]
                    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
                    content.append(image)
                else:
                    raise NotImplementedError(f"Unknown content type: {c['type']}")
            msg["content"] = content

        # import pdb; pdb.set_trace()
        res = model.chat(msgs = messages,tokenizer=tokenizer)
        print(res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())