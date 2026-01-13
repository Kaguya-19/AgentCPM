from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
processor = AutoProcessor.from_pretrained("/nfsdata/models/MiniCPM-V-4_5", trust_remote_code=True)

data = [
    [
        {"role": "user", "content": [
            {"type": "text", "text": "Hello, how are you?"},
            {"type": "image", "image": "file:///nfsdata/luyaxi/AgentRL/test.png"},
        ]}
    ],
    [
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]
]

from qwen_vl_utils import process_vision_info
texts, images, videos = [], [], []
for msgs in data:
    texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    imgs, vids = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
    images.append(imgs)   # 允许 None
    videos.append(vids)   # 允许 None

inputs = processor(text=texts, images=images, videos=videos, return_tensors="pt", padding=True)
print(inputs)
import pdb; pdb.set_trace()

def recursive_to(device, data):
    if isinstance(data, dict):
        return {k: recursive_to(device, v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to(device, v) for v in data]
    elif hasattr(data, "to"):
        return data.to(device)
    else:
        return data
inputs = recursive_to("cuda", inputs)

model = AutoModelForImageTextToText.from_pretrained("/nfsdata/models/MiniCPM-V-4_5", trust_remote_code=True).cuda()
outputs = model(**inputs)
