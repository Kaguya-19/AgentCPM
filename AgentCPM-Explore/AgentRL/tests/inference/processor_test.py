from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("/share_data/data1/models/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True)

# processor = AutoProcessor.from_pretrained("/share_data/data1/models/MiniCPM-V-4_5-64", trust_remote_code=True)

messages = [[
    {"role":"user","content":[
        {"type":"text","text":"hello!"},
        # {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAlUlEQVR4nO3QQREAIAzAMMC/501GHjQKer0zc372dIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAdoCVbYDfdVsQBUAAAAASUVORK5CYII="}}
        {"type":"image","image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAlUlEQVR4nO3QQREAIAzAMMC/501GHjQKer0zc372dIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAdoCVbYDfdVsQBUAAAAASUVORK5CYII="}
    ]},
    {"role":"assistant","content":"hi!","tool_calls":[{
        "id":"xxx",
        "index": -1,
        "type":"function",
        "function":{
            "name": "click",
            "arguments": '{"coordinate":[157,91]}'
        }
    }]}
]]*2
tools = [
    {"type":"function","function":{"name":"click","description":"Simulates a mouse click at the given coordinates.","parameters":{
        "type":"object",
        "properties":{
            "coordinate":{
                "type": "array",
                "items":{
                    "type":"integer"
                }
            }
        }
    }}}
]
import pdb;pdb.set_trace()

text = [processor.apply_chat_template(
    msg,
    tools=tools,
    return_dict=True,
    return_tensors="pt",
    tokenize=False,
) for msg in messages]
print(text)
images, videos = process_vision_info(messages, image_patch_size=16)
inputs = processor(text=text, images=images,
                return_tensors="pt",
                max_length=1024,
                truncation=True)

print(inputs)