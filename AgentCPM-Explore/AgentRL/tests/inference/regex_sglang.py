import openai

client = openai.Client(
    base_url="http://localhost:30000/v1",
    api_key="test"
)

res = client.chat.completions.create(
    model="/share_data/data1/models/Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Hi"}
    ],
    max_tokens=256,
    response_format={
        "type": "structural_tag",
        "structures":[
            {
                "begin":"</think>",
                "schema":{},
                "end":"This is mocked.",
            }
        ],
        "triggers":["</think>"],
    },
    extra_body={
        "chat_template_kwargs":{"enable_thinking": True}
    },
    # stop=["</think>"],
    logprobs=True
)

import pdb; pdb.set_trace()