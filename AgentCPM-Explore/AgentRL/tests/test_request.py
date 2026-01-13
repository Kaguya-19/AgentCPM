import openai

client = openai.OpenAI(
    api_key="sk-123",
    base_url="http://localhost:30000/v1"
)

ret = client.chat.completions.create(
    messages=[
        {"role": "user","content": "Write a hello world program in random languages."}
    ],
    model="/share_data/data1/models/Qwen/Qwen3-8B",
    logprobs=True,
    top_logprobs=5,
)
print(ret.choices[0].message.content)
import pdb; pdb.set_trace()