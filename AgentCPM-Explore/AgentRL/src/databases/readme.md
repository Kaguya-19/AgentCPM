# 上传 SFT 文件


AgentRL 将 SFT 数据上传至 mongoDB 上，通过异步采样数据库进行训练。完整的上传代码可见 `upload_sft_data.py`.

## 快速开始
在 `upload_sft_data.py` 中，补全 `math_config` 中的 `mongo_connection_url`, 并调整其他必要参数，直接运行即可。

## 使用说明：

**mongoDB 和 MinIO 的配置问题，请询问相应管理人员。**

对于你自己的数据，你需要继承基类，并实现自己的 `req_and_resp` 函数。该函数需要返回 `List[Tuple[Dict, Dict]]` 对于每个元组，第一个字典形式如下：
```python
{
    "messages": [
        {
            "role":"system",
            "content":""
        },
        {
            "role":"user",
            "content":""
        }
        ...
    ]
    "model": self.config.model_name
    "step": (int) # 仅对多轮数据
}
```

对于第二个元素，形式如下：
```python
{
    "role": "assistant",
    "content": ""
}
```


如果需要进一步控制数据的内容(如 prompt 长度等)，可以自行继承 `validate_check()` 函数，在基类中默认返回 `True`.
多轮数据中，只要有一个数据不满足条件，则一整个轨迹都不会被上传，但其中的图片仍会上传。


对于单轮数据，可以参考上传代码中的 `MathUploader`.