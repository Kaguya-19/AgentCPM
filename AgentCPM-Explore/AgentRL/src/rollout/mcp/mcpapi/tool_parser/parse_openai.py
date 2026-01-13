def to_dict(obj):
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    # 普通对象
    if hasattr(obj, "__dict__"):
        return vars(obj)
    # 已经是 dict
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot convert {type(obj)} to dict")

def parse_tool_for_openai(completion):
    if isinstance(completion, dict):
        tool_calls = completion.get("tool_calls", [])
    else:
        tool_calls = getattr(completion, "tool_calls", [])
    # 防御：确保 tool_calls 是列表
    if tool_calls is None:
        tool_calls = []
    # 统一转为 dict
    return [to_dict(tc) for tc in tool_calls] 