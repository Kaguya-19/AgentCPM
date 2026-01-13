def parse_tool_for_openai(completion:dict):
    return completion.get("tool_calls", []) 