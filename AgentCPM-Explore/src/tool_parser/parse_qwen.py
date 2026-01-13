from typing import Any, Dict, Optional
import re
import json


def parse_tool_for_qwen(completion: dict) -> Optional[Dict[str, Any]]:
    response = completion.get("response", "")
    try:
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        function_call_tuples = tool_call_regex.findall(response)
        function_call_tuples = [t.replace("\\", "\\\\") for t in function_call_tuples]

        # load the JSON, and then use it to build the Function and
        # Tool Call
        raw_function_calls = [
            json.loads(match[0] if match[0] else match[1])
            for match in function_call_tuples
        ]
        return raw_function_calls
    except Exception as e:
        return [] 