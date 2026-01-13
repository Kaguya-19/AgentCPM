from .parse_qwen import parse_tool_for_qwen
from .parse_openai import parse_tool_for_openai


class ToolParser :
    def __init__(self,model_name:str,is_openai_format:bool = False):
        self.model_name = model_name
        self.is_openai_format = is_openai_format

    
    def parse_tool_from_response(self,completion):
        if self.is_openai_format:
            return parse_tool_for_openai(completion=completion)
        else:
            if "qwen" in self.model_name:
                return parse_tool_for_qwen(completion=completion) 