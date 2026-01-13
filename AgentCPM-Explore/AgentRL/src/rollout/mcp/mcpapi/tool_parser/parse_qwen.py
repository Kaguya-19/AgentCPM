import re
import json5
import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def keep_after_think(response: str) -> str:
    """保留 </think> 之后的内容"""
    idx = response.find("</think>")
    if idx != -1:
        return response[idx + len("</think>") :]
    return response

def keep_after_last_think(response: str) -> str:
    """保留最后一个 </think> 之后的内容"""
    idx = response.rfind("</think>")
    if idx != -1:
        return response[idx + len("</think>") :]
    return response

def _extract_code_block(text: str) -> Optional[str]:
    """从文本中提取 <code> 标签内的内容"""
    match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def _clean_and_parse_json(json_candidate: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    尝试清理并解析 JSON 字符串。
    返回: (parsed_json_dict, error_message)
    """
    # 1. 尝试直接解析
    try:
        tool_call = json5.loads(json_candidate)
        return tool_call, None
    except Exception:
        pass

    # 2. 尝试通过括号计数截取有效 JSON
    json_start_idx = json_candidate.find('{')
    if json_start_idx == -1:
        return None, "No JSON start brace '{' found"
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(json_start_idx, len(json_candidate)):
        char = json_candidate[i]
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # 找到完整的 JSON 对象
                    try:
                        potential_json = json_candidate[json_start_idx:i+1]
                        return json5.loads(potential_json), None
                    except Exception as e:
                        return None, f"Brace matching found candidate but parse failed: {str(e)}"
    
    return None, "Failed to find matching closing brace"

def parse_tool_for_qwen(content: str) -> List[Dict[str, Any]]:
    response = keep_after_last_think(content)
    
    # 匹配 <tool_call> 块
    tool_call_regex = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = tool_call_regex.findall(response)
    
    raw_function_calls = []
    
    for match_content in matches:
        try:
            # 1. 检测代码块
            code_match = re.search(r'<code>(.*?)</code>', match_content, re.DOTALL)
            extracted_code = None
            json_candidate = match_content

            if code_match:
                extracted_code = code_match.group(1).strip()
                # 关键：截取 <code> 之前的部分作为 JSON 候选，避免 parse error
                json_candidate = match_content[:code_match.start()].strip()
            
            # 2. 解析 JSON
            parsed_json, error = _clean_and_parse_json(json_candidate)
            
            if not parsed_json:
                # 解析失败，创建一个 format error 标记的 tool_call
                error_message = f"Failed to parse JSON part, JSON parse error: {error}"
                logger.debug(f"Failed to parse JSON part: {match_content[:500] if len(match_content) > 500 else match_content}, JSON parse error: {error}")
                
                # 创建一个特殊的 tool_call 来标记 format error
                # function.name 为空会被 sampler.py 检测到并设置为 FORMATERROR
                format_error_tool_call = {
                    "id": f"call__{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": "",  # 空名称会被 sampler.py 检测为 format error
                        "arguments": json.dumps({
                            "_format_error": True,
                            "error_message": error_message
                        }, ensure_ascii=False)
                    }
                }
                raw_function_calls.append(format_error_tool_call)
                continue

            # 3. 处理代码注入 (针对 execute_code / PythonInterpreter)
            tool_name = parsed_json.get("name", "")
            if tool_name in ("execute_code", "PythonInterpreter"):
                # 确保 arguments 是字典
                if "arguments" not in parsed_json or not isinstance(parsed_json["arguments"], dict):
                    parsed_json["arguments"] = {}
                
                args = parsed_json["arguments"]
                existing_code = args.get("code")

                # 优先级逻辑：
                # 1. 如果 arguments 里有代码且非空，使用它
                # 2. 否则，使用从 <code> 标签提取的代码
                # 3. 再否则，尝试从原始 match 中再次提取 (fallback)
                if not existing_code or (isinstance(existing_code, str) and not existing_code.strip()):
                    if extracted_code:
                        args["code"] = extracted_code
                    else:
                        # Fallback: 再次尝试从原始文本提取，以防切分逻辑漏掉
                        fallback_code = _extract_code_block(match_content)
                        if fallback_code:
                            args["code"] = fallback_code

            # 4. 构造 OpenAI 格式的 Tool Call 对象
            # 注意：此处使用标准 json.dumps 序列化 arguments 字符串，确保下游兼容性
            if "name" in parsed_json and "arguments" in parsed_json:
                arguments_str = json.dumps(parsed_json["arguments"], ensure_ascii=False)
                tool_call = {
                    "id": f"call__{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": parsed_json["name"], 
                        "arguments": arguments_str
                    }
                }
                raw_function_calls.append(tool_call)
            
            elif "content" in parsed_json and isinstance(parsed_json["content"], dict):
                # 处理某些变体格式
                inner_content = parsed_json["content"]
                if "name" in inner_content and "arguments" in inner_content:
                    arguments_str = json.dumps(inner_content["arguments"], ensure_ascii=False)
                    tool_call = {
                        "id": f"call__{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": inner_content["name"], 
                            "arguments": arguments_str
                        }
                    }
                    raw_function_calls.append(tool_call)
                else:
                    raw_function_calls.append(parsed_json)
            else:
                raw_function_calls.append(parsed_json)

        except Exception as e:
            logger.error(f"Unexpected error parsing tool call: {e}", exc_info=True)
            raw_function_calls.append({"content": match_content})

    return raw_function_calls