#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Context Processor

This module is responsible for using the "secondary model" (Processor LLM) to
compress and summarize the main model's conversation history, generating Procedure objects.
"""

import logging
import json5 as json
import re
from typing import Any, List, Dict, Optional

from context.models import Procedure
from context.prompts import PROCEDURAL_MEMORY_SYSTEM_PROMPT, CONTEXT_COMPRESSION_SYSTEM_PROMPT, PROCEDURAL_MEMORY_SYSTEM_PROMPT_V4
from context import context_compression_tool_description  # import tool description


logger = logging.getLogger("context_processor")




def generate_procedure_summary(
    processor_llm_client: Any, 
    history_to_summarize: List[Dict]
) -> Optional[Procedure]:
    """
    Use "processor" LLM (secondary model) to generate history summary (Procedure).
    Adopt tool-based approach to obtain structured Procedure objects through tool_calls.

    Args:
        processor_llm_client: Initialized LLM client (e.g., ExtendedOpenAIClient).
        history_to_summarize: Raw historical message list exported from HistoryX with IDs included.

    Returns:
        A Procedure object (if successful), otherwise returns None.
    """
    if not hasattr(processor_llm_client, 'create_completion'):
        logger.error("Context processor error: The passed llm_client object does not have 'create_completion' method.")
        return None


    try:
        history_text = json.dumps(history_to_summarize, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to serialize history_to_summarize: {e}")
        history_text = str(history_to_summarize)
   
    final_user_instruction_template = f"""
        Below is the history with IDs that needs to be summarized:
        <history_text>
        {history_text}
        </history_text>
        
        Remember to remain the origin question by the user.
        """

    
    

    messages_for_json = [
        {"role": "system", "content": PROCEDURAL_MEMORY_SYSTEM_PROMPT},
        {"role": "user", "content": final_user_instruction_template} 
    ]
    


    messages_for_tool = [
        {"role": "system", "content": CONTEXT_COMPRESSION_SYSTEM_PROMPT},
        {"role": "user", "content": final_user_instruction_template} 
    ]


    procedure_obj: Optional[Procedure] = None
    


    for attempt in range(2):
        try:
            logger.info(f"Calling processor_llm_client (Markdown mode) to generate history summary...")

            procedure_obj = generate_procedure_summary_markdown(
                processor_llm_client,
                messages_for_json  # Either one is fine as long as user prompt is correct
            )
            logger.info("Markdown mode summarization successful.")
            return procedure_obj
        except Exception as e:
            logger.warning(f"History summary generation (Markdown mode) failed: {e}")

        # Attempt 2: JSON (fallback)
        try:
            logger.info(f"Calling processor_llm_client (JSON mode) to generate history summary...")
            procedure_obj = generate_procedure_summary_json(
                processor_llm_client,
                messages_for_json
            )
            logger.info("JSON mode summarization successful.")
            return procedure_obj
        except Exception as e:
            logger.warning(f"History summary generation (JSON mode) failed: {e}")

        try:
            logger.info(f"Calling processor_llm_client (Tool mode) to generate history summary...")
            procedure_obj = generate_context_compression_summary_tool(
                processor_llm_client,
                messages_for_tool
            )
            logger.info("Tool mode summarization successful.")
            return procedure_obj
        except Exception as e:
            logger.error(f"History summary generation (all modes) failed: {e}")
            return None
    return None

def generate_procedure_summary_json(
    processor_llm_client: Any, 
    summarizer_messages: List[Dict]) -> Optional[Procedure]:
    
    
    procedure_response_dict = processor_llm_client.create_completion(
        messages=summarizer_messages,
        tools=None 
    )
    response_content = procedure_response_dict.get("response", "")
    if not response_content:
        raise ValueError("Summarizer LLM returned empty content.")
    
    procedure_dict = extract_json(response_content) 
    if not isinstance(procedure_dict, dict):
            raise ValueError(f"extract_json did not return dict, but returned {type(procedure_dict)}")


    procedure_obj = Procedure.model_validate(procedure_dict) 
    return procedure_obj


def generate_context_compression_summary_tool(
    processor_llm_client: Any, 
    summarizer_messages: List[Dict]
) -> Optional[Procedure]:
    """
    Use "processor" LLM (secondary model) to generate history summary (Procedure).
    Adopt tool-based approach to obtain structured Procedure objects through tool_calls.

    Args:
        processor_llm_client: Initialized LLM client (e.g., ExtendedOpenAIClient).
    """
   
 
    procedure_response_dict = processor_llm_client.create_completion(
        messages=summarizer_messages,
        tools=[context_compression_tool_description],
        tool_choice="required"
    )
    

    tool_calls = procedure_response_dict.get("tool_calls", [])
    
    if not tool_calls:
        raise ValueError("Summarizer (secondary model) LLM returned empty tool_calls.")
    

    tool_call = tool_calls[0]
    args = tool_call.get("function", {}).get("arguments", {})
    

    if isinstance(args, str):
        args = json.loads(args)
    

    procedure_obj = Procedure.model_validate(args)
    
    return procedure_obj  

def extract_json(text: str) -> Any:
    """
    Extract JSON content from LLM output, automatically fix common pseudo-JSON formats, and convert to Python objects.
    Supports triple backtick wrapping and bare JSON.
    Returns Python objects (dict/list).

    """
    text = text.strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:

        json_str = text

        if not (json_str.startswith('{') or json_str.startswith('[')):
             raise ValueError(f"Unable to find JSON. Original content: {text[:200]}...")

    for _ in range(3):
        try:
            # Use json5.loads instead of json.loads
            obj = json.loads(json_str) 
            if isinstance(obj, str):
                json_str = obj
                continue
            return obj
        except Exception as e:

            last_brace = json_str.rfind('}')
            last_bracket = json_str.rfind(']')
            end_index = max(last_brace, last_bracket)
            
            if end_index > -1:
                truncated_json_str = json_str[:end_index + 1]
                try:
                    obj = json.loads(truncated_json_str)
                    if isinstance(obj, str):
                         json_str = obj
                         continue
                    logger.warning(f"Fixed truncated JSON. Original ending: {json_str[end_index+1:]}")
                    return obj
                except Exception:
                    pass 
            
            raise ValueError(f"Unable to parse as JSON: {e}\nOriginal content: {json_str[:500]}...")

    raise ValueError(f"Still not parsed as dict/list after multiple recursions, original content: {json_str[:500]}...")

def generate_procedure_summary_markdown(
    processor_llm_client: Any, 
    summarizer_messages: List[Dict]
) -> Procedure:
    """
    Use "processor" LLM (secondary model) to generate Markdown format summary.
    (Logic ported from manage_context.py)
    """
    

    messages_for_markdown = [msg for msg in summarizer_messages if msg["role"] != "system"]
    messages_for_markdown.insert(0, {"role": "system", "content": PROCEDURAL_MEMORY_SYSTEM_PROMPT_V4})

    if messages_for_markdown[-1]["role"] == "user":
        messages_for_markdown[-1]["content"] += "\n Please strictly obey the system prompt and output the summary in the required markdown format."
    else:

        messages_for_markdown.append({"role": "user", "content": "\n Please strictly obey the system prompt and output the summary in the required markdown format."})


    procedure_response_dict = processor_llm_client.create_completion(
        messages=messages_for_markdown,
        tools=None 
    )
    response_content = procedure_response_dict.get("response", "")
    if not response_content:
        raise ValueError("Markdown summarizer LLM returned empty content.")


    index = extract_replace_history_index(response_content)

    procedure_obj = Procedure(
        replace_history_index=index,
        procedures=response_content,
        step_goal=None,
        step_outcome=None,
        step_status=None
    )
    return procedure_obj

_RE_RHI = re.compile(
    r"""(?imx)               
    ^\s* 
    replace_history_index    
    \s*[:：]\s* 
    (\d+)                     
    (?:\s*[-~—–]\s*(\d+))?    
    \s*$                      
    """
)

def extract_replace_history_index(text: str) -> str:
    """
    Extract replace_history_index from Markdown text, return (start, end).
    (Ported from manage_context.py)
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    m = _RE_RHI.search(text)
    if not m:
        raise ValueError("replace_history_index not found (expected a line like 'replace_history_index: 2-8').")

    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) is not None else start

    if start > end:
        start, end = end, start  # normalize order

    if start <= 1: # Ensure not to replace id 0 (system) and id 1 (user)
        logger.warning(f"replace_history_index: {start}-{end} range too low, automatically corrected to 2-{end}")
        start = 2

    if start > end: # If start > end after correction (e.g., 2-1)
        raise ValueError(f"replace_history_index: {start}-{end} invalid range")

    return f"{start}-{end}"