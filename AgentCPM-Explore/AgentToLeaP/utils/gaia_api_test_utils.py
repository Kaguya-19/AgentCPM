#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("gaia_api_test_utils")

def visualize_conversation_history(conversation_history: List[Dict[str, Any]]) -> str:
    
    if not conversation_history:
        return "Empty conversation history."
        
    output = []
    
    for i, message in enumerate(conversation_history):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        
        
        if role == "system":
            continue
            
        if role == "user":
            output.append(f"[User {i}]:\n{content}\n")
        elif role == "assistant":
            
            if "tool_calls" in message:
                tool_calls = message.get("tool_calls", [])
                output.append(f"[Assistant {i}] (with {len(tool_calls)} tool calls):\n{content or 'No content, using tools'}\n")
                
                for tc_idx, tc in enumerate(tool_calls):
                    tc_func = tc.get("function", {})
                    tc_name = tc_func.get("name", "unknown_tool")
                    tc_args = tc_func.get("arguments", "{}")
                    output.append(f"  Tool call {tc_idx+1}: {tc_name}\n  Arguments: {tc_args}\n")
            else:
                output.append(f"[Assistant {i}]:\n{content}\n")
        elif role == "tool":
            tool_call_id = message.get("tool_call_id", "unknown")
            name = message.get("name", "unknown_tool")
            output.append(f"[Tool {i} - {name} (id: {tool_call_id})]: \n{content}\n")
    
    return "\n".join(output)


def format_gaia_dialog(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:

    dialog = []
    current_turn = {"user": "", "assistant": "", "tools": []}
    
    for msg in conversations:
        role = msg.get("role")
        
        if role == "system":
            continue
        elif role == "user":
            
            if current_turn.get("user") or current_turn.get("assistant") or current_turn.get("tools"):
                dialog.append(current_turn.copy())
                
            
            current_turn = {"user": msg.get("content", ""), "assistant": "", "tools": []}
        elif role == "assistant":
            current_turn["assistant"] = msg.get("content", "")
            
            
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                tc_func = tc.get("function", {})
                current_turn["tools"].append({
                    "name": tc_func.get("name", "unknown_tool"),
                    "arguments": tc_func.get("arguments", "{}"),
                    "id": tc.get("id", "unknown_id"),
                    "response": None  
                })
        elif role == "tool":
            
            tool_call_id = msg.get("tool_call_id")
            for tool in current_turn["tools"]:
                if tool["id"] == tool_call_id:
                    tool["response"] = msg.get("content")
                    break
    
    
    if current_turn.get("user") or current_turn.get("assistant") or current_turn.get("tools"):
        dialog.append(current_turn)
        
    return {"dialog": dialog}

def save_conversation_text(conversation_history: List[Dict[str, Any]], file_path: Path) -> None:
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                if role == "system":
                    continue
                
                f.write(f"[{role.upper()}]:\n")
                f.write(f"{content}\n\n")
                
                
                if role == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    for tc in tool_calls:
                        tc_func = tc.get("function", {})
                        tc_name = tc_func.get("name", "unknown_tool")
                        tc_args = tc_func.get("arguments", "{}")
                        f.write(f"  Tool call: {tc_name}\n")
                        f.write(f"  Arguments: {tc_args}\n\n")
    except Exception as e:
        logger.error(f"Error saving conversation text: {str(e)}")

def save_meta_config(
    task_dir: Path, 
    model: str, 
    provider: str, 
    tools_count: int,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None
) -> None:
    
    meta_config = {
        "model": model,
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "available_tools_count": tools_count
    }
    
    
    if system_prompt:
        meta_config["system_prompt"] = system_prompt
    
    
    if tools:
        
        tools_info = []
        for tool in tools:
            if "function" in tool:
                tool_info = {
                    "name": tool["function"].get("name", ""),
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {})
                }
                tools_info.append(tool_info)
        
        meta_config["tools"] = tools_info
    
    try:
        with open(task_dir / "meta_config.json", "w", encoding="utf-8") as f:
            json.dump(meta_config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving meta configuration: {str(e)}")

def save_tool_calls(task_dir: Path, tool_calls: List[Dict[str, Any]]) -> None:
  
    
    tool_calls_dir = task_dir / "tool_calls"
    os.makedirs(tool_calls_dir, exist_ok=True)
    
    
    calls_info = {
        "total_calls": len(tool_calls),
        "timestamp": datetime.now().isoformat(),
        "calls": []
    }
    
    for i, call in enumerate(tool_calls):
        
        call_id = call.get("id", f"call_{uuid.uuid4().hex[:8]}")
        function = call.get("function", {})
        name = function.get("name", "unknown_tool")
        arguments = function.get("arguments", "{}")
        
        
        server_id = call.get("server_id", None)
        
        
        if not server_id and "." in name:
            server_id, _ = name.split(".", 1)
        
        
        if not server_id:
            
            if name == "web_search":
                server_id = "zhipu-web-search"
            else:
                
                server_id = "default"
        
        
        if "server_id" not in call:
            call["server_id"] = server_id
        
        
        response_status = 200  
        response_detail = ""
        error_info = ""
        
        
        if "response" in call and isinstance(call["response"], dict):
            if "status_code" in call["response"]:
                response_status = call["response"]["status_code"]
            if "detail" in call["response"]:
                response_detail = call["response"]["detail"]
        
        
        if "error" in call:
            error_info = call["error"]
        
        
        call_info = {
            "id": call_id,
            "name": name,
            "server_id": server_id,
            "arguments": arguments,
            "response_status": response_status,
            "response_detail": response_detail,
            "error": error_info
        }
        
        calls_info["calls"].append(call_info)
        
       
        call_file = tool_calls_dir / f"call_{i+1}_{name}.json"
        with open(call_file, "w", encoding="utf-8") as f:
            json.dump(call, f, ensure_ascii=False, indent=2)
    
    
    summary_file = tool_calls_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(calls_info, f, ensure_ascii=False, indent=2)

def save_llm_interactions(task_dir: Path, conversation_history: List[Dict[str, Any]], model: str, provider: str) -> None:
    """
    Save LLM interaction information
    
    Args:
        task_dir: Task directory
        conversation_history: Conversation history
        model: Model name
        provider: Provider
    """
    # Create LLM interactions directory
    interactions_dir = task_dir / "llm_interactions"
    os.makedirs(interactions_dir, exist_ok=True)
    
    # Reorganize conversation history, grouped by user-assistant-tool order
    interactions = []
    current_interaction = None
    
    # Skip system message
    start_idx = 0
    for i, msg in enumerate(conversation_history):
        if msg.get("role") == "system":
            start_idx = i + 1
            break
    
    # Handle remaining conversation messages
    i = start_idx
    while i < len(conversation_history):
        # Get user message
        user_msg = None
        if i < len(conversation_history) and conversation_history[i].get("role") == "user":
            user_msg = conversation_history[i]
            i += 1
        else:
            # If no user message, create an empty one
            user_msg = {"role": "user", "content": ""}
        
        # Get assistant message
        assistant_msg = None
        if i < len(conversation_history) and conversation_history[i].get("role") == "assistant":
            assistant_msg = conversation_history[i]
            i += 1
        
        # If no assistant message, skip this turn
        if not assistant_msg:
            continue
        
        # Collect tool responses
        tool_responses = []
        while i < len(conversation_history) and conversation_history[i].get("role") == "tool":
            tool_responses.append(conversation_history[i])
            i += 1
        
        # Create interaction record
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "tool_responses": tool_responses,
            "model": model,
            "provider": provider
        }
        
        interactions.append(interaction)
    
    # Save each interaction
    for i, interaction in enumerate(interactions):
        interaction_count = i + 1
        
        # Save interaction file
        interaction_file = interactions_dir / f"interaction_{interaction_count}.json"
        with open(interaction_file, "w", encoding="utf-8") as f:
            json.dump(interaction, f, ensure_ascii=False, indent=2)
    
    # Save interaction summary
    summary = {
        "total_interactions": len(interactions),
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "provider": provider
    }
    
    summary_file = interactions_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def save_batch_summary(result_dir: Path, results: List[Dict[str, Any]], model: str, provider: str) -> None:
    """
    Save batch test summary
    
    Args:
        result_dir: Result directory
        results: List of test results
        model: Model name
        provider: Provider
    """
    # Calculate the number of completed and failed tasks
    completed = sum(1 for r in results if r.get("success", False))
    failed = len(results) - completed
    
    # Construct summary data
    summary = {
        "total_samples": len(results),
        "completed": completed,
        "failed": failed,
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    # Save summary file
    summary_file = result_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2) 