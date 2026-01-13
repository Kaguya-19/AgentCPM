"""
处理 discard_all_tools 和 discard_all 模式的上下文重置和总结功能。
"""
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Literal
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def reset_messages_discard_tools(
    messages: List[dict],
    task_system_prompt: str,
    task_query_prompt: str
) -> List[dict]:
    """
    重置消息（discard_all_tools 模式）：清除所有 tool 消息，保留 system、user 和 assistant 消息。
    
    处理流程：
    1. 过滤掉所有 role="tool" 的消息
    2. 删除最后 min(15, length-10) 条消息（保留至少10条）
    3. 确保有 system 和 user 提示
    4. 返回过滤后的消息列表
    
    Args:
        messages: 原始消息列表
        task_system_prompt: 任务的系统提示
        task_query_prompt: 任务的查询提示
        
    Returns:
        重置后的消息列表
    """
    # 统计消息数量
    tool_count = sum(1 for msg in messages if msg.get("role") == "tool")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    system_count = sum(1 for msg in messages if msg.get("role") == "system")
    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    
    # 过滤掉所有 tool 消息
    filtered_messages = [
        msg for msg in messages 
        if msg.get("role") != "tool"
    ]
    
    # 计算移除的 tool 消息数量
    removed_count = len(messages) - len(filtered_messages)
    kept_before_add = len(filtered_messages)
    
    # 删除最后 min(15, length-10) 条消息（保留至少10条）
    delete_count = min(15, max(0, len(filtered_messages) - 10))
    deleted_recent = 0
    if delete_count > 0:
        deleted_recent = delete_count
        filtered_messages = filtered_messages[:-delete_count]
        logger.info(f"Deleted last {deleted_recent} messages after tool filtering (kept {len(filtered_messages)} messages).")
    
    # 确保有 system 和 user 提示
    has_system = any(msg.get("role") == "system" for msg in filtered_messages)
    has_user = any(msg.get("role") == "user" for msg in filtered_messages)
    
    added_count = 0
    if not has_system:
        filtered_messages.insert(0, {"role": "system", "content": task_system_prompt})
        added_count += 1
    if not has_user:
        # 找到最后一个 user 消息或使用 query_prompt
        last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_msg:
            filtered_messages.append(last_user_msg)
        else:
            filtered_messages.append({"role": "user", "content": task_query_prompt})
        added_count += 1
    
    # 记录统计信息
    ratio_info = f"ratio assistant:tool = {assistant_count}:{tool_count}"
    if assistant_count > 0:
        ratio_info += f" ({tool_count/assistant_count:.2f} tools per assistant)"
    delete_info = f", deleted {deleted_recent} recent messages" if deleted_recent > 0 else ""
    logger.info(
        f"Reset messages using mode 'discard_all_tools': removed {removed_count} tool messages (was {tool_count}), "
        f"kept {kept_before_add} messages (system:{system_count}, user:{user_count}, assistant:{assistant_count}), "
        f"{ratio_info}{delete_info}"
        + (f", added {added_count} missing prompt(s)" if added_count > 0 else "")
        + f", total {len(filtered_messages)} messages after reset."
    )
    
    return filtered_messages


async def reset_messages_with_summary(
    messages: List[dict],
    task_system_prompt: str,
    task_query_prompt: str,
    tokenizer: Any,
    browse_agent_config: Dict[str, Any]
) -> List[dict]:
    """
    重置消息并生成总结。
    
    处理流程：
    1. 过滤掉所有 tool 消息
    2. 调用 summary model 总结所有发现
    3. 返回替换第三条及以后所有内容的消息列表（保留前2条，用总结替换第3条及以后的所有内容）
    
    Args:
        messages: 原始消息列表
        task_system_prompt: 任务的系统提示
        task_query_prompt: 任务的查询提示
        tokenizer: tokenizer 用于计算 token 数量
        
    Returns:
        重置后的消息列表
    """
    # 统计消息数量
    tool_count = sum(1 for msg in messages if msg.get("role") == "tool")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    system_count = sum(1 for msg in messages if msg.get("role") == "system")
    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    
    # 过滤掉所有 tool 消息
    filtered_messages = [
        msg for msg in messages 
        if msg.get("role") != "tool"
    ]
    
    removed_count = len(messages) - len(filtered_messages)
    kept_before_add = len(filtered_messages)
    
    # 确保有 system 和 user 提示
    has_system = any(msg.get("role") == "system" for msg in filtered_messages)
    has_user = any(msg.get("role") == "user" for msg in filtered_messages)
    
    added_count = 0
    if not has_system:
        filtered_messages.insert(0, {"role": "system", "content": task_system_prompt})
        added_count += 1
    if not has_user:
        # 找到最后一个 user 消息或使用 query_prompt
        last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_msg:
            filtered_messages.append(last_user_msg)
        else:
            filtered_messages.append({"role": "user", "content": task_query_prompt})
        added_count += 1
    
    # 准备总结内容：收集所有被过滤的消息内容
    # 从原始 messages 中提取所有 assistant 和 tool 消息的内容用于总结
    summary_content_parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ["assistant", "tool"]:
            content = msg.get("content", "")
            if content:
                if role == "assistant":
                    summary_content_parts.append(f"[Assistant]: {content}")
                elif role == "tool":
                    # 尝试解析 tool 消息的 JSON 内容
                    try:
                        if isinstance(content, str):
                            tool_data = json.loads(content)
                            summary_content_parts.append(f"[Tool {msg.get('name', 'unknown')}]: {json.dumps(tool_data, ensure_ascii=False)}")
                        else:
                            summary_content_parts.append(f"[Tool {msg.get('name', 'unknown')}]: {str(content)}")
                    except:
                        summary_content_parts.append(f"[Tool {msg.get('name', 'unknown')}]: {str(content)}")
    
    # 调用 summary model 生成总结
    summary_text = await generate_summary(summary_content_parts, task_query_prompt, browse_agent_config)
    
    # 构建最终消息列表：保留前2条，然后用总结替换第3条及以后的所有内容
    final_messages = []
    
    # 保留前2条消息（通常是 system, user）
    keep_count = min(2, len(filtered_messages))
    final_messages = filtered_messages[:keep_count].copy()
    
    # 添加总结作为一条 assistant message（替换第3条及以后的所有内容）
    final_messages.append({
        "role": "assistant",
        "content": summary_text
    })
    
    # 记录统计信息
    ratio_info = f"ratio assistant:tool = {assistant_count}:{tool_count}"
    if assistant_count > 0:
        ratio_info += f" ({tool_count/assistant_count:.2f} tools per assistant)"
    logger.info(
        f"Reset messages using mode 'discard_all': removed {removed_count} tool messages (was {tool_count}), "
        f"kept {kept_before_add} messages (system:{system_count}, user:{user_count}, assistant:{assistant_count}), "
        f"{ratio_info}"
        + (f", added {added_count} missing prompt(s)" if added_count > 0 else "")
        + f", total {len(final_messages)} messages after reset (kept first {keep_count}, replaced from 3rd message onwards with summary)."
    )
    
    return final_messages


async def generate_summary(content_parts: List[str], question: str, browse_agent_config: Dict[str, Any]) -> str:
    """
    使用 summary model 生成 Markdown 格式的总结。
    
    Args:
        content_parts: 要总结的内容部分列表
        question: 任务问题，用于指导总结
        browse_agent_config: browse_agent 配置字典，包含 models 列表和其他配置
        
    Returns:
        Markdown 格式的总结文本
    """
    if not content_parts:
        return "## Summary\n\nNo content to summarize."
    
    # 合并所有内容
    full_content = "\n\n".join(content_parts)
    
    # 构建提示词
    system_prompt = """You are an intelligent summarization assistant. Your task is to summarize all the findings and actions from the conversation history.

Please provide a concise summary in Markdown format that includes:
1. **Actions**: Actions taken by the agent
2. **Results**: Results of the actions
3. **Current Status**: What has been accomplished and what remains

Format your response as clean Markdown without code blocks."""
    
    user_prompt = f"""Please summarize all the findings from the following conversation history. The original task question is: "{question}"

## Conversation History
{full_content}

Please provide a concise summary in Markdown format covering all key findings, actions, and results."""
    
    models = browse_agent_config.get("models", [])
    max_retries = browse_agent_config.get("max_retries", 3)
    retry_delay = browse_agent_config.get("retry_delay", 2)
    timeout = browse_agent_config.get("timeout", 100)
    
    if not models:
        error_msg = "No models configured in browse_agent config"
        logger.error(error_msg)
        return f"## Summary\n\nError: {error_msg}"
    
    # 尝试配置中的每个模型，按顺序
    for i, model_config in enumerate(models):
        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model")
        
        if not api_key or not base_url or not model_name:
            logger.warning(f"Model {i+1} in browse_agent config is missing required fields (api_key, base_url, or model), skipping")
            continue
        
        try:
            llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"尝试使用 browse_agent model {i+1} 生成总结: {model_name}")
            
            current_retry_delay = retry_delay
            for attempt in range(max_retries):
                try:
                    response = await llm_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model_name,
                        timeout=timeout
                    )
                    
                    summary_text = response.choices[0].message.content
                    
                    # 清理可能的 Markdown 代码块标记
                    if summary_text.strip().startswith("```markdown"):
                        summary_text = summary_text.strip()[len("```markdown"):]
                    if summary_text.strip().startswith("```"):
                        summary_text = summary_text.strip()[3:]
                    if summary_text.strip().endswith("```"):
                        summary_text = summary_text.strip()[:-3]
                    
                    return summary_text.strip()
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Summary generation failed (attempt {attempt + 1}/{max_retries}) using model {i+1} ({model_name}): {e}, retrying in {current_retry_delay}s...")
                        await asyncio.sleep(current_retry_delay)
                        current_retry_delay *= 2
                    else:
                        logger.warning(f"Summary generation failed after {max_retries} attempts using model {i+1} ({model_name}): {e}")
                        break
        except Exception as e:
            logger.warning(f"Browse agent model {i+1} ({model_name}) 调用异常: {e}")
            continue
    
    # 所有模型都失败了
    error_msg = f"Failed to generate summary after trying all {len(models)} models in browse_agent config"
    logger.error(error_msg)
    return f"## Summary\n\nError: {error_msg}"


async def reset_messages(
    messages: List[dict],
    mode: Literal["discard_all_tools", "discard_all"],
    task_system_prompt: str,
    task_query_prompt: str,
    browse_agent_config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None
) -> List[dict]:
    """
    统一的消息重置函数，根据模式选择不同的处理方式。
    
    Args:
        messages: 原始消息列表
        mode: 重置模式，"discard_all_tools" 或 "discard_all"
        task_system_prompt: 任务的系统提示
        task_query_prompt: 任务的查询提示
        browse_agent_config: browse_agent 配置字典（仅 discard_all 模式需要）
        tokenizer: tokenizer（仅 discard_all 模式需要，但当前未使用）
        
    Returns:
        重置后的消息列表
    """
    if mode == "discard_all_tools":
        return await reset_messages_discard_tools(
            messages=messages,
            task_system_prompt=task_system_prompt,
            task_query_prompt=task_query_prompt
        )
    elif mode == "discard_all":
        if browse_agent_config is None:
            raise ValueError("browse_agent_config is required for 'discard_all' mode")
        return await reset_messages_with_summary(
            messages=messages,
            task_system_prompt=task_system_prompt,
            task_query_prompt=task_query_prompt,
            tokenizer=tokenizer,
            browse_agent_config=browse_agent_config
        )
    else:
        raise ValueError(f"Unknown reset mode: {mode}. Must be 'discard_all_tools' or 'discard_all'")

