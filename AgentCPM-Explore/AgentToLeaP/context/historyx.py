from typing import Union, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import json
from .promptx import PromptX
import os
from dataclasses import dataclass, field
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam


@dataclass
class HistoryX:
    """
    History message management class that supports independent storage of system prompt
    Internally stores only OpenAI param models, not IDs; IDs are calculated dynamically via indexing
    """
    system_prompt: PromptX
    chat_history: List[ChatCompletionMessageParam] = field(default_factory=list)

    @staticmethod
    def build_message_param(msg):
        """Convert dict or other formats to OpenAI param model"""
        if hasattr(msg, "model_dump"):
            return msg  # Already a param model
        
        if isinstance(msg, dict):
            role = msg.get("role")
            # Remove id field, OpenAI param does not support id
            clean_msg = {k: v for k, v in msg.items() if k != "id"}
            if role == "user":
                return ChatCompletionUserMessageParam(**clean_msg)
            elif role == "assistant":
                return ChatCompletionAssistantMessageParam(**clean_msg)
            elif role == "system":
                return ChatCompletionSystemMessageParam(**clean_msg)
            elif role == "tool":
                return ChatCompletionToolMessageParam(**clean_msg)
            elif role == "function":
                return ChatCompletionFunctionMessageParam(**clean_msg)
            elif role == "developer":
                return ChatCompletionDeveloperMessageParam(**clean_msg)
            else:
                raise ValueError(f"Unknown role: {role}")
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

    @classmethod
    def from_raw_history(cls, raw_history: Any) -> "HistoryX":
        """
        Supports initialization from raw chathistory (list[dict]) or file path.
        Automatically extracts the first message with role=system as system_prompt.
        """
        try:
            if isinstance(raw_history, str):
                if not os.path.isfile(raw_history):
                    raise FileNotFoundError(f"File not found: {raw_history}")
                with open(raw_history, 'r', encoding='utf-8') as f:
                    raw_history = json.load(f)
            if not isinstance(raw_history, list):
                raise ValueError("raw_history must be a list of dicts")
            if not raw_history or raw_history[0].get("role") != "system":
                raise ValueError("The first message must be a system prompt")
            
            # Extract system prompt
            system_msg = raw_history[0]
            system_prompt = PromptX(system_msg["content"])

            # Remaining messages become chat_history, all converted to param model
            chat_history = []
            for msg in raw_history[1:]:
                if not isinstance(msg, dict):
                    raise ValueError(f"Each message must be a dict, got {type(msg)}")
                chat_history.append(cls.build_message_param(msg))
            
            return cls(system_prompt=system_prompt, chat_history=chat_history)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as e:
            raise RuntimeError(f"Failed to initialize HistoryX: {e}")

    def to_raw_history(self, include_id: bool = False, include_system_prompt: bool = True) -> List[Dict]:
        """
        Export to raw chathistory format, automatically placing system prompt at the front.
        :param include_id: Whether to include id field (based on index)
        :param include_system_prompt: Whether to include system prompt
        """
        result = []
        
        # system prompt as the first message
        if include_system_prompt:
            system_msg = {
                "role": "system",
                "content": str(self.system_prompt)
            }
            if include_id:
                system_msg["id"] = 0
            result.append(system_msg)

        # chat_history messages
        for idx, msg in enumerate(self.chat_history):
            # msg itself is already dict, no need for model_dump()
            msg_dict = dict(msg)  # create copy
            if include_id:
                msg_dict["id"] = idx + 1  
            result.append(msg_dict)
        
        return result

    def _find_index_by_id(self, id: int) -> int:
        """Find the position in chat_history by id (starting from 1)"""
        idx = id - 1  # id=1 corresponds to chat_history[0]
        if 0 <= idx < len(self.chat_history):
            return idx
        raise ValueError(f"id {id} not found in chat_history")

    def add_message(self, message):
        """Add message, automatically convert to param model"""
        message = self.build_message_param(message)
        self.chat_history.append(message)

    def insert_after(self, idx: int, messages: List):
        """Insert messages after specified id, automatically convert to param model"""
        if not messages:
            raise ValueError("messages must be a non-empty list")
        
        # Convert all messages to param model
        converted_messages = [self.build_message_param(m) for m in messages]
        insert_pos = self._find_index_by_id(idx)
        self.chat_history[insert_pos+1:insert_pos+1] = converted_messages

    def replace_ids_with_message(self, ids: List[int], message):
        """Replace messages corresponding to specified multiple ids with one message"""
        if not ids:
            raise ValueError("ids must be a non-empty list")
        
        message = self.build_message_param(message)
        positions = [self._find_index_by_id(i) for i in ids]
        first_pos = min(positions)
        
      
        for i in sorted(positions, reverse=True):
            del self.chat_history[i]
        

        self.chat_history.insert(first_pos, message)

    def delete_by_ids(self, ids: List[int]):
        """Delete messages with specified ids"""
        if not ids:
            raise ValueError("ids must be a non-empty list")
        
        positions = [self._find_index_by_id(i) for i in ids]

        for i in sorted(positions, reverse=True):
            del self.chat_history[i]

    def save_to_json(self, path: str, include_id: bool = False, include_system_prompt: bool = True):
        """Save as JSON file in raw chathistory format"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.to_raw_history(include_id=include_id, include_system_prompt=include_system_prompt), 
                    f, ensure_ascii=False, indent=2
                )
        except OSError as e:
            raise RuntimeError(f"Failed to save to {path}: {e}")

    def to_openai_messages(self):
        """Directly used for openai.ChatCompletion.create(messages=...)"""
        return [
            {"role": "system", "content": str(self.system_prompt)}
        ] + list(self.chat_history) 

    def render_with_id(self):
        """Return message list with id field (id starts from 0 and increments)"""
        return self.to_raw_history(include_id=True, include_system_prompt=True)

    def __len__(self):
        return len(self.chat_history)

    def __getitem__(self, idx):
        """
        idx=0 returns system prompt, others return chat messages (id starts from 1)
        Supports negative indexing, -1 returns the last chat message
        """
        if idx == 0:
            return {
                "role": "system",
                "content": str(self.system_prompt)
            }
        
        # Handle negative indices
        if idx < 0:
            if idx == -1:
                # Return the last chat message
                if not self.chat_history:
                    raise IndexError("No chat messages available")
                return self.chat_history[-1]
            else:
                # Convert negative index to positive
                idx = len(self.chat_history) + idx + 1  # +1 because positive indices start from 1

        # Messages in chat_history are already dict
        return self.chat_history[idx - 1]

    def __setitem__(self, idx, value):
        """
        Supports modifying message content, especially historyx[-1]["content"] += "..."
        """
        if idx == 0:
            raise ValueError("Cannot modify system prompt through indexing")
        
        # Handle negative indices
        if idx < 0:
            if idx == -1:
                if not self.chat_history:
                    raise IndexError("No chat messages available")
                # Update the last message
                if isinstance(value, dict):
                    self.chat_history[-1].update(value)
                else:
                    raise ValueError("Value must be a dictionary")
                return
            else:
                # Convert negative index to positive
                idx = len(self.chat_history) + idx + 1

        # Update the message at the specified index
        if isinstance(value, dict):
            self.chat_history[idx - 1].update(value)
        else:
            raise ValueError("Value must be a dictionary")

    def append_to_last_message_content(self, text: str):
        """
        Safely append text to the content field of the last message
        Handle various content types (string, list, None, etc.)
        """
        if not self.chat_history:
            raise IndexError("No chat messages available")
        
        last_message = self.chat_history[-1]
        current_content = last_message.get("content")
        
        if current_content is None:
            # If content is None, directly set to string
            last_message["content"] = text
        elif isinstance(current_content, str):
            # If it's a string, directly concatenate
            last_message["content"] = current_content + text
        elif isinstance(current_content, list):
            # If it's a list (OpenAI content parts), convert to string then concatenate
            content_str = str(current_content)
            last_message["content"] = content_str + text
        else:
            # Other types, convert to string then concatenate
            content_str = str(current_content)
            last_message["content"] = content_str + text

    def __iter__(self):
        """Iterate over all messages, including system prompt"""
        yield {"role": "system", "content": str(self.system_prompt)}
        for msg in self.chat_history:
            # OpenAI param types are essentially dict, return directly
            yield msg

    def __call__(self):
        """Allow historyx() to directly get chathistory list"""
        return self.to_raw_history()

    def __repr__(self):
        """Display messages with id when printing"""
        return str(self.render_with_id())

    def __add__(self, other):
        """Merge two HistoryX objects or add message list"""
        if isinstance(other, HistoryX):
            # HistoryX + HistoryX
            new_history = []
            for msg in self.chat_history:
                new_history.append(self.build_message_param(msg))
            for msg in other.chat_history:
                new_history.append(self.build_message_param(msg))

            # system_prompt takes current object as primary
            return HistoryX(system_prompt=self.system_prompt, chat_history=new_history)

        elif isinstance(other, list):
            # HistoryX + list[dict]
            new_history = []
            for msg in self.chat_history:
                new_history.append(self.build_message_param(msg))
            for msg in other:
                new_history.append(self.build_message_param(msg))

            return HistoryX(system_prompt=self.system_prompt, chat_history=new_history)

        else:
            # Try to convert to HistoryX
            try:
                other = HistoryX.from_raw_history(other)
                return self.__add__(other)
            except Exception:
                return NotImplemented
