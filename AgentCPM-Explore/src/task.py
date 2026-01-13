# from client import MCPClient
from typing import List, Any, Dict
import uuid


class AgentTask:
    def __init__(self,
                 task_id: str = None,
                 prompt: str = None,  # the task requirements formatted by chat completions
                 file_path: str = None,  # the path to the input files
                 expectation: str = None, # the expected result when gt is not available
                 ground_truth: Any = None,  # the ground truth result
                 conversation_history: List[Dict] = None  # Conversation history
                 ):
        if task_id is None:
            self.task_id = str(uuid.uuid4())
        else:
            self.task_id = task_id
        
        self.prompt = prompt  # your requirements

        if file_path is not None:
            self.file_path = file_path

        if ground_truth is not None:
            self.expectation = ground_truth
        else:
            self.expectation = expectation

        if self.expectation is not None:
            self.task_type = "evaluation"
        else:
            self.task_type = "sampling"
        
        # If conversation history is provided, use it; otherwise, create a basic history with the user prompt
        if conversation_history is not None:
            self.conversation_history = conversation_history
        elif prompt is not None:
            self.conversation_history = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            self.conversation_history = []


class GAIATasks:
    def __init__(self, 
                 tasks: List[AgentTask] = None, # load all data samples
                 ):
        self.tasks = tasks 
    