import datetime
from enum import Enum
from pydantic import BaseModel, Field
from beanie import Document, Indexed, Link, before_event, ValidateOnSave
from typing import Iterator, Any, Callable, Optional, List, Literal, Dict, Union, Annotated
import uuid

class NoTaskAvailableError(Exception):
    """Exception raised when no task is available for sampling."""
    pass

class InferenceService(Document):
    """
    Beanie Document to represent an inference service.
    """
    class OpenAIConfig(BaseModel):
        host: str
        port: int
        base_url: str
        api_key: str = "sk-test"

    models: List[str]
    status: Literal["UP", "DOWN"] = "DOWN"
    
    connection_type: Literal["openai"] = "openai"
    configs: Optional[OpenAIConfig] = None
    
    recent_req_time: Optional[float] = None
    running_req_count: int = 0

    class Settings:
        indexes = [
            "service_id",
            "status",
        ]

class DispatchedSamplingTask(Document):
    """
    Beanie Document to represent a task to be dispatched.
    """
    task: Link["Task"]
    traj_id: int
    req_type: Literal["chatcompletions", "completions"]

    request: Optional[Dict[str, Any] | Any]
    response: Optional[Dict[str, Any] | Any] = None

    sampled_from: Optional[Link[InferenceService]] = None
    priority: int = 0
    creat_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    finish_time: Optional[datetime.datetime] = None
    score: Optional[float] = None
    advantage: Optional[float] = None
    created_at_step: int = 0
    is_minio_managed: Optional[bool] = False
    task_type: Optional[str] = None

    @before_event(ValidateOnSave)
    async def store_fields_in_minio(self):
        # MinIO functionality has been removed
        pass

    async def load_fields_from_minio(self):
        # MinIO functionality has been removed
        pass

    class Status(str, Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed" 
        TOOLFAILED = "toolfailed"  # Tool execution failed
        FORMATERROR = "formaterror" # Format error
        REPEATERROR = "repeaterror" # Repeat error
        CANCELLED = "cancelled"
        DROPPED = "dropped" # this would be useful if you do not want some samples to be trained.
    
    status: Status = Status.PENDING

    class Settings:
        is_root = True
        indexes = [
            "task_id",
            "traj_id",
            "status",
            "belongs_to",
        ]
        
class Task(Document):
    tid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    split: Optional[Literal["train", "valid", "test", "eval"]] = None
    epoch: int = 0
    added_step: int = 0
    description: str = ""
    num_samples: int = 0
    scores: List[float] = []
    
    class Status(str, Enum):
        CREATED = "created"
        RUNNING = "running"
        WAITING = "waiting"
        COMPLETED = "completed"
    status: Status = Status.CREATED
    
    class Settings:
        is_root = True
        indexes = [
            "tid",
            "split",
            "epoch",
            "added_step",
            "num_samples",
        ]
    
    
class Record(Document):
    task: Link[Task]
    split: Literal["train", "valid", "test", "eval"]

    traj_id: Optional[int] = None
    traj: List[Link["DispatchedSamplingTask"]] = Field(default_factory=list)
    meta_infos: dict[str,Any] = {}
    
    class Status(str, Enum):
        CREATED = "created"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        SCORING = "scoring"
        SCORED = "scored"
        READY = "ready"
        ABANDONED = "abandoned"
    
    status: Status = Status.CREATED
    
    score: Optional[float] = None    
    trained_count: int = 0
    last_trained_step: int = -1
    
    class Settings:
        is_root = True
        indexes = [
            "trained_count",
            "task_id",
            "status",
        ]



