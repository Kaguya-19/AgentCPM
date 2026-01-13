import dataclasses
from typing import List, Dict, Any, Tuple, Optional, Iterator
from collections.abc import Mapping
from beanie import Document
import torch

@dataclasses.dataclass
class RecordData:
    messages: List[Dict[str,str | List[Dict[str,Any]] | Any]]
    tools: Optional[List[Dict[str,Any]]]
    scores: Dict[int, float]
    advantages: Dict[int, float]
    created_at_step: Dict[int, int]
    reward: float
    step: int
    logprobs: Optional[Dict[int, Any]]
    record_id: Optional[str] = None

class DBRecordData(Document, RecordData):
    split: str
    fetched: bool = False # whether the data has been used
    class Settings:
        is_root = True
    
    @classmethod
    def from_record_data(
        cls,
        record_data: RecordData,
        split: str = "train",
    ) -> "DBRecordData":
        return cls(
            messages=record_data.messages,
            tools=record_data.tools,
            scores=record_data.scores,
            advantages=record_data.advantages,
            created_at_step=record_data.created_at_step,
            reward=record_data.reward,
            step=record_data.step,
            logprobs=record_data.logprobs,
            record_id=record_data.record_id,
            split=split,
        )
    
    def to_record_data(self) -> RecordData:
        return RecordData(
            messages=self.messages,
            tools=self.tools,
            scores=self.scores,
            advantages=self.advantages,
            created_at_step=self.created_at_step,
            reward=self.reward,
            step=self.step,
            logprobs=self.logprobs,
            record_id=self.record_id,
        )

@dataclasses.dataclass
class DataLabels(Mapping):
    """A container for RL data labels used in training.
    """
    scores: List[float]
    advantages: List[float]
    rewards: List[float]
    steps: List[int]
    target_ids: torch.Tensor
    assistant_mask: torch.Tensor
    advantage_mask: torch.Tensor
    create_step_mask: torch.Tensor
    per_token_logprobs: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        # iterate over field names
        return iter(field.name for field in dataclasses.fields(self))

    def __len__(self) -> int:
        return len(dataclasses.fields(self))

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self:
            yield key, getattr(self, key)
            