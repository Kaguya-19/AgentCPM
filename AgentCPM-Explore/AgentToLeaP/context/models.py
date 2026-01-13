from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict



class NextStep(BaseModel):
    processing_goal: str = Field(description="Current available condition information and specific goals to achieve in the next step")
    output_spec: str = Field(description="Detailed expected output")
    success_criteria: str = Field(description="Objective success criteria")
    detailed_actions: str = Field(description="Specific operational steps")

class Plan(BaseModel):
    plan: List[str] = Field(description="Display the key step sequence of the overall task")
    next_step: NextStep = Field(description="Provide detailed execution instructions for the agent's next step")
    next_step_tools: List[str] = Field(description="Tools required to execute the next step")



class ToolExperience(BaseModel):
    key: str
    value: str

class ToolExperienceList(BaseModel):
    tool_experience_list: List[Union[ToolExperience, Dict]] = Field(description="Tool usage experience")


class KVMemory(BaseModel):
    id: str = Field(description="Unique identifier for memory")
    key: str = Field(description="Memory key")
    value: str = Field(description="Memory value")
    event: str = Field(description="Memory operation type")
    old_value: Optional[str] = None

class KVMemoryList(BaseModel):
    memory_list: List[KVMemory] = Field(description="Memory list")

class Procedure(BaseModel):
    replace_history_index: str = Field(description="Index range of history records to be replaced, e.g. '2-5', please keep initial question and plan content")
    procedures: str = Field(description="Detailed process summary content, summary of execution results and valid information for each step")
    step_goal: Optional[str] = Field(description="Current step goal and what content the expected results should contain")
    step_outcome: Optional[str] = Field(description="Step execution result summary, and whether completed correctly")
    step_status: Optional[str] = Field(description="Step execution status (completed/partially completed/failed)")

