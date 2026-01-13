import asyncio
import torch
from torch.utils.data import DataLoader, IterableDataset
from beanie import init_beanie,UpdateResponse
import sys
sys.path.append("./")  
from src.models import models_to_initialize, Task

loop = asyncio.get_event_loop()

class RFTTask(Task):
    """
    A task for GUI task
    """
    inputs: list[dict[str, str]]
    outputs: str

loop.run_until_complete(init_beanie(
    connection_string="mongodb://10.0.1.8:27018,10.0.1.9:27018,10.0.1.11:27018/run",
    document_models=models_to_initialize+[RFTTask],
))
loop.run_until_complete(Task.find_all().delete())
loop.run_until_complete(RFTTask(goal="Test RFT Task", description="This is a test task for RFT",inputs=[{"role":"user","content":"hello world"}],outputs="test").save())

task = loop.run_until_complete(Task.find_one(Task.num_samples < 3,with_children=True).inc({Task.num_samples: 1},response_type=UpdateResponse.NEW_DOCUMENT))

import pdb;pdb.set_trace()