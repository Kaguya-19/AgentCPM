import sys

sys.path.append("./src")
sys.path.append("./")

import torch
import torch.utils.data

from src.training.datasets import DBIterableDataset, AgentTrainingConfig
from src.training.utils import _convert_data_into_inputs_labels, AutoProcessor
from src.databases import init_databases
processor = AutoProcessor.from_pretrained(
    "/nfsdata/models/Qwen3-VL-8B-Instruct",
    trust_remote_code=True,
)
def default_data_collator(inputs):
    if None in inputs:
        return None
    return _convert_data_into_inputs_labels(
        inputs,
        processor=processor,
        max_length=8196,
        pad_to_multiple_of=8,
    )


async def main():
    args = AgentTrainingConfig(
            max_trained_count=10,
            db_connection_string="mongodb://admin:2025AgentRL@172.16.1.37:27021/gui_trajectory_test?authSource=admin"
        )
    await init_databases(args)
    dataset = DBIterableDataset(
        args=args
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=default_data_collator,
        num_workers=0,
    )
    for item in dataloader:
        print(item)
        import pdb;pdb.set_trace()
        break

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())