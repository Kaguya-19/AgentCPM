# SFT Guide

This guide covers the minimal setup, common options, and runnable examples for **Supervised Fine-Tuning (SFT)** in AgentRL.

## When to use SFT
- Train on fixed labeled data (no online sampling / evaluation).
- Use CrossEntropy loss only (`loss_calculater=CrossEntropy`).
- Load data either from MongoDB `Record`s or local JSON / JSONL files.

## Prerequisites
1. Install dependencies and pass the basic import self-check (see `../README.md`).
2. If reading from DB: make sure `Record` and `DispatchedSamplingTask` are written following the document models in `../src/databases`.
3. If reading from local files: prepare JSON/JSONL that matches the message format expected by the data pipeline (examples in `../assets`).

## Recommended config
- Disable sampling: `--enable_sampling false`
- Disable online eval: `--eval_strategy no`
- Loss: `--loss_calculater CrossEntropy`
- Avoid re-training the same sample: `--max_trained_count 1`
- Tune memory: `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--bf16/--fp16`

## Examples

### 1) Train from local files

Option A (recommended): pass `train_dataset` into your Trainer directly (easy to customize).

```python
# python examples/sft_local_dataset.py
from pathlib import Path
from training.arl import ParquetDataset
from configs import AgentTrainingConfig
from training import QwenTrainer  # replace with your actual Trainer


def main():
  args = AgentTrainingConfig(
    model_name_or_path="/path/to/model",
    output_dir="output/sft-demo",
    enable_sampling=False,
    loss_calculater="CrossEntropy",
    eval_strategy="no",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1,
    save_steps=500,
  )

  # ParquetDataset yields RecordData objects
  train_dataset = ParquetDataset(
    file_paths=Path("assets/math500.jsonl").with_suffix(".parquet"),
    args=args,
    split="train",
  )

  trainer = QwenTrainer(
    model=None,  # Trainer loads the model internally according to args
    args=args,
    train_dataset=train_dataset,
  )
  trainer.train()


if __name__ == "__main__":
  main()
```

Option B: implement your own dataset that yields `RecordData`.

```python
from typing import Iterator

from configs import AgentTrainingConfig
from databases.dataset import RecordData
from training import QwenTrainer
from torch.utils.data import IterableDataset


class JsonlRecordDataDataset(IterableDataset):
  def __init__(self, jsonl_path: str):
    self.jsonl_path = jsonl_path

  def __iter__(self) -> Iterator[RecordData]:
    import json

    with open(self.jsonl_path, "r", encoding="utf-8") as f:
      for line in f:
        obj = json.loads(line)
        yield RecordData(
          messages=obj["messages"],
          tools=obj.get("tools"),
          scores=obj.get("scores", {}),
          advantages=obj.get("advantages", {}),
          created_at_step=obj.get("created_at_step", {}),
          reward=obj.get("reward", 1.0),
          step=obj.get("step", -1),
          logprobs=obj.get("logprobs"),
          record_id=obj.get("record_id"),
        )


args = AgentTrainingConfig(
  model_name_or_path="/path/to/model",
  output_dir="output/sft-demo-jsonl",
  enable_sampling=False,
  loss_calculater="CrossEntropy",
  eval_strategy="no",
)

train_dataset = JsonlRecordDataDataset("assets/math500.jsonl")
trainer = QwenTrainer(model=None, args=args, train_dataset=train_dataset)
trainer.train()
```

### 2) Train from MongoDB (sampling disabled)

```bash
torchrun --nproc_per_node=8 src/main.py \
  --model_name_or_path /path/to/model \
  --db_connection_string "mongodb://user:pwd@host:27017" \
  --output_dir output/sft-from-db \
  --enable_sampling false \
  --loss_calculater CrossEntropy \
  --eval_strategy no
```

Data must be written to MongoDB ahead of time. The document schema should match `DBRecordData` in `../src/databases/dataset.py`.

## Data format notes
- Multi-modal samples automatically convert image URLs to base64 during dataset loading (see `../src/training/datasets.py`).
- If you write directly into DB, write `DBRecordData` documents (avoid temporary schemas to prevent Beanie validation failures).
- For local/custom datasets, yield `RecordData` objects; the framework will convert to `DBRecordData` as needed.
- For Parquet, reuse `ParquetDataset` (`../src/training/datasets.py`), which already yields `RecordData`.
- For custom record-to-input conversion, customize `convert_record_to_data_func` in `../src/main.py`.

## Training tips
- If GPU memory is tight: try `--loss_seq_chunk_size` and/or `--activation_offloading`.
- For quick single-GPU verification: `python src/main.py ...` (distributed/parallel strategies are disabled).
- Use `--save_total_limit` to cap checkpoints.

## FAQ
- Import errors: re-install deps, then run `python -c "import sys; sys.path.insert(0, 'src'); import models, sampler"`.
- Empty data: verify `train_file` paths or check whether `Record`s exist in MongoDB.
- OOM: lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`, or enable `tp_size/pp_size/ep_size`.

