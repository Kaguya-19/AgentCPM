# Sampling and Task Customization Guide

This guide explains how to customize sampling tasks, extend `AsyncSampler`, and understand the inference service lifecycle in AgentRL.

## Concepts
- **Task**: a sampling task definition stored in MongoDB (`../src/databases/sampler.py`).
- **DispatchedSamplingTask**: a dispatch record created when a training/sampling worker claims a `Task`.
- **Record / DBRecordData**: generated trajectories and training samples consumed by datasets.
- **AsyncSampler**: abstract base class that encapsulates task claiming, inference calls, record writing, and evaluation (`../src/rollout/sampler.py`).
- **InferenceService**: registration info for inference subprocesses (`../src/training/inference.py`).

## Define a custom task type
1. Derive a new `Task` model in `../src/databases/sampler.py` and decorate it with `@register_new_model`:

```python
from databases import register_new_model, Task


@register_new_model
class RFTTask(Task):
    inputs: list[dict]
    target: str
```

2. Wrap your items as a dataset that returns `Task` instances:

```python
from torch.utils.data import Dataset


class RFTDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Return a Task instance; the scheduler will set split/epoch and write to DB.
        return RFTTask(**self.items[idx])
```

Then run it with `SamplingScheduler`:

```python
from rollout.scheduler import SamplingScheduler

dataset = RFTDataset(items)
scheduler = SamplingScheduler(
    config=cfg,
    sampler_class=MySampler,  # your custom sampler
    dataset=dataset,
    split="train",
)
scheduler.run()
```

Note: avoid manually calling `insert()` for tasks. The recommended approach is letting `SamplingScheduler` manage distributed locks and epoch counters, and write tasks into DB in a consistent way.

## Write a custom sampler

```python
from rollout.sampler import AsyncSampler


class MySampler(AsyncSampler):
    async def run(self):
        messages = [{"role": "user", "content": "Hello, who are you?"}]
        response = await self.create_chat_completions(messages)
        # process response and write a record here

    async def evaluate_record(self, record):
        # return a scalar score
        return 0
```

Use it via the async context manager (auto claim/return tasks):

```python
async with MySampler(cfg) as sampler:
    await sampler.run()
```

## Inference service lifecycle
- In each DP group, the TP leader process starts an SGLang subprocess when sampling is needed (`subprocess.Popen`) and registers an `InferenceService` document in MongoDB.
- After sampling, it frees GPU memory, terminates the subprocess, and resumes training. On failures, it attempts to terminate and recycle the subprocess.
- Service config (ports, parallelism, memory ratio) is controlled by `inf_tp_size`, `inf_ep_size`, `inf_mem_ratio`, `target_concurrency`, etc.

## Common debugging checklist
- Ensure there are available `Task`s in MongoDB, and `DispatchedSamplingTask` status updates over time.
- Inspect inference process logs for port conflicts or model loading failures.
- If sampling is stuck: lower `max_new_tokens` / `num_generations` or increase `target_concurrency`.
- If records are empty/low quality: adjust `evaluate_record` scoring logic or filter low-score samples in DB.

## Advanced customization
- To integrate another inference backend (local API or cloud service), modify `InferenceManager` logic in `../src/training/inference.py`.
- To customize `Record -> model input` conversion, replace `convert_record_to_data_func` in `../src/main.py`.
- For multi-modal preprocessing, see `preprocess_mm_messages_for_sample` in `../src/training/datasets.py`.

## Example: OSWorld + SamplingScheduler

This snippet shows how to wrap OSWorld as `OSworldDataset` and run the sampling loop with `SamplingScheduler`.

```python
from rollout.scheduler import SamplingScheduler
from rollout.osworld import OSWorldStatefulSampler, OSworldDataset
from configs import AgentTrainingConfig

cfg = AgentTrainingConfig(
    db_connection_string="mongodb://localhost:27017",
    oss_connection_string="http://minio:9000",
    train_file="assets/osworld_dataset.jsonl",
    target_concurrency=1,
    max_concurrent_samples_per_process=1,
)

dataset = OSworldDataset(cfg.train_file)

scheduler = SamplingScheduler(
    config=cfg,
    sampler_class=OSWorldStatefulSampler,
    dataset=dataset,
    split="train",
)

scheduler.run()
```

Key points:
- `OSworldDataset` reads JSON/JSONL line by line and yields `OSworldTask` with `task_config` and `instruction`. `task_config["proxy"]` is set to `True` automatically.
- `SamplingScheduler` pushes tasks epoch by epoch under a distributed lock for the current split; it starts new samplers when capacity and inference load allow it.
- If you want to pre-start OSWorld virtual environments, see `setup_osworld_envs` in `../src/sampling.py` and only start them on `LOCAL_RANK==0` to avoid duplicates.

