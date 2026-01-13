# 采样与任务定制指南

本文介绍如何在 AgentRL 中自定义采样任务、扩展 `AsyncSampler`，以及推理服务的关键机制。

## 概念速览
- **Task**：待采样任务的定义，存储在 MongoDB，位于 [src/databases/sampler.py](../src/databases/sampler.py)。
- **DispatchedSamplingTask**：训练/采样进程领取 `Task` 后生成的派发记录，标记状态与分配。
- **Record / DBRecordData**：实际生成的轨迹与训练样本，供数据集迭代。
- **AsyncSampler**：采样抽象基类，封装任务领取、推理调用、轨迹写入与评估逻辑，定义在 [src/rollout/sampler.py](../src/rollout/sampler.py)。
- **InferenceService**：推理子进程的注册信息，定义在 [src/training/inference.py](../src/training/inference.py)。

## 自定义任务类型
1. 在 [src/databases/sampler.py](../src/databases/sampler.py) 中为 `Task` 派生子类，并用 `@register_new_model` 装饰：
```python
from databases import register_new_model, Task

@register_new_model
class RFTTask(Task):
    inputs: list[dict]
    target: str
```
2. 将数据封装为 Dataset（返回 Task）：
```python
from torch.utils.data import Dataset

class RFTDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # 返回 Task 实例，Scheduler 会负责设置 split/epoch 并写入 DB
        return RFTTask(**self.items[idx])
```

然后交给 `SamplingScheduler`：
```python
from rollout.scheduler import SamplingScheduler

dataset = RFTDataset(items)
scheduler = SamplingScheduler(
    config=cfg,
    sampler_class=MySampler,  # 你的自定义采样器
    dataset=dataset,
    split="train",
)
scheduler.run()
```
说明：不建议手工 `insert()` 任务。标准做法是由 `SamplingScheduler` 在分布式锁与 epoch 计数的管理下，统一从 `Dataset` 取出 `Task` 并写入数据库。

## 编写自定义 Sampler
```python
from rollout.sampler import AsyncSampler

class MySampler(AsyncSampler):
    async def run(self):
        messages = [{"role": "user", "content": "你好，你是谁？"}]
        response = await self.create_chat_completions(messages)
        # 在此处理 response 并写入 record
    async def evaluate_record(self, record):
        # 返回标量分数
        return 0
```
- 通过异步上下文管理器自动领取/归还任务：
```python
async with MySampler(cfg) as sampler:
    await sampler.run()
```

## 推理服务生命周期
- 每个 DP 组内，TP 主进程在需要采样时启动 SGLang 子进程（`subprocess.Popen`），并在 Mongo 中注册 `InferenceService`。
- 采样结束后释放显存，终止子进程，再恢复训练；异常时会尝试终止并回收子进程。
- 服务配置（端口、并行度、显存占比）由训练参数中的 `inf_tp_size`、`inf_ep_size`、`inf_mem_ratio`、`target_concurrency` 等控制。

## 常用调试步骤
- 确认 Mongo 中存在可用 `Task`，且 `DispatchedSamplingTask` 状态随时间更新。
- 观察推理进程日志，检查是否有端口被占用或模型加载失败。
- 若采样阻塞，尝试降低 `max_new_tokens`、`num_generations`，或提升 `target_concurrency`。
- 采样结果空/质量差：调整 `evaluate_record` 打分逻辑，或在数据库侧过滤低分样本。

## 进阶定制
- 需要接入其他推理后端（如本地 API 或云端服务）时，可修改 [src/training/inference.py](../src/training/inference.py) 中 `InferenceManager` 相关逻辑。
- 如需自定义 `Record → 模型输入` 转换，可在 [src/main.py](../src/main.py) 中替换 `convert_record_to_data_func`。
- 对多模态样本的预处理可参考 [src/training/datasets.py](../src/training/datasets.py) 的 `preprocess_mm_messages_for_sample`。

## OSWorld 数据集接入 SamplingScheduler 示例
下面演示如何将 OSWorld 数据集包装为 `OSworldDataset` 并交给 `SamplingScheduler` 运行采样循环。

```python
from rollout.scheduler import SamplingScheduler
from rollout.osworld import OSWorldStatefulSampler, OSworldDataset
from configs import AgentTrainingConfig

# 准备训练/采样配置；根据实际环境填写 Mongo/MinIO 连接与并发控制
cfg = AgentTrainingConfig(
    db_connection_string="mongodb://localhost:27017",
    oss_connection_string="http://minio:9000",
    train_file="assets/osworld_dataset.jsonl",  # 也可用 osworld_filtered.jsonl 等
    target_concurrency=1,
    max_concurrent_samples_per_process=1,
)

# 读取数据集并构造成 Task 实例
dataset = OSworldDataset(cfg.train_file)

# 构建调度器（此处使用状态增强版采样器）
scheduler = SamplingScheduler(
    config=cfg,
    sampler_class=OSWorldStatefulSampler,
    dataset=dataset,
    split="train",
)

# 启动采样循环；内部会按 epoch 推送 Task 并调度 sampler
scheduler.run()
```

要点：
- `OSworldDataset` 会逐行读取 JSON/JSONL，生成含 `task_config` 与 `instruction` 的 `OSworldTask`。`task_config["proxy"]` 会自动设为 `True`。
- `SamplingScheduler` 会在当前 split 的分布式锁下逐 epoch 推送 Task；当 `max_concurrent_samples_per_process` 容量未满且推理服务负载低于 `target_concurrency` 时启动新的 sampler。
- 如果需要提前启动 OSWorld 虚拟环境，可参考 [src/sampling.py](../src/sampling.py#L5-L31) 中 `setup_osworld_envs` 的用法，仅在 `LOCAL_RANK==0` 时拉起环境以避免重复。
