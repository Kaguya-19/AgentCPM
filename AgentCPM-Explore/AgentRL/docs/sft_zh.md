# SFT 指南

本指南覆盖在 AgentRL 中进行监督微调（Supervised Fine-Tuning）的最小配置、常见选项与运行示例。

## 适用场景
- 需要在固定标注数据上做有监督训练，关闭采样与在线评估。
- 仅使用 CrossEntropy 损失（`loss_calculater=CrossEntropy`）。
- 数据来源可以是 MongoDB 中的 `Record`，也可以是本地 JSON/JSONL 文件。

## 前置准备
1. 安装依赖并完成基础自检（见仓库根目录 README）。
2. 可选：如果从数据库读取样本，确保已按 [src/databases](src/databases) 的文档模型写入 `Record` 与 `DispatchedSamplingTask`。
3. 如果使用本地数据文件，准备 JSON/JSONL，字段需与数据管道期望的消息格式对齐（示例见 `assets` 下文件）。

## 推荐配置
- 关闭采样：`--enable_sampling false`
- 关闭在线评估：`--eval_strategy no`
- 损失函数：`--loss_calculater CrossEntropy`
- 防止重复训练同一条样本：`--max_trained_count 1`
- 根据显存设置：`--per_device_train_batch_size`、`--gradient_accumulation_steps`、`--bf16/--fp16`

## 运行示例
### 1) 使用本地文件

两种方式其一：直接传 `train_dataset` 给 Trainer（推荐方式，便于自定义）。

```python
# python examples/sft_local_dataset.py
from pathlib import Path
from training.arl import ParquetDataset
from configs import AgentTrainingConfig
from training import QwenTrainer  # 你的具体 Trainer 类，若不同请替换

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

  # 以 Parquet 为例：ParquetDataset 会产出 RecordData
  train_dataset = ParquetDataset(
    file_paths=Path("assets/math500.jsonl").with_suffix(".parquet"),
    args=args,
    split="train",
  )

  trainer = QwenTrainer(
    model=None,              # 由 Trainer 内部按 args 加载
    args=args,
    train_dataset=train_dataset,
  )
  trainer.train()

if __name__ == "__main__":
  main()
```

如果你用 JSON/JSONL 或需要完全自定义读取逻辑，也可以自定义一个数据集，确保 `__iter__`/`__getitem__` 返回的是 `RecordData` 对象：

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
        # 将 JSON 映射为 RecordData 所需字段
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

### 2) 从数据库读取（关闭采样，仅训练）
```bash
torchrun --nproc_per_node=8 src/main.py \
  --model_name_or_path /path/to/model \
  --db_connection_string "mongodb://user:pwd@host:27017" \
  --output_dir output/sft-from-db \
  --enable_sampling false \
  --loss_calculater CrossEntropy \
  --eval_strategy no
```
> 数据需提前写入 MongoDB，文档需符合 [DBRecordData](src/databases/dataset.py#L27-L95) 结构（基于 [RecordData](src/databases/dataset.py#L15-L25) 扩展），对应文档模型见 [src/databases/sampler.py](src/databases/sampler.py)。

## 数据格式提示
- 多模态样本会在数据集加载阶段自动处理图片 URL → base64（参见 [src/training/datasets.py](src/training/datasets.py)）。
- 数据库写入请直接落库为 [DBRecordData](src/databases/dataset.py#L27-L95)，不要使用临时结构，避免 Beanie 校验失败。
- 本地文件或自定义 `Dataset` 需要产出 [RecordData](src/databases/dataset.py#L15-L25) 对象（如自定义 `convert_record_to_data_func`），框架会自动转换为 `DBRecordData` 后再入库/训练。
- 读取本地 Parquet 时可复用 [ParquetDataset](src/training/datasets.py#L350-L459)，其返回即为 `RecordData`，自定义实现也应遵守该返回类型以保证数据管道兼容。
- 若自行构造本地数据集（JSON/自定义迭代器），直接将 `train_dataset` 传入 Trainer（默认逻辑见 [src/training/arl.py](src/training/arl.py#L159-L200)），并保证迭代返回 `RecordData`。
- 如需自定义样本到模型输入的转换，可在 [src/main.py](src/main.py) 中传入 `convert_record_to_data_func` 示例进行修改。

## 训练技巧
- 大模型显存吃紧时，优先考虑 `--loss_seq_chunk_size`、`--activation_offloading`。
- 若仅做单机单卡快速验证，可直接使用 `python src/main.py ...`，但分布式/并行策略会被关闭。
- 使用 `--save_total_limit` 控制 checkpoint 数量，避免磁盘占满。

## 常见问题
- **导入失败**：重新安装依赖后运行 `python -c "import sys; sys.path.insert(0, 'src'); import models, sampler"`。
- **数据为空**：确认 `train_file` 路径正确，或检查 MongoDB 集合中 `Record` 是否存在。
- **显存不足**：降低 `per_device_train_batch_size`，提高 `gradient_accumulation_steps`，或开启分布式并行参数（`tp_size/pp_size/ep_size`）。
