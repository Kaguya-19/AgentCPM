# 强化学习 / GRPO 指南

本指南聚焦基于 AgentRL 的在线采样 + RL 训练流程，包含准备、运行与调优要点。适用于 GRPO/GSPO/MINIRL/CISPO 等算法（通过 `--loss_calculater` 选择）。

## 总体流程
1. 在 MongoDB 中写入可用的 `Task`（或自定义任务类型），训练时会创建 `DispatchedSamplingTask` 并生成 `Record`/`DBRecordData`。
2. 训练端按 TP 组启动 SGLang 推理子进程，接受采样请求并写回数据库；训练循环从数据库拉取轨迹继续优化。
3. 可选接入 MinIO（开启 `--enable_oss`）用于存储大对象或多模态数据。

## 前置条件
- 依赖安装与基础自检已通过。
- MongoDB 可访问，连接串填入 `--db_connection_string`；如使用对象存储，准备好 MinIO 并配置 `--oss_connection_string --enable_oss true`。
- 具备模型权重路径，或能从远端加载模型；确保推理显存预算由 `--inf_mem_ratio` 控制。

## 关键参数
- 采样：`enable_sampling true/false`, `sync_sampling`, `num_generations`, `max_new_tokens`, `max_prompt_tokens`, `dynamic_batching`, `target_concurrency`, `max_concurrent_samples_per_process`
- 并行：训练侧 `tp_size/pp_size/ep_size/cp_size`，推理侧 `inf_tp_size/inf_ep_size`
- 损失：`loss_calculater`（GRPO/GSPO/MINIRL/CISPO），`epsilon`、`epsilon_higher`、`beta1`、`beta2`、`importance_weight_cap_ratio`
- 数据过滤：`drop_zero_advantage`、`skip_non_positive_advantage`、`minimal_advantage`、`max_trained_count`、`retrained_interval`
- 内存：`inf_mem_ratio` 控制推理进程显存占比；训练侧可结合 `loss_seq_chunk_size`、`activation_offloading`

## 运行示例（单机多卡）
```bash
torchrun --nproc_per_node=8 src/main.py \
  --model_name_or_path /path/to/model \
  --output_dir output/rl-demo \
  --db_connection_string "mongodb://user:pwd@host:27017" \
  --enable_sampling true \
  --loss_calculater GRPO \
  --inf_tp_size 1 --inf_ep_size 1 --inf_mem_ratio 0.6 \
  --num_generations 8 --max_new_tokens 8192 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --save_steps 500
```
> 训练会为每个 DP 组的 TP 主进程启动推理服务；若 `eval_strategy!=no`，会额外启动评估采样进程。
> 也可不跟随启动推理进程，单独运行采样进程向数据库中存储数据同样可行。

## 数据准备
- 默认会从 MongoDB 自动拉取 `Record`, 并转换为训练数据。参考文档模型 [src/databases/sampler.py](../src/databases/sampler.py)。
- 如需自定义任务类型，按 [docs/sampling.md](sampling_zh.md) 注册新 `Task` 并提前写入待采样任务。
- 多模态样本会在数据加载阶段自动处理图片 URL → base64（见 [src/training/datasets.py](../src/training/datasets.py)）。

## 监控与故障
- 推理服务状态由 `InferenceService` 文档记录，可在 Mongo 中查看端口/状态。
- 如采样停滞，检查 `target_concurrency` 与 `max_concurrent_samples_per_process` 是否过低，以及推理服务是否存活。
- 若训练进程显存不足，降低 `per_device_train_batch_size` 或采用多种组合并行化，例如TP、CP、PP；若推理显存不足，降低 `inf_mem_ratio` 或 `max_new_tokens`。
- 采样长时间无结果：确认 `Task` 集合非空，或检查网络/鉴权配置。

## 清理与停止
- 训练完成后会自动清理推理子进程；异常中断时可手动终止相关 Python 进程或在 Mongo 中重置服务状态。
