# RL / GRPO Guide

This guide focuses on **online sampling + RL training** in AgentRL. It applies to GRPO / GSPO / MINIRL / CISPO via `--loss_calculater`.

## High-level flow
1. Prepare `Task`s in MongoDB (or custom task types). During training, AgentRL creates `DispatchedSamplingTask` and generates `Record` / `DBRecordData`.
2. Training launches SGLang inference subprocesses per TP group, serves sampling requests, and writes results back to DB; the training loop keeps fetching trajectories from DB and optimizing.
3. Optionally enable MinIO (`--enable_oss`) for large objects or multi-modal data (`--oss_connection_string`).

## Prerequisites
- Dependencies installed and basic self-check passed.
- MongoDB reachable (`--db_connection_string`).
- Optional: MinIO configured (`--oss_connection_string --enable_oss true`).
- Model weights available; inference GPU budget controlled by `--inf_mem_ratio`.

## Key arguments
- Sampling: `enable_sampling`, `sync_sampling`, `num_generations`, `max_new_tokens`, `max_prompt_tokens`, `dynamic_batching`, `target_concurrency`, `max_concurrent_samples_per_process`
- Parallelism: training `tp_size/pp_size/ep_size/cp_size`, inference `inf_tp_size/inf_ep_size`
- Loss: `loss_calculater` (GRPO/GSPO/MINIRL/CISPO), `epsilon`, `epsilon_higher`, `beta1`, `beta2`, `importance_weight_cap_ratio`
- Filtering: `drop_zero_advantage`, `skip_non_positive_advantage`, `minimal_advantage`, `max_trained_count`, `retrained_interval`
- Memory: inference `inf_mem_ratio`; training `loss_seq_chunk_size`, `activation_offloading`

## Example (single node, multi-GPU)

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

Notes:
- Training starts inference services on TP leader processes for each DP group. If `eval_strategy != no`, it will additionally start evaluation sampling processes.
- You can also run sampling separately to write data into DB without coupling it to the training process.

## Data preparation
- By default, training pulls `Record`s from MongoDB and converts them to training data. See models in `../src/databases/sampler.py`.
- For custom task types, follow [sampling.md](sampling.md) to register new `Task`s and write tasks ahead of time.
- Multi-modal samples automatically convert image URLs to base64 during loading (see `../src/training/datasets.py`).

## Monitoring and troubleshooting
- Inference service status is recorded as `InferenceService` documents in MongoDB (ports/status/metadata).
- If sampling stalls, check whether `target_concurrency` / `max_concurrent_samples_per_process` is too small and whether inference services are alive.
- If training OOMs: lower `per_device_train_batch_size` or enable parallelism (TP/CP/PP). If inference OOMs: lower `inf_mem_ratio` or `max_new_tokens`.
- If no results for a long time: make sure the `Task` collection is not empty; check network/auth config.

## Cleanup
- On normal exit, training cleans up inference subprocesses automatically. If interrupted, terminate the related Python processes, or reset service states in MongoDB if needed.

