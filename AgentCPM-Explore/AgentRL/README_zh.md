# AgentRL

<p align="center">
  【中文 | <a href="README.md">English</a>】
</p>

AgentRL是面向 LLM/VLM 的分布式强化学习与监督微调训练栈，覆盖采样、训练、推理服务和数据库存储全链路，支持 FSDP2/TP/PP/EP、SGLang 推理和 MongoDB 作为数据后端。我们在此基础上**完成了智能体多轮交互训练支持**，使得训练框架能够通过 MCP 协议连接和管理工具沙盒，实现多模型、多工具协作的智能体强化学习训练。

## 核心特性 (Key Features)

### 🛡️ 多轮次交互训练

支持智能体在训练中进行多轮工具调用与复杂状态交互。框架兼容**训练时启用上下文工程策略**，支持将**模型原生输出**直接作为上下文参与训练，有效提升智能体训练时的鲁棒性。

### 🔌 MCP 协议集成

通过 **MCP 协议**统一管理工具沙盒。利用 **MCP-Docker** 实现工具服务的标准化部署与连接，自定义工具可直接作为 RL 训练环境的一部分，**无需为每个工具编写特定适配器**。

### ⚡ 全异步训推解耦

* **训推完全解耦**：采用**训练与采样完全解耦**的设计。训练组件仅流式消费轨迹数据，不依赖采样过程的生命周期。采样端支持独立启动、停止、扩展和重启，为异构集群部署提供了极高的灵活性与可扩展性。
* **全异步流水线**：支持在**同一 GPU 上并发运行**异步采样和训练管道，显著提升硬件资源利用率。可启用框架内置轨迹新鲜度控制和重要性权重约束机制，有效缓解全异步场景下的 Off-Policy 分布漂移，确保训练稳定性。

### 🚀 极致性能与全链路审计

* **前缀合并加速**：引入类似推理端 KV Cache 的训练侧计算复用机制。系统统一管理多轮轨迹中的共享上下文，将 ReAct 等逐步推理型智能体的**多次计算压缩为单次前向传播**，大幅减少冗余计算并提升训练效率。
* **原生并行支持**：全面支持 PyTorch 原生并行生态，兼容 **FSDP2、Tensor Parallel (TP)、Context Parallel (CP)** 等多种策略，具备 **128K+ Tokens** 的超长上下文训练能力。
* **全链路审计**：构建可信数据闭环，所有采样轨迹、环境反馈及模型决策均**结构化持久存储**。支持对训练全过程进行精准回溯、调试与异常诊断。

## 安装

### 1 AgentRL Python 环境安装

1) 安装依赖（必要，推荐使用 uv）
```bash
uv init
uv pip install -r requirements.txt --no-build-isolation
```

2) 基础自检（无 GPU 亦可）
```bash
python -c "import sys; sys.path.insert(0, 'src'); import models, sampler; print('All dependencies working')"
```

3) 可选验证
- GPU：`python tests/dtensor_bench.py --world-size 1 --warmup 1`
- 数据库：`timeout 10 python tests/mongo_connection_test.py`

### 2 AgentRL-MCP 环境配置准备

完成 AgentRL 基础安装后，需要进行以下采样环境配置：

#### 2.1 服务部署

##### 2.1.1 MongoDB 部署（必需）

根据 `src/databases/deploy/mongo/docker-compose.yml` 部署 MongoDB 服务：

```bash
cd src/databases/deploy/mongo
docker-compose up -d
```

默认配置：

- 端口：27017
- 用户名：root（请根据实际环境修改）
- 密码：password（请根据实际环境修改）

##### 2.1.2 MCP-Docker 服务部署

请根据主项目 README 的说明部署 MCP-Docker 服务：

- 参考主 README 的 [QuickStart 部分](../README_zh.md#quickstart) 了解 MCP-Docker 部署方法
- 启动 MCP-Docker 服务
- 配置 MCP 工具服务器地址（用于训练脚本中的 `--mcp_manager_url` 参数，例如 `http://localhost:8000/mcpapi`）
- 确保 MCP 服务正常运行并可访问

##### 2.1.3 Browse Agent 部署（可选）

如果需要使用本地模型作为 Browse Agent（例如 SGLang），可以参考 `src/databases/deploy/sglang/docker-compose.yml` 部署：

```bash
cd src/databases/deploy/sglang
# 根据实际需求修改 docker-compose.yml 中的模型路径、端口等配置
docker-compose up -d
```

**注意**：如果使用 API 模型作为 Browse Agent，则无需部署此服务，只需在 `assets/agent_config_example.yml` 中配置 API 密钥和端点即可。

#### 2.2 Agent 配置文件准备

所有 agent 配置（browse agent 和 scorer agent）都在 `assets/agent_config.yml`中管理。配置文件参考 [`assets/agent_config_example.yml`](assets/agent_config_example.yml)。

##### 2.2.1 配置文件说明

- **配置文件示例**： `assets/agent_config_example.yml`
- **配置内容**：Browse Agent 和 Scorer Agent 的模型 API 密钥、端点、提示模板等

##### 2.2.2 Browse Agent 配置

- **功能**：通过 MCP 协议调用网页浏览、代码执行等工具，并对结果进行摘要
- **配置项**：
  - 模型 API 密钥和端点（支持多个模型，具有故障转移支持）
  - 提示长度限制（`max_prompt_length`, `min_prompt_length`）
  - 工具特定设置（`browse_tools`, `default_purpose`）
  - 摘要状态开关（`status`）

##### 2.2.3 Scorer Agent 配置

- **功能**：评估任务答案质量（在启用 LLM judge 时需要配置）
- **配置项**：
  - **模型配置**：`models` 列表，支持多个模型配置，具有故障转移支持
    - 每个模型需要 `api_key`、`base_url` 和 `model` 字段
    - 框架会按顺序尝试，直到成功或所有模型都失败
  - **Judge prompt 模板**：`judge_prompt_template`（可选）
    - 用于自定义 LLM judge 的评估提示词
    - 如果不提供，使用默认模板（见 [`src/rollout/mcp/scorer.py`](src/rollout/mcp/scorer.py)）

**注意**：关于如何在任务中指定 scorer 和自定义 scorer，请参考 [2. 数据准备 - 方式一：使用预定义的 MCP 采样器](#方式一使用预定义的-mcp-采样器推荐) 部分。

##### 2.2.4 本地模型配置

对于本地主机模型（例如 SGLang），将 `base_url` 配置为指向您的本地模型服务端点。框架将自动使用 `agent_config.yml` 中配置的模型，而不是环境变量。

**本地模型服务部署**：

- 参考 [2.1.3 Browse Agent 部署](#213-browse-agent-部署可选) 部分使用 `src/databases/deploy/sglang/docker-compose.yml` 部署 SGLang 服务
- 根据 `docker-compose.yml` 中的配置，设置 `base_url` 为对应的服务地址（例如：`http://localhost:38889`）
- 确保 `agent_config.yml` 中的 `served-model-name` 与 docker-compose 中的 `--served-model-name` 参数一致

## 训练

AgentRL-MCP 基于 AgentRL 框架，**通过 MCP 协议支持工具调用的强化学习训练**。按照以下步骤进行多节点、多 GPU 的 RL 训练：

### 1. 服务依赖确认

确保以下服务已部署并正常运行：

- **MongoDB**: 用于存储采样任务和训练轨迹（已在 AgentRL 安装步骤中配置）
- **MCP-Docker**: **AgentRL-MCP 核心依赖**，用于通过 MCP 协议管理工具沙盒和工具调用采样服务

确保 Agent 服务已配置（配置参考 `assets/agent_config_example.yml`，详细说明见 [2.2 Agent 配置文件准备](#22-agent-配置文件准备)）。

### 2. 数据准备

#### 方式一：使用预定义的 MCP 采样器（推荐）

**这是 AgentRL-MCP 的核心特性**：通过 MCP 协议连接工具沙盒进行采样。

准备包含以下字段的 JSONL 格式数据集：

- `id`: 任务唯一标识符
- `query`: 用户查询/问题（支持工具调用任务）
- `answer`: 期望答案（可选）
- `scorer`: 评分器名称（可选，默认为 `"agentcpm"`）
- `answer_schema`: 答案提取格式（可选，默认为 `"answer"`）

示例：

```json
{"id": "task_001", "query": "什么是机器学习？", "answer": "机器学习是...", "scorer": "agentcpm", "answer_schema": "answer"}
{"id": "task_002", "query": "计算 2+2 等于多少？", "answer": "4", "scorer": "math", "answer_schema": "boxed"}
```

**answer_schema 支持**：

- `"answer"`（默认）：使用 XML 标签格式 `<answer>...</answer>` 提取答案
- `"boxed"`：使用 LaTeX 格式 `\boxed{...}` 提取答案（适用于数学问题）
  - 也支持 `\fbox{...}` 格式
  - 自动匹配嵌套的大括号


**在 Task 中指定 Scorer**：

框架支持为每个任务单独指定 scorer，实现同一训练中不同类型任务使用不同的评分策略：

1. **在 JSONL 数据中指定**：通过 `scorer` 字段指定评分器名称
   - 如果任务数据中没有 `scorer` 字段，默认使用 `"agentcpm"`
   - 示例：`{"id": "task_001", "query": "问题", "answer": "答案", "scorer": "agentcpm"}`

2. **内置 Scorer**：
   - `agentcpm`：使用 LLM judge 进行答案评估（默认）

3. **自定义 Scorer**：

可以通过 `MCPScorerFactory.register_scorer()` 注册自定义 scorer。详细示例和用法请参考 [`src/rollout/mcp/scorer.py`](src/rollout/mcp/scorer.py) 中 `MCPScorerFactory.register_scorer()` 方法的文档注释。

**使用场景**：

- **多任务类型训练**：同一训练中可以混合不同类型的任务，每种任务使用最适合的 scorer，有效提高训练精度
  - 例如：`math` 任务使用数学专用 scorer，`deepresearch` 任务使用LLM judge scorer
- **渐进式评估**：可以根据任务难度或阶段选择不同的评估策略

详细实现参考 [`src/rollout/mcp/scorer.py`](src/rollout/mcp/scorer.py) 和 [`src/rollout/mcp/_rewarding.py`](src/rollout/mcp/_rewarding.py)。

#### 方式二：自定义采样器

在 `src/rollout` 目录下实现自定义采样器，继承 `AsyncSampler` 并实现以下方法：

- `run()` 方法：定义采样逻辑
- `evaluate_record()` 方法：评估采样结果并返回分数

如果需要自定义评分逻辑，可以在 `evaluate_record()` 中直接实现，或者通过 `MCPScorerFactory.register_scorer()` 注册自定义的 scorer 函数（scorer 是一个异步函数，接受 `model_answer` 和 `ground_truth` 参数，返回 float 分数）。

### 3. 配置训练脚本

参考此目录中的示例脚本获取完整的配置示例：

- **MINIRL 训练**: [`minirl_example.sh`](minirl_example.sh) - 使用 MINIRL 损失计算器
- **GRPO 训练**: [`grpo_example.sh`](grpo_example.sh) - 使用 GRPO 损失计算器
- **评估**: [`eval_example.sh`](eval_example.sh) - 用于模型评估

**推荐设置：**

- MINIRL: `--loss_calculater "MINIRL"`, `--token_level_loss false`, `--strict_in_bound true`, `--skip_length_normalization true`
- GRPO: `--loss_calculater "GRPO"`, `--token_level_loss true`

**注意**: 复制其中一个示例脚本，并根据您的环境修改占位符值（标记为 `<YOUR_...>`），包括：

- 多节点多 GPU 配置（Accelerate 设置）
- 模型和数据路径 `--model_name_or_path`
- Mongo URL：`--db_connection_string`
- MCP manager URL：`--mcp_manager_url` 
- Agent 配置：见 `--agent_config_path`
- 训练/推理超参数支持：详见 `src/configs.py`

### 4. 启动训练

确保您在 AgentRL 目录下，然后运行：

```bash
# 用法: ./minirl_example.sh <RUN_NAME>
# 或使用 GRPO: ./grpo_example.sh <RUN_NAME>
./minirl_example.sh my_rl_training_run
```

**训练输出说明**：

- **训练日志**：输出到 `logs/as/${RUN_NAME}_train_${MACHINE_RANK}.log`
  - 包含训练过程中的详细日志信息
  - 每个节点（MACHINE_RANK）都有独立的日志文件

- **模型检查点（Checkpoint）**：保存到 `output/${RUN_NAME}/`
  - 根据 `--save_steps` 参数定期保存模型检查点
  - 包含模型权重、优化器状态等训练状态信息
  - 可用于恢复训练或模型推理

- **训练轨迹和记录（Trajectory/Record）**：保存在 MongoDB 数据库中
  - 数据库名称：`${RUN_NAME}`（与运行名称相同）
  - 包含采样任务、执行轨迹、工具调用记录等完整训练数据
  - 可通过 MongoDB 客户端(Mongo Compass)或训练框架的数据接口访问
  - **数据链**：数据流转遵循 `Task → Record → DispatchedSamplingTask → DBRecordData` 的完整链路
    - `Task`：待采样任务的定义，存储在 MongoDB 中
    - `Record`：采样进程领取 `Task` 后创建的记录，用于存储该任务的完整多轮交互轨迹
    - `DispatchedSamplingTask`：采样过程中生成的派发记录，包含请求和响应，标记状态与分配，被添加到 `Record.traj` 中
    - `DBRecordData`：用于训练的数据格式，从 Record 转换而来，供数据集迭代使用


### 5. 训练流程

1. **初始化**: 加载模型，配置分布式训练
2. **采样**: 启动推理服务，使用 MCPSampler 进行数据采样
   - **多轮交互支持**：采样器通过 MCP 协议支持多轮工具调用，智能体可以在单次任务中进行多轮交互，每轮可以调用不同的工具并基于前一轮的结果继续执行
   - **上下文管理**：框架自动维护多轮对话的上下文，确保工具调用和响应的连贯性
3. **训练**: 从数据库读取采样结果（包括完整的多轮交互轨迹），计算损失并更新模型
4. **迭代**: 重复采样和训练步骤，持续优化多轮交互能力

## 参考资源

- 主项目文档: [`../README_zh.md`](../README_zh.md)
- 参数配置: `src/configs.py` 
- 示例脚本: `minirl_example.sh`, `grpo_example.sh`, `eval_example.sh`
- Accelerate 配置: `assets/fsdp2_dst.yml`
- Agent 配置: `assets/agent_config_example.yml`
