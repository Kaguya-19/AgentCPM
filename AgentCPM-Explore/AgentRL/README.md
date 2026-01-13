# AgentRL

<p align="center">
  【<a href="README_zh.md">中文</a> | English】
</p>

AgentRL is a distributed reinforcement learning and supervised fine-tuning training stack for LLM/VLM, covering the complete pipeline of sampling, training, inference services, and database storage, supporting FSDP2/TP/PP/EP, SGLang inference, and MongoDB as the data backend. Based on this foundation, we have **completed multi-turn interaction training support for agents**, enabling the training framework to connect and manage tool sandboxes through the MCP protocol, achieving multi-model, multi-tool collaborative reinforcement learning training for agents.

## Key Features

### 🛡️ Multi-Turn Interaction Training

Supports agents to perform multiple rounds of tool calls and complex state interactions during training. The framework is compatible with **context engineering strategies during training** and supports using **raw model outputs** directly as context for training, effectively improving the robustness of agent training.

### 🔌 MCP Protocol Integration

Unified management of tool sandboxes through the **MCP protocol**. Leverages **MCP-Docker** for standardized deployment and connection of tool services. Custom tools can be directly integrated as part of the RL training environment **without writing specific adapters for each tool**.

### ⚡ Fully Asynchronous Training-Sampling Decoupling

* **Complete Training-Sampling Decoupling**: Adopts a **fully decoupled design between training and sampling**. The training component only streams trajectory data consumption and does not depend on the lifecycle of the sampling process. The sampling side supports independent startup, shutdown, scaling, and restart, providing extremely high flexibility and scalability for heterogeneous cluster deployments.
* **Fully Asynchronous Pipeline**: Supports **concurrent asynchronous sampling and training pipelines on the same GPU**, significantly improving hardware resource utilization. Built-in trajectory freshness control and importance weight constraint mechanisms can be enabled to effectively mitigate off-policy distribution drift in fully asynchronous scenarios, ensuring training stability.

### 🚀 Ultimate Performance and Full-Link Auditing

* **Prefix Merging Acceleration**: Introduces a training-side computation reuse mechanism similar to inference-side KV Cache. The system uniformly manages shared contexts in multi-turn trajectories, compressing **multiple computations into a single forward pass** for step-by-step reasoning agents like ReAct, greatly reducing redundant computation and improving training efficiency.
* **Native Parallel Support**: Fully supports the PyTorch native parallel ecosystem, compatible with **FSDP2, Tensor Parallel (TP), Context Parallel (CP)** and other strategies, with **128K+ tokens** ultra-long context training capability.
* **Full-Link Auditing**: Establishes a trusted data closed loop where all sampling trajectories, environment feedback, and model decisions are **persistently stored in structured formats**. Supports precise backtracking, debugging, and anomaly diagnosis of the entire training process.

## Installation

### 1. AgentRL Python Environment Installation

1) Install dependencies (required, recommended using uv)
```bash
uv init
uv pip install -r requirements.txt --no-build-isolation
```

2) Basic self-check (GPU not required)
```bash
python -c "import sys; sys.path.insert(0, 'src'); import models, sampler; print('All dependencies working')"
```

3) Optional validation
- GPU: `python tests/dtensor_bench.py --world-size 1 --warmup 1`
- Database: `timeout 10 python tests/mongo_connection_test.py`

### 2. AgentRL-MCP Environment Configuration

After completing the basic AgentRL installation, the following sampling environment configuration is required:

#### 2.1 Service Deployment

##### 2.1.1 MongoDB Deployment (Required)

Deploy MongoDB service according to `src/databases/deploy/mongo/docker-compose.yml`:

```bash
cd src/databases/deploy/mongo
docker-compose up -d
```

Default configuration:

- Port: 27017
- Username: root (please modify according to your actual environment)
- Password: password (please modify according to your actual environment)

##### 2.1.2 MCP-Docker Service Deployment

Please deploy the MCP-Docker service according to the main project README:

- Refer to the [QuickStart section](../README.md#quickstart) of the main README for MCP-Docker deployment methods
- Start the MCP-Docker service
- Configure the MCP tool server address (used for the `--mcp_manager_url` parameter in training scripts, e.g., `http://localhost:8000/mcpapi`)
- Ensure the MCP service is running normally and accessible

##### 2.1.3 Browse Agent Deployment (Optional)

If you need to use a local model as a Browse Agent (e.g., SGLang), you can refer to `src/databases/deploy/sglang/docker-compose.yml` for deployment:

```bash
cd src/databases/deploy/sglang
# Modify the model path, port, and other configurations in docker-compose.yml according to actual needs
docker-compose up -d
```

**Note**: If using an API model as the Browse Agent, there's no need to deploy this service. Simply configure the API key and endpoint in `assets/agent_config_example.yml`.

#### 2.2 Agent Configuration File Preparation

All agent configurations (browse agent and scorer agent) are managed in `assets/agent_config.yml`. Reference the configuration file [`assets/agent_config_example.yml`](assets/agent_config_example.yml).

##### 2.2.1 Configuration File Description

- **Configuration file example**: `assets/agent_config_example.yml`
- **Configuration content**: Model API keys, endpoints, prompt templates, etc. for Browse Agent and Scorer Agent

##### 2.2.2 Browse Agent Configuration

- **Function**: Calls tools such as web browsing and code execution through the MCP protocol, and summarizes the results
- **Configuration items**:
  - Model API keys and endpoints (supports multiple models with failover support)
  - Prompt length limits (`max_prompt_length`, `min_prompt_length`)
  - Tool-specific settings (`browse_tools`, `default_purpose`)
  - Summary status switch (`status`)

##### 2.2.3 Scorer Agent Configuration

- **Function**: Evaluates the quality of task answers (configuration required when LLM judge is enabled)
- **Configuration items**:
  - **Model configuration**: `models` list, supports multiple model configurations with failover support
    - Each model requires `api_key`, `base_url`, and `model` fields
    - The framework will try in order until success or all models fail
  - **Judge prompt template**: `judge_prompt_template` (optional)
    - Used to customize the evaluation prompt for LLM judge
    - If not provided, uses the default template (see [`src/rollout/mcp/scorer.py`](src/rollout/mcp/scorer.py))

**Note**: For how to specify scorers in tasks and customize scorers, refer to the [2. Data Preparation - Method 1: Using Predefined MCP Sampler](#method-1-using-predefined-mcp-sampler-recommended) section.

##### 2.2.4 Local Model Configuration

For local host models (e.g., SGLang), configure `base_url` to point to your local model service endpoint. The framework will automatically use the model configured in `agent_config.yml` instead of environment variables.

**Local Model Service Deployment**:

- Refer to [2.1.3 Browse Agent Deployment](#213-browse-agent-deployment-optional) section to deploy SGLang service using `src/databases/deploy/sglang/docker-compose.yml`
- According to the configuration in `docker-compose.yml`, set `base_url` to the corresponding service address (e.g., `http://localhost:38889`)
- Ensure the `served-model-name` in `agent_config.yml` matches the `--served-model-name` parameter in docker-compose

## Training

AgentRL-MCP is based on the AgentRL framework and **supports reinforcement learning training with tool calls through the MCP protocol**. Follow these steps for multi-node, multi-GPU RL training:

### 1. Service Dependency Confirmation

Ensure the following services are deployed and running normally:

- **MongoDB**: Used for storing sampling tasks and training trajectories (configured in the AgentRL installation steps)
- **MCP-Docker**: **Core dependency of AgentRL-MCP**, used to manage tool sandboxes and tool call sampling services through the MCP protocol

Ensure the Agent service is configured (configuration reference `assets/agent_config_example.yml`, detailed instructions see [2.2 Agent Configuration File Preparation](#22-agent-configuration-file-preparation)).

### 2. Data Preparation

#### Method 1: Using Predefined MCP Sampler (Recommended)

**This is the core feature of AgentRL-MCP**: sampling through tool sandboxes connected via the MCP protocol.

Prepare a dataset in JSONL format with the following fields:

- `id`: Unique task identifier
- `query`: User query/question (supports tool call tasks)
- `answer`: Expected answer (optional)
- `scorer`: Scorer name (optional, defaults to `"agentcpm"`)
- `answer_schema`: Answer extraction format (optional, defaults to `"answer"`)

Example:

```json
{"id": "task_001", "query": "What is machine learning?", "answer": "Machine learning is...", "scorer": "agentcpm", "answer_schema": "answer"}
{"id": "task_002", "query": "Calculate what 2+2 equals?", "answer": "4", "scorer": "math", "answer_schema": "boxed"}
```

**answer_schema Support**:

- `"answer"` (default): Extracts answers using XML tag format `<answer>...</answer>`
- `"boxed"`: Extracts answers using LaTeX format `\boxed{...}` (suitable for math problems)
  - Also supports `\fbox{...}` format
  - Automatically matches nested braces


**Specifying Scorer in Tasks**:

The framework supports specifying a scorer for each task individually, enabling different types of tasks to use different scoring strategies within the same training:

1. **Specify in JSONL data**: Specify the scorer name through the `scorer` field
   - If the task data does not have a `scorer` field, defaults to `"agentcpm"`
   - Example: `{"id": "task_001", "query": "question", "answer": "answer", "scorer": "agentcpm"}`

2. **Built-in Scorers**:
   - `agentcpm`: Uses LLM judge for answer evaluation (default)

3. **Custom Scorers**:

You can register custom scorers through `MCPScorerFactory.register_scorer()`. For detailed examples and usage, refer to the documentation comments for the `MCPScorerFactory.register_scorer()` method in [`src/rollout/mcp/scorer.py`](src/rollout/mcp/scorer.py).

**Use Cases**:

- **Multi-task Type Training**: Different types of tasks can be mixed in the same training, with each type using the most suitable scorer, effectively improving training accuracy
  - For example: `math` tasks use a math-specific scorer, `deepresearch` tasks use an LLM judge scorer
- **Progressive Evaluation**: Different evaluation strategies can be selected based on task difficulty or stage

For detailed implementation, refer to [`src/rollout/mcp/scorer.py`](src/rollout/mcp/scorer.py) and [`src/rollout/mcp/_rewarding.py`](src/rollout/mcp/_rewarding.py).

#### Method 2: Custom Sampler

Implement a custom sampler in the `src/rollout` directory by inheriting `AsyncSampler` and implementing the following methods:

- `run()` method: Define sampling logic
- `evaluate_record()` method: Evaluate sampling results and return scores

If you need custom scoring logic, you can implement it directly in `evaluate_record()`, or register a custom scorer function through `MCPScorerFactory.register_scorer()` (a scorer is an async function that accepts `model_answer` and `ground_truth` parameters and returns a float score).

### 3. Configure Training Script

Refer to the example scripts in this directory for complete configuration examples:

- **MINIRL Training**: [`minirl_example.sh`](minirl_example.sh) - Uses MINIRL loss calculator
- **GRPO Training**: [`grpo_example.sh`](grpo_example.sh) - Uses GRPO loss calculator
- **Evaluation**: [`eval_example.sh`](eval_example.sh) - For model evaluation

**Recommended Settings:**

- MINIRL: `--loss_calculater "MINIRL"`, `--token_level_loss false`, `--strict_in_bound true`, `--skip_length_normalization true`
- GRPO: `--loss_calculater "GRPO"`, `--token_level_loss true`

**Note**: Copy one of the example scripts and modify the placeholder values (marked as `<YOUR_...>`) according to your environment, including:

- Multi-node multi-GPU configuration (Accelerate settings)
- Model and data paths `--model_name_or_path`
- Mongo URL: `--db_connection_string`
- MCP manager URL: `--mcp_manager_url` 
- Agent configuration: see `--agent_config_path`
- Training/inference hyperparameter support: see `src/configs.py` for details

### 4. Start Training

Ensure you are in the AgentRL directory, then run:

```bash
# Usage: ./minirl_example.sh <RUN_NAME>
# Or use GRPO: ./grpo_example.sh <RUN_NAME>
./minirl_example.sh my_rl_training_run
```

**Training Output Explanation**:

- **Training Logs**: Output to `logs/as/${RUN_NAME}_train_${MACHINE_RANK}.log`
  - Contains detailed log information during the training process
  - Each node (MACHINE_RANK) has an independent log file

- **Model Checkpoints**: Saved to `output/${RUN_NAME}/`
  - Model checkpoints are saved periodically according to the `--save_steps` parameter
  - Contains model weights, optimizer state, and other training state information
  - Can be used for resuming training or model inference

- **Training Trajectories and Records**: Saved in the MongoDB database
  - Database name: `${RUN_NAME}` (same as the run name)
  - Contains complete training data including sampling tasks, execution trajectories, tool call records, etc.
  - Can be accessed through MongoDB client (Mongo Compass) or the training framework's data interface
  - **Data Chain**: Data flow follows the complete chain `Task → Record → DispatchedSamplingTask → DBRecordData`
    - `Task`: Definition of tasks to be sampled, stored in MongoDB
    - `Record`: Record created after the sampling process claims a `Task`, used to store the complete multi-turn interaction trajectory of that task
    - `DispatchedSamplingTask`: Dispatch record generated during the sampling process, contains requests and responses, marks status and allocation, added to `Record.traj`
    - `DBRecordData`: Data format used for training, converted from Record, for dataset iteration


### 5. Training Process

1. **Initialization**: Load model, configure distributed training
2. **Sampling**: Start inference service, use MCPSampler for data sampling
   - **Multi-turn Interaction Support**: The sampler supports multi-turn tool calls through the MCP protocol. Agents can perform multiple rounds of interaction in a single task, calling different tools in each round and continuing execution based on the results of the previous round
   - **Context Management**: The framework automatically maintains context across multi-turn conversations, ensuring coherence of tool calls and responses
3. **Training**: Read sampling results from the database (including complete multi-turn interaction trajectories), calculate loss, and update the model
4. **Iteration**: Repeat sampling and training steps, continuously optimizing multi-turn interaction capabilities

## Reference Resources

- Main project documentation: [`../README.md`](../README.md)
- Parameter configuration: `src/configs.py` 
- Example scripts: `minirl_example.sh`, `grpo_example.sh`, `eval_example.sh`
- Accelerate configuration: `assets/fsdp2_dst.yml`
- Agent configuration: `assets/agent_config_example.yml`
