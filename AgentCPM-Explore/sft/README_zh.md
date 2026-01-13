# AgentCPM SFT 训练（TRL + Accelerate + DeepSpeed）

<p align="center">
  【中文 | <a href="README.md">English</a>】
</p>

本目录提供一个用于因果语言模型的监督微调（SFT）训练脚本，基于：

- Hugging Face Transformers（`Trainer`）
- TRL（`SFTConfig`）
- Accelerate（启动器 + 分布式运行时）
- DeepSpeed ZeRO-3（推荐）
- 可选：SwanLab 实验跟踪

训练采用 **选择性监督（selective supervision）**：只对特定 assistant 片段范围内的 token 计算监督损失；其余 token 会被置为 `ignore_index = -100` 并在 loss 中屏蔽。

---

## 环境要求

- Python 3.10+
- `torch`, `transformers`, `datasets`, `trl`, `accelerate`, `deepspeed`, `numpy`, `pandas`, `tqdm`
- （可选）`swanlab`

安装依赖（示例）：

```bash
pip install torch transformers datasets trl accelerate deepspeed numpy pandas tqdm
pip install swanlab  # 可选
```

---

## 数据格式

脚本输入为一个或多个 **pickle** 文件。每个 pickle 文件应包含一个 Python `list[str]`，其中每个字符串是一条**已完整渲染为 chat 格式的样本**（即已经包含你希望使用的 chat template）。

代码中的示例：

```python
sft_data_path_list = ["./input/sft_data_001.pkl", "./input/sft_data_002.pkl"]
```

---

## 快速开始

### 1) 配置 Accelerate（DeepSpeed ZeRO-3）

推荐只需运行一次（建议选择 DeepSpeed ZeRO Stage 3）：

```bash
accelerate config
```

如果你使用自定义的 Accelerate 配置文件：

```bash
accelerate launch --config_file path/to/accelerate_config.yaml AgentCPM_SFT.py
```

或使用默认配置：

```bash
accelerate launch AgentCPM_SFT.py
```

---

## 关键参数

所有参数都定义在 `AgentCPM_SFT.py` 的 `main()` 中。

### 模型与 I/O

- `model_path`：本地模型路径或 Hugging Face 模型名
- `experiment_name`：实验/运行标识
- `output_path`：checkpoint 与日志输出目录
- `sft_data_path_list`：pickle 文件列表

### 监督模式

collator 支持两种模式：

- `supervise_mode="all"`：监督所有由 marker 标记出来的 assistant 片段
- `supervise_mode="last"`：只监督最后一个 assistant 片段

### Markers（标记符）

默认值：

- `loss_start_token = "<|im_start|>assistant"`
- `loss_end_token   = "<|im_end|>"`

只有位于上述 markers 之间的 token 才会写入 `labels`；其他位置会被设为 `-100`。

---

## （可选）SwanLab

如需启用 SwanLab，可通过环境变量登录并启动脚本：

```bash
export SWANLAB_API_KEY="YOUR_KEY"
accelerate launch AgentCPM_SFT.py
```

当检测到 `SWANLAB_API_KEY` 被设置时，脚本会调用 `swanlab.login(api_key=os.getenv("SWANLAB_API_KEY"), save=False)`。

---

## 数据示例

（示例内容与英文版一致）

```text
<|im_start|>system
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

## Task Strategy

1. **Analyze the user's request** to clarify the task objective, break it down into clear sub-goals, and arrange them in logical order.
2. **If the task does not require tool use, think step by step and answer the user directly.**
3. **If the task requires tool use, develop a concise step-by-step plan** (e.g., 1., 2., 3.), with each step corresponding to a specific sub-goal, obey tool-use guidelines to solve the task.

## Tool-Use Guidelines
4. **Call only one tool per step**, prioritizing the tool that best advances the current sub-goal.
5. **Tool Prioritization Rule: To access any online resource via a URL (like http:// or https://), including webpages and online PDFs, you must use the fetch_url tool. The read_file tool should only be used for local file URIs (e.g., file:///...).
6. **After each tool call, stop responding immediately** and wait for user feedback or tool results. Do not assume results or continue analysis.
7. **Extract and summarize key information from tool results** to inform the next step.
8. **Adjust your plan promptly when new information or challenges arise**, ensuring all sub-goals are covered and nothing is missed.
9. **For key conclusions, you must cross-validate using multiple tools or methods** to ensure the accuracy and consistency of the answer.
10. **After you have verified the answer, output the final answer in the specified format**.

## Answer Format
- **Answers should be direct and concise**, preferably using single words, numbers with commas and unit, or brief phrases.
- **Strictly follow the format requirements**, wrapping the final answer in `<answer>
</answer>` tags.

**Your goal: Minimize unnecessary thinking, act decisively, continuously use tools to gather information, and cross-validate with multiple tools until you can confidently provide the most concise and accurate answer.**

Where:
- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- Only one tool call is allowed per responses


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function…4077 chars truncated… must provide the final answer by turn 8, so I need to be efficient.

Let me structure my response:
</think>

Zazie Beetz played the character Domino in a film produced by 20th Century Fox. I'll search for this information to find the specific movie title.

I'll use the search tool with a query that includes Zazie Beetz, Domino, and 20th Century Fox to narrow down the results.

<tool_call>
{"name": "search", "arguments": {"query": ["Zazie Beetz Domino 20th Century Fox film"]}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"query": "Zazie Beetz Domino 20th Century Fox film", "engine": "auto", "time_ms": 369, "num_results": 10, "results": [{"title": "Zazie Beetz's Domino Dominates the First Deadpool 2 Clip - The ...", "link": "https://www.motionpictures.org/2018/05/zazie-beetzs-domino-dominates-the-first-deadpool-2-clip/", "snippet": "May 3, 2018 ... ... film, which highlights newcomer Zazie Beetz as the ... Featured image: Zazie Beetz as Domino in Twentieth Century Fox's DEADPOOL 2.", "source": "google", "metadata": {"display_link": "www.motionpictures.org"}}, {"title": "Zazie Beetz As Domino In The Final \"Deadpool 2\" Trailer Healed My ...", "link": "https://www.buzzfeed.com/alliehayes/zazie-beetz-domino-deadpool-2", "snippet": "Apr 19, 2018 ... 20th Century Fox. Because she is the ONLY big deal. I'm saying this now: Zazie Beetz as Domino is my favorite thing to happen in a movie this ...", "source": "google", "metadata": {"display_link": "www.buzzfeed.com"}}, {"title": "'Deadpool 2' Star Teases Return for MCU-set Third Movie", "link": "https://www.menshealth.com/entertainment/a40834865/deadpool-2-zazie-beetz-return-3-tease/", "snippet": "Aug 8, 2022 ... She, however, was tight-lipped about reprising the role. ryan reynolds as deadpool, zazie beetz as domino, deadpool 2. 20th Century Fox. \"Well ...", "source": "google", "metadata": {"display_link": "www.menshealth.com"}}, {"title": "Deadpool 2 - Wikipedia", "link": "https://en.wikipedia.org/wiki/Deadpool_2", "snippet": "Domino, a character with connections to Cable in the comics, was also believed to be featured in the sequel. By the first film's release, 20th Century Fox ...", "source": "google", "metadata": {"display_link": "en.wikipedia.org"}}, {"title": "REVIEW: “DEADPOOL 2” (2018) 20th Century Fox | Peggy at the ...", "link": "https://peggyatthemovies.com/2018/05/17/review-deadpool2-2018-20th-century-fox/", "snippet": "May 17, 2018 ... My personal favourite and one of the last standing through it all, is bad-ass Domino (Zazie Beetz), whose superpower is undeniably “luck”.", "source": "google", "metadata": {"display_link": "peggyatthemovies.com"}}, {"title": "Give Zazie Beetz Her Own Domino Marvel Movie, Immediately - The ...", "link": "https://www.theringer.com/2018/05/21/movies/domino-deadpool-2-zazie-beetz-marvel-movie-franchise", "snippet": "May 21, 2018 ... The best part of 'Deadpool 2' is a “lucky” superhero with whom we'd like to spend a lot more time. 20th Century Fox/Ringer illustration.", "source": "google", "metadata": {"display_link": "www.theringer.com"}}, {"title": "Review: 'Deadpool 2' Has More Swearing, Slicing and Dicing From ...", "link": "https://www.nytimes.com/2018/05/14/movies/deadpool-2-review-ryan-reynolds.html", "snippet": "May 14, 2018 ... Zazie Beetz as the fortunate Domino in “Deadpool 2.”Credit...Joe Lederer/20th Century Fox. Grief and despair drive Wade first to seek revenge ...", "source": "google", "metadata": {"display_link": "www.nytimes.com"}}, {"title": "Ryan Reynolds Shows Off Domino From 'Deadpool 2' On Twitter", "link": "https://deadline.com/2017/07/ryan-reynolds-shows-off-domino-from-deadpool-2-1202139482/", "snippet": "Jul 31, 2017 ... Ryan Reynolds tweeted out a photo today of Zazie Beetz as Domino from 20th Century Fox's \"Deadpool 2.\"", "source": "google", "metadata": {"display_link": "deadline.com"}}, {"title": "“She can really keep up with Ryan”: Deadpool & Wolverine isn't ...", "link": "https://www.imdb.com/news/ni64735550/", "snippet": "Zazie Beetz as popular X-Men Character Dominio | Twentieth Century Fox Not ... Why Didn't Deadpool & Wolverine Bring Back Zazie Beetz's Domino? While ...", "source": "google", "metadata": {"display_link": "www.imdb.com"}}, {"title": "'Deadpool 2' Director David Leitch On Domino's Powers In The Film ...", "link": "https://heroichollywood.com/deadpool-2-leitch-domino-powers/", "snippet": "Aug 1, 2017 ... Just yesterday we got our first look at Atlanta star Zazie Beetz as Domino in 20th Century Fox's upcoming sequel Deadpool 2 and now today we ...", "source": "google", "metadata": {"display_link": "heroichollywood.com"}}], "timestamp": "2025-09-15T13:46:50.475374", "success": true}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
I can see from the search results that Zazie Beetz played Domino in Deadpool 2, which was produced by 20th Century Fox. The search results consistently mention this film and her role as Domino.
</think>
```

