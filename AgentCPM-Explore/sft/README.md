# AgentCPM SFT Training (TRL + Accelerate + DeepSpeed)

<p align="center">
  【<a href="README_zh.md">中文</a> | English】
</p>

This repo contains a supervised fine-tuning (SFT) training script for a causal language model using:

- Hugging Face Transformers (`Trainer`)
- TRL (`SFTConfig`)
- Accelerate (launcher + distributed runtime)
- DeepSpeed ZeRO-3 (recommended)
- Optional experiment tracking via SwanLab

The training uses **selective supervision**: only tokens inside specific assistant spans are supervised; all other tokens are masked with `ignore_index = -100`.

---

## Requirements

- Python 3.10+
- `torch`, `transformers`, `datasets`, `trl`, `accelerate`, `deepspeed`, `numpy`, `pandas`, `tqdm`
- (Optional) `swanlab`

Install dependencies (example):

```bash
pip install torch transformers datasets trl accelerate deepspeed numpy pandas tqdm
pip install swanlab  # optional
```

---

## Data Format

The script expects one or more **pickle** files. Each pickle should contain a Python `list[str]`, where each string is a **fully rendered chat-formatted sample** (already includes the desired chat template).

Example in code:

```python
sft_data_path_list = ["./input/sft_data_001.pkl", "./input/sft_data_002.pkl"]
```

---

## Quick Start

### 1) Configure Accelerate (DeepSpeed ZeRO-3)

Run once (recommended: choose DeepSpeed ZeRO Stage 3):

```bash
accelerate config
```

If you use a custom Accelerate config file:

```bash
accelerate launch --config_file path/to/accelerate_config.yaml AgentCPM_SFT.py
```

Or use the default config:

```bash
accelerate launch AgentCPM_SFT.py
```

---

## Key Parameters

All parameters are defined inside `main()` in `AgentCPM_SFT.py`.

### Model and I/O

- `model_path`: local path or HF model name
- `experiment_name`: run identifier
- `output_path`: output directory for checkpoints/logs
- `sft_data_path_list`: list of pickle files

### Supervision Mode

The collator supports two modes:

- `supervise_mode="all"`: supervise **all** assistant spans between markers
- `supervise_mode="last"`: supervise only the **last** assistant span

### Markers

By default:

- `loss_start_token = "<|im_start|>assistant"`
- `loss_end_token   = "<|im_end|>"`

Only tokens between these markers are included in `labels`; other positions are set to `-100`.

---

## Optional: SwanLab

To enable SwanLab, log in via environment variable and run the script:

```bash
export SWANLAB_API_KEY="YOUR_KEY"
accelerate launch AgentCPM_SFT.py
```

The script will call `swanlab.login(api_key=os.getenv("SWANLAB_API_KEY"), save=False)` when `SWANLAB_API_KEY` is set.

---


## Data Example

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
{"type": "function", "function": {"name": "execute_code", "description": "Execute Python code in the conda environment. For short code snippets only. For longer code, use initialize_code_file and append_to_code_file instead.\n                Packages installed in the environment: PyPDF2 geopy PyMuPDF docx2txt pdfminer.six rdkit python-chess stockfish yfinance CoolProp seaborn python-pptx python-docx pdfplumber geopandas biopython pubchempy googletrans pyshp selenium waybackpy networkx wbdata and their dependencies and other common python packages", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "read_file", "description": "\n    universal file processing tool that converts various formats to structured markdown.\n\n    ## Supported Formats\n    local_source:\n    - Office documents: Word, Excel, PowerPoint, PDF\n    - Images: AI OCR and AI description\n    - Media: audio transcription through google api, AI video description\n    - Archives: ZIP, RAR extraction\n    online_source:\n    - Online images\n    ", "parameters": {"properties": {"uri": {"description": "The URI of the file to convert to markdown. Examples: 'file:///path/to/document.pdf'", "title": "Uri", "type": "string"}, "purpose": {"description": "The purpose of the file processing, what information you want to get from the file", "title": "Purpose", "type": "string"}, "process_type": {"default": "others", "description": "The type of the file to process.'", "enum": ["audio", "image", "video", "others"], "title": "Process Type", "type": "string"}}, "required": ["uri", "purpose"], "title": "read_fileArguments", "type": "object"}}}
{"type": "function", "function": {"name": "fetch_url", "description": "\n    Fetch webpage(s) and online pdf(s) and return the content with AI summary.\n    Supports parallel processing of multiple (at most 3) URLs. \n    ", "parameters": {"properties": {"url": {"description": "The URL(s) of the webpage(s) and online pdf(s) to visit. Can be a single URL or a list of URLs.", "items": {"type": "string"}, "title": "Url", "type": "array"}, "purpose": {"description": "The purpose of the visit, what information you want to get from the webpage", "title": "Purpose", "type": "string"}}, "required": ["url", "purpose"], "title": "fetch_urlArguments", "type": "object"}}}
{"type": "function", "function": {"name": "search", "description": "\n    Google search supports parallel processing of multiple (at most 3) queries. \n    The tool retrieves the top 10 results for each query in parallel.\n    ", "parameters": {"properties": {"query": {"description": "Array of query strings. Include multiple complementary search queries in a single call.", "items": {"type": "string"}, "title": "Query", "type": "array"}}, "required": ["query"], "title": "searchArguments", "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Which 20th Century Fox film cast Zazie Beetz as Domino?<|im_end|>
<|im_start|>assistant
<think>
First, the user asks: "Which 20th Century Fox film cast Zazie Beetz as Domino?" I need to find the film from 20th Century Fox that featured Zazie Beetz in the role of Domino.

I have access to tools like web search, so I should use the `search` tool to look up information about Zazie Beetz and her roles, specifically in 20th Century Fox films.

My plan:

1. **Think:** I need to search for Zazie Beetz's filmography, focusing on her role as Domino in a 20th Century Fox film.

2. **Act:** Use the `search` tool with a query that combines "Zazie Beetz", "Domino", and "20th Century Fox".

The query should be specific: "Zazie Beetz Domino 20th Century Fox film"

I must follow the alternate thinking and acting modes. Each turn, I need to think and then act with a tool call.

This is turn 1, so I'll start with thinking.

Finally, I must provide the final answer by turn 8, so I need to be efficient.

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

<answer>Deadpool 2</answer><|im_end|>
```

---

## Notes

- The script prints token-length percentiles during preprocessing to help you choose a reasonable `max_seq_length`.
- If your samples do not contain the marker tokens, you will get zero supervised tokens and training will not behave as intended.
