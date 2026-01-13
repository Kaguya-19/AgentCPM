#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentCPM QuickStart Script

This script helps you quickly set up and run a custom demo task using the AgentCPM framework.
It creates a temporary benchmark environment and executes the evaluation pipeline.

Usage:
1. Modify the [USER CONFIGURATION] section below with your API key, model, and query.
2. Run this script: `python quickstart.py`
3. Check the output directory for the execution trace (dialog.json, etc.).
"""

import os
import sys
import json
import subprocess
import stat
from pathlib import Path

# =====================================================================================
# --- USER CONFIGURATION SECTION ---
# Please modify the variables below before running.
# =====================================================================================

# 1. The Task Query
# Define the task you want the agent to perform.
QUERY = "Check today's arXiv computer science papers (cs.AI or cs.CL) and list the titles of the top 3 most interesting ones."

# 2. LLM Configuration
# Set your Model Name, API Key, and Base URL.
MODEL_NAME = ""                # e.g., "gpt-4o", "deepseek-chat", "claude-3-5-sonnet"
API_KEY = "sk-your-api-key-here"     # Your API Key
BASE_URL = "https://api.openai.com/v1" # API Base URL
PROVIDER = "openai"                  # Provider name (used for internal format adaptation, e.g., "openai")

# 3. System Configuration
# URL where your Tool Server (MCP Manager) is running.
MANAGER_URL = "http://localhost:8000"

# Directory where the execution results (logs, dialog.json) will be saved.
OUTPUT_DIR = "./outputs"

# 4. Processor Model Configuration (Optional)
# The processor model handles specialized tasks (e.g., summarizing browser content).
# By default, it reuses the Main Model configuration for quickstart, we recommend using a different model,like qwen3-14B.
# Set USE_SEPARATE_PROCESSOR = True to use a different model.
USE_SEPARATE_PROCESSOR = False

# Only if USE_SEPARATE_PROCESSOR is True, configure these:
PROCESSOR_MODEL_NAME = "gpt-3.5-turbo"
PROCESSOR_API_KEY = "sk-..."
PROCESSOR_BASE_URL = "https://api.openai.com/v1"
PROCESSOR_PROVIDER = "openai"

# =====================================================================================
# --- INTERNAL SCRIPT LOGIC (Do not modify unless necessary) ---
# =====================================================================================

# Paths
ROOT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT_DIR / "AgentToLeaP" / "benchmarks" / "quickstart"
DATA_FILE = BENCHMARK_DIR / "quickstart.jsonl"
RUN_SCRIPT = BENCHMARK_DIR / "run.sh"

def create_benchmark_dir():
    """Create the temporary benchmark directory if it doesn't exist."""
    if not BENCHMARK_DIR.exists():
        os.makedirs(BENCHMARK_DIR)
        print(f"[Info] Created benchmark directory: {BENCHMARK_DIR}")

def create_task_file():
    """Generate the JSONL task file containing the user query."""
    # Annotator result is empty because we skip the scoring phase.
    task_data = {
        "task_id": "quickstart_demo_001",
        "Question": QUERY,
        "annotator_result": "", 
        "file_name": "",
        "level": 1
    }
    
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(task_data, ensure_ascii=False) + "\n")
    print(f"[Info] Generated task file: {DATA_FILE}")

def create_run_script():
    """Generate the run.sh script with the user configuration."""
    
    # Determine Processor Configuration
    if USE_SEPARATE_PROCESSOR:
        p_provider = PROCESSOR_PROVIDER
        p_model = PROCESSOR_MODEL_NAME
        p_base_url = PROCESSOR_BASE_URL
        p_api_key = PROCESSOR_API_KEY
        print(f"[Config] Using SEPARATE Processor Model: {p_model}")
    else:
        # Reuse Main Model configuration
        p_provider = PROVIDER
        p_model = MODEL_NAME
        p_base_url = BASE_URL
        p_api_key = API_KEY
        print(f"[Config] Using Main Model as Processor: {p_model}")

    # We use --skip-post-eval to avoid the automatic scoring phase since we don't have ground truth.
    script_content = f"""#!/bin/bash

# =====================================================================================
# --- QuickStart Configuration (Auto-Generated) ---
# =====================================================================================

export PROVIDER="{PROVIDER}"
export MODEL_NAME="{MODEL_NAME}"
export RESULT_DIR_NAME="{MODEL_NAME}_quickstart"
export BASE_URL="{BASE_URL}"
export API_KEY="{API_KEY}"

# --- Processor Model Config ---
export PROCESSOR_PROVIDER="{p_provider}"
export PROCESSOR_MODEL_NAME="{p_model}"
export PROCESSOR_BASE_URL="{p_base_url}"
export PROCESSOR_API_KEY="{p_api_key}"

# --- System Config ---
export MANAGER_URL="{MANAGER_URL}"
export EVALUATION_ROOT_DIR="{str(Path(OUTPUT_DIR).resolve()).replace(os.sep, '/')}"
export FILES_DIR=""

# --- Parameters ---
export PASS_K=1
export TEMPERATURE=1
export TOP_P=1
export MAX_TOKENS=4096
export MAX_INTERACTIONS=30

# --- Switches ---
export USE_BROWSER_PROCESSOR="true"
export RETURN_THOUGHT_TO_LLM="true"
export USE_CONTEXT_MANAGER="false"
export USE_LLM_JUDGE="false"

# --- Constants ---
export TOOL_START_TAG="<tool_call>"
export TOOL_END_TAG="</tool_call>"
# Get the directory of this script
export SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export BENCHMARK_NAME="quickstart"
export INPUT_FILE="$SCRIPT_DIR/quickstart.jsonl"
export RAW_OUTPUT_DIR="$EVALUATION_ROOT_DIR/quickstart_results"

# --- MongoDB (Placeholder) ---
export MONGO_URI=""
export MONGO_DB=""
export MONGO_COL="quickstart_logs"
export MONGO_CREATE_INDEX="0"

# =====================================================================================
# --- Execution ---
# =====================================================================================
echo "========================================="
echo "Running QuickStart Evaluation"
echo "Model: $MODEL_NAME"
echo "Output Dir: $RAW_OUTPUT_DIR"
echo "========================================="

# Call run_evaluation.py from the root/evaluation directory
# Added --skip-post-eval to skip scoring

python "$SCRIPT_DIR/../../run_evaluation.py" \\
    --provider "$PROVIDER" \\
    --model "$MODEL_NAME" \\
    --model-name "$RESULT_DIR_NAME" \\
    --base-url "$BASE_URL" \\
    --api-key "$API_KEY" \\
    --processor-provider "$PROCESSOR_PROVIDER" \\
    --processor-model "$PROCESSOR_MODEL_NAME" \\
    --processor-base-url "$PROCESSOR_BASE_URL" \\
    --processor-api-key "$PROCESSOR_API_KEY" \\
    --benchmark-name "$BENCHMARK_NAME" \\
    --manager-url "$MANAGER_URL" \\
    --input-file "$INPUT_FILE" \\
    --output-dir "$RAW_OUTPUT_DIR" \\
    --output-base-dir "$EVALUATION_ROOT_DIR" \\
    --max-samples 1 \\
    --num-processes 1 \\
    --max-interactions "$MAX_INTERACTIONS" \\
    --tool-start-tag "$TOOL_START_TAG" \\
    --tool-end-tag "$TOOL_END_TAG" \\
    --files-dir "$FILES_DIR" \\
    --use-browser-processor \\
    --return-thought \\
    --skip-post-eval \\
    --k "$PASS_K" \\
    --temperature "$TEMPERATURE" \\
    --top-p "$TOP_P" \\
    --max-tokens "$MAX_TOKENS"

echo "Done. Results saved to $RAW_OUTPUT_DIR"
# Pause to let user see the output
read -p "Press any key to exit..."
"""
    
    # Write run.sh
    # Ensure line endings are LF for bash compatibility
    with open(RUN_SCRIPT, "w", encoding="utf-8", newline='\n') as f:
        f.write(script_content)
    
    # Make executable on Unix/Linux/Mac
    try:
        st = os.stat(RUN_SCRIPT)
        os.chmod(RUN_SCRIPT, st.st_mode | stat.S_IEXEC)
    except Exception:
        pass
    
    print(f"[Info] Generated run script: {RUN_SCRIPT}")

def main():
    print("========================================")
    print("      AgentCPM QuickStart Script        ")
    print("========================================")
    
    # Validation
    if not API_KEY or "your-api-key" in API_KEY:
        print("[Error] API_KEY is missing or invalid.")
        print("        Please edit this script and set your API_KEY in the configuration section.")
        sys.exit(1)

    if not MODEL_NAME:
        print("[Error] MODEL_NAME is missing.")
        print("        Please edit this script and set the MODEL_NAME (e.g., 'gpt-4o').")
        sys.exit(1)

    print(f"[Config] Model: {MODEL_NAME}")
    print(f"[Config] Query: {QUERY}")
    print("\n[Status] Setting up environment...")
    
    create_benchmark_dir()
    create_task_file()
    create_run_script()

    print("\n[Status] Executing run script...")
    print(f"Script Path: {RUN_SCRIPT}")
    
    # Execute run.sh
    try:
        if os.name == 'nt':
            # Try running with bash (requires Git Bash or WSL on Windows)
            print("[Info] Attempting to run with 'bash'...")
            subprocess.check_call(["bash", str(RUN_SCRIPT)])
        else:
            subprocess.check_call([str(RUN_SCRIPT)])
            
    except FileNotFoundError:
        print("\n[Error] 'bash' command not found.")
        print("If you are on Windows, please ensure Git Bash is installed and in your PATH.")
        print("Alternatively, run the generated script manually from a bash terminal:")
        print(f"bash {RUN_SCRIPT}")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Execution failed with exit code: {e.returncode}")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

