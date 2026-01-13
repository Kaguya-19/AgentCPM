#!/bin/bash

# Get port parameter, default 9000
PORT=${1:-9000}

# Set log file path
LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/uvicorn-$PORT.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Change to working directory
cd /app

# Activate conda environment
CONDA_ENV_NAME="mcp-agent"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "Activated conda env: $CONDA_ENV_NAME"
echo "Starting uvicorn server on port $PORT"
echo "Logs will be written to: $LOG_FILE"

python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" >> "$LOG_FILE" 2>&1
