#!/bin/bash
set -e

# Ensure Node.js and npm are available
export PATH="$PATH:/usr/bin:/usr/local/bin"

# Display environment info
echo "Node.js version: $(node --version)"
echo "NPM version: $(npm --version)"
echo "Python version: $(python --version)"

# Check if config file exists
CONFIG_FILE="/app/config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file $CONFIG_FILE not found, using default config"
fi

# Activate conda environment
source /opt/conda/bin/activate mcp-agent

# Display Python version after activation
echo "Activated Python version: $(python --version)"
echo "Checking for uvicorn: $(which uvicorn || echo 'Not found')"

# Output current directory for debugging
echo "Current directory: $(pwd)"

# Set environment variables
export PYTHONPATH=/app:$PYTHONPATH
export CODE_STORAGE_DIR=/app/code_storage
export CONDA_ENV_NAME=mcp-executor
export UV_PATH=/usr/local/bin/uv

# Create necessary directories
mkdir -p /app/mcp_server_logs

# Start main server
echo "Starting main server..."
cd /app
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /app/mcp_server_logs/main_server.log 2>&1 &

# Wait for main server to start
sleep 5

# Start Streamable HTTP MCP server
echo "Starting Streamable HTTP MCP server..."
cd /app
python -m uvicorn streamable_http_server:app --host 0.0.0.0 --port 8088 --log-level debug

# Execute passed commands
exec "$@"
