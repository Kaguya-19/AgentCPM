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

# Install mcp_code_executor dependencies
echo "Checking and installing mcp_code_executor dependencies..."
cd /app/mcp_servers/mcp_code_executor
npm install
npm run build

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

# Check if GAIA folder exists, create if not
if [ ! -d "/app/filesystem/GAIA" ]; then
    echo "Creating GAIA directory in filesystem..."
    mkdir -p /app/filesystem/GAIA
fi

# Copy data folder to filesystem directory if exists
if [ -d "/app/data/GAIA" ]; then
    echo "Copying GAIA data to filesystem directory..."
    cp -r /app/data/GAIA /app/filesystem/
fi

# Start mcp_code_executor server
echo "Starting mcp_code_executor server..."
cd /app/mcp_servers/mcp_code_executor
nohup node build/index.js > /app/mcp_server_logs/mcp_code_executor.log 2>&1 &

# Start filesystem server
echo "Starting filesystem server..."
nohup npx @modelcontextprotocol/server-filesystem /app/filesystem > /app/mcp_server_logs/filesystem.log 2>&1 &

# Start nlp-search-infra-server
echo "Starting nlp-search-infra-server with proxy settings..."
cd /app/mcp_servers/nlp-search-infra-server
nohup ./start_with_proxy.sh > /app/mcp_server_logs/nlp-search-infra-server.log 2>&1 &

# Wait for all sub-servers to start
echo "Waiting for all servers to start..."
sleep 5

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
