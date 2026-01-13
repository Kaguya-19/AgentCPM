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

# Verify mcp_code_executor build exists (already built in Dockerfile)
if [ -f "/app/mcp_servers/mcp_code_executor/build/index.js" ]; then
    echo "✅ mcp_code_executor build found"
else
    echo "⚠️ mcp_code_executor build not found, attempting rebuild..."
    cd /app/mcp_servers/mcp_code_executor
    npm install && npm run build
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

# Start main server
echo "Starting main server..."
cd /app
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /app/mcp_server_logs/main_server.log 2>&1 &

# Wait for main server to start
sleep 10

# Final verification
echo "=== Service Startup Status Check ==="
echo "Main Server (8000): $(curl -s http://localhost:8000 > /dev/null && echo 'Running' || echo 'Not started')"

# Start Streamable HTTP MCP server
echo "Starting Streamable HTTP MCP server..."
cd /app
python -m uvicorn streamable_http_server:app --host 0.0.0.0 --port 8088 --log-level debug

# Execute passed commands
exec "$@"
