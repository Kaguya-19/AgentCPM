#!/bin/bash

# Activate conda environment
source /opt/conda/bin/activate mcp-agent

# Check Python version
python --version

# Get config file path (from env var or default)
CONFIG_FILE_PATH="${MCP_SERVER_CONFIG_PATH:-/app/config.toml}"

# Config file format detection
FILE_EXT="${CONFIG_FILE_PATH##*.}"
IS_TOML=false
if [ "$FILE_EXT" = "toml" ]; then
    IS_TOML=true
    echo "Detected TOML format config file"
    
    # Check if Python toml module is installed
    if ! python -c "import toml" &> /dev/null; then
        echo "Installing Python toml module..."
        pip install toml
    fi
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE_PATH" ]; then
    echo "Error: Config file $CONFIG_FILE_PATH not found!"
    exit 1
fi

echo "Using server config file: $CONFIG_FILE_PATH"

# Create log directory
LOG_DIR="/app/mcp_server_logs"
mkdir -p "$LOG_DIR"

# If TOML format, use Python script to parse
if [ "$IS_TOML" = true ]; then
    # Use Python to read TOML and output server list
    SERVER_NAMES=$(python -c "
import toml
config = toml.load('$CONFIG_FILE_PATH')
for name in config.get('mcpServers', {}).keys():
    print(name)
")

    # Start services
    for SERVER_NAME in $SERVER_NAMES; do
        echo "Preparing to start server: $SERVER_NAME"
        
        # Use Python script to get server config
        SERVER_CONFIG=$(python -c "
import toml, json, sys
config = toml.load('$CONFIG_FILE_PATH')
server = config.get('mcpServers', {}).get('$SERVER_NAME', {})
print(json.dumps(server))
")

        COMMAND=$(echo "$SERVER_CONFIG" | python -c "import json, sys; print(json.loads(sys.stdin.read()).get('command', ''))")
        ARGS_JSON=$(echo "$SERVER_CONFIG" | python -c "import json, sys; cfg=json.loads(sys.stdin.read()); print(json.dumps(cfg.get('args', [])))")
        
        # Convert JSON array to bash array
        ARGS=()
        if [[ "$ARGS_JSON" != "null" ]]; then
            while IFS= read -r line; do
                ARGS+=("$line")
            done < <(echo "$ARGS_JSON" | python -c "import json, sys; print('\n'.join(json.loads(sys.stdin.read())))")
        fi

        # Get environment variables
        ENV_PREFIX=""
        ENV_JSON=$(echo "$SERVER_CONFIG" | python -c "
import json, sys
cfg = json.loads(sys.stdin.read())
env_dict = {}
for k, v in cfg.items():
    if k.startswith('env.'):
        env_dict[k[4:]] = v
print(json.dumps(env_dict))
")

        if [[ "$ENV_JSON" != "{}" && "$ENV_JSON" != "null" ]]; then
            while IFS='=' read -r key value; do
                # Remove possible quotes
                key=$(echo "$key" | sed 's/^"\|"$//g')
                value=$(echo "$value" | sed 's/^"\|"$//g')
                if [ ! -z "$key" ]; then
                    ENV_PREFIX+="$key='$value' "
                fi
            done < <(echo "$ENV_JSON" | python -c "
import json, sys
env = json.loads(sys.stdin.read())
for k, v in env.items():
    print(f'{k}={json.dumps(v)}')
")
        fi

        LOG_FILE="$LOG_DIR/${SERVER_NAME}.log"
        
        # Check if command is python, if so use python -u for unbuffered output
        if [[ "$COMMAND" == *"python"* ]]; then
            if [[ "$COMMAND" == *"-m"* ]]; then
                COMMAND_WITH_UNBUFFERED=$(echo "$COMMAND" | sed 's/python/python -u/')
            else
                COMMAND_WITH_UNBUFFERED=$(echo "$COMMAND" | awk '{ $1 = $1 " -u"; print }')
            fi
            echo "Starting server $SERVER_NAME (Python unbuffered): eval $ENV_PREFIX $COMMAND_WITH_UNBUFFERED ${ARGS[@]} >> $LOG_FILE 2>&1 &"
            eval $ENV_PREFIX $COMMAND_WITH_UNBUFFERED "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        else
            echo "Starting server $SERVER_NAME: eval $ENV_PREFIX $COMMAND ${ARGS[@]} >> $LOG_FILE 2>&1 &"
            eval $ENV_PREFIX $COMMAND "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        fi

        # Record PID
        SERVER_PID=$!
        echo "Server $SERVER_NAME started, PID: $SERVER_PID, Log: $LOG_FILE"
        echo $SERVER_PID > "$LOG_DIR/${SERVER_NAME}.pid"
    done

else
    # Original JSON processing logic
    SERVER_NAMES=$(jq -r '.mcpServers | keys[]' "$CONFIG_FILE_PATH")

    # Start all servers defined in config.json
    for SERVER_NAME in $SERVER_NAMES; do
        echo "Preparing to start server: $SERVER_NAME"

        COMMAND=$(jq -r --arg SERVER_NAME "$SERVER_NAME" '.mcpServers[$SERVER_NAME].command' "$CONFIG_FILE_PATH")
        ARGS_JSON=$(jq -r --arg SERVER_NAME "$SERVER_NAME" '.mcpServers[$SERVER_NAME].args | tojson' "$CONFIG_FILE_PATH")
        
        # Convert JSON array to bash array
        ARGS=()
        if [[ "$ARGS_JSON" != "null" ]]; then
            while IFS= read -r line; do
                ARGS+=("$line")
            done < <(echo "$ARGS_JSON" | jq -r '.[]')
        fi

        # Get environment variables
        ENV_VARS_JSON=$(jq -r --arg SERVER_NAME "$SERVER_NAME" '.mcpServers[$SERVER_NAME].env | tojson' "$CONFIG_FILE_PATH")
        ENV_PREFIX=""
        if [[ "$ENV_VARS_JSON" != "null" ]]; then
            while IFS='=' read -r key value; do
                key=$(echo $key | jq -r '.')
                value=$(echo $value | jq -r '.')
                ENV_PREFIX+="$key='$value' "
            done < <(echo "$ENV_VARS_JSON" | jq -r 'to_entries | .[] | "\(.key)=\(.value)"')
        fi

        LOG_FILE="$LOG_DIR/${SERVER_NAME}.log"
        
        # Check if command is python
        if [[ "$COMMAND" == *"python"* ]]; then
            if [[ "$COMMAND" == *"-m"* ]]; then
                COMMAND_WITH_UNBUFFERED=$(echo "$COMMAND" | sed 's/python/python -u/')
            else
                COMMAND_WITH_UNBUFFERED=$(echo "$COMMAND" | awk '{ $1 = $1 " -u"; print }')
            fi
            echo "Starting server $SERVER_NAME (Python unbuffered): eval $ENV_PREFIX $COMMAND_WITH_UNBUFFERED ${ARGS[@]} >> $LOG_FILE 2>&1 &"
            eval $ENV_PREFIX $COMMAND_WITH_UNBUFFERED "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        else
            echo "Starting server $SERVER_NAME: eval $ENV_PREFIX $COMMAND ${ARGS[@]} >> $LOG_FILE 2>&1 &"
            eval $ENV_PREFIX $COMMAND "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        fi

        # Record PID
        SERVER_PID=$!
        echo "Server $SERVER_NAME started, PID: $SERVER_PID, Log: $LOG_FILE"
        echo $SERVER_PID > "$LOG_DIR/${SERVER_NAME}.pid"
    done
fi

echo "All servers started. Check logs in $LOG_DIR."

# Keep script running so Docker container doesn't exit
# Gracefully shutdown services on SIGTERM and SIGINT
trap "echo 'Stop signal received, shutting down services...'; for pidfile in $LOG_DIR/*.pid; do if [ -f \"\$pidfile\" ]; then kill \$(cat \"\$pidfile\") && rm \"\$pidfile\"; fi; done; exit 0" SIGTERM SIGINT

# Infinite loop, wait for signal
while true; do
    tail -f /dev/null & wait $!
done
