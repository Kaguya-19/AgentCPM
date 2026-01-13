# AgentCPM MCP 服务器

这是一个支持 MCP (Model Context Protocol) 的服务器实现，提供了 SSE 和 streamable-http 两种连接方式。

## 功能特点

- 支持 streamable-http 和 SSE 两种连接协议
- 提供基本的工具调用功能
- 可与 MCP Inspector 工具兼容
- 支持 RL-Factory 框架的 MCPManager 集成

## 快速开始

### 安装依赖

```bash
pip install fastapi uvicorn httpx sse-starlette toml
```

对于简化版服务器，只需要标准库：

```bash
# 无需额外依赖
python simple_server.py
```

### 启动服务器

启动完整版服务器：

```bash
python sse_server.py
```

或启动简化版服务器（仅使用标准库）：

```bash
python simple_server.py
```

默认情况下，服务器将在 `0.0.0.0:8088` 上运行。

## 使用 MCP Inspector CLI 测试

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) 是一个用于测试 MCP 服务器的工具，提供了命令行和 UI 两种使用方式。

### 安装 MCP Inspector

```bash
npm install -g @modelcontextprotocol/inspector
```

### 使用 CLI 模式测试

我们提供了一个测试脚本，演示如何使用 MCP Inspector CLI 工具：

```bash
./test_inspector_cli.sh
```

或者手动运行以下命令：

```bash
# 连接到服务器
npx @modelcontextprotocol/inspector --cli http://localhost:8088 --transport http

# 列出可用工具
npx @modelcontextprotocol/inspector --cli http://localhost:8088 --transport http --method tools/list

# 调用工具
npx @modelcontextprotocol/inspector --cli http://localhost:8088 --transport http --method tools/call --tool-name search --tool-arg query=测试查询
```

### 使用 UI 模式测试

启动 MCP Inspector UI：

```bash
npx @modelcontextprotocol/inspector
```

然后在浏览器中访问 `http://localhost:6274`，配置连接：

- Transport: Streamable HTTP
- Server URL: http://localhost:8088/mcp

## 使用 Python 测试脚本

我们还提供了一个 Python 测试脚本，可以测试服务器连接：

```bash
# 使用 streamable-http 连接（推荐）
python test_sse_server.py --type streamable-http

# 直接连接模式（不使用 MCPManager）
python test_sse_server.py --type streamable-http --direct

# 使用 SSE 连接（已弃用）
python test_sse_server.py --type sse
```

## 与 RL-Factory 框架集成

本服务器可以与 RL-Factory 框架中的 MCPManager 集成。在 RL-Factory 配置中使用以下设置：

```python
config = {
    "mcpServers": {
        "streamable-http-agentmcp": {
            "type": "streamable-http",
            "url": "http://localhost:8088/mcp",
            "headers": {"Content-Type": "application/json"},
            "sse_read_timeout": 30
        }
    }
}

# 初始化 MCPManager
from envs.utils.mcp_manager import MCPManager
manager = MCPManager()
tools = manager.initConfig(config)
```

## 服务器端点

- `/` - 根路径，返回简单的 HTML 页面
- `/health` - 健康检查端点
- `/sse` - SSE 连接端点（已弃用）
- `/messages` - SSE 消息处理端点（已弃用）
- `/mcp` - streamable-http 端点（推荐）

## 注意事项

- 推荐使用 streamable-http 连接方式，性能更好且更可靠
- 如果在 Docker 容器中运行，请确保端口映射正确
- 服务器支持 CORS，可以从任何源访问 