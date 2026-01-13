# NLP-Search-Infra-Server

多引擎搜索聚合MCP服务器，提供高可用性和故障转移功能。

## 特性

- **多搜索引擎整合**：按优先级轮询调用不同的搜索引擎
- **自动故障转移**：当高优先级搜索引擎出错或遇到限速时，自动切换到低优先级引擎
- **限速处理**：智能检测限速错误，并对受限搜索引擎进行冷却
- **统一响应格式**：所有搜索引擎返回的结果统一格式化
- **高可用性**：支持每分钟100+ RPM的高频请求
- **LLM友好内容**：集成Jina AI Reader，提供更适合LLM处理的网页内容

## 支持的搜索引擎

按照默认优先级从高到低排序：

1. **Google搜索** - 高性能收费API，需要API密钥
2. **DuckDuckGo搜索** - 免费搜索，不需要API密钥
3. **Exa搜索** - 需要API密钥
4. **Jina AI搜索** - 通过s.jina.ai提供的搜索服务
5. **Bing搜索** - 需要API密钥
6. **百度搜索** - 需要API密钥
7. **Wayback Machine搜索** - 互联网档案馆的历史网页搜索
8. **本地Wiki搜索** - 在本地Wiki数据库中搜索

## 安装

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置搜索引擎：

复制并编辑配置文件：

```bash
cp config.json.template config.json
```

根据需要编辑 `config.json` 添加API密钥等配置。

## 配置

配置文件 `config.json` 示例：

```json
{
  "google": {
    "api_key": "YOUR_GOOGLE_API_KEY",
    "cse_id": "YOUR_GOOGLE_CSE_ID"
  },
  "bing": {
    "api_key": "YOUR_BING_API_KEY"
  },
  "baidu": {
    "api_key": "YOUR_BAIDU_API_KEY",
    "secret_key": "YOUR_BAIDU_SECRET_KEY"
  },
  "exa": {
    "api_key": "YOUR_EXA_API_KEY"
  },
  "wiki": {
    "path": "/path/to/your/local/wiki/data"
  },
  "rate_limits": {
    "default_cooldown": 300,
    "max_error_count": 3
  }
}
```

## Jina AI Reader 集成

本服务器集成了 [Jina AI Reader](https://github.com/jina-ai/reader)，提供更适合LLM处理的网页内容和搜索结果。

### Jina Reader 功能

- **网页内容优化**：通过 r.jina.ai 将网页内容转换为更适合LLM处理的格式
- **图片描述生成**：可选择为网页中的图片生成文本描述，帮助LLM理解图片内容
- **搜索功能**：通过 s.jina.ai 直接搜索网页内容

### 使用方法

- 在 `fetch_url` 工具中设置 `use_jina=True`（默认）以获取LLM友好的网页内容
- 在 `search` 工具中设置 `engine="jina"` 或 `use_jina=True` 以使用Jina AI搜索

## MCP工具

该MCP服务器提供以下工具：

1. **search** - 执行网络搜索并返回结果
   - `query`: 搜索查询词 (必需)
   - `num_results`: 返回结果数量，默认为10
   - `engine`: 指定搜索引擎类型 (auto/google/duckduckgo/exa/jina/bing/baidu/wayback/wiki)
   - `use_jina`: 是否使用Jina AI搜索增强结果，默认为False

2. **web_archive_search** - 搜索网站的历史归档版本
   - `url`: 要搜索的网站URL (必需)
   - `num_results`: 返回的历史版本数量，默认为5

3. **fetch_url** - 抓取网页内容并返回文本
   - `url`: 要抓取的网页URL (必需)
   - `timeout`: 超时时间，单位为秒，默认10秒
   - `use_jina`: 是否使用Jina Reader优化内容（适合LLM），默认为True
   - `with_image_alt`: 是否为图片生成替代文本描述，默认为False

4. **get_available_engines** - 获取当前可用的搜索引擎列表和状态

## 配置接入

### 接入MCPManager配置

在 `MCPManager/node/config.toml` 中添加以下配置：

```toml
[mcpServers."nlp-search-infra-server"]
command = "/opt/conda/envs/mcp-agent/bin/python"
args = ["/app/AgentCPM-MCP/src/mcp_servers/nlp-search-infra-server/src/index.py"]
```

如果需要设置环境变量，可以添加：

```toml
[mcpServers."nlp-search-infra-server".env]
CONFIG_PATH = "/app/AgentCPM-MCP/src/mcp_servers/nlp-search-infra-server/config.json"
LOG_LEVEL = "INFO"
```

### 接入mcp-cli配置

在 `MCPMAgent/src/mcp-cli/server_config.json` 中添加以下配置：

```json
{
  "mcp_servers": [
    {
      "name": "nlp-search-infra-server",
      "command": "python",
      "args": ["/path/to/AgentCPM-MCP/src/mcp_servers/nlp-search-infra-server/src/index.py"]
    }
  ]
}
```

## 运行

启动服务器：

```bash
cd src
python index.py
```

服务器默认使用MCP协议启动，会自动监听默认端口。

## 环境变量配置

服务器支持通过环境变量进行配置：

- `CONFIG_PATH`: 配置文件路径，默认为 `../config.json`
- `LOG_LEVEL`: 日志级别，默认为 `INFO`
- `LOG_FILE`: 日志文件路径，默认为 `nlp_search_infra.log`
- `LOG_ROTATION`: 日志轮转设置，默认为 `100 MB`

## 压测

可以使用 `benchmark` 目录下的压测脚本进行性能测试：

```bash
# 自动化测试（从配置文件读取API密钥）
cd benchmark
./run_test.sh

# 自定义参数运行测试
./run_test.sh -r 20 -d 60 -c 10
```

该压测脚本会模拟高并发请求，确保服务器能够处理指定RPM的查询负载。 