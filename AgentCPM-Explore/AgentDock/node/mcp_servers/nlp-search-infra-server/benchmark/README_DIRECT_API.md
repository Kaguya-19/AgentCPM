# 搜索引擎直接API压力测试工具

这个目录包含用于直接对搜索引擎API进行压力测试的工具，无需通过NLP搜索基础设施服务。这些工具可以帮助你评估各个搜索引擎API的性能、稳定性和可扩展性，特别是在高负载情况下（如每分钟100个请求）的表现。

## 功能特点

- 直接测试搜索引擎API，绕过服务器限制
- 支持多种搜索引擎（Exa、DuckDuckGo、Bing、百度等）
- 可配置的请求速率（RPM）、持续时间和并发数
- 详细的性能指标报告，包括响应时间、成功率等
- 支持批量测试多个搜索引擎API
- 美观的进度显示和结果报告
- 自动保存测试结果为JSON格式
- 支持从配置文件自动读取API密钥

## 安装依赖

在运行测试工具之前，请确保安装了所需的依赖：

```bash
pip install httpx rich loguru duckduckgo_search exa-py
```

## 快速开始（推荐）

使用自动化测试脚本，从配置文件中读取API密钥并运行测试：

```bash
# 使脚本可执行
chmod +x run_test.sh

# 使用默认参数运行测试
./run_test.sh

# 自定义参数运行测试
./run_test.sh -r 20 -d 60 -c 10
```

参数说明：

- `-r, --rpm NUMBER`: 每分钟请求数（默认：10）
- `-d, --duration NUMBER`: 测试持续时间，单位秒（默认：30）
- `-c, --concurrency NUMBER`: 并发请求数（默认：5）
- `-n, --num-results NUMBER`: 每次搜索返回的结果数量（默认：3）
- `-q, --queries FILE`: 测试查询文件（默认：example_queries.txt）
- `-o, --output-dir DIR`: 结果输出目录（默认：benchmark_results/当前时间）
- `--no-duckduckgo`: 不包含DuckDuckGo搜索引擎测试
- `-h, --help`: 显示帮助信息

## 直接使用自动化测试脚本

如果你想直接使用Python脚本而不是Shell脚本，可以使用`auto_benchmark.py`：

```bash
python auto_benchmark.py --rpm 10 --duration 30 --include-duckduckgo
```

参数说明：

- `--config`: 配置文件路径（默认：../config.json）
- `--rpm`: 每分钟请求数（默认：10）
- `--duration`: 测试持续时间，单位秒（默认：30）
- `--concurrency`: 并发请求数（默认：5）
- `--num-results`: 每次搜索返回的结果数量（默认：3）
- `--queries-file`: 包含测试查询的文件路径（默认：example_queries.txt）
- `--output-dir`: 结果输出目录（默认：auto_benchmark_results）
- `--include-duckduckgo`: 包含DuckDuckGo搜索引擎测试
- `--region`: DuckDuckGo搜索区域（默认：wt-wt）
- `--safesearch`: DuckDuckGo安全搜索级别（默认：moderate）

## 单一引擎测试

### Exa搜索引擎

使用`direct_exa_benchmark.py`脚本对Exa搜索引擎API进行压力测试：

```bash
python direct_exa_benchmark.py --api-key YOUR_EXA_API_KEY --rpm 100 --duration 60
```

参数说明：

- `--api-key`: Exa API密钥（必需）
- `--rpm`: 每分钟请求数（默认：100）
- `--duration`: 测试持续时间，单位秒（默认：60）
- `--concurrency`: 并发请求数（默认：10）
- `--num-results`: 每次搜索返回的结果数量（默认：5）
- `--queries-file`: 包含自定义测试查询的文件路径（可选）

### DuckDuckGo搜索引擎

使用`direct_duckduckgo_benchmark.py`脚本对DuckDuckGo搜索引擎API进行压力测试：

```bash
python direct_duckduckgo_benchmark.py --rpm 100 --duration 60
```

参数说明：

- `--rpm`: 每分钟请求数（默认：100）
- `--duration`: 测试持续时间，单位秒（默认：60）
- `--concurrency`: 并发请求数（默认：10）
- `--max-results`: 每次搜索返回的结果数量（默认：5）
- `--region`: 搜索区域（默认：wt-wt）
- `--safesearch`: 安全搜索级别（默认：moderate，可选：on、moderate、off）
- `--queries-file`: 包含自定义测试查询的文件路径（可选）

### Bing搜索引擎

使用`direct_bing_benchmark.py`脚本对Bing搜索引擎API进行压力测试：

```bash
python direct_bing_benchmark.py --api-key YOUR_BING_API_KEY --rpm 100 --duration 60
```

参数说明：

- `--api-key`: Bing API密钥（必需）
- `--rpm`: 每分钟请求数（默认：100）
- `--duration`: 测试持续时间，单位秒（默认：60）
- `--concurrency`: 并发请求数（默认：10）
- `--count`: 每次搜索返回的结果数量（默认：5）
- `--queries-file`: 包含自定义测试查询的文件路径（可选）

### 百度搜索引擎

使用`direct_baidu_benchmark.py`脚本对百度搜索引擎API进行压力测试：

```bash
python direct_baidu_benchmark.py --api-key YOUR_BAIDU_API_KEY --secret-key YOUR_BAIDU_SECRET_KEY --rpm 100 --duration 60
```

参数说明：

- `--api-key`: 百度API密钥（必需）
- `--secret-key`: 百度Secret Key（必需）
- `--rpm`: 每分钟请求数（默认：100）
- `--duration`: 测试持续时间，单位秒（默认：60）
- `--concurrency`: 并发请求数（默认：10）
- `--rn`: 每次搜索返回的结果数量（默认：5）
- `--queries-file`: 包含自定义测试查询的文件路径（可选）

## 批量测试多个引擎

### 使用run_direct_benchmarks.py（旧版）

使用`run_direct_benchmarks.py`脚本对Exa和DuckDuckGo搜索引擎API进行批量测试：

```bash
python run_direct_benchmarks.py --exa --exa-api-key YOUR_EXA_API_KEY --duckduckgo
```

### 使用run_all_direct_benchmarks.py（新版）

使用`run_all_direct_benchmarks.py`脚本对所有支持的搜索引擎API进行批量测试：

```bash
python run_all_direct_benchmarks.py --all --exa-api-key YOUR_EXA_API_KEY --bing-api-key YOUR_BING_API_KEY --baidu-api-key YOUR_BAIDU_API_KEY --baidu-secret-key YOUR_BAIDU_SECRET_KEY
```

参数说明：

- `--rpm`: 每分钟请求数（默认：100）
- `--duration`: 测试持续时间，单位秒（默认：60）
- `--concurrency`: 并发请求数（默认：10）
- `--num-results`: 每次搜索返回的结果数量（默认：5）
- `--queries-file`: 包含自定义测试查询的文件路径（可选）
- `--output-dir`: 结果输出目录（默认：benchmark_results）

搜索引擎选择：
- `--all`: 测试所有可用的搜索引擎API
- `--exa`: 测试Exa搜索引擎API
- `--exa-api-key`: Exa API密钥
- `--duckduckgo`: 测试DuckDuckGo搜索引擎API
- `--region`: DuckDuckGo搜索区域（默认：wt-wt）
- `--safesearch`: DuckDuckGo安全搜索级别（默认：moderate）
- `--bing`: 测试Bing搜索引擎API
- `--bing-api-key`: Bing API密钥
- `--baidu`: 测试百度搜索引擎API
- `--baidu-api-key`: 百度API密钥
- `--baidu-secret-key`: 百度Secret Key

## 自定义测试查询

你可以创建一个文本文件，每行包含一个搜索查询，然后使用`--queries-file`参数指定该文件：

```bash
python auto_benchmark.py --queries-file example_queries.txt
```

## 测试结果

测试结果将显示在控制台中，并自动保存为JSON文件。结果包括：

- 总请求数、成功请求数、失败请求数
- 成功率和实际RPM
- 平均响应时间、P50/P90/P99响应时间
- 空结果数
- 错误信息统计

批量测试完成后，会生成一个汇总报告，比较所有引擎的性能。

## 示例用法

1. 使用自动化脚本测试所有可用的搜索引擎：

```bash
./run_test.sh
```

2. 测试单个搜索引擎：

```bash
# 测试Exa搜索引擎
python direct_exa_benchmark.py --api-key YOUR_EXA_API_KEY --rpm 100 --duration 60

# 测试DuckDuckGo搜索引擎
python direct_duckduckgo_benchmark.py --rpm 50 --duration 120

# 测试Bing搜索引擎
python direct_bing_benchmark.py --api-key YOUR_BING_API_KEY --rpm 80 --duration 60

# 测试百度搜索引擎
python direct_baidu_benchmark.py --api-key YOUR_BAIDU_API_KEY --secret-key YOUR_BAIDU_SECRET_KEY --rpm 60 --duration 60
```

3. 批量测试多个搜索引擎：

```bash
# 测试所有可用的搜索引擎
python run_all_direct_benchmarks.py --all --exa-api-key YOUR_EXA_API_KEY --bing-api-key YOUR_BING_API_KEY --baidu-api-key YOUR_BAIDU_API_KEY --baidu-secret-key YOUR_BAIDU_SECRET_KEY

# 只测试特定的搜索引擎
python run_all_direct_benchmarks.py --exa --exa-api-key YOUR_EXA_API_KEY --bing --bing-api-key YOUR_BING_API_KEY

# 使用自定义参数
python run_all_direct_benchmarks.py --all --rpm 50 --duration 30 --concurrency 5 --exa-api-key YOUR_EXA_API_KEY --bing-api-key YOUR_BING_API_KEY
```

## 注意事项

- 高强度压力测试可能会导致搜索引擎API限流，请确保有适当的错误处理和冷却机制
- 请遵守各搜索引擎API的使用条款和限制
- 对于需要API密钥的搜索引擎（如Exa、Bing、百度），请确保提供有效的API密钥
- 测试前确保网络连接稳定，以获得更准确的结果
- 如果遇到频繁的限速错误，请尝试降低RPM和并发数 