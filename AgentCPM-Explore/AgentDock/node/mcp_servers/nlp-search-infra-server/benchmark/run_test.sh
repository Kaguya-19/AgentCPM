#!/bin/bash

# 设置默认参数
RPM=10
DURATION=30
CONCURRENCY=5
NUM_RESULTS=3
INCLUDE_DUCKDUCKGO=true
OUTPUT_DIR="benchmark_results/$(date +%Y%m%d_%H%M%S)"

# 显示帮助信息
show_help() {
    echo "搜索引擎API测试工具快速启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --rpm NUMBER         每分钟请求数 (默认: $RPM)"
    echo "  -d, --duration NUMBER    测试持续时间(秒) (默认: $DURATION)"
    echo "  -c, --concurrency NUMBER 并发请求数 (默认: $CONCURRENCY)"
    echo "  -n, --num-results NUMBER 每次搜索返回的结果数量 (默认: $NUM_RESULTS)"
    echo "  -q, --queries FILE       测试查询文件 (默认: example_queries.txt)"
    echo "  -o, --output-dir DIR     结果输出目录 (默认: $OUTPUT_DIR)"
    echo "  --no-duckduckgo          不包含DuckDuckGo搜索引擎测试"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                       使用默认参数运行测试"
    echo "  $0 -r 20 -d 60           每分钟20个请求，持续60秒"
    echo "  $0 -q my_queries.txt     使用自定义查询文件"
    echo ""
}

# 解析命令行参数
QUERIES_FILE="example_queries.txt"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -r|--rpm)
            RPM="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -n|--num-results)
            NUM_RESULTS="$2"
            shift 2
            ;;
        -q|--queries)
            QUERIES_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-duckduckgo)
            INCLUDE_DUCKDUCKGO=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $key"
            show_help
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python auto_benchmark.py --rpm $RPM --duration $DURATION --concurrency $CONCURRENCY --num-results $NUM_RESULTS --queries-file $QUERIES_FILE --output-dir $OUTPUT_DIR"

# 添加DuckDuckGo选项
if [ "$INCLUDE_DUCKDUCKGO" = true ]; then
    CMD="$CMD --include-duckduckgo"
fi

# 显示将要执行的命令
echo "执行命令: $CMD"
echo ""

# 执行命令
eval $CMD

# 显示结果位置
echo ""
echo "测试结果已保存到: $OUTPUT_DIR" 