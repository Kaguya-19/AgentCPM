#!/usr/bin/env python

import asyncio
import argparse
import os
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import List, Dict, Any, Optional

# 导入各个搜索引擎的测试工具
try:
    from direct_exa_benchmark import ExaBenchmark
    HAS_EXA = True
except ImportError:
    HAS_EXA = False

try:
    from direct_duckduckgo_benchmark import DuckDuckGoBenchmark
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False

try:
    from direct_bing_benchmark import BingBenchmark
    HAS_BING = True
except ImportError:
    HAS_BING = False

try:
    from direct_baidu_benchmark import BaiduBenchmark
    HAS_BAIDU = True
except ImportError:
    HAS_BAIDU = False

# 配置控制台输出
console = Console()

# 加载默认查询
DEFAULT_QUERIES = [
    "人工智能最新技术",
    "大语言模型研究进展",
    "强化学习算法优化",
    "计算机视觉识别技术",
    "自然语言处理应用场景",
    "机器学习在医疗领域的应用",
    "深度学习框架比较",
    "神经网络结构设计",
    "知识图谱构建方法",
    "推荐系统算法评估"
]

async def run_exa_benchmark(api_key: str, 
                           queries: List[str], 
                           rpm: int, 
                           duration: int, 
                           concurrency: int,
                           num_results: int) -> Dict[str, Any]:
    """运行Exa搜索引擎API压力测试
    
    Args:
        api_key: Exa API密钥
        queries: 测试查询列表
        rpm: 每分钟请求数
        duration: 测试持续时间（秒）
        concurrency: 并发请求数
        num_results: 每次搜索返回的结果数量
        
    Returns:
        测试结果
    """
    if not HAS_EXA:
        console.print("[red]未找到Exa测试工具，请确保已安装exa-py库[/red]")
        return {
            "engine": "exa",
            "error": "未找到Exa测试工具，请确保已安装exa-py库"
        }
    
    try:
        benchmark = ExaBenchmark(
            api_key=api_key,
            queries=queries,
            rpm=rpm,
            duration=duration,
            num_results=num_results
        )
        
        await benchmark.run(concurrency=concurrency)
        
        # 提取结果
        result = {
            "engine": "exa",
            "total_requests": benchmark.results["total_requests"],
            "successful_requests": benchmark.results["successful_requests"],
            "failed_requests": benchmark.results["failed_requests"],
            "empty_results": benchmark.results["empty_results"],
            "success_rate": benchmark.results["successful_requests"] * 100 / benchmark.results["total_requests"] if benchmark.results["total_requests"] > 0 else 0,
            "response_times": {
                "avg": sum(benchmark.results["response_times"]) / len(benchmark.results["response_times"]) if benchmark.results["response_times"] else 0,
                "count": len(benchmark.results["response_times"])
            },
            "errors": benchmark.results["errors"]
        }
        
        return result
    
    except Exception as e:
        console.print(f"[red]Exa测试失败: {str(e)}[/red]")
        return {
            "engine": "exa",
            "error": str(e)
        }

async def run_duckduckgo_benchmark(queries: List[str], 
                                  rpm: int, 
                                  duration: int, 
                                  concurrency: int,
                                  max_results: int,
                                  region: str = "wt-wt",
                                  safesearch: str = "moderate") -> Dict[str, Any]:
    """运行DuckDuckGo搜索引擎API压力测试
    
    Args:
        queries: 测试查询列表
        rpm: 每分钟请求数
        duration: 测试持续时间（秒）
        concurrency: 并发请求数
        max_results: 每次搜索返回的结果数量
        region: 搜索区域
        safesearch: 安全搜索级别
        
    Returns:
        测试结果
    """
    if not HAS_DUCKDUCKGO:
        console.print("[red]未找到DuckDuckGo测试工具，请确保已安装duckduckgo_search库[/red]")
        return {
            "engine": "duckduckgo",
            "error": "未找到DuckDuckGo测试工具，请确保已安装duckduckgo_search库"
        }
    
    try:
        benchmark = DuckDuckGoBenchmark(
            queries=queries,
            rpm=rpm,
            duration=duration,
            max_results=max_results,
            region=region,
            safesearch=safesearch
        )
        
        await benchmark.run(concurrency=concurrency)
        
        # 提取结果
        result = {
            "engine": "duckduckgo",
            "total_requests": benchmark.results["total_requests"],
            "successful_requests": benchmark.results["successful_requests"],
            "failed_requests": benchmark.results["failed_requests"],
            "empty_results": benchmark.results["empty_results"],
            "success_rate": benchmark.results["successful_requests"] * 100 / benchmark.results["total_requests"] if benchmark.results["total_requests"] > 0 else 0,
            "response_times": {
                "avg": sum(benchmark.results["response_times"]) / len(benchmark.results["response_times"]) if benchmark.results["response_times"] else 0,
                "count": len(benchmark.results["response_times"])
            },
            "errors": benchmark.results["errors"]
        }
        
        return result
    
    except Exception as e:
        console.print(f"[red]DuckDuckGo测试失败: {str(e)}[/red]")
        return {
            "engine": "duckduckgo",
            "error": str(e)
        }

async def run_bing_benchmark(api_key: str,
                            queries: List[str], 
                            rpm: int, 
                            duration: int, 
                            concurrency: int,
                            count: int) -> Dict[str, Any]:
    """运行Bing搜索引擎API压力测试
    
    Args:
        api_key: Bing API密钥
        queries: 测试查询列表
        rpm: 每分钟请求数
        duration: 测试持续时间（秒）
        concurrency: 并发请求数
        count: 每次搜索返回的结果数量
        
    Returns:
        测试结果
    """
    if not HAS_BING:
        console.print("[red]未找到Bing测试工具[/red]")
        return {
            "engine": "bing",
            "error": "未找到Bing测试工具"
        }
    
    try:
        benchmark = BingBenchmark(
            api_key=api_key,
            queries=queries,
            rpm=rpm,
            duration=duration,
            count=count
        )
        
        await benchmark.run(concurrency=concurrency)
        
        # 提取结果
        result = {
            "engine": "bing",
            "total_requests": benchmark.results["total_requests"],
            "successful_requests": benchmark.results["successful_requests"],
            "failed_requests": benchmark.results["failed_requests"],
            "empty_results": benchmark.results["empty_results"],
            "success_rate": benchmark.results["successful_requests"] * 100 / benchmark.results["total_requests"] if benchmark.results["total_requests"] > 0 else 0,
            "response_times": {
                "avg": sum(benchmark.results["response_times"]) / len(benchmark.results["response_times"]) if benchmark.results["response_times"] else 0,
                "count": len(benchmark.results["response_times"])
            },
            "errors": benchmark.results["errors"]
        }
        
        return result
    
    except Exception as e:
        console.print(f"[red]Bing测试失败: {str(e)}[/red]")
        return {
            "engine": "bing",
            "error": str(e)
        }

async def run_baidu_benchmark(api_key: str,
                             secret_key: str,
                             queries: List[str], 
                             rpm: int, 
                             duration: int, 
                             concurrency: int,
                             rn: int) -> Dict[str, Any]:
    """运行百度搜索引擎API压力测试
    
    Args:
        api_key: 百度API密钥
        secret_key: 百度Secret Key
        queries: 测试查询列表
        rpm: 每分钟请求数
        duration: 测试持续时间（秒）
        concurrency: 并发请求数
        rn: 每次搜索返回的结果数量
        
    Returns:
        测试结果
    """
    if not HAS_BAIDU:
        console.print("[red]未找到百度测试工具[/red]")
        return {
            "engine": "baidu",
            "error": "未找到百度测试工具"
        }
    
    try:
        benchmark = BaiduBenchmark(
            api_key=api_key,
            secret_key=secret_key,
            queries=queries,
            rpm=rpm,
            duration=duration,
            rn=rn
        )
        
        await benchmark.run(concurrency=concurrency)
        
        # 提取结果
        result = {
            "engine": "baidu",
            "total_requests": benchmark.results["total_requests"],
            "successful_requests": benchmark.results["successful_requests"],
            "failed_requests": benchmark.results["failed_requests"],
            "empty_results": benchmark.results["empty_results"],
            "success_rate": benchmark.results["successful_requests"] * 100 / benchmark.results["total_requests"] if benchmark.results["total_requests"] > 0 else 0,
            "response_times": {
                "avg": sum(benchmark.results["response_times"]) / len(benchmark.results["response_times"]) if benchmark.results["response_times"] else 0,
                "count": len(benchmark.results["response_times"])
            },
            "errors": benchmark.results["errors"]
        }
        
        return result
    
    except Exception as e:
        console.print(f"[red]百度测试失败: {str(e)}[/red]")
        return {
            "engine": "baidu",
            "error": str(e)
        }

async def run_all_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """运行所有搜索引擎API的压力测试
    
    Args:
        args: 命令行参数
        
    Returns:
        汇总测试结果
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载查询
    queries = []
    if args.queries_file:
        try:
            with open(args.queries_file, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            console.print(f"[red]无法加载查询文件: {str(e)}[/red]")
            queries = []
    
    # 如果没有指定查询文件或加载失败，使用默认查询
    if not queries:
        queries = DEFAULT_QUERIES
    
    # 汇总结果
    summary = {
        "timestamp": str(datetime.now()),
        "rpm": args.rpm,
        "duration": args.duration,
        "concurrency": args.concurrency,
        "results": {}
    }
    
    # 运行测试
    engines_to_test = []
    
    if args.exa and args.exa_api_key:
        engines_to_test.append(("exa", {"api_key": args.exa_api_key}))
    
    if args.duckduckgo:
        engines_to_test.append(("duckduckgo", {}))
    
    if args.bing and args.bing_api_key:
        engines_to_test.append(("bing", {"api_key": args.bing_api_key}))
    
    if args.baidu and args.baidu_api_key and args.baidu_secret_key:
        engines_to_test.append(("baidu", {
            "api_key": args.baidu_api_key,
            "secret_key": args.baidu_secret_key
        }))
    
    for engine, params in engines_to_test:
        console.print(Panel(f"[bold blue]开始测试 {engine} 搜索引擎API[/bold blue]"))
        
        if engine == "exa":
            result = await run_exa_benchmark(
                api_key=params["api_key"],
                queries=queries,
                rpm=args.rpm,
                duration=args.duration,
                concurrency=args.concurrency,
                num_results=args.num_results
            )
        elif engine == "duckduckgo":
            result = await run_duckduckgo_benchmark(
                queries=queries,
                rpm=args.rpm,
                duration=args.duration,
                concurrency=args.concurrency,
                max_results=args.num_results,
                region=args.region,
                safesearch=args.safesearch
            )
        elif engine == "bing":
            result = await run_bing_benchmark(
                api_key=params["api_key"],
                queries=queries,
                rpm=args.rpm,
                duration=args.duration,
                concurrency=args.concurrency,
                count=args.num_results
            )
        elif engine == "baidu":
            result = await run_baidu_benchmark(
                api_key=params["api_key"],
                secret_key=params["secret_key"],
                queries=queries,
                rpm=args.rpm,
                duration=args.duration,
                concurrency=args.concurrency,
                rn=args.num_results
            )
        else:
            result = {"engine": engine, "error": "不支持的搜索引擎"}
        
        summary["results"][engine] = result
        
        # 等待一段时间，让系统恢复
        if engine != engines_to_test[-1][0]:  # 如果不是最后一个引擎
            console.print("[yellow]等待10秒钟，让系统恢复...[/yellow]")
            await asyncio.sleep(10)
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"all_direct_benchmark_summary_{timestamp}.json")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]汇总结果已保存到 {summary_file}[/green]")
    
    # 显示简要汇总
    show_summary(summary)
    
    return summary

def show_summary(summary: Dict[str, Any]):
    """显示测试结果汇总
    
    Args:
        summary: 汇总测试结果
    """
    console.print(Panel("[bold green]测试完成[/bold green]"))
    
    table = Table(title="搜索引擎API性能汇总")
    table.add_column("引擎", style="cyan")
    table.add_column("总请求数", style="green")
    table.add_column("成功率", style="green")
    table.add_column("平均响应时间", style="green")
    table.add_column("空结果数", style="yellow")
    
    for engine, result in summary["results"].items():
        if "error" in result:
            table.add_row(
                engine,
                "N/A",
                "N/A",
                "N/A",
                f"错误: {result['error'][:30]}..."
            )
        else:
            success_rate = result["success_rate"]
            avg_time = result["response_times"]["avg"]
            
            table.add_row(
                engine,
                str(result["total_requests"]),
                f"{success_rate:.2f}%",
                f"{avg_time:.3f}秒",
                str(result["empty_results"])
            )
    
    console.print(table)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="搜索引擎API直接压力测试工具")
    
    # 通用参数
    parser.add_argument("--rpm", type=int, default=100, help="每分钟请求数")
    parser.add_argument("--duration", type=int, default=60, help="测试持续时间（秒）")
    parser.add_argument("--concurrency", type=int, default=10, help="并发请求数")
    parser.add_argument("--num-results", type=int, default=5, help="每次搜索返回的结果数量")
    parser.add_argument("--queries-file", help="包含测试查询的文件路径")
    parser.add_argument("--output-dir", default="benchmark_results", help="结果输出目录")
    
    # 搜索引擎选择
    parser.add_argument("--all", action="store_true", help="测试所有可用的搜索引擎API")
    
    parser.add_argument("--exa", action="store_true", help="测试Exa搜索引擎API")
    parser.add_argument("--exa-api-key", help="Exa API密钥")
    
    parser.add_argument("--duckduckgo", action="store_true", help="测试DuckDuckGo搜索引擎API")
    parser.add_argument("--region", default="wt-wt", help="DuckDuckGo搜索区域")
    parser.add_argument("--safesearch", default="moderate", choices=["on", "moderate", "off"], help="DuckDuckGo安全搜索级别")
    
    parser.add_argument("--bing", action="store_true", help="测试Bing搜索引擎API")
    parser.add_argument("--bing-api-key", help="Bing API密钥")
    
    parser.add_argument("--baidu", action="store_true", help="测试百度搜索引擎API")
    parser.add_argument("--baidu-api-key", help="百度API密钥")
    parser.add_argument("--baidu-secret-key", help="百度Secret Key")
    
    args = parser.parse_args()
    
    # 如果指定了--all，测试所有可用的搜索引擎
    if args.all:
        args.exa = HAS_EXA
        args.duckduckgo = HAS_DUCKDUCKGO
        args.bing = HAS_BING
        args.baidu = HAS_BAIDU
    
    # 如果没有指定任何搜索引擎，默认测试所有可用的搜索引擎
    if not (args.exa or args.duckduckgo or args.bing or args.baidu):
        args.exa = HAS_EXA
        args.duckduckgo = HAS_DUCKDUCKGO
        args.bing = HAS_BING
        args.baidu = HAS_BAIDU
    
    # 检查API密钥
    if args.exa and not args.exa_api_key:
        console.print("[yellow]警告: 未提供Exa API密钥，将跳过Exa搜索引擎测试[/yellow]")
        args.exa = False
    
    if args.bing and not args.bing_api_key:
        console.print("[yellow]警告: 未提供Bing API密钥，将跳过Bing搜索引擎测试[/yellow]")
        args.bing = False
    
    if args.baidu and (not args.baidu_api_key or not args.baidu_secret_key):
        console.print("[yellow]警告: 未提供百度API密钥或Secret Key，将跳过百度搜索引擎测试[/yellow]")
        args.baidu = False
    
    # 运行测试
    await run_all_benchmarks(args)

if __name__ == "__main__":
    asyncio.run(main()) 