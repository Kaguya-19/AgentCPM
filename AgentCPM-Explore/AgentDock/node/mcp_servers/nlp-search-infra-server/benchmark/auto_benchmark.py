#!/usr/bin/env python

import asyncio
import argparse
import os
import json
import sys
from rich.console import Console
from rich.panel import Panel
from typing import Dict, Any, Optional

# 配置控制台输出
console = Console()

# 配置文件默认路径
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.json"
)

def load_api_keys(config_path: str) -> Dict[str, Any]:
    """从配置文件中加载API密钥
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        API密钥字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 提取API密钥
        api_keys = {
            "exa": config.get("exa", {}).get("api_key"),
            "bing": config.get("bing", {}).get("api_key"),
            "baidu_api_key": config.get("baidu", {}).get("api_key"),
            "baidu_secret_key": config.get("baidu", {}).get("secret_key")
        }
        
        # 检查API密钥是否有效
        for key, value in api_keys.items():
            if value in [None, "", "YOUR_BING_API_KEY", "YOUR_BAIDU_API_KEY", "YOUR_BAIDU_SECRET_KEY"]:
                api_keys[key] = None
        
        return api_keys
    
    except Exception as e:
        console.print(f"[bold red]错误: 无法加载配置文件 {config_path}: {str(e)}[/bold red]")
        return {
            "exa": None,
            "bing": None,
            "baidu_api_key": None,
            "baidu_secret_key": None
        }

async def run_benchmark(args: argparse.Namespace):
    """运行搜索引擎API测试
    
    Args:
        args: 命令行参数
    """
    # 加载API密钥
    api_keys = load_api_keys(args.config)
    
    # 构建命令行参数
    base_cmd = [
        "python", "run_all_direct_benchmarks.py",
        "--rpm", str(args.rpm),
        "--duration", str(args.duration),
        "--concurrency", str(args.concurrency),
        "--num-results", str(args.num_results),
        "--output-dir", args.output_dir
    ]
    
    # 添加搜索引擎选择
    engines = []
    
    # 检查哪些搜索引擎可用
    if api_keys["exa"]:
        engines.append("exa")
        base_cmd.extend(["--exa", "--exa-api-key", api_keys["exa"]])
    else:
        console.print("[yellow]警告: 未找到有效的Exa API密钥，将跳过Exa搜索引擎测试[/yellow]")
    
    # DuckDuckGo不需要API密钥
    if args.include_duckduckgo:
        engines.append("duckduckgo")
        base_cmd.extend(["--duckduckgo", "--region", args.region, "--safesearch", args.safesearch])
    
    if api_keys["bing"] and api_keys["bing"] != "YOUR_BING_API_KEY":
        engines.append("bing")
        base_cmd.extend(["--bing", "--bing-api-key", api_keys["bing"]])
    else:
        console.print("[yellow]警告: 未找到有效的Bing API密钥，将跳过Bing搜索引擎测试[/yellow]")
    
    if api_keys["baidu_api_key"] and api_keys["baidu_secret_key"] and \
       api_keys["baidu_api_key"] != "YOUR_BAIDU_API_KEY" and \
       api_keys["baidu_secret_key"] != "YOUR_BAIDU_SECRET_KEY":
        engines.append("baidu")
        base_cmd.extend([
            "--baidu", 
            "--baidu-api-key", api_keys["baidu_api_key"],
            "--baidu-secret-key", api_keys["baidu_secret_key"]
        ])
    else:
        console.print("[yellow]警告: 未找到有效的百度API密钥，将跳过百度搜索引擎测试[/yellow]")
    
    if not engines:
        console.print("[bold red]错误: 未找到任何有效的API密钥，无法进行测试[/bold red]")
        return
    
    # 添加查询文件
    if args.queries_file:
        base_cmd.extend(["--queries-file", args.queries_file])
    
    # 显示测试信息
    console.print(Panel(f"[bold green]开始自动化搜索引擎API测试[/bold green]\n"
                        f"将测试以下搜索引擎: {', '.join(engines)}\n"
                        f"RPM: {args.rpm}\n"
                        f"持续时间: {args.duration}秒\n"
                        f"并发数: {args.concurrency}\n"
                        f"每次返回结果数: {args.num_results}"))
    
    # 运行测试命令
    cmd_str = " ".join([arg if " " not in arg else f'"{arg}"' for arg in base_cmd])
    console.print(f"[cyan]执行命令: {cmd_str}[/cyan]")
    
    # 使用系统命令执行
    process = await asyncio.create_subprocess_exec(
        *base_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # 实时输出结果
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        console.print(line.decode().strip())
    
    # 等待进程结束
    await process.wait()
    
    # 检查是否有错误
    if process.returncode != 0:
        stderr = await process.stderr.read()
        console.print(f"[bold red]测试失败，错误信息: {stderr.decode()}[/bold red]")
    else:
        console.print("[bold green]自动化测试完成！[/bold green]")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自动化搜索引擎API测试工具（从配置文件读取API密钥）")
    
    # 通用参数
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--rpm", type=int, default=10, help="每分钟请求数（默认：10，建议小于20）")
    parser.add_argument("--duration", type=int, default=30, help="测试持续时间，单位秒（默认：30）")
    parser.add_argument("--concurrency", type=int, default=5, help="并发请求数（默认：5）")
    parser.add_argument("--num-results", type=int, default=3, help="每次搜索返回的结果数量（默认：3）")
    parser.add_argument("--queries-file", default="example_queries.txt", help="包含测试查询的文件路径")
    parser.add_argument("--output-dir", default="auto_benchmark_results", help="结果输出目录")
    
    # DuckDuckGo相关参数（不需要API密钥）
    parser.add_argument("--include-duckduckgo", action="store_true", help="是否包含DuckDuckGo搜索引擎测试")
    parser.add_argument("--region", default="wt-wt", help="DuckDuckGo搜索区域")
    parser.add_argument("--safesearch", default="moderate", choices=["on", "moderate", "off"], help="DuckDuckGo安全搜索级别")
    
    args = parser.parse_args()
    
    # 运行测试
    asyncio.run(run_benchmark(args))

if __name__ == "__main__":
    main() 