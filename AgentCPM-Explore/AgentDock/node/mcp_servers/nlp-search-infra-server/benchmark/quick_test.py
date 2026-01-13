#!/usr/bin/env python

import asyncio
import argparse
import os
import sys
from rich.console import Console
from rich.panel import Panel

# 配置控制台输出
console = Console()

async def run_quick_test(args):
    """运行快速测试
    
    Args:
        args: 命令行参数
    """
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
    if args.all:
        base_cmd.append("--all")
    else:
        engines = []
        if args.exa and args.exa_api_key:
            engines.append("exa")
            base_cmd.extend(["--exa", "--exa-api-key", args.exa_api_key])
        
        if args.duckduckgo:
            engines.append("duckduckgo")
            base_cmd.extend(["--duckduckgo", "--region", args.region, "--safesearch", args.safesearch])
        
        if args.bing and args.bing_api_key:
            engines.append("bing")
            base_cmd.extend(["--bing", "--bing-api-key", args.bing_api_key])
        
        if args.baidu and args.baidu_api_key and args.baidu_secret_key:
            engines.append("baidu")
            base_cmd.extend(["--baidu", "--baidu-api-key", args.baidu_api_key, "--baidu-secret-key", args.baidu_secret_key])
        
        if not engines:
            console.print("[bold red]错误: 未选择任何搜索引擎或未提供必要的API密钥[/bold red]")
            return
    
    # 添加查询文件
    if args.queries_file:
        base_cmd.extend(["--queries-file", args.queries_file])
    
    # 显示测试信息
    console.print(Panel(f"[bold green]开始快速测试搜索引擎API[/bold green]\n"
                        f"RPM: {args.rpm}\n"
                        f"持续时间: {args.duration}秒\n"
                        f"并发数: {args.concurrency}\n"
                        f"每次返回结果数: {args.num_results}"))
    
    # 运行测试命令
    cmd_str = " ".join(base_cmd)
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
        console.print("[bold green]快速测试完成！[/bold green]")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="搜索引擎API快速测试工具")
    
    # 通用参数
    parser.add_argument("--rpm", type=int, default=10, help="每分钟请求数（默认：10，建议小于20）")
    parser.add_argument("--duration", type=int, default=30, help="测试持续时间，单位秒（默认：30）")
    parser.add_argument("--concurrency", type=int, default=5, help="并发请求数（默认：5）")
    parser.add_argument("--num-results", type=int, default=3, help="每次搜索返回的结果数量（默认：3）")
    parser.add_argument("--queries-file", help="包含测试查询的文件路径")
    parser.add_argument("--output-dir", default="quick_test_results", help="结果输出目录")
    
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
    
    # 运行测试
    asyncio.run(run_quick_test(args))

if __name__ == "__main__":
    main() 