#!/usr/bin/env python

import asyncio
import json
import time
import random
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import print as rprint
from loguru import logger

# 导入Exa搜索API
try:
    from exa_py import Exa
except ImportError:
    logger.error("未找到exa-py库，请使用'pip install exa-py'安装")
    raise ImportError("请先安装exa-py: pip install exa-py")

# 配置日志
logger.add("exa_benchmark_results.log", rotation="100 MB", level="INFO")

# 测试查询列表
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
    "推荐系统算法评估",
    "元学习最新研究",
    "联邦学习隐私保护",
    "图神经网络应用",
    "迁移学习技术",
    "自监督学习方法",
    "多模态融合技术",
    "对比学习在图像处理中的应用",
    "注意力机制改进",
    "生成对抗网络发展",
    "强化学习在自动驾驶中的应用"
]

class ExaBenchmark:
    """Exa搜索引擎直接API压力测试工具"""
    
    def __init__(self, 
                 api_key: str,
                 queries: List[str] = None, 
                 rpm: int = 100, 
                 duration: int = 60,
                 num_results: int = 5):
        """初始化压测工具
        
        Args:
            api_key: Exa API密钥
            queries: 测试查询列表
            rpm: 每分钟请求数
            duration: 测试持续时间（秒）
            num_results: 每次搜索返回的结果数量
        """
        self.api_key = api_key
        self.exa_client = Exa(api_key=api_key)
        self.queries = queries or DEFAULT_QUERIES
        self.rpm = rpm
        self.duration = duration
        self.num_results = num_results
        self.console = Console()
        
        # 计算请求间隔（秒）
        self.request_interval = 60 / rpm if rpm > 0 else 1
        
        # 结果统计
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": {},
            "empty_results": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def _make_request(self, query: str) -> Tuple[bool, float, Dict]:
        """执行单个搜索请求
        
        Args:
            query: 搜索查询
            
        Returns:
            (成功标志, 响应时间, 响应内容)
        """
        start_time = time.time()
        try:
            # 使用asyncio.to_thread将同步API调用转换为异步操作
            result = await asyncio.to_thread(
                self.exa_client.search_and_contents,
                query=query,
                text=True,
                num_results=self.num_results
            )
            
            response_time = time.time() - start_time
            
            # 检查结果是否为空
            if hasattr(result, 'results'):
                items = result.results
            elif isinstance(result, dict) and 'results' in result:
                items = result['results']
            else:
                items = result if isinstance(result, list) else []
                
            if not items:
                return True, response_time, {"results": [], "empty": True}
                
            return True, response_time, {"results": items, "empty": False}
                
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            # 检查是否是限速错误
            is_rate_limit = "rate" in error_msg.lower() or "limit" in error_msg.lower() or "429" in error_msg
            
            return False, response_time, {
                "error": error_msg, 
                "is_rate_limit": is_rate_limit
            }
    
    async def _worker(self, worker_id: int, progress: Progress):
        """工作线程，持续发送请求
        
        Args:
            worker_id: 工作线程ID
            progress: 进度条对象
        """
        task_id = progress.add_task(f"[cyan]Worker {worker_id}", total=self.duration)
        
        start_time = time.time()
        end_time = start_time + self.duration
        
        while time.time() < end_time:
            # 随机选择一个查询
            query = random.choice(self.queries)
            
            # 发送请求
            success, response_time, result = await self._make_request(query)
            
            # 更新统计信息
            self.results["total_requests"] += 1
            
            if success:
                self.results["successful_requests"] += 1
                self.results["response_times"].append(response_time)
                
                # 检查结果是否为空
                if result.get("empty", False):
                    self.results["empty_results"] += 1
            else:
                self.results["failed_requests"] += 1
                error_msg = result.get("error", "未知错误")
                self.results["errors"][error_msg] = self.results["errors"].get(error_msg, 0) + 1
                
                # 如果是限速错误，可能需要暂停一下
                if result.get("is_rate_limit", False):
                    logger.warning(f"Worker {worker_id} 遇到限速错误，暂停5秒")
                    await asyncio.sleep(5)
            
            # 更新进度
            elapsed = time.time() - start_time
            progress.update(task_id, completed=min(elapsed, self.duration))
            
            # 等待下一个请求时间
            next_request_time = start_time + (self.results["total_requests"] / (self.rpm / 60))
            sleep_time = max(0, next_request_time - time.time())
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    async def run(self, concurrency: int = 10):
        """运行压力测试
        
        Args:
            concurrency: 并发请求数
        """
        self.results["start_time"] = datetime.now()
        
        self.console.print(Panel(f"[bold green]开始Exa搜索引擎直接API压力测试[/bold green]\n"
                                f"API密钥: {self.api_key[:8]}...{self.api_key[-4:] if len(self.api_key) > 12 else ''}\n"
                                f"目标RPM: {self.rpm}\n"
                                f"持续时间: {self.duration}秒\n"
                                f"并发数: {concurrency}\n"
                                f"每次返回结果数: {self.num_results}"))
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            # 创建工作线程
            workers = [self._worker(i, progress) for i in range(concurrency)]
            
            # 等待所有工作线程完成
            await asyncio.gather(*workers)
        
        self.results["end_time"] = datetime.now()
        
        # 显示测试结果
        self._show_results()
    
    def _show_results(self):
        """显示测试结果"""
        # 计算统计数据
        total_time = (self.results["end_time"] - self.results["start_time"]).total_seconds()
        actual_rpm = self.results["total_requests"] * 60 / total_time if total_time > 0 else 0
        success_rate = self.results["successful_requests"] * 100 / self.results["total_requests"] if self.results["total_requests"] > 0 else 0
        
        response_times = self.results["response_times"]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p50_response_time = statistics.median(response_times) if response_times else 0
        p90_response_time = statistics.quantiles(response_times, n=10)[-1] if len(response_times) >= 10 else 0
        p99_response_time = statistics.quantiles(response_times, n=100)[-1] if len(response_times) >= 100 else 0
        
        # 创建结果表格
        table = Table(title="Exa搜索引擎API压力测试结果")
        
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("总请求数", str(self.results["total_requests"]))
        table.add_row("成功请求数", str(self.results["successful_requests"]))
        table.add_row("失败请求数", str(self.results["failed_requests"]))
        table.add_row("空结果数", str(self.results["empty_results"]))
        table.add_row("成功率", f"{success_rate:.2f}%")
        table.add_row("实际RPM", f"{actual_rpm:.2f}")
        table.add_row("平均响应时间", f"{avg_response_time:.3f}秒")
        table.add_row("中位响应时间 (P50)", f"{p50_response_time:.3f}秒")
        table.add_row("P90响应时间", f"{p90_response_time:.3f}秒")
        table.add_row("P99响应时间", f"{p99_response_time:.3f}秒")
        
        self.console.print(table)
        
        # 显示错误信息
        if self.results["errors"]:
            error_table = Table(title="错误统计")
            error_table.add_column("错误信息", style="red")
            error_table.add_column("次数", style="yellow")
            
            for error, count in sorted(self.results["errors"].items(), key=lambda x: x[1], reverse=True):
                error_table.add_row(error[:100] + "..." if len(error) > 100 else error, str(count))
            
            self.console.print(error_table)
        
        # 保存结果到文件
        self._save_results()
    
    def _save_results(self):
        """将测试结果保存到文件"""
        timestamp = self.results["start_time"].strftime("%Y%m%d_%H%M%S")
        filename = f"exa_benchmark_result_{timestamp}.json"
        
        # 转换datetime对象为字符串
        result_copy = self.results.copy()
        result_copy["start_time"] = str(result_copy["start_time"])
        result_copy["end_time"] = str(result_copy["end_time"])
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
        
        self.console.print(f"[green]测试结果已保存到 {filename}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Exa搜索引擎API直接压力测试工具")
    parser.add_argument("--api-key", required=True, help="Exa API密钥")
    parser.add_argument("--rpm", type=int, default=100, help="每分钟请求数")
    parser.add_argument("--duration", type=int, default=60, help="测试持续时间（秒）")
    parser.add_argument("--concurrency", type=int, default=10, help="并发请求数")
    parser.add_argument("--num-results", type=int, default=5, help="每次搜索返回的结果数量")
    parser.add_argument("--queries-file", help="包含测试查询的文件路径")
    
    args = parser.parse_args()
    
    # 加载自定义查询
    queries = DEFAULT_QUERIES
    if args.queries_file:
        try:
            with open(args.queries_file, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"无法加载查询文件: {e}")
    
    # 创建并运行压测工具
    benchmark = ExaBenchmark(
        api_key=args.api_key,
        queries=queries,
        rpm=args.rpm,
        duration=args.duration,
        num_results=args.num_results
    )
    
    await benchmark.run(concurrency=args.concurrency)


if __name__ == "__main__":
    asyncio.run(main()) 