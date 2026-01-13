#!/usr/bin/env python

import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from mcp.server.fastmcp import FastMCP, Context
from loguru import logger
import httpx
from search_manager import SearchManager
from jina_reader import JinaReader

# 从环境变量获取配置
CONFIG_PATH = os.environ.get("CONFIG_PATH")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = os.environ.get("LOG_FILE", "nlp_search_infra.log")
LOG_ROTATION = os.environ.get("LOG_ROTATION", "100 MB")

# 配置日志记录器
logger.add(LOG_FILE, rotation=LOG_ROTATION, level=LOG_LEVEL)
logger.info(f"启动服务器，配置路径: {CONFIG_PATH or '默认'}, 日志级别: {LOG_LEVEL}")

# 创建MCP服务器
mcp = FastMCP("NLP-Search-Infra")

# 全局搜索管理器
search_manager = None
# 全局Jina Reader客户端
jina_reader = JinaReader()
last_init_attempt = 0
init_cooldown = 30  # 初始化失败后的冷却时间（秒）

async def initialize_search_manager():
    """初始化搜索管理器"""
    global search_manager, last_init_attempt
    current_time = time.time()
    
    # 如果已经初始化或者冷却期未过，则跳过
    if search_manager is not None:
        return True
    
    if current_time - last_init_attempt < init_cooldown:
        logger.warning(f"搜索管理器初始化冷却中，还需等待 {init_cooldown - (current_time - last_init_attempt):.0f} 秒")
        return False
    
    last_init_attempt = current_time
    
    try:
        # 尝试读取配置文件
        config_path = CONFIG_PATH
        if not config_path or not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
            
        if not os.path.exists(config_path):
            config_path = None
            logger.warning("配置文件不存在，使用默认配置")
        
        search_manager = SearchManager(config_path)
        logger.info(f"搜索管理器初始化成功，可用引擎：{len(search_manager.engines)}")
        return True
    except Exception as e:
        logger.error(f"搜索管理器初始化失败: {str(e)}")
        search_manager = None
        return False

@mcp.tool()
async def search(query: str, num_results: int = 10, engine: str = "auto", use_jina: bool = True) -> str:
    """执行网络搜索并返回结果。
    
    Args:
        query: 搜索查询词
        num_results: 返回结果数量，默认为10
        engine: 搜索引擎类型，可选值:
            - "auto": 自动选择最佳可用搜索引擎（默认）
            - "google": 优先使用Google搜索（需要API密钥）
            - "duckduckgo": 优先使用DuckDuckGo搜索
            - "exa": 优先使用Exa搜索（需要API密钥）
            - "bing": 优先使用Bing搜索（需要API密钥）
            - "baidu": 优先使用百度搜索（需要API密钥）
            - "wiki": 优先使用本地Wiki搜索
            - "jina": 使用Jina AI搜索（s.jina.ai）
        use_jina: 是否使用Jina AI搜索增强结果，默认为True
    """
    global search_manager, jina_reader
    
    try:
        # 如果指定使用jina搜索引擎
        if engine.lower() == "jina":
            logger.info(f"使用Jina AI搜索: {query}")
            start_time = time.time()
            
            try:
                # 使用Jina Reader进行搜索
                jina_results = await jina_reader.search(query, json_format=True)
                elapsed_time = time.time() - start_time
                
                if isinstance(jina_results, list):
                    # 转换为标准格式
                    results = []
                    for item in jina_results:
                        results.append({
                            "title": item.get("title", ""),
                            "link": item.get("url", ""),
                            "snippet": item.get("content", "")[:200] + "...",
                            "source": "jina"
                        })
                    
                    response = {
                        "query": query,
                        "engine": "jina",
                        "time_ms": int(elapsed_time * 1000),
                        "num_results": len(results),
                        "results": results
                    }
                    
                    logger.success(f"Jina搜索 '{query}' 成功，找到 {len(results)} 个结果")
                    return json.dumps(response, ensure_ascii=False)
                else:
                    # 如果返回的不是列表，可能是错误信息
                    logger.warning(f"Jina搜索返回非列表结果: {jina_results}")
                    
            except Exception as e:
                logger.error(f"Jina搜索出错: {str(e)}")
                # 如果Jina搜索失败，继续使用常规搜索引擎
        
        # 使用常规搜索引擎
        if not await initialize_search_manager():
            logger.error("搜索管理器初始化失败")
            return json.dumps({
                "error": "搜索管理器初始化失败，请稍后再试",
                "query": query,
                "engine": engine
            }, ensure_ascii=False)
        
        if not search_manager or not search_manager.engines:
            logger.error("没有可用的搜索引擎")
            return json.dumps({
                "error": "没有可用的搜索引擎",
                "query": query,
                "engine": engine
            }, ensure_ascii=False)
        
        start_time = time.time()
        results = await search_manager.search(query, num_results, engine)
        elapsed_time = time.time() - start_time
        
        # 如果需要使用Jina增强结果
        if use_jina and results:
            logger.info(f"使用Jina Reader增强搜索结果")
            enhanced_results = []
            
            for result in results:
                # 保存原始搜索引擎信息
                original_source = result.get("source", "unknown")
                
                # 构建增强后的结果
                enhanced_result = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": f"jina_reader-{original_source}"  # 组合来源信息
                }
                
                enhanced_results.append(enhanced_result)
            
            # 使用增强后的结果
            results = enhanced_results
        
        # 准备响应对象
        response = {
            "query": query,
            "engine": engine,
            "time_ms": int(elapsed_time * 1000),
            "num_results": len(results),
            "results": results
        }
        
        # 如果是空结果，添加更详细的信息
        if not results:
            logger.warning(f"搜索 '{query}' 未找到结果，引擎: {engine}")
            
            # 获取可用引擎状态
            available_engines = []
            if search_manager:
                engine_status = search_manager.get_available_engines()
                available_engines = [e["name"] for e in engine_status if e["available"]]
            
            response["message"] = "未找到相关结果，请尝试修改搜索词或使用其他搜索引擎"
            response["available_engines"] = available_engines
            response["suggestion"] = "尝试使用更具体的搜索词，或检查搜索引擎是否可用"
        else:
            logger.success(f"搜索 '{query}' 成功，找到 {len(results)} 个结果")
        
        return json.dumps(response, ensure_ascii=False)
    
    except Exception as e:
        logger.exception(f"搜索时出现错误: {str(e)}")
        return json.dumps({
            "error": f"搜索失败: {str(e)}",
            "query": query,
            "engine": engine,
            "suggestion": "请稍后重试，或尝试使用其他搜索引擎"
        }, ensure_ascii=False)

@mcp.tool()
async def fetch_url(url: str, timeout: int = 10, use_jina: bool = True, with_image_alt: bool = False) -> str:
    """抓取网页内容并返回文本。
    
    Args:
        url: 要抓取的网页URL
        timeout: 超时时间，单位为秒，默认10秒
        use_jina: 是否使用Jina Reader优化内容（适合LLM），默认为True
        with_image_alt: 是否为图片生成替代文本描述，默认为False
    """
    global jina_reader
    
    try:
        # 使用Jina Reader获取内容
        if use_jina:
            logger.info(f"使用Jina Reader抓取网页: {url}")
            content = await jina_reader.fetch_url(
                url=url, 
                with_image_alt=with_image_alt,
                response_format="markdown"
            )
            
            result = {
                "url": url,
                "title": f"从 {url} 获取的内容",
                "content": content,
                "content_type": "text/markdown",
                "length": len(content),
                "source": "jina_reader"
            }
            
            logger.success(f"Jina Reader成功抓取网页: {url}")
            return json.dumps(result, ensure_ascii=False)
        
        # 使用原始方法获取内容
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"开始抓取网页: {url}")
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            
            # 设置来源信息
            source = "direct"
            
            # 如果是HTML内容，解析并提取主要文本
            if "text/html" in content_type:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 移除脚本和样式元素
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()
                
                # 提取标题
                title = soup.title.string if soup.title else "无标题"
                
                # 提取正文内容
                main_content = soup.find("main") or soup.find("article") or soup.find("body")
                
                if main_content:
                    paragraphs = main_content.find_all("p")
                    text_content = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                else:
                    paragraphs = soup.find_all("p")
                    text_content = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                
                # 如果找不到段落，使用所有文本
                if not text_content:
                    text_content = soup.get_text().strip()
                
                result = {
                    "url": url,
                    "title": title,
                    "content": text_content[:50000],  # 限制内容长度
                    "content_type": "text/html",
                    "length": len(text_content),
                    "source": source
                }
            else:
                # 非HTML内容，返回原始文本（如果是文本类型）
                if "text/" in content_type:
                    content = response.text[:50000]  # 限制内容长度
                else:
                    content = f"[无法解析的内容类型: {content_type}]"
                
                result = {
                    "url": url,
                    "title": url,
                    "content": content,
                    "content_type": content_type,
                    "length": len(content),
                    "source": source
                }
            
            logger.success(f"成功抓取网页: {url}")
            return json.dumps(result, ensure_ascii=False)
    
    except Exception as e:
        logger.exception(f"抓取网页时出现错误: {str(e)}")
        return json.dumps({
            "error": f"抓取网页失败: {str(e)}",
            "url": url,
            "suggestion": "请检查URL是否有效，或增加超时时间"
        }, ensure_ascii=False)

@mcp.tool()
async def get_available_engines() -> str:
    """获取当前可用的搜索引擎列表和状态"""
    global search_manager
    
    try:
        if not await initialize_search_manager():
            return json.dumps({
                "error": "搜索管理器初始化失败，请稍后再试"
            }, ensure_ascii=False)
            
        engines = search_manager.get_available_engines()
        
        return json.dumps({
            "engines": engines,
            "total": len(engines),
            "available": sum(1 for e in engines if e["available"])
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.exception(f"获取搜索引擎列表失败: {str(e)}")
        return json.dumps({
            "error": f"获取搜索引擎列表失败: {str(e)}"
        }, ensure_ascii=False)

async def cleanup():
    """清理资源"""
    logger.info("执行清理操作")
    # 这里可以添加需要清理的资源

if __name__ == "__main__":
    try:
        # 使用默认配置启动MCP服务器
        mcp.run()
    finally:
        # 确保在退出时执行清理操作
        asyncio.run(cleanup()) 