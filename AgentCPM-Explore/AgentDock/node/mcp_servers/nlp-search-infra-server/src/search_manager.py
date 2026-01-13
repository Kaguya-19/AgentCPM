#!/usr/bin/env python

from typing import List, Dict, Optional, Any, Type
import json
import os
import time
import asyncio
from loguru import logger
from datetime import datetime

from search_engines import (
    SearchEngine, 
    SearchResult,
    GoogleSearch,
    DuckDuckGoSearch,
    BingSearch,
    BaiduSearch,
    WikiSearch,
    ExaSearch
)

class SearchManager:
    """搜索管理器，负责管理多个搜索引擎并按优先级轮询调用"""
    
    def __init__(self, config_path: str = None):
        self.engines: List[SearchEngine] = []
        self.last_used_engine: Optional[SearchEngine] = None
        self.last_search_time = 0
        self.cooldown_period = 60  # 冷却时间，单位秒
        
        # 尝试从配置文件加载配置
        config = {}
        if config_path:
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"已从 {config_path} 加载配置")
            except Exception as e:
                logger.error(f"无法加载配置文件 {config_path}: {str(e)}")
        
        self._initialize_engines(config)
        
    def _initialize_engines(self, config: Dict[str, Any]):
        """初始化搜索引擎"""
        
        # 1. 添加高性能收费API（如Google）
        if "google" in config:
            google_config = config["google"]
            api_key = google_config.get("api_key")
            cse_id = google_config.get("cse_id")
            if api_key and cse_id:
                self.engines.append(GoogleSearch(api_key=api_key, cse_id=cse_id))
                logger.info("已初始化Google搜索引擎")
            else:
                logger.warning("Google搜索配置不完整，跳过初始化")
        
        # 2. 添加DuckDuckGo搜索（自己实现的API）
        self.engines.append(DuckDuckGoSearch())
        logger.info("已初始化DuckDuckGo搜索引擎")
        
        # 3. 添加Exa搜索引擎
        if "exa" in config:
            api_key = config["exa"].get("api_key")
            if api_key:
                try:
                    self.engines.append(ExaSearch(api_key=api_key))
                    logger.info("已初始化Exa搜索引擎")
                except Exception as e:
                    logger.error(f"初始化Exa搜索引擎失败: {str(e)}")
            else:
                logger.warning("Exa搜索配置不完整，跳过初始化")
        
        # 4. 添加其他搜索引擎API（如Bing，百度）
        if "bing" in config:
            api_key = config["bing"].get("api_key")
            if api_key:
                self.engines.append(BingSearch(api_key=api_key))
                logger.info("已初始化Bing搜索引擎")
            else:
                logger.warning("Bing搜索配置不完整，跳过初始化")
                
        if "baidu" in config:
            baidu_config = config["baidu"]
            api_key = baidu_config.get("api_key")
            secret_key = baidu_config.get("secret_key")
            if api_key and secret_key:
                self.engines.append(BaiduSearch(api_key=api_key, secret_key=secret_key))
                logger.info("已初始化百度搜索引擎")
            else:
                logger.warning("百度搜索配置不完整，跳过初始化")
        
        # 5. 添加本地Wiki搜索
        if "wiki" in config:
            wiki_path = config["wiki"].get("path")
            if wiki_path and os.path.exists(wiki_path):
                self.engines.append(WikiSearch(wiki_path=wiki_path))
                logger.info(f"已初始化Wiki搜索引擎，数据路径: {wiki_path}")
            else:
                logger.warning(f"Wiki搜索配置不完整或路径不存在: {wiki_path}")
        
        # 按优先级排序
        self.engines.sort(key=lambda e: getattr(e, "priority", 999))
        
        engine_names = [f"{e.__class__.__name__}(priority={getattr(e, 'priority', 999)})" for e in self.engines]
        logger.info(f"搜索引擎初始化完成，按优先级排序: {engine_names}")
    
    async def search(self, query: str, num_results: int = 10, preferred_engine_type: str = None) -> List[Dict]:
        """执行搜索并按优先级依次调用搜索引擎
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            preferred_engine_type: 优先使用的搜索引擎类型，默认按优先级顺序
            
        Returns:
            搜索结果列表
        """
        if not self.engines:
            logger.error("没有可用的搜索引擎")
            return []
        
        logger.info(f"开始搜索: {query}, 引擎类型: {preferred_engine_type or '默认顺序'}")
        
        # 按照指定引擎类型搜索
        if preferred_engine_type and preferred_engine_type.lower() != "auto":
            for engine in self.engines:
                engine_type = engine.__class__.__name__.lower()
                if preferred_engine_type.lower() in engine_type:
                    if engine.is_available():
                        logger.info(f"使用指定的引擎类型: {engine_type}")
                        try:
                            results = await engine.search(query, num_results)
                            self.last_used_engine = engine
                            self.last_search_time = time.time()
                            if results:
                                return [r.to_dict() for r in results]
                        except Exception as e:
                            logger.error(f"使用 {engine_type} 搜索失败: {str(e)}")
        
        # 如果没有指定引擎类型或者指定的引擎类型搜索失败，按优先级顺序搜索
        current_time = time.time()
        cooldown_elapsed = self.last_used_engine is None or (current_time - self.last_search_time) > self.cooldown_period
        
        # 尝试优先使用上次成功的引擎，如果冷却时间已过
        if self.last_used_engine and cooldown_elapsed and self.last_used_engine.is_available():
            try:
                logger.info(f"尝试使用上次成功的引擎: {self.last_used_engine.__class__.__name__}")
                results = await self.last_used_engine.search(query, num_results)
                if results:
                    self.last_search_time = current_time
                    return [r.to_dict() for r in results]
            except Exception as e:
                logger.warning(f"使用上次成功的引擎失败: {str(e)}")
        
        # 按优先级顺序尝试所有可用引擎
        for engine in self.engines:
            if engine.is_available():
                engine_name = engine.__class__.__name__
                logger.info(f"尝试使用引擎: {engine_name}")
                
                try:
                    results = await engine.search(query, num_results)
                    if results:
                        logger.success(f"{engine_name} 搜索成功，获取到 {len(results)} 个结果")
                        self.last_used_engine = engine
                        self.last_search_time = current_time
                        return [r.to_dict() for r in results]
                    else:
                        logger.warning(f"{engine_name} 搜索返回空结果")
                except Exception as e:
                    logger.error(f"{engine_name} 搜索出错: {str(e)}")
        
        logger.warning("所有搜索引擎都失败，无法获取结果")
        return []

    def get_available_engines(self) -> List[Dict]:
        """获取可用的搜索引擎列表"""
        engines = []
        for engine in self.engines:
            engine_name = engine.__class__.__name__
            engine_type = engine_name.lower().replace("search", "")
            available = engine.is_available()
            cooldown = None
            if engine.rate_limited_until:
                cooldown = (engine.rate_limited_until - datetime.now()).seconds if engine.rate_limited_until > datetime.now() else 0
                
            engines.append({
                "name": engine_name,
                "type": engine_type,
                "available": available,
                "priority": getattr(engine, "priority", 999),
                "cooldown_remaining": cooldown
            })
            
        return engines 