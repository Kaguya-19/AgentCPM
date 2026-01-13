#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import httpx
import json
import os
import time
import asyncio
from loguru import logger
import backoff
from duckduckgo_search import DDGS
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests

# 添加Exa导入
try:
    from exa_py import Exa
except ImportError:
    logger.warning("未找到exa-py库，ExaSearch将不可用，请使用'pip install exa-py'安装")


class SearchResult:
    """统一的搜索结果格式"""
    
    def __init__(self, title: str, link: str, snippet: str, source: str):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.source = source

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "source": self.source
        }


class SearchEngine(ABC):
    """搜索引擎接口"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.error_count = 0
        self.rate_limited_until = None
        self.max_error_count = 3
        self.rate_limit_cooldown = 300  # 默认冷却时间，单位秒
    
    def is_available(self) -> bool:
        """检查搜索引擎是否可用"""
        if self.rate_limited_until and datetime.now() < self.rate_limited_until:
            logger.warning(f"{self.name} 处于限速冷却中，还需等待 {(self.rate_limited_until - datetime.now()).seconds} 秒")
            return False
        if self.error_count >= self.max_error_count:
            # 重置错误计数，并设置冷却时间
            logger.warning(f"{self.name} 错误次数过多，设置冷却期")
            self.error_count = 0
            self.rate_limited_until = datetime.now() + timedelta(seconds=self.rate_limit_cooldown)
            return False
        return True
        
    def record_error(self, is_rate_limit: bool = False):
        """记录错误，如果是限速错误则立即进入冷却期"""
        self.error_count += 1
        if is_rate_limit:
            self.rate_limited_until = datetime.now() + timedelta(seconds=self.rate_limit_cooldown)
            logger.warning(f"{self.name} 遇到限速错误，设置冷却期 {self.rate_limit_cooldown} 秒")
            self.error_count = 0  # 重置错误计数
    
    def record_success(self):
        """记录成功的请求，重置错误计数"""
        self.error_count = 0
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """执行搜索并返回结果"""
        pass


class GoogleSearch(SearchEngine):
    """Google搜索实现"""
    
    def __init__(self, api_key: str, cse_id: str):
        super().__init__()
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.priority = 1  # 最高优先级

    @backoff.on_exception(backoff.expo, 
                         (httpx.RequestError, httpx.HTTPStatusError), 
                         max_tries=3, 
                         giveup=lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [403, 429])
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用Google Custom Search API执行搜索"""
        if not self.api_key or not self.cse_id:
            logger.warning("Google搜索凭据未配置")
            return []
            
        if not self.is_available():
            return []
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                params = {
                    'key': self.api_key,
                    'cx': self.cse_id,
                    'q': query,
                    'num': min(num_results, 10)  # Google API限制每页最多10个结果
                }
                
                logger.info(f"发送Google搜索请求: {query}")
                response = await client.get(self.base_url, params=params)
                
                # 检查限速和错误
                if response.status_code in [403, 429]:
                    self.record_error(is_rate_limit=True)
                    logger.error(f"Google搜索限速错误: {response.status_code}")
                    return []
                    
                response.raise_for_status()
                data = response.json()
                
                self.record_success()  # 记录成功请求
                logger.info("Google搜索请求成功")
                
                results = []
                for item in data.get('items', []):
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        link=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source='google'
                    ))
                    
                return results
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code in [403, 429]:
                    self.record_error(is_rate_limit=True)
                    logger.error(f"Google搜索限速错误: {status_code}")
                else:
                    self.record_error()
                    logger.error(f"Google搜索HTTP错误: {status_code}")
                return []
            except Exception as e:
                self.record_error()
                logger.error(f"Google搜索失败: {str(e)}")
                return []


class DuckDuckGoSearch(SearchEngine):
    """DuckDuckGo搜索实现"""
    
    def __init__(self):
        super().__init__()
        self.ddgs = DDGS()
        self.priority = 2  # 第二优先级
        self.rate_limit_cooldown = 60  # 降低冷却时间到60秒
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用DuckDuckGo搜索"""
        if not self.is_available():
            return []
            
        try:
            # 使用duckduckgo_search库进行搜索
            raw_results = self.ddgs.text(
                keywords=query,
                region="wt-wt",  # 全球
                safesearch="moderate",
                max_results=num_results
            )
            
            # 直接使用结果迭代，不预先转换为列表
            results = []
            has_results = False
            
            for item in raw_results:
                has_results = True
                results.append(SearchResult(
                    title=item.get('title', ''),
                    link=item.get('href', ''),  # duckduckgo_search使用'href'作为链接字段
                    snippet=item.get('body', ''),  # duckduckgo_search使用'body'作为摘要字段
                    source='duckduckgo'
                ))
            
            # 如果没有结果，抛出异常
            if not has_results:
                raise Exception(f"DuckDuckGo搜索 '{query}' 返回空结果")
                
            self.record_success()  # 记录成功请求
            logger.info(f"DuckDuckGo搜索成功: {query}")
            
            return results
            
        except Exception as e:
            error_str = str(e).lower()
            # 检查是否是限速错误
            if "rate" in error_str or "limit" in error_str or "429" in error_str or "too many requests" in error_str:
                self.record_error(is_rate_limit=True)
                logger.error(f"DuckDuckGo搜索限速错误: {str(e)}")
            else:
                self.record_error()
                logger.error(f"DuckDuckGo搜索失败: {str(e)}")
            return []


class ExaSearch(SearchEngine):
    """Exa搜索实现"""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.exa_client = None
        self.priority = 2  # 与DuckDuckGo相同优先级
        self.rate_limit_cooldown = 60  # 冷却时间60秒
        
        # 初始化Exa客户端
        try:
            from exa_py import Exa
            self.exa_client = Exa(api_key=self.api_key)
            logger.info("Exa搜索客户端初始化成功")
        except ImportError:
            logger.error("未找到exa-py库，请使用'pip install exa-py'安装")
        except Exception as e:
            logger.error(f"Exa客户端初始化失败: {str(e)}")
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用Exa API执行搜索"""
        if not self.api_key or not self.exa_client:
            logger.warning("Exa搜索凭据未配置或客户端初始化失败")
            return []
            
        if not self.is_available():
            return []
            
        try:
            logger.info(f"发送Exa搜索请求: {query}")
            
            # 使用asyncio.to_thread将同步API调用转换为异步操作
            result = await asyncio.to_thread(
                self.exa_client.search_and_contents,
                query=query,
                text=True,
                num_results=num_results
            )
            
            self.record_success()  # 记录成功请求
            logger.info("Exa搜索请求成功")
            
            results = []
            if hasattr(result, 'results'):
                items = result.results
            elif isinstance(result, dict) and 'results' in result:
                items = result['results']
            else:
                items = result if isinstance(result, list) else []
                
            for item in items:
                # 提取文本内容
                text_content = ""
                if hasattr(item, 'text'):
                    text_content = item.text
                elif isinstance(item, dict):
                    text_content = item.get('text', '')
                
                # 提取标题和链接
                title = ""
                link = ""
                if hasattr(item, 'title'):
                    title = item.title
                elif isinstance(item, dict):
                    title = item.get('title', '')
                    
                if hasattr(item, 'url'):
                    link = item.url
                elif isinstance(item, dict):
                    link = item.get('url', '')
                
                # 如果没有标题，使用链接作为标题
                if not title and link:
                    title = link
                
                # 限制摘要长度
                snippet = text_content[:500] + "..." if len(text_content) > 500 else text_content
                
                results.append(SearchResult(
                    title=title,
                    link=link,
                    snippet=snippet,
                    source='exa'
                ))
                
            return results
                
        except Exception as e:
            error_str = str(e).lower()
            # 检查是否是限速错误
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                self.record_error(is_rate_limit=True)
                logger.error(f"Exa搜索限速错误: {str(e)}")
            else:
                self.record_error()
                logger.error(f"Exa搜索失败: {str(e)}")
            return []


class BingSearch(SearchEngine):
    """Bing搜索实现"""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        self.priority = 3  # 第三优先级

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用Bing API执行搜索"""
        if not self.api_key:
            logger.warning("Bing搜索凭据未配置")
            return []
            
        if not self.is_available():
            return []
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                headers = {
                    "Ocp-Apim-Subscription-Key": self.api_key
                }
                params = {
                    "q": query,
                    "count": min(num_results, 50)  # Bing API每页最多50个结果
                }
                
                logger.info(f"发送Bing搜索请求: {query}")
                response = await client.get(self.base_url, headers=headers, params=params)
                
                # 检查限速和错误
                if response.status_code in [403, 429]:
                    self.record_error(is_rate_limit=True)
                    logger.error(f"Bing搜索限速错误: {response.status_code}")
                    return []
                    
                response.raise_for_status()
                data = response.json()
                
                self.record_success()  # 记录成功请求
                logger.info("Bing搜索请求成功")
                
                results = []
                for item in data.get("webPages", {}).get("value", []):
                    results.append(SearchResult(
                        title=item.get("name", ""),
                        link=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        source="bing"
                    ))
                    
                return results
                
            except Exception as e:
                self.record_error()
                logger.error(f"Bing搜索失败: {str(e)}")
                return []


class BaiduSearch(SearchEngine):
    """百度搜索实现"""
    
    def __init__(self, api_key: str, secret_key: str):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = None
        self.token_expires = 0
        self.base_url = "https://aip.baidubce.com/rpc/2.0/creation/v1/search"
        self.priority = 3  # 第三优先级
        
    async def _ensure_token(self) -> bool:
        """确保有效的访问令牌"""
        current_time = time.time()
        if self.access_token and current_time < self.token_expires:
            return True
            
        if not self.api_key or not self.secret_key:
            logger.warning("百度搜索凭据未配置")
            return False
            
        token_url = f"https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(token_url, params=params)
                response.raise_for_status()
                result = response.json()
                self.access_token = result.get("access_token")
                expires_in = result.get("expires_in", 2592000)  # 默认30天
                self.token_expires = current_time + expires_in - 60  # 提前60秒过期
                return bool(self.access_token)
        except Exception as e:
            logger.error(f"获取百度访问令牌失败: {str(e)}")
            return False

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用百度搜索"""
        if not self.is_available():
            return []
            
        if not await self._ensure_token():
            return []
            
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            params = {
                "access_token": self.access_token
            }
            
            data = {
                "query": query,
                "domain": "general",
                "top_k": min(num_results, 50)  # 限制最大结果数
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.base_url, 
                    headers=headers, 
                    params=params,
                    json=data
                )
                
                # 检查限速和错误
                if response.status_code in [403, 429]:
                    self.record_error(is_rate_limit=True)
                    logger.error(f"百度搜索限速错误: {response.status_code}")
                    return []
                
                response.raise_for_status()
                result = response.json()
                
                self.record_success()  # 记录成功请求
                
                results = []
                for item in result.get("result", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        link=item.get("url", ""),
                        snippet=item.get("summary", ""),
                        source="baidu"
                    ))
                
                return results
                
        except Exception as e:
            self.record_error()
            logger.error(f"百度搜索失败: {str(e)}")
            return []


class WikiSearch(SearchEngine):
    """本地Wiki搜索实现"""
    
    def __init__(self, wiki_path: str):
        super().__init__()
        self.priority = 5  # 最低优先级
        self.wiki_path = wiki_path
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """在本地Wiki数据中搜索"""
        if not self.is_available():
            return []
            
        if not os.path.exists(self.wiki_path):
            logger.warning(f"Wiki路径不存在: {self.wiki_path}")
            return []
            
        try:
            # 简单的关键词匹配搜索
            results = []
            query_terms = query.lower().split()
            
            # 遍历wiki文件目录
            for root, dirs, files in os.walk(self.wiki_path):
                if len(results) >= num_results:
                    break
                    
                for file in files:
                    if not file.endswith('.json'):
                        continue
                        
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            wiki_data = json.load(f)
                            
                        for page in wiki_data.get('pages', []):
                            title = page.get('title', '')
                            text = page.get('text', '')
                            
                            # 简单的相关性评分
                            score = 0
                            for term in query_terms:
                                title_count = title.lower().count(term)
                                text_count = text.lower().count(term)
                                score += title_count * 3 + text_count
                                
                            if score > 0:
                                # 提取包含查询词的片段
                                snippet = ""
                                for term in query_terms:
                                    term_index = text.lower().find(term)
                                    if term_index >= 0:
                                        start = max(0, term_index - 100)
                                        end = min(len(text), term_index + 100)
                                        context = text[start:end].strip()
                                        if context:
                                            snippet = f"{context}..."
                                            break
                                
                                if not snippet and text:
                                    snippet = text[:200] + "..." if len(text) > 200 else text
                                
                                results.append((score, SearchResult(
                                    title=title,
                                    link=f"wiki://{file_path}#{title.replace(' ', '_')}",
                                    snippet=snippet,
                                    source="wiki"
                                )))
                            
                            if len(results) >= num_results * 2:  # 收集足够的候选结果
                                break
                    except Exception as e:
                        logger.error(f"处理Wiki文件 {file_path} 时出错: {str(e)}")
            
            # 按相关性分数排序
            results.sort(key=lambda x: x[0], reverse=True)
            
            self.record_success()
            return [result[1] for result in results[:num_results]]
            
        except Exception as e:
            self.record_error()
            logger.error(f"Wiki搜索失败: {str(e)}")
            return [] 