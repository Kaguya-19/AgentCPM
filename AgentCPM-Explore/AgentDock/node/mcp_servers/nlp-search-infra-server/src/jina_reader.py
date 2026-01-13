#!/usr/bin/env python

import httpx
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from urllib.parse import quote

# 配置日志
logger = logging.getLogger("jina_reader")

# Jina Reader API 端点
JINA_READ_API = "https://r.jina.ai/"
JINA_SEARCH_API = "https://s.jina.ai/"

# 默认超时时间（秒）
DEFAULT_TIMEOUT = 30

class JinaReader:
    """Jina Reader API 客户端，用于获取适合 LLM 的网页内容和搜索结果"""
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """初始化 Jina Reader 客户端
        
        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
    
    async def fetch_url(self, url: str, with_image_alt: bool = False, 
                       response_format: str = "markdown") -> str:
        """获取网页内容，转换为适合 LLM 的格式
        
        Args:
            url: 要获取的网页 URL
            with_image_alt: 是否为图片生成替代文本
            response_format: 响应格式，可选值: markdown, html, text
            
        Returns:
            处理后的网页内容
        """
        # 构建 Jina Reader URL
        jina_url = f"{JINA_READ_API}{url}"
        
        # 设置请求头
        headers = {}
        
        # 添加图片替代文本选项
        if with_image_alt:
            headers["x-with-generated-alt"] = "true"
        
        # 添加响应格式选项
        if response_format in ["markdown", "html", "text"]:
            headers["x-respond-with"] = response_format
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(jina_url, headers=headers)
                response.raise_for_status()
                return response.text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP错误: {e.response.status_code} - {e.response.text}")
            return f"获取URL内容失败: HTTP错误 {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"请求错误: {str(e)}")
            return f"获取URL内容失败: {str(e)}"
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            return f"获取URL内容失败: {str(e)}"
    
    async def search(self, query: str, sites: Optional[List[str]] = None, 
                   json_format: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """搜索网页内容
        
        Args:
            query: 搜索查询
            sites: 限制搜索的网站列表
            json_format: 是否返回 JSON 格式
            
        Returns:
            搜索结果，如果 json_format 为 True，则返回结果列表，否则返回 Markdown 文本
        """
        # 对查询进行 URL 编码
        encoded_query = quote(query)
        
        # 构建 Jina Search URL
        jina_url = f"{JINA_SEARCH_API}{encoded_query}"
        
        # 添加站点参数
        params = {}
        if sites:
            params = {f"site": site for site in sites}
        
        # 设置请求头
        headers = {}
        if json_format:
            headers["Accept"] = "application/json"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(jina_url, params=params, headers=headers)
                response.raise_for_status()
                
                if json_format:
                    return response.json()
                else:
                    return response.text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP错误: {e.response.status_code} - {e.response.text}")
            return f"搜索失败: HTTP错误 {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"请求错误: {str(e)}")
            return f"搜索失败: {str(e)}"
        except Exception as e:
            logger.error(f"未知错误: {str(e)}")
            return f"搜索失败: {str(e)}"

# 使用示例
async def main():
    reader = JinaReader()
    
    # 获取网页内容示例
    content = await reader.fetch_url("https://github.com/jina-ai/reader", with_image_alt=True)
    print("网页内容:", content[:500], "...\n")
    
    # 搜索示例
    search_results = await reader.search("Jina AI Reader", sites=["github.com"])
    print("搜索结果:", search_results[:500], "...")

if __name__ == "__main__":
    asyncio.run(main()) 