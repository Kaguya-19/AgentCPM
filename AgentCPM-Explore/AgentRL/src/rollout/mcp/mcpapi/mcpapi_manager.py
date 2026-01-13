#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCPAPIManager Client

This client is used to interact with the MCPAPIManager HTTP API.
"""

import json
import logging
import asyncio
import httpx
import copy
from log import logger
from typing import Dict, List, Any, Optional, Union
from pathlib import Path



class MCPAPIManager:
    """
    MCPAPIManager Client Class
    
    Used to interact with the MCPAPIManager HTTP API, including querying servers,
    tools, and executing tool calls.
    """

    def __init__(self, manager_url: str = "http://localhost:8000/mcpapi", timeout: int = 150):
        """
        Initialize MCPAPIManager client

        Args:
            manager_url: Base URL of the MCPAPIManager API, defaults to http://localhost:8000/mcpapi
            timeout: Request timeout in seconds
        """
        self.manager_url = manager_url
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self.servers = {}
        self.tools_by_server = {} 
        self.all_tools = []
        # Compatible with original MCPHandler interface
        self.openai_tools = []
        # Blocked tools list (these tools will not appear in the tool list and cannot be called)
        self.blocked_tools = {"read_file"} #,"execute_code","PythonInterpreter"}#,"search","fetch_url"}
        # Tool name mapping: maps display names shown to the model to actual tool names
        # Example: model sees "visit", actually calls "fetch_url"
        self.tool_name_mapping = {
            "fetch_url": "fetch_url"  # display_name -> actual_tool_name (no mapping, use original tool name directly)
        }
        self.reverse_tool_name_mapping = {v: k for k, v in self.tool_name_mapping.items()}  # actual_tool_name -> display_name


    async def initialize(self) -> bool:
        """
        Initialize the client

        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Get available server list
            servers_response = await self.http_client.get(f"{self.manager_url}/servers")
            servers_response.raise_for_status()
            
            servers_data = servers_response.json()
            self.servers = {server: {} for server in servers_data.get("servers", [])}

            # Get tools from all servers
            tools_response = await self.http_client.get(f"{self.manager_url}/tools")
            tools_response.raise_for_status()
            
            tools_data = tools_response.json()
            self.all_tools = tools_data.get("tools", [])
            
            # Organize tools by server
            for server in self.servers.keys():
                try:
                    server_tools_response = await self.http_client.get(f"{self.manager_url}/server/{server}/tools")
                    server_tools_response.raise_for_status()
                    
                    server_tools_data = server_tools_response.json()
                    self.tools_by_server[server] = server_tools_data.get("tools", [])
                    
                    # Convert to OpenAI format tools for compatibility with existing code
                    for tool in server_tools_data.get("tools", []):
                        if "function" in tool:
                            tool_function = tool["function"]
                            actual_tool_name = tool_function.get("name")
                            
                            # Apply tool name mapping: replace actual tool name with display name
                            display_tool_name = self.reverse_tool_name_mapping.get(actual_tool_name, actual_tool_name)
                            
                            # Modify fetch_url tool description, add full content feature description
                            description = tool_function.get("description", "")
                            parameters = copy.deepcopy(tool_function.get("parameters", {}))
                            
                            # if actual_tool_name == "fetch_url":
                            #     description += "\n    Note: If you need the complete/full content without summarization, include keywords like 'full content', 'complete content', 'raw content', or 'entire content' in the 'purpose' parameter. The tool will return the raw content without AI summarization, even for long pages."
                            #     # Update purpose parameter description
                            #     if "properties" in parameters and "purpose" in parameters["properties"]:
                            #         purpose_desc = parameters["properties"]["purpose"].get("description", "")
                            #         if "full content" not in purpose_desc.lower() and "complete content" not in purpose_desc.lower():
                            #             parameters["properties"]["purpose"]["description"] = purpose_desc + " To get the complete/full content without summarization, include keywords like 'full content', 'complete content', 'raw content', or 'entire content' in this parameter."
                            
                            # Process execute_code tool definition conversion
                            display_tool_name, description, parameters = self._process_execute_code_tool(
                                actual_tool_name, display_tool_name, description, parameters
                            )
                            
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": display_tool_name,  # Use display name
                                    "description": description,
                                    "parameters": parameters
                                }
                            }
                            self.openai_tools.append(openai_tool)
                    
                except Exception as e:
                    logger.error(f"Error getting tools for server '{server}': {str(e)}")
                    self.tools_by_server[server] = []

            # Remove blocked tools from openai_tools
            self.openai_tools = [
                tool for tool in self.openai_tools
                if tool.get("function", {}).get("name") not in self.blocked_tools
            ]
            # if self.blocked_tools:
            #     logger.info(f"Removed blocked tools from tool list: {self.blocked_tools}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MCPAPIManager client: {str(e)}")
            return False
        
    async def list_servers(self) -> List[str]:
        """
        List all available MCP servers

        Returns:
            List[str]: List of server IDs
        """
        return list(self.servers.keys())
    
    async def get_server_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """
        Get tools on a specific server

        Args:
            server_id: Server ID

        Returns:
            List[Dict[str, Any]]: List of tool schemas
        """
        if server_id not in self.tools_by_server:
            logger.warning(f"Unknown server ID: {server_id}")
            return []
        
        return self.tools_by_server[server_id]
    
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from all servers

        Returns:
            List[Dict[str, Any]]: List of tool schemas
        """
        return self.all_tools
    
    async def call_tool(self, 
                       tool_id: str, 
                       arguments: Dict[str, Any], 
                       server_id: Optional[str] = None,
                       timeout: Optional[int] = None,
                       ) -> Dict[str, Any]:
        """
        Call an MCP tool

        Args:
            tool_id: Tool ID
            arguments: Tool arguments
            server_id: (Optional) Server ID, if not provided will try to auto-detect

        Returns:
            Dict[str, Any]: Tool response, always contains a "status" field ("success" or "error")
        """
        try:
            # If server ID and tool ID are provided, construct full tool call ID
            if server_id is not None and "." not in tool_id:
                full_tool_id = f"{server_id}.{tool_id}"
            else:
                full_tool_id = tool_id
            
            # Try to call tool, retry on timeout or request error
            max_retries = 3
            retry_count = 0
            last_exception = None
            backoff_factor = 1.5
            base_delay = 1
            
            while retry_count <= max_retries:
                try:
                    response = await self.http_client.post(
                        f"{self.manager_url}/tool/{full_tool_id}",
                        json=arguments,
                        timeout = timeout or self.timeout
                    )
                    
                    # Check HTTP status code
                    response.raise_for_status()
                    
                    # Parse response
                    result = response.json()
                    
                    # Check if response contains error information
                    if isinstance(result, dict):
                        # First check if response body has status_code and detail fields (server-side error format)
                        # Even if HTTP status code is 200, response body may contain status_code >= 400
                        status_code = result.get("status_code")
                        detail = result.get("detail")
                        
                        if status_code is not None and status_code >= 400 and detail:
                            # Server-side error, put detail into content and log
                            error_msg = detail if isinstance(detail, str) else str(detail)
                            logger.error(f"Tool {full_tool_id} returned error status code {status_code}: {error_msg}")
                            
                            # Construct standard error response format
                            result = {
                                "status": "error",
                                "content": {
                                    "error": error_msg,
                                    "detail": detail,
                                    "status_code": status_code,
                                    "tool_id": full_tool_id
                                }
                            }
                        # If status field already exists, check if it's an error
                        elif "status" in result and result["status"] == "error":
                            error_msg = result.get("content", {}).get("error", "Unknown error")
                            error_detail = result.get("content", {}).get("detail", "")
                            logger.error(f"Tool {full_tool_id} returned error status: {error_msg}")
                            if error_detail:
                                logger.error(f"Error detail: {error_detail}")
                        # If content contains error field, set status to error
                        elif isinstance(result.get("content"), dict) and "error" in result["content"]:
                            error_msg = result["content"]["error"]
                            logger.error(f"Tool {full_tool_id} response content contains error: {error_msg}")
                            # Add status field
                            if "status" not in result:
                                result["status"] = "error"
                        # If no error information, ensure status field exists
                        elif "status" not in result:
                            result["status"] = "success"
                    
                    # Check if content is empty dict, log error and return error message
                    if isinstance(result, dict) and result.get("status") == "success":
                        content = result.get("content", None)
                        if isinstance(content, dict) and len(content) == 0:
                            logger.error(f"Tool {full_tool_id} returned empty dict content{{}}")
                            # Build detailed error message
                            error_detail = f"Tool {full_tool_id} returned empty content"
                            if "fetch_url" in full_tool_id and isinstance(arguments, dict):
                                url = arguments.get('url', 'unknown')
                                purpose = arguments.get('purpose', 'unknown')
                                error_detail += f" (URL: {url}, purpose: {purpose})"
                            elif "search" in full_tool_id and isinstance(arguments, dict):
                                query = arguments.get('query', 'unknown')
                                error_detail += f" (query: {query})"
                            
                            result = {
                                "status": "error",
                                "content": {
                                    "error": "Tool returned empty content",
                                    "detail": error_detail
                                }
                            }
                    
                    return result
                    
                except httpx.ReadTimeout as e:
                    last_exception = e
                    # execute_code timeout does not retry, as code execution timeout usually means code has issues (e.g., infinite loop)
                    if "execute_code" in full_tool_id:
                        logger.error(f"Tool {full_tool_id} timed out, execute_code does not retry")
                        break
                    retry_count += 1
                    if retry_count <= max_retries:
                        delay = base_delay * (backoff_factor ** (retry_count - 1))
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Tool {full_tool_id} timed out, reached maximum retry count")
                        break
                except httpx.RequestError as e:
                    last_exception = e
                    retry_count += 1
                    if retry_count <= max_retries:
                        delay = base_delay * (backoff_factor ** (retry_count - 1))
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Tool {full_tool_id} request error, reached maximum retry count: {str(e)}")
                        break
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    # 5xx errors (server errors) can retry, 4xx errors (client errors) do not retry
                    if 500 <= status_code < 600:
                        last_exception = e
                        retry_count += 1
                        if retry_count <= max_retries:
                            delay = base_delay * (backoff_factor ** (retry_count - 1))
                            logger.warning(f"Tool {full_tool_id} HTTP {status_code} error, retrying after {delay:.2f} seconds ({retry_count}/{max_retries})")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"Tool {full_tool_id} HTTP {status_code} error, reached maximum retry count")
                            break
                    else:
                        # 4xx errors (client errors) do not retry, return directly
                        logger.error(f"Tool {full_tool_id} HTTP error: {status_code} - {e.response.text}")
                        return {
                            "status": "error",
                            "content": {
                                "error": f"HTTP error {status_code}",
                                "detail": e.response.text,
                            }
                        }
            
            # If all retries failed, return error
            if isinstance(last_exception, httpx.ReadTimeout):
                error_type = "timeout"
            elif isinstance(last_exception, httpx.HTTPStatusError):
                error_type = f"HTTP {last_exception.response.status_code} error"
            else:
                error_type = "request error"
            return {
                "status": "error",
                "content": {
                    "error": f"Tool call {error_type}",
                    "detail": f"Calling {full_tool_id} {error_type}, retried {max_retries} times: {str(last_exception) if last_exception else 'Unknown error'}",
                }
            }
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_id}': {str(e)}")
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error details: {tb}")
            
            return {
                "status": "error",
                "content": {
                    "error": str(e),
                    "detail": "Internal error occurred while calling tool",
                    "traceback": tb,
                    "tool_id": tool_id,
                    "server_id": server_id
                }
            }
    
    async def _reconnect(self):
        """
        Rebuild HTTP client connection
        Call this method when connection issues occur
        """
        try:
            # Close old connection
            if self.http_client:
                try:
                    await self.http_client.aclose()
                except Exception as e:
                    logger.warning(f"Error closing old connection: {e}")
            
            # Create new connection
            self.http_client = httpx.AsyncClient(timeout=self.timeout)
            logger.info("HTTP client connection rebuilt")
        except Exception as e:
            logger.error(f"Error rebuilding HTTP client connection: {e}")
            raise
    
    async def close(self):
        """close HTTP client"""
        await self.http_client.aclose() 

    def _process_execute_code_tool(
        self,
        actual_tool_name: str,
        display_tool_name: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> tuple[str, str, Dict[str, Any]]:
        """
        Process execute_code tool definition conversion, aligned with pre-training
        
        Args:
            actual_tool_name: Actual tool name
            display_tool_name: Display tool name
            description: Tool description
            parameters: Tool parameters
            
        Returns:
            tuple: (Updated display_tool_name, updated description, updated parameters)
        """
        if actual_tool_name == "execute_code":
            display_tool_name = "PythonInterpreter"
            description = QWEN_CODE_TOOL_DESCRIPTION
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        return display_tool_name, description, parameters

QWEN_CODE_TOOL_DESCRIPTION = """Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function like : print(f"The result is: {np.mean([1,2,3])}").

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>"""
