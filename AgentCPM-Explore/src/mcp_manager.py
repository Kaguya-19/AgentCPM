#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCPManager Client

This client is used to interact with the HTTP API of MCPManager
"""

import os
import json
import logging
import asyncio
import httpx
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger("mcp_manager")


class MCPManager:
    """
    MCPManager Client class
    Used to interact with MCPManager's HTTP API, including querying servers, tools, and executing tool calls
    """

    def __init__(self, manager_url: str = "http://localhost:8000/mcpapi", timeout: int = 30):
        """
        Initialize MCPManager client

        Args:
            manager_url: Base URL for MCPManager API, default is http://localhost:8000/mcpapi
            timeout: Request timeout (seconds)
        """
        self.manager_url = manager_url
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self.servers = {}
        self.tools_by_server = {}
        self.all_tools = []
        
        # Compatibility with original MCPHandler interface
        self.openai_tools = []

    async def initialize(self) -> bool:
        """
        Initialize client

        Returns:
            bool: Initialization success status
        """
        try:
            # Get list of available servers
            logger.info(f"Initializing MCPManager client from {self.manager_url}...")
            servers_response = await self.http_client.get(f"{self.manager_url}/servers")
            servers_response.raise_for_status()
            
            servers_data = servers_response.json()
            self.servers = {server: {} for server in servers_data.get("servers", [])}
            logger.info(f"Fetched {len(self.servers)} MCP servers")

            # Get tools on all servers
            tools_response = await self.http_client.get(f"{self.manager_url}/tools")
            tools_response.raise_for_status()
            
            tools_data = tools_response.json()
            self.all_tools = tools_data.get("tools", [])
            logger.info(f"Fetched {len(self.all_tools)} MCP tools")
            
            # Organize tools by server
            for server in self.servers.keys():
                try:
                    server_tools_response = await self.http_client.get(f"{self.manager_url}/server/{server}/tools")
                    server_tools_response.raise_for_status()
                    
                    server_tools_data = server_tools_response.json()
                    self.tools_by_server[server] = server_tools_data.get("tools", [])
                    logger.info(f"Server '{server}' has {len(self.tools_by_server[server])} tools")
                    
                    # Convert to OpenAI format tools for compatibility with existing code
                    for tool in server_tools_data.get("tools", []):
                        if "function" in tool:
                            tool_function = tool["function"]
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool_function.get("name"),
                                    "description": tool_function.get("description", ""),
                                    "parameters": tool_function.get("parameters", {})
                                }
                            }
                            self.openai_tools.append(openai_tool)
                    
                except Exception as e:
                    logger.error(f"Error fetching tools for server '{server}': {str(e)}")
                    self.tools_by_server[server] = []

            logger.info(f"Converted {len(self.openai_tools)} tools to OpenAI format")
            return True
        except Exception as e:
            logger.error(f"Error initializing MCPManager client: {str(e)}")
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
        Get all tools on all servers

        Returns:
            List[Dict[str, Any]]: List of tool schemas
        """
        return self.all_tools
    
    async def call_tool(self, 
                       tool_id: str, 
                       arguments: Dict[str, Any], 
                       server_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Call MCP tool

        Args:
            tool_id: Tool ID
            arguments: Tool arguments
            server_id: (Optional) Server ID, if not provided, attempts to find automatically

        Returns:
            Dict[str, Any]: Tool response, always includes status field ("success" or "error")
        """
        try:
            # If server ID and tool ID are provided, construct full tool call ID
            if server_id is not None and "." not in tool_id:
                full_tool_id = f"{server_id}.{tool_id}"
            else:
                full_tool_id = tool_id
            
            # Call tool
            logger.info(f"Calling tool: {full_tool_id}, arguments: {arguments}")
            
            try:
                response = await self.http_client.post(
                    f"{self.manager_url}/tool/{full_tool_id}",
                    json=arguments,
                    timeout=self.timeout
                )
                
                # Check HTTP status code
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                logger.debug(f"Tool call result: {result}")
                
                # Check if response contains error information
                if isinstance(result, dict):
                    # If status field already exists, check if it's error
                    if "status" in result and result["status"] == "error":
                        error_msg = result.get("content", {}).get("error", "Unknown error")
                        error_detail = result.get("content", {}).get("detail", "")
                        logger.error(f"Tool {full_tool_id} returned error status: {error_msg}")
                        if error_detail:
                            logger.error(f"Error details: {error_detail}")
                    # If content contains error field, set status to error
                    elif isinstance(result.get("content"), dict) and "error" in result["content"]:
                        error_msg = result["content"]["error"]
                        logger.error(f"Response content of tool {full_tool_id} contains error: {error_msg}")
                        # Add status field
                        if "status" not in result:
                            result["status"] = "error"
                    # If no error information, ensure status field exists
                    elif "status" not in result:
                        result["status"] = "success"
                
                return result
                
            except httpx.ReadTimeout:
                logger.error(f"Call to tool {full_tool_id} timed out")
                return {
                    "status": "error",
                    "content": {
                        "error": "Tool call timeout",
                        "detail": f"Call to {full_tool_id} timed out, please try again later",
                        "tool_id": full_tool_id
                    }
                }
            except httpx.HTTPStatusError as e:
                logger.error(f"Call to tool {full_tool_id} HTTP error: {e.response.status_code} - {e.response.text}")
                return {
                    "status": "error",
                    "content": {
                        "error": f"HTTP Error {e.response.status_code}",
                        "detail": e.response.text,
                        "tool_id": full_tool_id
                    }
                }
            except httpx.RequestError as e:
                logger.error(f"Call to tool {full_tool_id} request error: {str(e)}")
                return {
                    "status": "error",
                    "content": {
                        "error": f"Request error",
                        "detail": str(e),
                        "tool_id": full_tool_id
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
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose() 
