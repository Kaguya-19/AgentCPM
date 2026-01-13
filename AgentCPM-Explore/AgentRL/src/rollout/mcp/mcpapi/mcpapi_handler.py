import os
import sys
import json
import asyncio
import argparse
from log import logger
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import MCPAPIManager
from .mcpapi_manager import MCPAPIManager

class MCPAPIHandler:
    """
    MCP Handler Class
    
    Handles connections with MCP servers and tool calls, using MCPAPIManager API
    """
    
    def __init__(self, server_name: str = "all",
                 config_file: str = "config.toml",
                 manager_url: str = "http://localhost:9876/mcpapi"):
        """
        Initialize MCP handler
        
        Args:
            server_name: Server name
            config_file: Config file path (for compatibility with old API, now replaced by manager_url)
            manager_url: MCPAPIManager API URL
        """
        self.server_name = server_name
        self.config_file = config_file
        self.manager_url = manager_url
        self.tools = []
        self.openai_tools = []
        self.tool_to_server_map = {}
        
        # Tool name mapping: maps display names shown to the model to actual tool names
        # Example: model sees "visit", actually calls "fetch_url"
        self.tool_name_mapping = {
            "fetch_url": "fetch_url"  # display_name -> actual_tool_name
        }
        self.reverse_tool_name_mapping = {v: k for k, v in self.tool_name_mapping.items()}  # actual_tool_name -> display_name
        
        # Blocked tools list (these tools will not appear in the tool list and cannot be called)
        # Use set to avoid NoneType iteration errors
        self.blocked_tools = set()
        
        # MCPAPIManager client instance
        self.manager = None
        self.session_id = str(uuid.uuid4())
        
    async def initialize(self) -> bool:
        """
        Initialize connection and tools
        
        Returns:
            Whether initialization was successful
        """
        try:
            # Ensure blocked_tools is an iterable set
            if self.blocked_tools is None:
                self.blocked_tools = set()

            # Create and initialize MCPAPIManager client
            self.manager = MCPAPIManager(manager_url=self.manager_url)
            
            if not await self.manager.initialize():
                logger.error("MCPAPIManager initialization failed")
                return False
            
            # Get tool list
            if self.server_name.lower() == "all":
                servers = await self.manager.list_servers()
                
                # Get all tools
                for server in servers:
                    server_tools = await self.manager.get_server_tools(server)
                    for tool in server_tools:
                        if "function" in tool:
                            # Extract name from tool (this is the actual tool name, e.g., fetch_url)
                            actual_tool_name = tool["function"].get("name", "unknown")
                            
                            # Skip blocked tools
                            if actual_tool_name in self.blocked_tools:
                                continue
                            
                        # Save mapping between actual tool name and server
                        self.tool_to_server_map[actual_tool_name] = server
                        # logger.debug(f"Saved tool mapping: {actual_tool_name} -> {server}")
                        # If tool name has mapping, also save display name to server mapping
                        if actual_tool_name in self.reverse_tool_name_mapping:
                            display_name = self.reverse_tool_name_mapping[actual_tool_name]
                            self.tool_to_server_map[display_name] = server
                            # logger.debug(f"Saved tool display name mapping: {display_name} -> {server} (actual tool: {actual_tool_name})")
            else:
                # Only get tools from specific server
                server_tools = await self.manager.get_server_tools(self.server_name)
                for tool in server_tools:
                    if "function" in tool:
                        # Extract name from tool (this is the actual tool name, e.g., fetch_url)
                        actual_tool_name = tool["function"].get("name", "unknown")
                        
                        # Skip blocked tools
                        if actual_tool_name in self.blocked_tools:
                            continue
                        
                        # Save mapping between actual tool name and server
                        self.tool_to_server_map[actual_tool_name] = self.server_name
                        # logger.debug(f"Saved tool mapping: {actual_tool_name} -> {self.server_name}")
                        # If tool name has mapping, also save display name to server mapping
                        if actual_tool_name in self.reverse_tool_name_mapping:
                            display_name = self.reverse_tool_name_mapping[actual_tool_name]
                            self.tool_to_server_map[display_name] = self.server_name
                            # logger.debug(f"Saved tool display name mapping: {display_name} -> {self.server_name} (actual tool: {actual_tool_name})")
            
            # Get all OpenAI format tools (MCPAPIManager has already applied tool name mapping and filtering)
            self.openai_tools = self.manager.openai_tools
            
            # Filter again to ensure blocked tools are not in the list
            self.openai_tools = [
                tool for tool in self.openai_tools
                if tool.get("function", {}).get("name") not in self.blocked_tools
            ]
            
            return True
            
        except Exception as e:
            logger.error(f"MCP handler initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool
        
        Args:
            tool_name: Tool name (may be display name, e.g., "visit")
            arguments: Arguments dictionary
            
        Returns:
            Tool call result
        """
        try:
            #arguments["session_id"] = self.session_id
            
            # Check if tool is blocked
            if tool_name in self.blocked_tools:
                logger.warning(f"Attempted to call blocked tool: {tool_name}")
                return {
                    "status": "error",
                    "content": {
                        "error": f"Tool {tool_name} is blocked and cannot be used"
                    }
                }
            
            # Map display name to actual tool name
            actual_tool_name = self.tool_name_mapping.get(tool_name, tool_name)
            # logger.debug(f"Tool call mapping: {tool_name} -> {actual_tool_name}")
            
            # Check if actual tool name is blocked
            if actual_tool_name in self.blocked_tools:
                logger.warning(f"Attempted to call blocked tool: {tool_name} (actual: {actual_tool_name})")
                return {
                    "status": "error",
                    "content": {
                        "error": f"Tool {actual_tool_name} is blocked and cannot be used"
                    }
                }
            
            # Find server for tool (using actual tool name)
            server_id = self.tool_to_server_map.get(actual_tool_name)
            if not server_id:
                # If not found with actual tool name, try using display name
                server_id = self.tool_to_server_map.get(tool_name)
            
            # logger.debug(f"Tool {tool_name} (actual: {actual_tool_name}) mapped to server: {server_id}")
            logger.debug(f"Available tool mappings: {list(self.tool_to_server_map.keys())}")
            
            if not server_id:
                # If mapping not found, try direct call (MCPAPIManager will try to find it)
                logger.warning(f"Server mapping not found for tool {tool_name} (actual: {actual_tool_name}), trying direct call")
                result = await self.manager.call_tool(actual_tool_name, arguments)
            else:
                # Call tool using found server ID and actual tool name
                result = await self.manager.call_tool(actual_tool_name, arguments, server_id)
            
            # Process return result
            if result.get("status") == "error":
                error_content = result.get("content", {})
                error_message = error_content.get("error", "Unknown error")
                logger.error(f"Tool call error: {error_message}")
                
                # Construct error response
                return {
                    "status": "error",
                    "content": {
                        "error": error_message
                    }
                }
                
            # Tool call successful, return content
            content = result.get("content", {})
            
            formatted_response = self._format_tool_response(content)
            
            # No longer truncating, return full content directly

            return {
                "status": "success",
                "content": formatted_response
            }
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error details: {tb}")
            
            return {
                "status": "error",
                "content": {
                    "error": str(e)
                }
            }
            
    def _format_tool_response(self, response_content):
        """
        Format tool response
        
        Args:
            response_content: Tool response content
            
        Returns:
            Formatted response
        """
        # If response is already a dict, return directly
        if isinstance(response_content, dict):
            return response_content
            
        # If response is a string, try to parse as JSON
        if isinstance(response_content, str):
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                # If not valid JSON, wrap as dict
                return {"result": response_content}
                
        # Other types, wrap directly
        return {"result": response_content}
             
    async def close(self) -> None:
        """
        Close connection and resources
        """
        if self.manager:
            try:
                await self.manager.close()
            except Exception as e:
                logger.error(f"Error closing MCPAPIManager client: {str(e)}")
