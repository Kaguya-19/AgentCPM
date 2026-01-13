import os
import sys
import json
import asyncio
import argparse
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("mcp_handler")
logger.setLevel(logging.DEBUG)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MCPManager
from mcp_manager import MCPManager

class MCPHandler:
    """
    MCP Handler class
    
    Handles connection to MCP servers and tool calls using the MCPManager API
    """
    
    def __init__(self, server_name: str = "all",
                 config_file: str = "config.toml",
                 manager_url: str = "http://localhost:8000/mcpapi"):
        """
        Initialize MCP handler
        
        Args:
            server_name: Server name
            config_file: Configuration file path (for compatibility with old API, now replaced by manager_url)
            manager_url: URL for MCPManager API
        """
        self.server_name = server_name
        self.config_file = config_file
        self.manager_url = manager_url
        self.tools = []
        self.openai_tools = []
        self.tool_to_server_map = {}
        
        # MCPManager client instance
        self.manager = None
        
    async def initialize(self) -> bool:
        """
        Initialize connection and tools
        
        Returns:
            Initialization success status
        """
        try:
            # Create and initialize MCPManager client
            self.manager = MCPManager(manager_url=self.manager_url)
            
            if not await self.manager.initialize():
                logger.error("MCPManager initialization failed")
                return False
            
            # Get tool list
            if self.server_name.lower() == "all":
                servers = await self.manager.list_servers()
                logger.info(f"Connected to {len(servers)} servers")
                
                # Get all tools
                for server in servers:
                    server_tools = await self.manager.get_server_tools(server)
                    for tool in server_tools:
                        if "function" in tool:
                            # Extract name from tool
                            tool_name = tool["function"].get("name", "unknown")
                            # Save mapping between tool and server
                            self.tool_to_server_map[tool_name] = server
            else:
                # Get tools for a specific server only
                server_tools = await self.manager.get_server_tools(self.server_name)
                for tool in server_tools:
                    if "function" in tool:
                        # Extract name from tool
                        tool_name = tool["function"].get("name", "unknown")
                        # Save mapping between tool and server
                        self.tool_to_server_map[tool_name] = self.server_name
            
            # Get all OpenAI format tools
            self.openai_tools = self.manager.openai_tools
            logger.info(f"Initialized {len(self.openai_tools)} OpenAI format tools")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP handler: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call tool
        
        Args:
            tool_name: Tool name
            arguments: Argument dictionary
            
        Returns:
            Tool call result
        """
        try:
            logger.info(f"Calling tool: {tool_name}")
            logger.debug(f"Tool arguments: {arguments}")

            # Find corresponding server for the tool
            server_id = self.tool_to_server_map.get(tool_name)
            
            if not server_id:
                # If mapping is not found, try calling directly (MCPManager will attempt to find it)
                logger.warning(f"Server mapping for tool {tool_name} not found, trying to call directly")
                result = await self.manager.call_tool(tool_name, arguments)
            else:
                # Call tool using the found server ID
                result = await self.manager.call_tool(tool_name, arguments, server_id)
            
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
        # If response is already a dictionary, return directly
        if isinstance(response_content, dict):
            return response_content
            
        # If response is a string, try parsing as JSON
        if isinstance(response_content, str):
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                # If not valid JSON, wrap as dictionary
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
                logger.info("Closed MCPManager client")
            except Exception as e:
                logger.error(f"Error closing MCPManager client: {str(e)}")

# Simple test function
async def test_mcp_handler():
    """Test MCPHandler functionality"""
    handler = MCPHandler(server_name="all", manager_url="http://localhost:8000/mcpapi")
    
    if await handler.initialize():
        logger.info(f"Successfully initialized, found {len(handler.openai_tools)} tools")
        
        # Test calling tool
        if handler.openai_tools:
            sample_tool = handler.openai_tools[0]["function"]["name"]
            logger.info(f"Attempting to call tool: {sample_tool}")
            
            # Simple example parameters, real usage requires correct parameters based on tool needs
            result = await handler.call_tool(sample_tool, {"text": "Hello World"})
            logger.info(f"Tool call result: {result}")
    else:
        logger.error("Initialization failed")
    
    await handler.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP tool handler")
    parser.add_argument("--server", default="all", help="Server name to use")
    parser.add_argument("--manager-url", default="http://localhost:8000/mcpapi", help="MCPManager API URL")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_mcp_handler()) 
