#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Call the search tool of nlp-search-infra-server via MCPHandler
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # AgentCPM-MCP/
sys.path.insert(0, str(project_root))

# Import MCPHandler
from src.mcp_handler import MCPHandler

async def main():

    query = "sarabhai vs sarabhai monisha becomes sophisticated episode number"
    num_results = 10
    engine = "auto"
    use_jina = True
    
    # Create arguments dictionary
    arguments = {
        "query": query,
        "num_results": num_results,
        "engine": engine,
        "use_jina": use_jina
    }
    
    print(f"Preparing to call search tool with arguments: {json.dumps(arguments, ensure_ascii=False)}")
    
    # Note: Use "all" instead of a specific server name to fetch all available tools
    handler = MCPHandler(
        server_name="all", 
        manager_url="http://localhost:8000/mcpapi"
    )
    
    try:
        if not await handler.initialize():
            print("MCPHandler initialization failed")
            return 1
        
        print(f"Number of available tools: {len(handler.openai_tools)}")
        tool_names = [tool.get("function", {}).get("name", "") for tool in handler.openai_tools]
        print(f"Tool names list: {tool_names}")
        
        if "search" not in tool_names:
            print("Error: search tool is not available in the current server")
            return 1
        
        print("Starting to call search tool...")
        result = await handler.call_tool("search", arguments)
        
        if result.get("status") == "error":
            error_message = result.get("content", {}).get("error", "Unknown error")
            print(f"Failed to call search tool: {error_message}")
            return 1
        
        content = result.get("content", {})
        print("\nSearch Results:")
        print(json.dumps(content, ensure_ascii=False, indent=2))
        
        return 0
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    finally:
        # Close MCPHandler
        await handler.close()

if __name__ == "__main__":
    asyncio.run(main()) 
