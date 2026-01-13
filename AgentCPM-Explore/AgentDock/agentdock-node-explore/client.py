"""
AgentDock MCP Client
Multi-server MCP client supporting stdio, SSE, and streamable HTTP protocols
"""
import asyncio
import json5
import json
import jsonschema
import os
import logging
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool, Prompt, Resource
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


SERVER_RES_SPLIT = os.environ.get("SERVER_RES_SPLIT", '.')
logger = logging.getLogger()


class MCPClient:
    def __init__(self, config: dict[str, Any]):
        """Initialize MCP client with server configuration.
        
        Args:
            config: Dictionary of server configurations
        """
        self.sessions: Dict[str, ClientSession] = {}
        self.session_tools: Dict[str, List[Tool]] = {}
        self.session_prompts: Dict[str, List[Prompt]] = {}
        self.session_resources: Dict[str, List[Resource]] = {}
        self.exit_stack = AsyncExitStack()
        self.config = config

    async def connect_to_server(self, server_id: str):
        """Connect to an MCP server
        
        Args:
            server_id: ID of the server from config
        """
        assert server_id in self.config, f"Server ID '{server_id}' not found in config"

        server_config = self.config[server_id]
        if "command" in server_config:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            
        elif "url" in server_config:
            # Detect SSE URL - check if URL contains 'sse' or ends with '/sse'
            if "sse" in server_config["url"] or server_config["url"].endswith("/sse"):
                logger.info(f"Using SSE client to connect to server: {server_config['url']}")
                transports = await self.exit_stack.enter_async_context(sse_client(**server_config, timeout=150.0))
                session = await self.exit_stack.enter_async_context(ClientSession(*transports))
            else:
                logger.info(f"Using HTTP client to connect to server: {server_config['url']}")
                istream, ostream, get_session_id = await self.exit_stack.enter_async_context(
                    streamablehttp_client(**server_config, timeout=150.0)
                )
                session = await self.exit_stack.enter_async_context(ClientSession(istream, ostream))
        else:
            raise ValueError(f"Invalid server configuration for '{server_id}'")
        
        await session.initialize()
        
        # Store session with its ID
        self.sessions[server_id] = session
        
        # List available tools
        self.session_tools[server_id] = (await session.list_tools()).tools
        logger.warning(f"Connected to server '{server_id}' with tools: " + str([tool.name for tool in self.session_tools[server_id]]))
        
        # List available prompts
        try:
            self.session_prompts[server_id] = (await session.list_prompts()).prompts
        except Exception as e:
            logger.error(f"Error listing prompts for server '{server_id}': {e}")
            self.session_prompts[server_id] = []
            
        # List available resources
        try:
            self.session_resources[server_id] = (await session.list_resources()).resources
        except Exception as e:
            logger.error(f"Error listing resources for server '{server_id}': {e}")
            self.session_resources[server_id] = []

        return server_id

    async def init_all_sessions(self):
        """Initialize all sessions from config"""
        tasks = [self.connect_to_server(server_id) for server_id in self.config.keys()]
        await asyncio.gather(*tasks)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
    
    def list_tools(self):
        """List all tools and their descriptions"""
        tools_desc = []
        
        for server_id, tools in self.session_tools.items():
            for tool in tools:
                tools_desc.append({
                    "type": "function",
                    "function": {
                        "name": SERVER_RES_SPLIT.join([server_id, tool.name]),
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
        
        return tools_desc
    
    def list_prompts(self):
        """List all prompts from all sessions"""
        prompts = []
        
        for server_id, session in self.sessions.items():
            for prompt in self.session_prompts[server_id]:
                prompts.append({
                    "name": SERVER_RES_SPLIT.join([server_id, prompt.name]),
                    "description": prompt.description,
                    "parameters": [arg.model_dump(mode='json') for arg in prompt.arguments]
                })
        
        return prompts
    
    def list_resources(self):
        """List all resources from all sessions"""
        resources = []
        
        for server_id, session in self.sessions.items():
            for resource in self.session_resources[server_id]:
                resources.append({
                    'uri': resource.uri,
                    "name": SERVER_RES_SPLIT.join([server_id, resource.name]),
                    "description": resource.description,
                    'mimeType': resource.mimeType,
                    'size': resource.size,
                })
        
        return resources
    
    async def get_prompt(self, prompt_name: str, prompt_args: Optional[str] = None):
        """Get a prompt by ID
        
        Args:
            prompt_name: Name of the prompt to get (format: server_id.prompt_name)
            prompt_args: Optional arguments for the prompt
        """
        server_id, prompt_name = prompt_name.split(SERVER_RES_SPLIT)
        assert server_id in self.sessions, f"Session for server '{server_id}' not found"
        assert prompt_name in [prompt.name for prompt in self.session_prompts[server_id]], \
            f"Prompt '{prompt_name}' not found in server '{server_id}'"
        session = self.sessions[server_id]
        
        prompt = await session.get_prompt(prompt_name, prompt_args)
        return prompt
    
    async def read_resource(self, resource_name: str):
        """Get a resource by URI
        
        Args:
            resource_name: Name of the resource to get (format: server_id.resource_name)
        """
        server_id, resource_name = resource_name.split(SERVER_RES_SPLIT)
        assert server_id in self.sessions, f"Session for server '{server_id}' not found"
        
        resource_uri = None
        for resource in self.session_resources[server_id]:
            if resource.name == resource_name:
                resource_uri = resource.uri
                break
        assert resource_uri is not None, f"Resource '{resource_name}' not found in server '{server_id}'"
        session = self.sessions[server_id]
        
        resource = await session.read_resource(resource_uri)
        return resource
     
    async def call_tool(self, tool_name: str, tool_args: str):
        """Call a tool with given arguments
        
        Args:
            tool_name: Name of the tool to call (format: server_id.tool_name)
            tool_args: Arguments for the tool as JSON string
        """
        server_id, tool_name = tool_name.split(SERVER_RES_SPLIT)
        assert server_id in self.sessions, f"Session for server '{server_id}' not found"
        tool = None
        for t in self.session_tools[server_id]:
            if t.name == tool_name:
                tool = t
                break
        assert tool is not None, f"Tool '{tool_name}' not found in server '{server_id}'"
        session = self.sessions[server_id]
        
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError as e:
            try:
                tool_args = json5.loads(tool_args)
            except:
                raise e
        jsonschema.validate(tool_args, tool.inputSchema)
        print(tool_args)
        
        # Add timeout control
        try:
            response = await asyncio.wait_for(
                session.call_tool(tool.name, tool_args),
                timeout=180.0  # 180 second timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool call timed out after 180 seconds")
        
        return response
