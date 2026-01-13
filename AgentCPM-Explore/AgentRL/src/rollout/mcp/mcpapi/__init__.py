"""
MCPAPI module - A customized MCP (Model Context Protocol) communication protocol.

This module implements a customized MCP communication protocol that extends the standard
MCP with the following features:
- Multi-server aggregation: Supports merging tools from multiple MCP servers
- Server selection: Allows dynamic selection of MCP servers
- Tool selection: Allows dynamic selection and filtering of available tools

Main components:
- MCPAPIHandler: Handles communication with individual MCP servers
- MCPAPIManager: Manages multiple MCP servers and aggregates their tools
- parse_tool: Parses tool calls from various model formats
- to_dict: Converts tool call objects to dictionaries
"""

from .mcpapi_handler import MCPAPIHandler
from .mcpapi_manager import MCPAPIManager
from .tool_parser import parse_tool
from .tool_parser.parse_openai import to_dict

__all__ = ["MCPAPIHandler", "MCPAPIManager", "parse_tool", "to_dict"]