#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AgentDock Node Explore - Simple MCP Server
Simple MCP server implemented with Python standard library, supporting basic HTTP request handling
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)s: %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("simple-mcp-server")

# Store sessions
sessions = {}

# Load config file
def load_config(config_path=None):
    """Load configuration file"""
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    
    try:
        import toml
        config = toml.load(config_path)
        logger.info(f"Loaded config file: {config_path}")
        return config
    except ImportError:
        logger.warning("Cannot import toml library, using empty config")
        return {"mcpServers": {}}
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {"mcpServers": {}}

# Mock tool list
TOOLS = [
    {
        "name": "search",
        "description": "Search tool for finding information on the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "execute_code",
        "description": "Execute code",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "enum": ["python", "javascript", "bash"]
                },
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                }
            },
            "required": ["language", "code"]
        }
    }
]

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for handling requests"""
    daemon_threads = True

class MCPHandler(BaseHTTPRequestHandler):
    """MCP request handler"""
    
    def log_message(self, format, *args):
        """Custom log format"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-MCP-Session-ID, X-MCP-Streaming')
        self.send_header('Access-Control-Max-Age', '86400')
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def read_json_body(self):
        """Read and parse request body"""
        content_length = int(self.headers['Content-Length']) if 'Content-Length' in self.headers else 0
        if content_length > 0:
            body = self.rfile.read(content_length)
            try:
                return json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                logger.error("Cannot parse request body as JSON")
                return None
        return {}
    
    def do_OPTIONS(self):
        """Handle OPTIONS request"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Handle GET request"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Handle health check request
        if path == '/health':
            self.send_json_response({
                "status": "ok",
                "version": "1.0.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "connections": len(sessions)
            })
        
        # Handle SSE connection request (mock, not actual SSE stream)
        elif path == '/sse':
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "type": "sse",
                "created_at": time.time(),
                "last_active": time.time()
            }
            logger.info(f"New SSE session: {session_id}")
            
            self.send_json_response({
                "session_id": session_id,
                "message": "SSE connection established, use /messages endpoint to send requests"
            })
        
        # Handle root path request
        elif path == '/':
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AgentDock Simple MCP Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                </style>
            </head>
            <body>
                <h1>AgentDock Simple MCP Server</h1>
                <p>This is a simple MCP server implemented with Python standard library.</p>
                <p>Available endpoints:</p>
                <ul>
                    <li>/health - Health check</li>
                    <li>/sse - Establish SSE connection</li>
                    <li>/messages - Handle SSE messages</li>
                    <li>/mcp - Streamable HTTP endpoint</li>
                </ul>
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        
        else:
            self.send_json_response({"error": "Resource not found"}, 404)
    
    def do_POST(self):
        """Handle POST request"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        # Handle streamable-http request
        if path == '/mcp':
            # Get or create session ID
            session_id = self.headers.get('X-MCP-Session-ID')
            is_new_session = False
            
            if not session_id:
                session_id = str(uuid.uuid4())
                is_new_session = True
                logger.info(f"New streamable-http session: {session_id}")
            
            # Handle session initialization
            if is_new_session or session_id not in sessions:
                sessions[session_id] = {
                    "type": "streamable-http",
                    "created_at": time.time(),
                    "last_active": time.time()
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('X-MCP-Session-ID', session_id)
                self.send_cors_headers()
                self.end_headers()
                
                response = json.dumps({
                    "sessionId": session_id,
                    "message": "Connection successful"
                }).encode('utf-8')
                
                self.wfile.write(response)
                return
            
            # Update session active time
            sessions[session_id]["last_active"] = time.time()
            
            # Parse request body
            body = self.read_json_body()
            if not body:
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error: Invalid JSON"
                    },
                    "id": None
                }, 400)
                return
            
            # Handle JSON-RPC request
            jsonrpc = body.get("jsonrpc")
            method = body.get("method")
            params = body.get("params", {})
            id = body.get("id")
            
            if not jsonrpc or not method or id is None:
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid JSON-RPC request"
                    },
                    "id": id if id is not None else None
                }, 400)
                return
            
            # Handle tool list request
            if method == "tools/list" or method == "listTools":
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "result": {
                        "tools": TOOLS
                    },
                    "id": id
                })
            
            # Handle tool call request
            elif method == "tools/call" or method == "callTool":
                # Parse tool name
                tool_name = None
                tool_args = {}
                
                if "name" in params:
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})
                else:
                    tool_name = params.get("tool")
                    tool_args = params.get("params", {})
                
                if not tool_name:
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32602,
                            "message": "Missing tool name"
                        },
                        "id": id
                    }, 400)
                    return
                
                # Mock tool call
                if tool_name == "search":
                    query = tool_args.get("query", "")
                    result = f"Mock search result: '{query}'"
                    
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result
                                }
                            ]
                        },
                        "id": id
                    })
                
                elif tool_name == "execute_code":
                    language = tool_args.get("language", "")
                    code = tool_args.get("code", "")
                    result = f"Mock code execution ({language}):\n{code}\nExecution successful!"
                    
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result
                                }
                            ]
                        },
                        "id": id
                    })
                
                else:
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        },
                        "id": id
                    }, 400)
            
            # Handle other JSON-RPC methods
            else:
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not supported: {method}"
                    },
                    "id": id
                }, 400)
        
        # Handle SSE message request
        elif path == '/messages':
            # Get session ID from query params
            session_id = query.get('sessionId', [''])[0]
            
            if not session_id:
                self.send_json_response({
                    "error": "Missing session ID"
                }, 400)
                return
            
            if session_id not in sessions:
                self.send_json_response({
                    "error": "Invalid session ID"
                }, 404)
                return
            
            # Update session active time
            sessions[session_id]["last_active"] = time.time()
            
            # Parse request body
            body = self.read_json_body()
            if not body:
                self.send_json_response({
                    "error": "Invalid JSON request body"
                }, 400)
                return
            
            # Handle JSON-RPC request (similar to /mcp endpoint)
            jsonrpc = body.get("jsonrpc")
            method = body.get("method")
            params = body.get("params", {})
            id = body.get("id")
            
            if not jsonrpc or not method or id is None:
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid JSON-RPC request"
                    },
                    "id": id if id is not None else None
                }, 400)
                return
            
            # Handle same request types as /mcp
            if method == "tools/list" or method == "listTools":
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "result": {
                        "tools": TOOLS
                    },
                    "id": id
                })
            
            elif method == "tools/call" or method == "callTool":
                # Parse tool name
                tool_name = None
                tool_args = {}
                
                if "name" in params:
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})
                else:
                    tool_name = params.get("tool")
                    tool_args = params.get("params", {})
                
                if not tool_name:
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32602,
                            "message": "Missing tool name"
                        },
                        "id": id
                    }, 400)
                    return
                
                # Mock tool call
                if tool_name == "search":
                    query = tool_args.get("query", "")
                    result = f"Mock search result: '{query}'"
                    
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result
                                }
                            ]
                        },
                        "id": id
                    })
                
                elif tool_name == "execute_code":
                    language = tool_args.get("language", "")
                    code = tool_args.get("code", "")
                    result = f"Mock code execution ({language}):\n{code}\nExecution successful!"
                    
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result
                                }
                            ]
                        },
                        "id": id
                    })
                
                else:
                    self.send_json_response({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        },
                        "id": id
                    }, 400)
            
            # Handle other JSON-RPC methods
            else:
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not supported: {method}"
                    },
                    "id": id
                }, 400)
        
        else:
            self.send_json_response({"error": "Resource not found"}, 404)

def cleanup_sessions():
    """Clean up expired sessions"""
    while True:
        time.sleep(60)  # Check every minute
        now = time.time()
        inactive_timeout = 300  # 5 minutes inactive timeout
        
        inactive_sessions = []
        for session_id, session in sessions.items():
            if now - session["last_active"] > inactive_timeout:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            sessions.pop(session_id, None)

def run(port=8088):
    """Start server"""
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, MCPHandler)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
    cleanup_thread.start()
    
    logger.info(f"Starting server, listening on port {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        logger.info("Server closed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9088))
    run(port)
