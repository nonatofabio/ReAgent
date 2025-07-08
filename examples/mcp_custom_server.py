"""
Example of creating a custom MCP server and integrating it with ReAgent.

This example demonstrates:
1. How to create a custom MCP server
2. How to configure ReAgent to use the custom MCP server
3. How to execute a task with the MCP-enabled orchestrator

For more information on the Model Context Protocol, see:
https://modelcontextprotocol.io/
"""

import asyncio
import json
import os
import sys
import tempfile
from typing import Dict, List, Any, Optional

from reagent import ReactiveSwarmOrchestrator
from reagent.core.orchestrator import SwarmConfiguration, CoordinationPattern
from reagent.utils.mcp import create_subprocess_mcp_transport


# Example MCP server implementation
class SimpleMCPServer:
    """
    A simple MCP server implementation that provides file system access.
    
    This is a minimal example to demonstrate the MCP protocol. In a real
    implementation, you would want to add proper error handling, security
    measures, and more robust functionality.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.tools = {
            "list_files": {
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory"
                        }
                    },
                    "required": ["path"]
                }
            },
            "read_file": {
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        if request.get("type") == "list_tools":
            return self._handle_list_tools()
        elif request.get("type") == "tool_call":
            return self._handle_tool_call(request)
        else:
            return {
                "type": "error",
                "error": {
                    "message": f"Unknown request type: {request.get('type')}"
                }
            }
    
    def _handle_list_tools(self) -> Dict[str, Any]:
        """Handle a list_tools request."""
        return {
            "type": "list_tools_response",
            "tools": [
                {
                    "name": name,
                    "description": tool["description"],
                    "input_schema": tool["parameters"]
                }
                for name, tool in self.tools.items()
            ]
        }
    
    def _handle_tool_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool_call request."""
        tool_name = request.get("name")
        tool_id = request.get("id")
        tool_params = request.get("parameters", {})
        
        if tool_name not in self.tools:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {tool_name}"
                    }
                ]
            }
        
        try:
            if tool_name == "list_files":
                return self._list_files(tool_id, tool_params)
            elif tool_name == "read_file":
                return self._read_file(tool_id, tool_params)
            else:
                return {
                    "type": "tool_call_response",
                    "id": tool_id,
                    "status": "error",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool not implemented: {tool_name}"
                        }
                    ]
                }
        except Exception as e:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool: {str(e)}"
                    }
                ]
            }
    
    def _list_files(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """List files in a directory."""
        path = params.get("path", ".")
        
        try:
            files = os.listdir(path)
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "success",
                "content": [
                    {
                        "type": "text",
                        "text": f"Files in {path}:\n" + "\n".join(files)
                    }
                ]
            }
        except Exception as e:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Error listing files: {str(e)}"
                    }
                ]
            }
    
    def _read_file(self, tool_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read the contents of a file."""
        path = params.get("path")
        
        try:
            with open(path, "r") as f:
                content = f.read()
            
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "success",
                "content": [
                    {
                        "type": "text",
                        "text": f"Contents of {path}:\n{content}"
                    }
                ]
            }
        except Exception as e:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Error reading file: {str(e)}"
                    }
                ]
            }


def run_mcp_server():
    """Run the MCP server."""
    server = SimpleMCPServer()
    
    # Read from stdin and write to stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = server.handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            print(json.dumps({
                "type": "error",
                "error": {
                    "message": "Invalid JSON"
                }
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "type": "error",
                "error": {
                    "message": f"Error: {str(e)}"
                }
            }), flush=True)


async def main():
    """Run the example."""
    # Create a temporary script file for the MCP server
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write("""
import json
import os
import sys

class SimpleMCPServer:
    def __init__(self):
        self.tools = {
            "list_files": {
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory"
                        }
                    },
                    "required": ["path"]
                }
            },
            "read_file": {
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    
    def handle_request(self, request):
        if request.get("type") == "list_tools":
            return self._handle_list_tools()
        elif request.get("type") == "tool_call":
            return self._handle_tool_call(request)
        else:
            return {
                "type": "error",
                "error": {
                    "message": f"Unknown request type: {request.get('type')}"
                }
            }
    
    def _handle_list_tools(self):
        return {
            "type": "list_tools_response",
            "tools": [
                {
                    "name": name,
                    "description": tool["description"],
                    "input_schema": tool["parameters"]
                }
                for name, tool in self.tools.items()
            ]
        }
    
    def _handle_tool_call(self, request):
        tool_name = request.get("name")
        tool_id = request.get("id")
        tool_params = request.get("parameters", {})
        
        if tool_name not in self.tools:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {tool_name}"
                    }
                ]
            }
        
        try:
            if tool_name == "list_files":
                return self._list_files(tool_id, tool_params)
            elif tool_name == "read_file":
                return self._read_file(tool_id, tool_params)
            else:
                return {
                    "type": "tool_call_response",
                    "id": tool_id,
                    "status": "error",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Tool not implemented: {tool_name}"
                        }
                    ]
                }
        except Exception as e:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool: {str(e)}"
                    }
                ]
            }
    
    def _list_files(self, tool_id, params):
        path = params.get("path", ".")
        
        try:
            files = os.listdir(path)
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "success",
                "content": [
                    {
                        "type": "text",
                        "text": f"Files in {path}:\\n" + "\\n".join(files)
                    }
                ]
            }
        except Exception as e:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Error listing files: {str(e)}"
                    }
                ]
            }
    
    def _read_file(self, tool_id, params):
        path = params.get("path")
        
        try:
            with open(path, "r") as f:
                content = f.read()
            
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "success",
                "content": [
                    {
                        "type": "text",
                        "text": f"Contents of {path}:\\n{content}"
                    }
                ]
            }
        except Exception as e:
            return {
                "type": "tool_call_response",
                "id": tool_id,
                "status": "error",
                "content": [
                    {
                        "type": "text",
                        "text": f"Error reading file: {str(e)}"
                    }
                ]
            }

# Run the MCP server
server = SimpleMCPServer()

# Read from stdin and write to stdout
for line in sys.stdin:
    try:
        request = json.loads(line)
        response = server.handle_request(request)
        print(json.dumps(response), flush=True)
    except json.JSONDecodeError:
        print(json.dumps({
            "type": "error",
            "error": {
                "message": "Invalid JSON"
            }
        }), flush=True)
    except Exception as e:
        print(json.dumps({
            "type": "error",
            "error": {
                "message": f"Error: {str(e)}"
            }
        }), flush=True)
""")
        server_script_path = f.name
    
    try:
        print(f"Created temporary MCP server script at: {server_script_path}")
        
        # Create MCP transport for the custom server
        mcp_transport = create_subprocess_mcp_transport(["python", server_script_path])
        
        # Create orchestrator with MCP support
        orchestrator = ReactiveSwarmOrchestrator(
            mcp_transport_callable=mcp_transport
        )
        
        # Execute task with swarm - MCP tools will be available
        result = await orchestrator.execute_reactive_swarm(
            "List the files in the current directory and analyze their contents",
            config=SwarmConfiguration(
                initial_size=2,
                max_size=4,
                coordination_pattern=CoordinationPattern.ADAPTIVE
            )
        )
        
        print(f"Success: {result.success}")
        print(f"Agents used: {result.agents_used}")
        print(f"Adaptations made: {result.adaptations_made}")
        print(f"Results: {result.content}")
    
    finally:
        # Clean up the temporary script file
        try:
            os.unlink(server_script_path)
            print(f"Removed temporary MCP server script: {server_script_path}")
        except Exception as e:
            print(f"Error removing temporary script: {str(e)}")


if __name__ == "__main__":
    # If run directly as a script, run the MCP server
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_mcp_server()
    # Otherwise, run the example
    else:
        asyncio.run(main())
