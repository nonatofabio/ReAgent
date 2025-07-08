"""
Simplified MCP utilities for ReAgent - leveraging Strands' native MCP support.

This module provides minimal utilities for MCP integration, following the principle
of extending Strands rather than reimplementing functionality.

ReAgent is an EXTENSION to Strands, not a new framework - we favor minimal code
lines and maximal reuse of Strands capabilities.
"""

import json
import os
from typing import Optional, Dict, Any, List

try:
    from strands.tools.mcp import MCPClient
    from strands.tools.mcp.mcp_types import MCPTransport
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPClient = None


def load_mcp_client_from_config(config_path: str) -> Optional[MCPClient]:
    """
    Load MCP client from standard MCP configuration file.
    
    This function reads a standard MCP configuration file and creates an MCPClient
    using Strands' native MCP support. The configuration should follow the standard
    MCP format:
    
    ```json
    {
      "mcpServers": {
        "serverName": {
          "command": "python",
          "args": ["-m", "server_module"],
          "transport": {"type": "stdio"}
        }
      }
    }
    ```
    
    Args:
        config_path: Path to the standard MCP configuration file
        
    Returns:
        MCPClient instance or None if configuration is invalid
        
    Raises:
        ImportError: If Strands MCP support is not available
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "strands.tools.mcp module not available. "
            "Make sure you have the latest version of strands installed."
        )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"MCP configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate standard MCP configuration format
        if 'mcpServers' not in config:
            raise ValueError("Configuration must contain 'mcpServers' section")
        
        # Use Strands' native MCP client creation
        # Note: This assumes Strands has a from_config method or similar
        # If not available, we'll use the first server for simplicity
        servers = config['mcpServers']
        if not servers:
            raise ValueError("No MCP servers configured")
        
        # For simplicity, use the first configured server
        # In a more complete implementation, this could support multiple servers
        server_name, server_config = next(iter(servers.items()))
        
        # Create transport callable for Strands MCPClient
        def create_transport() -> MCPTransport:
            import subprocess
            command = [server_config['command']] + server_config.get('args', [])
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            return process.stdout, process.stdin
        
        return MCPClient(create_transport)
        
    except Exception as e:
        raise ValueError(f"Failed to load MCP configuration: {str(e)}")


def get_mcp_tools_from_config(config_path: str) -> List[Any]:
    """
    Get MCP tools from configuration file.
    
    Args:
        config_path: Path to MCP configuration file
        
    Returns:
        List of MCP tools
    """
    client = load_mcp_client_from_config(config_path)
    if client is None:
        return []
    
    try:
        client.start()
        tools = client.list_tools_sync()
        return tools
    except Exception:
        return []
    finally:
        try:
            client.stop(None, None, None)
        except Exception:
            pass


# Backward compatibility - simple factory function
def create_mcp_transport_from_config(config_path: str) -> Optional[callable]:
    """
    Backward compatibility function.
    
    Returns a transport callable for the old interface.
    """
    try:
        client = load_mcp_client_from_config(config_path)
        if client is None:
            return None
        
        # Return the transport callable from the client
        return client._transport_callable if hasattr(client, '_transport_callable') else None
    except Exception:
        return None
