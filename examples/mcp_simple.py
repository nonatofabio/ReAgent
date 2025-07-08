"""
Simple example demonstrating MCP integration with ReAgent using standard configuration.

This example shows how to use ReAgent with MCP tools using the standard MCP
configuration format for enhanced capabilities.
"""

import asyncio
import os
from reagent import ReactiveSwarmOrchestrator
from reagent.core.orchestrator import SwarmConfiguration, CoordinationPattern


async def main():
    """Run a simple example of MCP integration using standard configuration."""
    # Get the path to the standard MCP configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "mcp_config.json")
    
    print(f"Using standard MCP configuration from: {config_path}")
    
    try:
        # Create orchestrator with standard MCP configuration
        orchestrator = ReactiveSwarmOrchestrator(
            mcp_config_path=config_path
        )
        
        # Execute task with swarm - MCP tools will be available
        result = await orchestrator.execute_reactive_swarm(
            "Analyze the current directory structure and suggest improvements",
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
    
    except FileNotFoundError:
        print(f"Error: MCP configuration file not found at {config_path}")
        print("Please create a standard MCP configuration file with the following format:")
        print("""
        {
          "mcpServers": {
            "your-server": {
              "command": "python",
              "args": ["-m", "your_mcp_server"],
              "transport": {
                "type": "stdio"
              }
            }
          }
        }
        """)
    
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("Make sure you have the latest version of strands installed.")
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
