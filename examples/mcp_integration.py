"""
Example implementation of MCP integration using Strands' native MCP support.

This example demonstrates how to integrate Model Context Protocol (MCP) tools
with ReAgent's reactive swarm orchestration using both direct Strands integration
and ReAgent's simplified MCP configuration approach.

Reference: https://strandsagents.com/latest/api-reference/tools/#strands.tools.mcp
"""

import asyncio
import os
from typing import Optional

from strands import Agent
from strands.tools import swarm
from strands.tools.mcp import MCPClient

from reagent import ReactiveSwarmOrchestrator
from reagent.core.orchestrator import SwarmConfiguration, CoordinationPattern
from reagent.utils.mcp import load_mcp_client_from_config


async def example_direct_strands_mcp(config_path: str):
    """Example of using MCP directly with Strands using standard configuration."""
    print("=== Direct Strands MCP Integration ===")
    
    try:
        # Load MCP client from standard configuration
        mcp_client = load_mcp_client_from_config(config_path)
        if not mcp_client:
            print("Failed to create MCP client from configuration")
            return
        
        mcp_client.start()
        
        # Get MCP tools
        mcp_tools = mcp_client.list_tools_sync()
        print(f"Loaded {len(mcp_tools)} MCP tools")
        
        # Create agent with swarm tool and MCP tools
        agent = Agent(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            tools=[swarm] + mcp_tools,
            system_prompt="You are a helpful assistant with access to MCP tools."
        )
        
        # Use the agent with MCP tools
        result = await agent.acomplete(
            "Please analyze the current directory structure and suggest improvements."
        )
        
        print("Direct Strands Result:")
        print(result)
        
    except Exception as e:
        print(f"Error in direct Strands MCP integration: {str(e)}")
    finally:
        # Clean up MCP client
        if 'mcp_client' in locals() and mcp_client:
            try:
                mcp_client.stop(None, None, None)
            except Exception:
                pass


async def example_reagent_mcp(config_path: str):
    """Example of using MCP with ReAgent's reactive swarm orchestration."""
    print("\n=== ReAgent MCP Integration ===")
    
    try:
        # Create orchestrator with standard MCP configuration
        orchestrator = ReactiveSwarmOrchestrator(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            mcp_config_path=config_path
        )
        
        # Execute task with swarm - MCP tools will be available
        result = await orchestrator.execute_reactive_swarm(
            "Analyze this repository structure and provide recommendations for organization",
            config=SwarmConfiguration(
                initial_size=2,
                max_size=4,
                coordination_pattern=CoordinationPattern.ADAPTIVE
            )
        )
        
        print("ReAgent Integration Result:")
        print(f"Success: {result.success}")
        print(f"Agents used: {result.agents_used}")
        print(f"Adaptations made: {result.adaptations_made}")
        print(f"Results: {result.content}")
        
    except Exception as e:
        print(f"Error in ReAgent MCP integration: {str(e)}")


async def main():
    """Run both MCP integration examples."""
    # Get the path to the standard MCP configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "mcp_config.json")
    
    if not os.path.exists(config_path):
        print(f"MCP configuration file not found at: {config_path}")
        print("Please create a standard MCP configuration file.")
        return
    
    print(f"Using MCP configuration: {config_path}")
    
    # Run both examples
    await example_direct_strands_mcp(config_path)
    await example_reagent_mcp(config_path)


if __name__ == "__main__":
    asyncio.run(main())
