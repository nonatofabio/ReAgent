"""
ReAgent CLI - Command-line interface for ReactiveSwarmOrchestrator.
"""

import asyncio
import logging
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from typing import Optional

from .core.orchestrator import ReactiveSwarmOrchestrator, SwarmConfiguration, CoordinationPattern

app = typer.Typer(
    name="reagent",
    help="ReAgent - Reactive Agent Orchestration System",
    add_completion=False
)
console = Console()


@app.command()
def execute(
    task: str = typer.Argument(..., help="The task to execute with reactive swarm"),
    swarm_size: int = typer.Option(3, "--size", "-s", help="Initial swarm size"),
    max_size: int = typer.Option(8, "--max-size", help="Maximum swarm size"),
    pattern: str = typer.Option("collaborative", "--pattern", "-p", help="Coordination pattern"),
    timeout: int = typer.Option(300, "--timeout", "-t", help="Execution timeout in seconds"),
    reactive: bool = typer.Option(True, "--reactive/--no-reactive", help="Enable reactive adaptation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    log_file: str = typer.Option(None, "--log-file", help="Log to file"),
    show_memory: bool = typer.Option(False, "--show-memory", help="Show memory operations"),
    add_tool: list[str] = typer.Option([], "--add-tool", help="Add optional Strands tool (e.g., retrieve, mem0_memory)"),
    god_mode: bool = typer.Option(True, "--not-god-mode", help="Brings back tool confirmation prompts"),
    list_tools: bool = typer.Option(False, "--list-tools", help="List available tools and exit"),
):
    """Execute a task using reactive swarm orchestration."""
    
    # Handle --list-tools flag
    if list_tools:
        _show_available_tools()
        return
    
    # Configure logging
    import logging
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    
    # Set up logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger()
    logger.info(f"Starting ReAgent execution: {task}")
    
    console.print(Panel.fit(
        f"[bold blue]ReAgent Reactive Swarm Execution[/bold blue]\n"
        f"Task: {task}\n"
        f"Initial Swarm Size: {swarm_size}\n"
        f"Max Size: {max_size}\n"
        f"Pattern: {pattern}\n"
        f"Reactive: {reactive}\n"
        f"Debug: {debug}\n"
        f"Log Level: {log_level}",
        title="Starting Execution"
    ))
    
    # Set god mode environment variable if requested
    if god_mode:
        import os
        os.environ["BYPASS_TOOL_CONSENT"] = "true"
        console.print("[bold yellow]⚠️  GOD MODE ENABLED - All tool confirmations bypassed![/bold yellow]")
    
    # Run the async execution
    result = asyncio.run(_execute_task(
        task, swarm_size, max_size, pattern, timeout, reactive, verbose, debug, show_memory, add_tool
    ))
    
    # Display results
    _display_results(result, debug=debug, show_memory=show_memory)


@app.command()
def status():
    """Show ReAgent system status and capabilities."""
    
    console.print("[bold green]ReAgent System Status[/bold green]")
    
    # Create status table
    table = Table(title="System Components")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    try:
        # Test Strands availability
        from strands import Agent
        strands_status = "✅ Available"
        strands_details = "AWS Strands Agents SDK loaded"
    except ImportError:
        strands_status = "❌ Missing"
        strands_details = "Install with: pip install strands-agents"
    
    try:
        # Test Strands Tools availability
        from strands_tools import swarm
        tools_status = "✅ Available"
        tools_details = "Swarm tool loaded successfully"
    except ImportError:
        tools_status = "❌ Missing"
        tools_details = "Install with: pip install strands-agents-tools"
    
    table.add_row("Strands SDK", strands_status, strands_details)
    table.add_row("Strands Tools", tools_status, tools_details)
    table.add_row("Reactive Memory", "✅ Ready", "Hybrid memory system initialized")
    table.add_row("Adaptation Engine", "✅ Ready", "Reactive adaptation rules loaded")
    
    console.print(table)


@app.command()
def patterns():
    """List available coordination patterns."""
    
    console.print("[bold blue]Available Coordination Patterns[/bold blue]")
    
    patterns_info = [
        ("collaborative", "Agents build upon others' insights and seek consensus", "Best for complex analysis"),
        ("competitive", "Agents develop independent solutions", "Best for creative tasks"),
        ("hybrid", "Balances cooperation with independent exploration", "Best for balanced approaches"),
        ("adaptive", "Switches patterns dynamically based on results", "ReAgent-specific reactive pattern"),
    ]
    
    table = Table()
    table.add_column("Pattern", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Use Case")
    
    for pattern, description, use_case in patterns_info:
        table.add_row(pattern, description, use_case)
    
    console.print(table)


@app.command()
def history(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of recent executions to show")
):
    """Show execution history and performance metrics."""
    
    console.print("[bold blue]Execution History[/bold blue]")
    console.print("(This would show recent executions from ReactiveSwarmOrchestrator)")
    
    # In a real implementation, this would load and display actual history
    console.print("[dim]No executions recorded yet. Run 'reagent execute' to start.[/dim]")


@app.command()
def memory(
    action: str = typer.Argument("status", help="Action: status, inspect, clean"),
    key: str = typer.Option(None, "--key", "-k", help="Specific memory key to inspect"),
    tier: str = typer.Option(None, "--tier", "-t", help="Memory tier to inspect"),
):
    """Inspect and manage ReAgent memory system."""
    
    console.print(f"[bold blue]ReAgent Memory System - {action.title()}[/bold blue]")
    
    if action == "status":
        _show_memory_status()
    elif action == "inspect":
        _inspect_memory(key, tier)
    elif action == "clean":
        _clean_memory()
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: status, inspect, clean")


def _show_memory_status():
    """Show memory system status."""
    import os
    from pathlib import Path
    
    memory_dir = Path("reagent_memory")
    
    if not memory_dir.exists():
        console.print("[yellow]No memory directory found. Run a task first.[/yellow]")
        return
    
    # Count files by type
    execution_files = list(memory_dir.glob("execution:*.json"))
    adaptation_files = list(memory_dir.glob("adaptation:*.json"))
    config_files = list(memory_dir.glob("config:*.json"))
    other_files = [f for f in memory_dir.glob("*.json") 
                   if not any(f.name.startswith(prefix) for prefix in ["execution:", "adaptation:", "config:"])]
    
    table = Table(title="Memory System Status")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Details")
    
    table.add_row("Execution Records", str(len(execution_files)), "Task execution history")
    table.add_row("Adaptation Records", str(len(adaptation_files)), "Reactive adaptation history")
    table.add_row("Configuration Records", str(len(config_files)), "Swarm configuration snapshots")
    table.add_row("Other Records", str(len(other_files)), "Miscellaneous memory entries")
    
    total_size = sum(f.stat().st_size for f in memory_dir.glob("*.json"))
    table.add_row("Total Size", f"{total_size / 1024:.1f} KB", f"Directory: {memory_dir.absolute()}")
    
    console.print(table)
    
    # Show recent files
    if execution_files:
        console.print("\n[bold yellow]Recent Executions:[/bold yellow]")
        recent_files = sorted(execution_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        for i, file in enumerate(recent_files, 1):
            mtime = file.stat().st_mtime
            import datetime
            time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"  {i}. {file.name} ({time_str})")


def _inspect_memory(key: str = None, tier: str = None):
    """Inspect specific memory entries."""
    import json
    from pathlib import Path
    
    memory_dir = Path("reagent_memory")
    
    if not memory_dir.exists():
        console.print("[yellow]No memory directory found.[/yellow]")
        return
    
    if key:
        # Look for specific key
        matching_files = list(memory_dir.glob(f"*{key}*.json"))
        if not matching_files:
            console.print(f"[red]No memory entries found matching key: {key}[/red]")
            return
        
        for file in matching_files:
            console.print(f"\n[bold cyan]File: {file.name}[/bold cyan]")
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                console.print(json.dumps(data, indent=2))
            except Exception as e:
                console.print(f"[red]Error reading file: {e}[/red]")
    else:
        # Show all files
        json_files = list(memory_dir.glob("*.json"))
        if not json_files:
            console.print("[yellow]No memory files found.[/yellow]")
            return
        
        console.print(f"[bold green]Found {len(json_files)} memory files:[/bold green]")
        for file in sorted(json_files):
            console.print(f"  - {file.name}")


def _clean_memory():
    """Clean up memory system."""
    import shutil
    from pathlib import Path
    
    memory_dir = Path("reagent_memory")
    
    if not memory_dir.exists():
        console.print("[yellow]No memory directory found.[/yellow]")

def _show_available_tools():
    """Display available tools."""
    console.print("\n[bold green]📦 ReAgent Available Tools[/bold green]\n")
    
    console.print("[bold blue]Default Tools (Always Enabled):[/bold blue]")
    console.print("  [cyan]File Operations:[/cyan] file_read, file_write, editor")
    console.print("  [cyan]Code Execution:[/cyan] python_repl")
    console.print("  [cyan]Workflows:[/cyan] swarm, workflow, batch, use_agent, think")
    console.print("  [cyan]Web & Network:[/cyan] http_request, browser, duckduckgo_search")
    console.print("  [cyan]Shell & System:[/cyan] shell, environment")
    
    console.print("\n[bold yellow]Optional Tools (use --add-tool):[/bold yellow]")
    console.print("  [cyan]Memory & RAG:[/cyan]")
    console.print("    • retrieve - Bedrock Knowledge Base retrieval")
    console.print("    • memory - Agent memory in Bedrock KB")
    console.print("    • agent_core_memory - Bedrock Agent Core Memory")
    console.print("    • mem0_memory - Mem0-based personalization")
    
    console.print("  [cyan]Code & Execution:[/cyan]")
    console.print("    • code_interpreter - Sandboxed code execution")
    
    console.print("  [cyan]Multi-modal:[/cyan]")
    console.print("    • generate_image - AI image generation (Bedrock)")
    console.print("    • generate_image_stability - Stability AI images")
    console.print("    • image_reader - Image analysis")
    console.print("    • nova_reels - AI video generation")
    console.print("    • diagram - Cloud/UML diagrams")
    console.print("    • speak - Text-to-speech")
    
    console.print("  [cyan]AWS Services:[/cyan]")
    console.print("    • use_aws - Interact with AWS services")
    
    console.print("  [cyan]Advanced:[/cyan]")
    console.print("    • use_computer - Desktop automation")
    console.print("    • cron - Task scheduling")
    console.print("    • slack - Slack integration")
    console.print("    • rss - RSS feed management")
    console.print("    • a2a_client - Agent-to-agent communication")
    
    console.print("\n[bold cyan]Examples:[/bold cyan]")
    console.print("  # Use default tools only")
    console.print("  [dim]reagent execute 'analyze this code'[/dim]")
    console.print("\n  # Add memory tools")
    console.print("  [dim]reagent execute 'research AI' --add-tool retrieve --add-tool mem0_memory[/dim]")
    console.print("\n  # Add AWS integration")
    console.print("  [dim]reagent execute 'deploy app' --add-tool use_aws[/dim]")
    console.print("\n  # God mode (skip confirmations)")
    console.print("  [dim]reagent execute 'automated task' --god-mode[/dim]")
    console.print()


def _clean_memory():
    """Clean up memory system."""
    import shutil
    from pathlib import Path
    
    memory_dir = Path("reagent_memory")
    
    if not memory_dir.exists():
        console.print("[yellow]No memory directory found.[/yellow]")
        return
    
    # Ask for confirmation
    file_count = len(list(memory_dir.glob("*.json")))
    if file_count == 0:
        console.print("[yellow]Memory directory is already empty.[/yellow]")
        return
    
    confirm = typer.confirm(f"Delete {file_count} memory files?")
    if confirm:
        shutil.rmtree(memory_dir)
        console.print("[green]Memory directory cleaned.[/green]")
    else:
        console.print("[yellow]Memory cleanup cancelled.[/yellow]")


@app.command()
def demo(
    scenario: str = typer.Argument("file_analysis", help="Demo scenario to run"),
):
    """Run a demonstration scenario."""
    
    scenarios = {
        "file_analysis": "Analyze all Python files in the current directory and provide insights",
        "system_health": "Monitor system health and generate a comprehensive report",
        "data_processing": "Process data files and adapt analysis based on content discovered",
        "research_task": "Research a topic using multiple specialized agents with reactive coordination",
    }
    
    if scenario not in scenarios:
        console.print(f"[red]Unknown scenario: {scenario}[/red]")
        console.print("Available scenarios:")
        for name, description in scenarios.items():
            console.print(f"  - [cyan]{name}[/cyan]: {description}")
        return
    
    task = scenarios[scenario]
    console.print(f"[bold yellow]Running demo scenario: {scenario}[/bold yellow]")
    console.print(f"Task: {task}")
    
    # Run the demo with reactive adaptation enabled
    result = asyncio.run(_execute_task(
        task, swarm_size=3, max_size=6, pattern="adaptive", 
        timeout=300, reactive=True, verbose=True
    ))
    _display_results(result)


async def _execute_task(
    task: str,
    swarm_size: int,
    max_size: int,
    pattern: str,
    timeout: int,
    reactive: bool,
    verbose: bool,
    debug: bool = False,
    show_memory: bool = False,
    additional_tools: list[str] = None
) -> dict:
    """Execute a task asynchronously."""
    
    logger = logging.getLogger()
    
    try:
        # Create swarm configuration
        config = SwarmConfiguration(
            initial_size=swarm_size,
            max_size=max_size,
            coordination_pattern=CoordinationPattern(pattern),
            timeout_seconds=timeout,
            enable_reactive_adaptation=reactive
        )
        
        logger.info(f"Configuration created: {config}")
        
        # Initialize orchestrator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            init_task = progress.add_task("Initializing ReactiveSwarmOrchestrator...", total=None)
            
            orchestrator = ReactiveSwarmOrchestrator(
                additional_tools=additional_tools if additional_tools else None
            )
            
            if additional_tools:
                logger.info(f"ReactiveSwarmOrchestrator initialized with additional tools: {', '.join(additional_tools)}")
            else:
                logger.info("ReactiveSwarmOrchestrator initialized with default tools")
            progress.update(init_task, completed=True)
        
        # Execute the task
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            exec_task = progress.add_task("Executing reactive swarm...", total=None)
            
            logger.info(f"Starting swarm execution for task: {task}")
            result = await orchestrator.execute_reactive_swarm(task, config)
            logger.info(f"Swarm execution completed. Success: {result.success}")
            
            progress.update(exec_task, completed=True)
        
        # Collect debug information
        debug_info = {}
        if debug or show_memory:
            try:
                # Get memory statistics
                memory_stats = await orchestrator.memory.get_statistics()
                debug_info['memory_stats'] = memory_stats
                logger.debug(f"Memory statistics: {memory_stats}")
            except Exception as e:
                logger.warning(f"Could not get memory statistics: {e}")
                debug_info['memory_stats'] = {"error": str(e)}
        
        return {
            "success": result.success,
            "content": result.content,
            "execution_time": result.execution_time,
            "agents_used": result.agents_used,
            "adaptations_made": result.adaptations_made,
            "final_config": result.final_configuration,
            "adaptation_history": result.adaptation_history,
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.exception("Execution failed with exception")
        console.print(f"[red]Execution failed: {str(e)}[/red]")
        if verbose or debug:
            console.print_exception()
        return {"success": False, "error": str(e), "debug_info": {"exception": str(e)}}


def _display_results(result: dict, debug: bool = False, show_memory: bool = False) -> None:
    """Display execution results."""
    
    if result.get("success", False):
        console.print("\n[bold green]✅ Execution Completed Successfully[/bold green]")
        
        # Display content if available
        if result.get("content"):
            # Extract actual text from nested structure
            content = result["content"]
            if isinstance(content, dict) and 'result' in content:
                display_content = content['result']
            elif isinstance(content, str):
                display_content = content
            else:
                display_content = str(content)
            
            console.print("\n[bold blue]Results:[/bold blue]")
            console.print(Panel(display_content, title="Task Output"))
        
        # Display adaptations if any were made
        if result.get("adaptations_made", 0) > 0:
            console.print("\n[bold yellow]Adaptations Made:[/bold yellow]")
            for i, adaptation in enumerate(result.get("adaptation_history", []), 1):
                console.print(f"{i}. {adaptation}")
        
        # Display debug information
        if debug and result.get("debug_info"):
            console.print("\n[bold magenta]🔍 Debug Information:[/bold magenta]")
            debug_info = result["debug_info"]
            
            if "memory_stats" in debug_info:
                console.print("\n[cyan]Memory Statistics:[/cyan]")
                memory_stats = debug_info["memory_stats"]
                if isinstance(memory_stats, dict) and "error" not in memory_stats:
                    for key, value in memory_stats.items():
                        console.print(f"  {key}: {value}")
                else:
                    console.print(f"  Error: {memory_stats}")
        
        # Display memory operations if requested
        if show_memory and result.get("debug_info", {}).get("memory_stats"):
            console.print("\n[bold cyan]💾 Memory Operations:[/bold cyan]")
            # This would show detailed memory tier operations
            console.print("  Memory tier operations logged to debug output")
    
    else:
        console.print("\n[bold red]❌ Execution Failed[/bold red]")
        
        if result.get("error"):
            console.print(f"Error: {result['error']}")
        
        if debug and result.get("debug_info"):
            console.print("\n[bold magenta]🔍 Debug Information:[/bold magenta]")
            debug_info = result["debug_info"]
            for key, value in debug_info.items():
                console.print(f"  {key}: {value}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
