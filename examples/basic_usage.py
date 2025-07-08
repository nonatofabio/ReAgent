"""
Basic usage example for ReactiveSwarmOrchestrator.

This example demonstrates how to use ReAgent for reactive multi-agent coordination.
"""

import asyncio
from reagent import ReactiveSwarmOrchestrator
from reagent.core.orchestrator import SwarmConfiguration, CoordinationPattern


async def main():
    """Demonstrate basic ReactiveSwarmOrchestrator usage."""
    
    print("🚀 ReAgent - Reactive Swarm Orchestration Example")
    print("=" * 50)
    
    # Create orchestrator
    print("\n1. Initializing ReactiveSwarmOrchestrator...")
    try:
        orchestrator = ReactiveSwarmOrchestrator()
        print("✅ Orchestrator initialized successfully")
    except ImportError as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Install required packages:")
        print("   pip install strands-agents strands-agents-tools")
        return
    
    # Example 1: Simple task execution
    print("\n2. Executing simple task...")
    simple_task = "Analyze the benefits and challenges of renewable energy"
    
    config = SwarmConfiguration(
        initial_size=3,
        max_size=5,
        coordination_pattern=CoordinationPattern.COLLABORATIVE,
        enable_reactive_adaptation=True
    )
    
    result = await orchestrator.execute_reactive_swarm(simple_task, config)
    
    print(f"✅ Task completed: {result.success}")
    print(f"⏱️  Execution time: {result.execution_time:.2f}s")
    print(f"🤖 Agents used: {result.agents_used}")
    print(f"🔄 Adaptations made: {result.adaptations_made}")
    
    if result.content:
        print(f"\n📋 Result preview: {str(result.content)[:200]}...")
    
    # Example 2: Complex task with adaptive pattern
    print("\n3. Executing complex task with adaptive coordination...")
    complex_task = """
    Research and analyze the impact of artificial intelligence on job markets,
    considering multiple perspectives including economic, social, and technological factors.
    Provide recommendations for policy makers and individuals.
    """
    
    adaptive_config = SwarmConfiguration(
        initial_size=2,
        max_size=8,
        coordination_pattern=CoordinationPattern.ADAPTIVE,  # ReAgent-specific
        enable_reactive_adaptation=True
    )
    
    result2 = await orchestrator.execute_reactive_swarm(complex_task, adaptive_config)
    
    print(f"✅ Complex task completed: {result2.success}")
    print(f"⏱️  Execution time: {result2.execution_time:.2f}s")
    print(f"🤖 Agents used: {result2.agents_used}")
    print(f"🔄 Adaptations made: {result2.adaptations_made}")
    
    # Show performance metrics
    print("\n4. Performance metrics...")
    metrics = await orchestrator.get_performance_metrics()
    print(f"📊 Total executions: {metrics['total_executions']}")
    print(f"📈 Success rate: {metrics['success_rate']:.1%}")
    print(f"⚡ Average execution time: {metrics['average_execution_time']:.2f}s")
    print(f"🔄 Total adaptations: {metrics['total_adaptations']}")
    
    # Show memory statistics
    print("\n5. Memory system statistics...")
    memory_stats = await orchestrator.shared_memory.get_memory_stats()
    print(f"💾 Total entries: {memory_stats['total_entries']}")
    print(f"🏠 Local cache: {memory_stats['tier_counts']['local']}")
    print(f"💿 Persistent: {memory_stats['tier_counts']['persistent']}")
    print(f"📦 Archive: {memory_stats['tier_counts']['archive']}")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Try the CLI interface:")
    print("   reagent execute 'Your task here'")
    print("   reagent demo file_analysis")
    print("   reagent status")


if __name__ == "__main__":
    asyncio.run(main())
