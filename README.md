# ReAgent - Reactive Agent Orchestration System

**A reactive orchestration layer built on AWS Strands Agents SDK with dynamic adaptation capabilities**

## Overview

ReAgent provides reactive multi-agent coordination by extending AWS Strands Agents with dynamic planning, hybrid memory management, and adaptive execution patterns. Built on proven production frameworks, ReAgent delivers enterprise-ready reactive agent orchestration with real-time swarm adaptation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAgent System                           │
├─────────────────────────────────────────────────────────────┤
│  ReactiveSwarmOrchestrator (Strands Agent)                 │
│  ├── LLM-Based Task Analysis (no hardcoded complexity)     │
│  ├── Dynamic Swarm Sizing (1-8 agents)                     │
│  ├── Pattern Switching (collaborative/competitive/hybrid)   │
│  ├── Real-time Adaptation Engine (LLM-driven)              │
│  └── Performance Monitoring & Optimization                 │
├─────────────────────────────────────────────────────────────┤
│  ReactiveSharedMemory                                      │
│  ├── LLM-Based Tier Selection (no hardcoded rules)         │
│  ├── Real-time Agent Coordination                          │
│  ├── Tiered Storage (local/persistent/shared/archive)      │
│  ├── Automatic Tier Management                             │
│  └── Access Pattern Optimization                           │
├─────────────────────────────────────────────────────────────┤
│  AWS Strands Foundation                                     │
│  ├── Agent SDK + Swarm Tool                                │
│  ├── Multi-Agent Patterns                                  │
│  └── Model Integration (100+ LLM providers)                │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 🧠 LLM-Driven Intelligence
- **No Hardcoded Logic**: All decisions use LLM analysis for context-aware orchestration
- **Task Complexity Analysis**: LLM evaluates task requirements and recommends optimal configurations
- **Adaptive Triggers**: LLM determines when and how to adapt swarm behavior
- **Memory Management**: LLM-based tier selection for optimal data placement

### 🔄 Reactive Orchestration
- **Dynamic Swarm Sizing**: Automatically adjusts agent count (1-8) based on LLM task analysis
- **Pattern Switching**: Switches between collaborative, competitive, and hybrid coordination
- **Real-time Adaptation**: Modifies swarm configuration during execution based on LLM evaluation
- **Performance Monitoring**: Tracks metrics and triggers LLM-based optimizations

### 🧠 Built on AWS Strands
- **Production-Proven Foundation**: Uses the same SDK that powers Amazon Q Developer
- **Native Swarm Integration**: Extends `strands_tools.swarm` with reactive capabilities
- **Model Flexibility**: Support for 100+ LLM providers via Strands integration
- **Enterprise Ready**: Built-in observability and AWS deployment patterns

### 💾 Reactive Shared Memory
- **Real-time Coordination**: Agents share findings and coordinate through memory during execution
- **Tiered Storage**: Local cache → Persistent → Shared → Archive with LLM-based placement
- **Automatic Management**: Promotes/demotes data based on LLM analysis of access patterns
- **Thread-Safe**: Concurrent access for multi-agent coordination
- **Phase Continuity**: Memory maintains context between execution phases

### 🎯 Adaptation Intelligence
- **LLM-Based Triggers**: 6 adaptation triggers evaluated by LLM instead of hardcoded thresholds
- **Contextual Rules**: LLM analyzes execution patterns to determine adaptation needs
- **Cooldown Management**: Prevents excessive adaptations
- **Historical Learning**: Improves based on past executions stored in memory

## Quick Start

### Prerequisites

ReAgent uses [uv](https://docs.astral.sh/uv/) for Python package management. Install uv first:

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew (macOS)
brew install uv
```

### Installation

```bash
# Install ReAgent with uv (recommended)
uv pip install -e .

# Or install with pip
pip install -e .

# Set up AWS credentials for Bedrock (required for LLM analysis)
aws configure
```

### Basic Usage

```python
from reagent import ReactiveSwarmOrchestrator
from reagent.core.orchestrator import SwarmConfiguration, CoordinationPattern

# Create reactive orchestrator
orchestrator = ReactiveSwarmOrchestrator()

# Execute reactive task with LLM-based optimization
result = await orchestrator.execute_reactive_swarm(
    "Analyze renewable energy trends and provide policy recommendations",
    config=SwarmConfiguration(
        initial_size=3,
        max_size=6,
        coordination_pattern=CoordinationPattern.ADAPTIVE,
        enable_reactive_adaptation=True
    )
)

print(f"Success: {result.success}")
print(f"Agents used: {result.agents_used}")
print(f"Adaptations made: {result.adaptations_made}")
print(f"Results: {result.content}")
```

### CLI Usage

```bash
# Execute reactive goal
reagent execute "Research AI impact on job markets and provide recommendations"

# Execute with specific parameters
reagent execute "Analyze system performance" --size 4 --pattern adaptive

# Run demo scenarios
reagent demo file_analysis
reagent demo system_health

# Check system status
reagent status

# View memory usage
reagent memory status

# If using uv for development, prefix with uv run:
uv run reagent execute "Your task here"
uv run reagent status
```

## How It Works

### 4-Phase Reactive Execution

1. **Task Analysis Phase**: LLM analyzes task complexity and optimizes initial configuration
2. **Initial Execution Phase**: Swarm executes with agents coordinating through shared memory
3. **Adaptive Phase**: LLM evaluates results and adapts configuration if needed
4. **Finalization Phase**: Results stored in memory for future learning

### Real-time Agent Coordination

Agents use shared memory throughout execution:
```python
# Agents store findings for others to build upon
store_swarm_memory("findings:phase:agent_1", research_data)

# Agents retrieve context to avoid duplication
previous_work = retrieve_swarm_memory("findings:phase:agent_2")

# Agents coordinate work division
store_swarm_memory("coordination:phase", "Agent 1: economics, Agent 2: policy")
```

### LLM-Based Adaptation

Instead of hardcoded rules, LLM evaluates:
- Task complexity changes during execution
- Performance patterns and bottlenecks
- Quality issues requiring different coordination
- Resource constraints needing optimization

## Core Components

### ReactiveSwarmOrchestrator
Main orchestration engine with LLM-driven adaptation:

```python
# LLM analyzes task and recommends optimal configuration
config = SwarmConfiguration(
    initial_size=2,
    max_size=8,
    coordination_pattern=CoordinationPattern.ADAPTIVE,  # LLM-selected
    adaptation_triggers=[
        AdaptationTrigger.COMPLEXITY_INCREASE,
        AdaptationTrigger.PERFORMANCE_DEGRADATION
    ]
)
```

### ReactiveSharedMemory
Tiered memory system with LLM-based optimization:

```python
# LLM determines optimal tier based on data characteristics
await memory.store_with_tier("execution_result", result, tier="auto")

# Retrieves with access tracking and LLM-based tier promotion
data = await memory.retrieve_with_context("key", include_history=True)
```

### AdaptationEngine
Real-time adaptation using LLM evaluation:

```python
# LLM analyzes performance and determines if adaptations are needed
adaptations = engine.analyze_and_adapt(
    current_config=config,
    intermediate_results=results,
    performance_metrics=metrics
)
```

## Examples

### File Analysis with Reactive Adaptation
```python
# The orchestrator will:
# 1. LLM analyzes file structure complexity → recommends 3 agents
# 2. Agents coordinate through shared memory during analysis
# 3. LLM detects complexity increase → adapts to 5 agents
# 4. Agents share findings in memory → generate comprehensive insights

result = await orchestrator.execute_reactive_swarm(
    "Analyze all Python files in this project and provide optimization recommendations"
)
```

### System Monitoring with Dynamic Response
```python
# The orchestrator will:
# 1. LLM determines monitoring strategy → deploys specialized agents
# 2. Agents share monitoring data through memory in real-time
# 3. LLM detects anomalies → spawns additional response agents
# 4. Agents coordinate response actions through shared memory

result = await orchestrator.execute_reactive_swarm(
    "Monitor system health and respond to anomalies",
    config=SwarmConfiguration(
        initial_size=2,
        max_size=8,
        coordination_pattern=CoordinationPattern.ADAPTIVE
    )
)
```

## Performance

### Benchmarks
- **Agent Spawn Time**: < 1 second per agent
- **LLM Analysis Time**: 2-3 seconds for task complexity analysis
- **Memory Access**: Local < 1ms, Persistent < 10ms, Shared < 50ms
- **Adaptation Time**: 1-2 seconds for LLM-based adaptation decisions
- **Coordination Overhead**: < 50ms per memory operation

### Scalability
- **Current**: 1-8 concurrent agents with shared memory coordination
- **Memory Usage**: ~50MB per agent + shared memory pools
- **Throughput**: 10-100 tasks/minute depending on complexity
- **LLM Calls**: Optimized with caching and fallback mechanisms

## Development

### Project Structure
```
reagent/
├── __init__.py                    # Main package exports
├── core/
│   ├── orchestrator.py           # ReactiveSwarmOrchestrator (LLM-driven)
│   ├── memory.py                 # ReactiveSharedMemory (LLM tier selection)
│   ├── adaptation.py             # AdaptationEngine (LLM-based triggers)
│   └── tools/
│       └── task_analysis.py      # LLM-based task complexity analysis
├── cli.py                        # Command-line interface
examples/
└── basic_usage.py                # Usage examples
tests/
└── test_orchestrator.py          # Test suite
```

### Running Tests
```bash
# Install development dependencies with uv
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/

# Test LLM integration
uv run python test_llm_integration.py

# Test memory usage
uv run python test_memory_simple.py

# Run with coverage
uv run pytest tests/ --cov=reagent
```

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd ReAgent

# Install in development mode with uv (recommended)
uv pip install -e ".[dev]"

# Set up AWS credentials for Bedrock
aws configure

# Run example
uv run python examples/basic_usage.py

# Run CLI
uv run reagent status
```

## Deployment

### Local Development
```bash
# Run basic example
uv run python examples/basic_usage.py

# Use CLI interface
reagent execute "Your reactive task here"

# Or with uv for development
uv run reagent execute "Your reactive task here"
```

### AWS Deployment
ReAgent is designed to leverage Strands' AWS deployment capabilities:

```python
# Deploy using Strands deployment patterns
# (Future enhancement - see Strands documentation)
```

## Contributing

We welcome contributions! ReAgent is designed to be a community-driven project for advancing reactive agent orchestration.

### 🎯 Priority Areas for Contribution:

- **LLM Integration**: Enhanced prompts and analysis techniques
- **Adaptation Strategies**: New LLM-based adaptation patterns
- **Memory Optimization**: Advanced tiered storage algorithms
- **Coordination Patterns**: Novel agent coordination strategies
- **Performance**: Optimization and benchmarking
- **Documentation**: Examples and use cases

### 🚀 How to Contribute:

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the LLM-first design principles
3. **Add tests** to verify your changes work correctly
4. **Submit a pull request** with a clear description

### 📋 Development Guidelines:

- **No Hardcoded Logic**: All decision-making must use LLM analysis
- **Memory Integration**: New features should leverage reactive shared memory
- **Test Coverage**: Include tests for both success and failure scenarios
- **Documentation**: Update relevant docs and examples

## TODO & Feature Requests

### 🔧 Current Development TODOs:

- [ ] Implement semantic memory tier 
- [ ] Add more sophisticated context to LLM analysis prompts
- [ ] Create learning system that improves adaptation based on outcomes
- [ ] Add support for custom LLM models beyond Bedrock
- [ ] Implement swarm execution rollback for failed adaptations
- [ ] Add metrics dashboard for swarm performance monitoring
- [ ] Create integration tests with real-world scenarios

### 💡 Feature Request Process:

**Have an idea for ReAgent?** We'd love to hear from you!

1. **Check existing issues** to see if your idea is already being discussed
2. **Create a new issue** using our feature request template
3. **Describe your use case** and how it would benefit reactive agent orchestration
4. **Engage with the community** to refine and prioritize the feature

**[📝 Submit Feature Request →](https://github.com/nonatofabio/ReAgent/issues/new?template=feature_request.md)**

### 🌟 Community Ideas Welcome:

- Novel coordination patterns for specific domains
- Integration with other agent frameworks
- Advanced memory management strategies
- Performance optimization techniques
- Real-world use case implementations
- Educational examples and tutorials

### 🤝 Join the Discussion:

- **Issues**: Bug reports and feature requests
- **Discussions**: Architecture ideas and use cases
- **Wiki**: Community knowledge and best practices
- **Discord**: Real-time community chat (coming soon)

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **AWS Strands Team** - For the foundational agent SDK and swarm capabilities
- **Anthropic** - For the Model Context Protocol specification and Claude models
- **Open Source Community** - For inspiration and best practices
- **Contributors** - Everyone who helps make ReAgent better

## Related Projects

- [AWS Strands Agents](https://github.com/strands-agents/sdk-python) - Foundation agent SDK
- [Strands Tools](https://github.com/strands-agents/tools) - Swarm tool and other utilities
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standardized LLM integration