"""
ReactiveSwarmOrchestrator - Main orchestration engine for reactive multi-agent systems.

Built on AWS Strands Agents SDK with reactive adaptation capabilities.
"""

import asyncio
import time
import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from strands import Agent, tool
from strands.models import BedrockModel
try:
    from strands_tools import swarm
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    swarm = None

# Import MCP support
try:
    from strands.tools.mcp import MCPClient, MCPAgentTool
    from strands.tools.mcp.mcp_types import MCPTransport
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .memory import ReactiveSharedMemory
from .adaptation import AdaptationEngine, AdaptationTrigger


class CoordinationPattern(Enum):
    """Coordination patterns for swarm execution."""
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive" 
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"  # ReAgent-specific: switches patterns dynamically


@dataclass
class SwarmConfiguration:
    """Configuration for swarm execution."""
    initial_size: int = 3
    max_size: int = 8
    min_size: int = 1
    coordination_pattern: CoordinationPattern = CoordinationPattern.COLLABORATIVE
    adaptation_triggers: List[AdaptationTrigger] = field(default_factory=list)
    timeout_seconds: int = 300
    enable_reactive_adaptation: bool = True


@dataclass
class SwarmResult:
    """Result from swarm execution."""
    success: bool
    content: Any
    execution_time: float
    agents_used: int
    adaptations_made: int
    final_configuration: SwarmConfiguration
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)


class ReactiveSwarmOrchestrator:
    """
    Reactive orchestration layer built on Strands Agents SDK.
    
    Extends Strands swarm capabilities with:
    - Dynamic swarm sizing based on task complexity
    - Adaptive coordination pattern switching
    - Reactive memory management
    - Real-time adaptation based on intermediate results
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        enable_logging: bool = True,
        mcp_config_path: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        mcp_transport_callable: Optional[Callable[[], MCPTransport]] = None  # Deprecated, kept for backward compatibility
    ):
        self.logger = logging.getLogger(f"{__name__}.ReactiveSwarmOrchestrator")
        
        # Initialize Strands agent with swarm capability
        if not SWARM_AVAILABLE:
            raise ImportError(
                "strands_tools package not available. "
                "Install with: pip install strands-agents-tools"
            )
        
        # Default to Bedrock Claude if no model specified
        if model is None:
            model = BedrockModel(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name="us-west-2"
            )
        
        # Default system prompt for reactive orchestration
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        # Initialize MCP client - simplified approach using standard configuration
        self.mcp_client = None
        self.mcp_tools = []
        
        # Prefer standard MCP configuration over deprecated transport callable
        if mcp_config_path:
            self._initialize_mcp_from_config(mcp_config_path)
        elif mcp_transport_callable:
            # Backward compatibility - deprecated approach
            self.logger.warning("mcp_transport_callable is deprecated. Use mcp_config_path with standard MCP configuration instead.")
            self._initialize_mcp_from_transport(mcp_transport_callable)
        
        # Create base agent with swarm tool and MCP tools
        self.base_agent = Agent(
            model=model,
            tools=[swarm] + self._create_reactive_tools() + self.mcp_tools + tools if tools else [],
            system_prompt=system_prompt
        )
        
        # Initialize reactive components
        self.shared_memory = ReactiveSharedMemory()
        self.memory = self.shared_memory  # Alias for CLI compatibility
        self.adaptation_engine = AdaptationEngine()
        
        # Track execution state
        self.execution_history: List[SwarmResult] = []
        
        if enable_logging:
            self.logger.info("ReactiveSwarmOrchestrator initialized")
    
    def _initialize_mcp_from_config(self, config_path: str):
        """Initialize MCP client from standard configuration file."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "strands.tools.mcp module not available. "
                "Make sure you have the latest version of strands installed."
            )
        
        try:
            from ..utils.mcp import load_mcp_client_from_config
            
            self.logger.info(f"Loading MCP configuration from: {config_path}")
            self.mcp_client = load_mcp_client_from_config(config_path)
            
            if self.mcp_client:
                self.mcp_client.start()
                self.mcp_tools = self.mcp_client.list_tools_sync()
                self.logger.info(f"Loaded {len(self.mcp_tools)} MCP tools")
            else:
                self.logger.warning("Failed to create MCP client from configuration")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP from config: {str(e)}")
            # Don't fail initialization, just continue without MCP
    
    def _initialize_mcp_from_transport(self, mcp_transport_callable: Callable[[], MCPTransport]):
        """Initialize MCP client from transport callable (deprecated)."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "strands.tools.mcp module not available. "
                "Make sure you have the latest version of strands installed."
            )
        
        try:
            self.logger.info("Initializing MCP client from transport callable")
            self.mcp_client = MCPClient(mcp_transport_callable)
            self.mcp_client.start()
            self.mcp_tools = self.mcp_client.list_tools_sync()
            self.logger.info(f"Loaded {len(self.mcp_tools)} MCP tools")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP from transport: {str(e)}")
            # Don't fail initialization, just continue without MCP
    
    def __del__(self):
        """Clean up resources when the orchestrator is deleted."""
        if self.mcp_client:
            try:
                self.mcp_client.stop(None, None, None)
            except Exception as e:
                # Just log the error, don't raise during cleanup
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error stopping MCP client: {str(e)}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for reactive orchestration."""
        return """You are a Reactive Swarm Orchestrator powered by AWS Strands Agents.

Your role is to:
1. Analyze complex tasks and determine optimal swarm configuration
2. Coordinate multiple specialized agents working together
3. Adapt swarm behavior based on intermediate results and changing conditions
4. Optimize resource usage and task distribution dynamically
5. Ensure high-quality outcomes through collaborative intelligence

You have access to:
- Swarm coordination tools for managing multiple agents
- Reactive adaptation capabilities for real-time optimization
- Shared memory systems for knowledge persistence
- Pattern switching for different coordination strategies
- MCP tools for extended capabilities (when available)

Always consider:
- Task complexity when determining swarm size
- Intermediate results when adapting coordination patterns
- Resource efficiency and execution time
- Quality of collaborative outcomes
- Appropriate MCP tools for specialized tasks

Be adaptive, efficient, and focused on achieving the best possible results through intelligent swarm coordination."""
    
    def _create_reactive_tools(self) -> List[Any]:
        """Create ReAgent-specific tools for reactive capabilities."""
        
        # Import Strands tools directly
        from .tools.task_analysis import analyze_task_complexity
        
        @tool
        def adapt_swarm_configuration(
            current_config: Dict[str, Any],
            intermediate_results: List[Dict[str, Any]],
            performance_metrics: Dict[str, float]
        ) -> Dict[str, Any]:
            """Adapt swarm configuration based on intermediate results."""
            
            adaptations = self.adaptation_engine.analyze_and_adapt(
                current_config, intermediate_results, performance_metrics
            )
            
            return {
                'adaptations': adaptations,
                'new_configuration': adaptations.get('new_config', current_config),
                'reasoning': adaptations.get('reasoning', 'No adaptations needed')
            }
        
        @tool
        def store_swarm_memory(key: str, value: Any, tier: str = "auto") -> Dict[str, Any]:
            """Store information in reactive shared memory."""
            return self.shared_memory.store_with_tier(key, value, tier)
        
        @tool
        def retrieve_swarm_memory(key: str, include_history: bool = False) -> Dict[str, Any]:
            """Retrieve information from reactive shared memory."""
            return self.shared_memory.retrieve_with_context(key, include_history)
        
        return [
            analyze_task_complexity,
            adapt_swarm_configuration, 
            store_swarm_memory,
            retrieve_swarm_memory
        ]
    
    async def execute_reactive_swarm(
        self,
        task: str,
        config: Optional[SwarmConfiguration] = None,
        **kwargs
    ) -> SwarmResult:
        """
        Execute a task using reactive swarm orchestration.
        
        Args:
            task: The task to execute
            config: Swarm configuration (uses defaults if not provided)
            **kwargs: Additional parameters passed to the swarm tool
            
        Returns:
            SwarmResult with execution details and adaptations
        """
        start_time = time.time()
        
        if config is None:
            config = SwarmConfiguration()
        
        self.logger.info(f"Starting reactive swarm execution: task={task}, config={config}")
        
        try:
            # Phase 1: Initial task analysis and configuration
            analysis_result = await self._analyze_task_and_configure(task, config)
            optimized_config = analysis_result['optimized_config']
            
            # Phase 2: Execute swarm with initial configuration
            initial_result = await self._execute_swarm_phase(
                task, optimized_config, phase="initial", **kwargs
            )
            
            # Phase 3: Reactive adaptation based on results
            if config.enable_reactive_adaptation:
                adapted_result = await self._adaptive_execution_phase(
                    task, optimized_config, initial_result, **kwargs
                )
                final_result = adapted_result
            else:
                final_result = initial_result
            
            # Phase 4: Finalize and store results
            execution_time = time.time() - start_time
            
            swarm_result = SwarmResult(
                success=final_result.get('success', True),
                content=final_result.get('content', final_result),
                execution_time=execution_time,
                agents_used=final_result.get('agents_used', optimized_config.initial_size),
                adaptations_made=len(final_result.get('adaptations', [])),
                final_configuration=optimized_config,
                adaptation_history=final_result.get('adaptation_history', [])
            )
            
            # Store in execution history
            self.execution_history.append(swarm_result)
            
            # Store in shared memory for future reference
            await self.shared_memory.store_execution_result(task, swarm_result)
            
            self.logger.info(
                f"Reactive swarm execution completed: success={swarm_result.success}, "
                f"execution_time={execution_time}, adaptations={swarm_result.adaptations_made}"
            )
            
            return swarm_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Reactive swarm execution failed: error={str(e)}")
            
            return SwarmResult(
                success=False,
                content=f"Execution failed: {str(e)}",
                execution_time=execution_time,
                agents_used=0,
                adaptations_made=0,
                final_configuration=config
            )
    
    async def _analyze_task_and_configure(
        self, task: str, base_config: SwarmConfiguration
    ) -> Dict[str, Any]:
        """Analyze task and optimize initial configuration."""
        
        # Use the agent's task analysis tool
        analysis_prompt = f"""
        Analyze this task for optimal swarm configuration:
        
        Task: {task}
        
        Current configuration:
        - Initial size: {base_config.initial_size}
        - Max size: {base_config.max_size}
        - Pattern: {base_config.coordination_pattern.value}
        
        Use the analyze_task_complexity tool to determine optimal settings.
        """
        
        analysis_response = await asyncio.to_thread(self.base_agent, analysis_prompt)
        
        # Extract configuration recommendations from the analysis response
        from .models import TaskComplexityAnalysis
        
        try:
            # Try to extract the tool result from the response
            import re
            import json
            
            # Look for tool result in the response
            tool_result_pattern = r'analyze_task_complexity.*?```json\s*(.*?)\s*```'
            match = re.search(tool_result_pattern, str(analysis_response), re.DOTALL)
            
            if match:
                # Parse the JSON from the tool result
                analysis_data = json.loads(match.group(1))
                analysis = TaskComplexityAnalysis(**analysis_data)
                
                # Create optimized configuration based on analysis
                optimized_config = SwarmConfiguration(
                    initial_size=min(base_config.max_size, max(base_config.min_size, analysis.recommended_swarm_size)),
                    max_size=base_config.max_size,
                    min_size=base_config.min_size,
                    coordination_pattern=getattr(CoordinationPattern, analysis.recommended_pattern.upper(), base_config.coordination_pattern),
                    adaptation_triggers=base_config.adaptation_triggers,
                    timeout_seconds=base_config.timeout_seconds,
                    enable_reactive_adaptation=base_config.enable_reactive_adaptation
                )
                
                self.logger.info(
                    f"Task complexity analysis: score={analysis.complexity_score}, "
                    f"level={analysis.complexity_level}, "
                    f"recommended_size={analysis.recommended_swarm_size}, "
                    f"pattern={analysis.recommended_pattern}"
                )
            else:
                # Fallback to default configuration if no tool result found
                self.logger.warning("Could not extract task complexity analysis from response")
                optimized_config = base_config
        except Exception as e:
            # Log the error and fall back to the base configuration
            self.logger.error(f"Failed to parse task complexity analysis: {str(e)}")
            optimized_config = base_config
        
        return {
            'analysis': analysis_response,
            'optimized_config': optimized_config
        }
    
    async def _execute_swarm_phase(
        self,
        task: str,
        config: SwarmConfiguration,
        phase: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a single swarm phase."""
        
        self.logger.info(f"Executing swarm phase: {phase}, config={config}")
        
        # Prepare swarm execution parameters
        swarm_params = {
            'task': task,
            'swarm_size': config.initial_size,
            'coordination_pattern': config.coordination_pattern.value,
            **kwargs
        }
        
        # Execute using Strands swarm tool
        swarm_prompt = f"""
        Execute this task using swarm coordination:
        
        Task: {task}
        Configuration: {swarm_params}
        
        Use the swarm tool with the specified parameters.
        """
        
        result = await asyncio.to_thread(self.base_agent, swarm_prompt)
        
        return {
            'phase': phase,
            'result': result,
            'config_used': config,
            'success': True  # Would be determined from actual result parsing
        }
    
    async def _adaptive_execution_phase(
        self,
        task: str,
        config: SwarmConfiguration,
        initial_result: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute adaptive phase based on initial results."""
        
        self.logger.info("Starting adaptive execution phase")
        
        # Analyze initial results for adaptation opportunities
        adaptation_prompt = f"""
        Analyze the initial swarm execution results and determine if adaptations are needed:
        
        Original task: {task}
        Initial result: {initial_result}
        Current configuration: {config}
        
        Use the adapt_swarm_configuration tool to determine if changes are needed.
        If adaptations are recommended, execute another swarm phase with the new configuration.
        """
        
        adaptation_response = await asyncio.to_thread(self.base_agent, adaptation_prompt)
        
        # For now, return the initial result with adaptation metadata
        # In a full implementation, this would potentially trigger another swarm execution
        return {
            **initial_result,
            'adaptations': [],
            'adaptation_analysis': adaptation_response,
            'adaptation_history': [
                {
                    'phase': 'adaptive_analysis',
                    'timestamp': time.time(),
                    'analysis': adaptation_response
                }
            ]
        }
    
    async def get_execution_history(self) -> List[SwarmResult]:
        """Get history of swarm executions."""
        return self.execution_history.copy()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all executions."""
        if not self.execution_history:
            return {'message': 'No executions recorded yet'}
        
        successful_executions = [r for r in self.execution_history if r.success]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'average_execution_time': sum(r.execution_time for r in successful_executions) / len(successful_executions) if successful_executions else 0,
            'average_agents_used': sum(r.agents_used for r in successful_executions) / len(successful_executions) if successful_executions else 0,
            'total_adaptations': sum(r.adaptations_made for r in self.execution_history)
        }