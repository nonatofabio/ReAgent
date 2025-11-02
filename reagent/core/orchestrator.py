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
    from strands_tools.swarm import Swarm
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
        self.logger = logging.getLogger()
        
        # Initialize Strands agent with swarm capability
        if not SWARM_AVAILABLE:
            raise ImportError(
                "strands_tools package not available. "
                "Install with: pip install strands-agents-tools"
            )
        
        # Default to Bedrock Claude if no model specified
        if model is None:
            model = BedrockModel(
                model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                region_name="us-west-2"
            )
        
        # Store model for agent creation
        self.model = model
        
        # Default system prompt for reactive orchestration
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        self.logger.debug(f"Using system prompt: {system_prompt[:100]}...")
        
        # Initialize MCP client - simplified approach using standard configuration
        self.mcp_client = None
        self.mcp_tools = []
        self.logger.info(f"ReactiveSwarmOrchestrator model initialized")
        # Prefer standard MCP configuration over deprecated transport callable
        if mcp_config_path:
            self._initialize_mcp_from_config(mcp_config_path)
        elif mcp_transport_callable:
            # Backward compatibility - deprecated approach
            self.logger.warning("mcp_transport_callable is deprecated. Use mcp_config_path with standard MCP configuration instead.")
            self._initialize_mcp_from_transport(mcp_transport_callable)
        
        # Create base agent with swarm tool and MCP tools
        tools = [swarm] + self._create_reactive_tools() + self.mcp_tools + (tools if tools else [])
        self.base_agent = Agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt
        )
        self.logger.info(f"Base agent created with {len(tools)} tools")
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
        from .prompts import PromptTemplates
        return PromptTemplates.swarm_orchestrator_system()
    
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
            # Run async method synchronously
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a task and wait for it
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self.shared_memory.store_with_tier(key, value, tier))
                        return future.result()
                else:
                    return loop.run_until_complete(self.shared_memory.store_with_tier(key, value, tier))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.shared_memory.store_with_tier(key, value, tier))
        
        @tool
        def retrieve_swarm_memory(key: str, include_history: bool = False) -> Dict[str, Any]:
            """Retrieve information from reactive shared memory."""
            # Run async method synchronously
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a task and wait for it
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self.shared_memory.retrieve_with_context(key, include_history))
                        return future.result()
                else:
                    return loop.run_until_complete(self.shared_memory.retrieve_with_context(key, include_history))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.shared_memory.retrieve_with_context(key, include_history))
        
        return [
            analyze_task_complexity,
            adapt_swarm_configuration, 
            store_swarm_memory,
            retrieve_swarm_memory
        ]
    def _format_tool_context(self) -> str:
        """Format available tools for agent awareness."""
        
        tool_descriptions = []
        
        # Memory tools (ReAgent value-add)
        tool_descriptions.append("""
Memory Coordination Tools (for agent collaboration):
- store_swarm_memory(key, value, tier="auto"): Save findings for other agents
- retrieve_swarm_memory(key, include_history=False): Access shared data
- Use keys like: "findings:agent_X", "progress", "results"
""")
        
        # MCP tools (if available)
        if self.mcp_tools:
            mcp_names = [t.__name__ if hasattr(t, '__name__') else 'mcp_tool' 
                         for t in self.mcp_tools[:5]]  # First 5
            tool_descriptions.append(f"""
External Research Tools (via MCP):
- {', '.join(mcp_names)}
- Use these for current information, web search, etc.
""")
        
        # Strands native tools
        tool_descriptions.append("""
Swarm Coordination Tools (native to strands):
- handoff_to_agent(agent_name, context): Transition work to another agent
- complete_swarm_task(result): Deliver final result when done
""")
        
        return "\n".join(tool_descriptions)
    
    
    async def execute_reactive_swarm(
        self,
        task: str,
        config: Optional[SwarmConfiguration] = None,
        **kwargs
    ) -> SwarmResult:
        """
        Execute task using hybrid ReAgent intelligence + Strands execution.
        
        ReAgent provides:
        - Task complexity analysis
        - Custom agent role specifications
        - Tool context and coordination strategy
        
        Strands provides:
        - Agent creation and management
        - Coordination via handoff_to_agent
        - Final delivery via complete_swarm_task
        """
        start_time = time.time()
        
        if config is None:
            config = SwarmConfiguration()
        
        self.logger.info(f"Starting ReAgent execution: {task}")
        
        try:
            # Phase 1: Intelligence Layer (ReAgent's value)
            analysis_result = await self._analyze_task_and_configure(task, config)
            optimized_config = analysis_result['optimized_config']
            analysis_obj = analysis_result['analysis_obj']
            
            # If analysis extraction failed, create a default one
            if analysis_obj is None:
                from .models import TaskComplexityAnalysis, TaskStep, DomainAssessment
                analysis_obj = TaskComplexityAnalysis(
                    complexity_score=5,
                    complexity_level="medium",
                    recommended_swarm_size=optimized_config.initial_size,
                    recommended_pattern=optimized_config.coordination_pattern.value,
                    steps=[
                        TaskStep(
                            description="Analyze task requirements",
                            estimated_time_minutes=5,
                            complexity=0.5
                        )
                    ],
                    domains=[
                        DomainAssessment(
                            domain="general",
                            relevance=0.5,
                            complexity=0.5
                        )
                    ],
                    reasoning="Default analysis used due to extraction failure",
                    recommendations=["Use default swarm configuration"]
                )
                self.logger.debug("Using default TaskComplexityAnalysis due to extraction failure")
            
            # Generate specialized agent roles based on analysis
            agent_roles = await self._generate_agent_role_specifications(task, analysis_obj)
            self.logger.info(f"Generated {optimized_config.initial_size} specialized agent roles")
            
            # Phase 2: Execution Layer (Delegate to Strands)
            result = await self._execute_swarm_phase(
                task,
                optimized_config,
                analysis_obj,
                agent_roles,
                **kwargs
            )
            
            # Package results
            swarm_result = SwarmResult(
                success=True,
                content=result['result'],
                execution_time=time.time() - start_time,
                agents_used=optimized_config.initial_size,
                adaptations_made=0,
                final_configuration=optimized_config
            )
            
            # Store for learning
            self.execution_history.append(swarm_result)
            await self.shared_memory.store_execution_result(task, swarm_result)
            
            self.logger.info(f"Execution completed: {swarm_result.execution_time:.2f}s")
            return swarm_result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            return SwarmResult(
                success=False,
                content=f"Execution failed: {str(e)}",
                execution_time=time.time() - start_time,
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
        
        # Extract configuration recommendations and analysis object using refactored method
        optimized_config, analysis_obj = self._extract_and_apply_analysis(analysis_response, base_config)
        
        return {
            'analysis': analysis_response,
            'optimized_config': optimized_config,
            'analysis_obj': analysis_obj
        }
    def _extract_and_apply_analysis(
        self, response, base_config: SwarmConfiguration
    ) -> tuple[SwarmConfiguration, Optional['TaskComplexityAnalysis']]:
        """
        Extract task complexity analysis from agent response and apply it to configuration.
        
        Tries multiple extraction strategies in order:
        1. Extract from Strands response object attributes (.content, .text)
        2. Look for tool result XML tags
        3. Search for JSON blocks in markdown
        4. Find raw JSON in response text
        
        Args:
            response: Agent response (may be Strands response object or string)
            base_config: Base configuration to use as fallback
            
        Returns:
            Tuple of (optimized SwarmConfiguration, TaskComplexityAnalysis or None)
        """
        import re
        import json
        from .models import TaskComplexityAnalysis
        
        analysis_data = None
        response_text = None
        
        try:
            # Step 1: Extract text from Strands response object
            if hasattr(response, 'content'):
                response_text = str(response.content)
            elif hasattr(response, 'text'):
                response_text = str(response.text)
            elif hasattr(response, 'message'):
                response_text = str(response.message)
            else:
                response_text = str(response)
            
            self.logger.debug(f"Response text for analysis extraction: {response_text[:500]}...")
            
            # Step 2: Try multiple extraction patterns
            extraction_attempts = [
                # Pattern 1: XML-style tool result tags
                (r'<analyze_task_complexity[^>]*>\s*({.*?})\s*</analyze_task_complexity>', "XML tool tags"),
                # Pattern 2: JSON in markdown code blocks
                (r'```json\s*({.*?})\s*```', "JSON markdown block"),
                # Pattern 3: Raw JSON object
                (r'({[\s\S]*?"complexity_score"[\s\S]*?"recommended_pattern"[\s\S]*?})', "Raw JSON"),
            ]
            
            for pattern, description in extraction_attempts:
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        analysis_data = json.loads(match.group(1))
                        self.logger.debug(f"Successfully extracted analysis using: {description}")
                        break
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"Failed to parse JSON from {description}: {e}")
                        continue
            
            # Step 3: If no patterns matched, try finding JSON boundaries
            if not analysis_data:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    try:
                        potential_json = response_text[json_start:json_end]
                        analysis_data = json.loads(potential_json)
                        self.logger.debug("Successfully extracted analysis from JSON boundaries")
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"Failed to parse JSON from boundaries: {e}")
            
            # Step 4: Validate and apply analysis
            if analysis_data:
                analysis = TaskComplexityAnalysis(**analysis_data)
                
                # Create optimized configuration
                optimized_config = SwarmConfiguration(
                    initial_size=min(
                        base_config.max_size,
                        max(base_config.min_size, analysis.recommended_swarm_size)
                    ),
                    max_size=base_config.max_size,
                    min_size=base_config.min_size,
                    coordination_pattern=getattr(
                        CoordinationPattern,
                        analysis.recommended_pattern.upper(),
                        base_config.coordination_pattern
                    ),
                    adaptation_triggers=base_config.adaptation_triggers,
                    timeout_seconds=base_config.timeout_seconds,
                    enable_reactive_adaptation=base_config.enable_reactive_adaptation
                )
                
                self.logger.info(
                    f"Task complexity analysis applied: "
                    f"score={analysis.complexity_score}, "
                    f"level={analysis.complexity_level}, "
                    f"recommended_size={analysis.recommended_swarm_size}, "
                    f"pattern={analysis.recommended_pattern}"
                )
                
                return optimized_config, analysis
            else:
                # No analysis data found - use base config silently
                self.logger.debug("No task complexity analysis found in response, using base configuration")
                return base_config, None
                
        except Exception as e:
            # Log error and fall back to base configuration
            self.logger.debug(f"Error during analysis extraction: {str(e)}, using base configuration")
            return base_config, None
    
    async def _generate_agent_role_specifications(
        self,
        task: str,
        analysis: 'TaskComplexityAnalysis'
    ) -> str:
        """Generate specialized agent role descriptions based on task analysis."""
        
        role_prompt = f"""
Based on this task analysis, generate {analysis.recommended_swarm_size} specialized agent roles:

Task: {task}
Complexity: {analysis.complexity_level}
Domains: {[d.domain for d in analysis.domains]}

For each agent, provide:
1. Role name (e.g., "Research Specialist", "Analysis Expert")
2. Primary responsibilities (what they focus on)
3. What they should NOT do (boundaries)
4. Coordination notes (how they work with others)

Format as clear, structured text that will be passed to strands.swarm.
"""
        
        result = await asyncio.to_thread(self.base_agent, role_prompt)
        return str(result)
    
    async def _execute_swarm_phase(
        self,
        task: str,
        config: SwarmConfiguration,
        analysis: 'TaskComplexityAnalysis',
        agent_roles: str,
        phase: str = "execution",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute using strands.swarm with ReAgent-generated specifications."""
        
        self.logger.info(f"Executing swarm: {config.initial_size} agents, pattern={config.coordination_pattern.value}")
        
        # Format tool context once
        tool_context = self._format_tool_context()
        
        # Create rich prompt combining YOUR intelligence with strands execution
        swarm_prompt = f"""
TASK: {task}

SWARM CONFIGURATION:
- Number of agents: {config.initial_size}
- Coordination pattern: {config.coordination_pattern.value}
- Task complexity: {analysis.complexity_level} (score: {analysis.complexity_score}/10)

SPECIALIZED AGENT ROLES (Generated by ReAgent Intelligence Layer):
{agent_roles}

{tool_context}

EXECUTION INSTRUCTIONS:
1. Each agent should work on their specialized role
2. Use memory tools to coordinate and share findings
3. Use handoff_to_agent when transitioning between roles
4. Use complete_swarm_task to deliver the final, complete result

Execute this task with the specialized agents using the tools provided.
The swarm tool will handle agent creation and coordination.
"""
        
        # Execute via base agent (which invokes strands.swarm)
        self.logger.info("Invoking strands.swarm tool...")
        result = await asyncio.to_thread(self.base_agent, swarm_prompt)
        
        return {
            'phase': phase,
            'result': str(result),
            'config_used': config,
            'success': True
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
