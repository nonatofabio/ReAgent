"""Task analysis tools for ReAgent using Strands tools directly."""

import json
import logging
import re
from typing import Dict, Any, List

from strands import tool
from ..models import TaskComplexityAnalysis
from ..prompts import PromptTemplates

logger = logging.getLogger(__name__)

@tool
def analyze_task_complexity(task: str) -> Dict[str, Any]:
    """
    Analyze task complexity to determine optimal swarm configuration.
    
    This tool analyzes the given task to determine:
    - Complexity score (1-10)
    - Complexity level (low, medium, high)
    - Recommended swarm size (1-8)
    - Recommended coordination pattern
    - Required steps for completion
    - Domain-specific assessments
    
    Args:
        task: The task description to analyze
        
    Returns:
        Dict containing complexity analysis and recommendations
    """
    # Use LLM-based analysis directly in the tool
    try:
        from strands import Agent
        from strands.models import BedrockModel
        import json
        import re
        
        # Create agent for analysis
        model = BedrockModel(
            model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-west-2"
        )
        
        agent = Agent(
            model=model,
            system_prompt="You are a task complexity analyzer. Analyze tasks and provide structured complexity assessments."
        )
        
        analysis_prompt = PromptTemplates.task_complexity_analysis(task)
        
        response = agent(analysis_prompt)
        
        # Extract JSON from response
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, str(response), re.DOTALL)
        
        if match:
            analysis_json = match.group(1)
        else:
            # Try to find JSON in the response
            json_start = str(response).find('{')
            json_end = str(response).rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                analysis_json = str(response)[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
        
        analysis = json.loads(analysis_json)
        
        # Validate required fields
        required_fields = ['complexity_score', 'complexity_level', 'recommended_swarm_size', 'recommended_pattern']
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Task complexity analysis failed: {str(e)}")
        # Return fallback analysis
        return generate_fallback_analysis(task)

def extract_and_validate_analysis(response: str) -> Dict[str, Any]:
    """
    Extract and validate task complexity analysis from agent response.
    
    Args:
        response: The raw response from the agent
        
    Returns:
        Validated analysis as a dictionary
    """
    try:
        # Try to extract JSON from the response
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, str(response), re.DOTALL)
        
        if match:
            analysis_json = match.group(1)
        else:
            # If no JSON block found, try to parse the entire response
            analysis_json = str(response)
        
        # Parse the JSON
        analysis = json.loads(analysis_json)
        
        # Validate with Pydantic model
        validated_analysis = TaskComplexityAnalysis(**analysis)
        
        # Return as dictionary
        return validated_analysis.model_dump()
        
    except Exception as e:
        logger.error(f"Failed to extract and validate analysis: {str(e)}")
        # Return a fallback analysis
        return generate_fallback_analysis(task)

def generate_fallback_analysis(task: str) -> Dict[str, Any]:
    """
    Generate LLM-based analysis when extraction or validation fails.
    
    Args:
        task: The task description
        
    Returns:
        LLM-generated analysis as a dictionary
    """
    from strands import Agent
    from strands.models import BedrockModel
    
    try:
        # Create a simple agent for analysis
        model = BedrockModel(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-west-2"
        )
        
        agent = Agent(
            model=model,
            system_prompt="You are a task complexity analyzer. Analyze tasks and provide structured complexity assessments."
        )
        
        analysis_prompt = f"""
        Analyze this task for complexity and provide a JSON response with the following structure:
        {{
            "complexity_score": <number 1-10>,
            "complexity_level": "<low|medium|high>",
            "recommended_swarm_size": <number 1-8>,
            "recommended_pattern": "<collaborative|competitive|adaptive>",
            "steps": [
                {{"description": "<step description>", "estimated_time_minutes": <number>}}
            ],
            "domains": [
                {{"domain": "<domain name>", "relevance": <0-1>, "complexity": <0-1>}}
            ],
            "reasoning": "<explanation of analysis>",
            "recommendations": ["<recommendation 1>", "<recommendation 2>"]
        }}
        
        Task to analyze: {task}
        
        Consider:
        - Number of distinct subtasks required
        - Domain expertise needed
        - Coordination complexity
        - Time sensitivity
        - Quality requirements
        
        Respond with only the JSON structure.
        """
        
        response = agent(analysis_prompt)
        
        # Extract JSON from response
        import json
        import re
        
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, str(response), re.DOTALL)
        
        if match:
            analysis_json = match.group(1)
        else:
            # Try to find JSON in the response
            json_start = str(response).find('{')
            json_end = str(response).rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                analysis_json = str(response)[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
        
        analysis = json.loads(analysis_json)
        
        # Validate required fields
        required_fields = ['complexity_score', 'complexity_level', 'recommended_swarm_size', 'recommended_pattern']
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"LLM-based analysis failed: {str(e)}, falling back to minimal analysis")
        
        # Minimal fallback that doesn't use heuristics
        return {
            "complexity_score": 5.0,
            "complexity_level": "medium",
            "recommended_swarm_size": 3,
            "recommended_pattern": "adaptive",
            "steps": [
                {"description": "Analyze and execute task", "estimated_time_minutes": 30}
            ],
            "domains": [
                {"domain": "general", "relevance": 1.0, "complexity": 0.5}
            ],
            "reasoning": "LLM analysis unavailable, using conservative defaults",
            "recommendations": [
                "Use adaptive pattern with 3 agents",
                "Monitor and adjust based on results"
            ]
        }
