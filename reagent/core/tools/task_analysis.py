"""Task analysis tools for ReAgent using Strands tools directly."""

import json
import logging
import re
from typing import Dict, Any, List

from strands import tool
from ..models import TaskComplexityAnalysis
from ..prompts import TASK_COMPLEXITY_ANALYSIS_PROMPT

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
    # The implementation will be handled by the Strands agent
    # This is just a tool definition that will be registered with the agent
    pass

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
    Generate a fallback analysis when extraction or validation fails.
    
    Args:
        task: The task description
        
    Returns:
        Fallback analysis as a dictionary
    """
    # Simple heuristic based on task length and keyword detection
    task_length = len(task)
    
    # Detect complexity indicators
    complexity_keywords = [
        "analyze", "research", "compare", "evaluate", "optimize",
        "design", "develop", "implement", "architect", "complex",
        "comprehensive", "detailed", "thorough", "extensive"
    ]
    
    keyword_count = sum(1 for keyword in complexity_keywords if keyword.lower() in task.lower())
    
    # Calculate complexity score (1-10)
    length_factor = min(5, task_length / 100)  # Max 5 points for length
    keyword_factor = min(5, keyword_count)     # Max 5 points for keywords
    
    complexity_score = length_factor + keyword_factor
    
    # Determine complexity level
    if complexity_score < 3:
        complexity_level = "low"
        recommended_size = 1
        recommended_pattern = "collaborative"
    elif complexity_score < 7:
        complexity_level = "medium"
        recommended_size = 3
        recommended_pattern = "adaptive"
    else:
        complexity_level = "high"
        recommended_size = 5
        recommended_pattern = "adaptive"
    
    # Create basic steps
    steps = [
        {"description": "Analyze requirements", "estimated_time_minutes": 10},
        {"description": "Execute primary task", "estimated_time_minutes": 30},
        {"description": "Review and finalize", "estimated_time_minutes": 15}
    ]
    
    # Create basic domain assessment
    domains = [
        {"domain": "general", "relevance": 1.0, "complexity": complexity_score / 10}
    ]
    
    # Return analysis in the expected format
    return {
        "complexity_score": round(complexity_score, 1),
        "complexity_level": complexity_level,
        "recommended_swarm_size": recommended_size,
        "recommended_pattern": recommended_pattern,
        "steps": steps,
        "domains": domains,
        "reasoning": "Heuristic analysis based on task length and keyword detection",
        "recommendations": [
            f"Use {recommended_size} agents with {recommended_pattern} pattern",
            "Monitor performance and adjust as needed"
        ]
    }
