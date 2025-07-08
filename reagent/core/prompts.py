"""Prompt templates for ReAgent."""

# Task complexity analysis prompt
TASK_COMPLEXITY_ANALYSIS_PROMPT = """
You are an expert AI system analyzer specializing in task complexity assessment.

Analyze the following task to determine its complexity and optimal swarm configuration:

Task: {task}

Provide a detailed analysis with the following:
1. Complexity score (1-10 scale)
2. Complexity level (low, medium, high)
3. Recommended swarm size (1-8 agents)
4. Recommended coordination pattern (collaborative, competitive, hybrid, adaptive)
5. Steps required to complete the task
6. Domain assessments
7. Reasoning behind your analysis
8. Recommendations for execution

Your response should be in JSON format with the following structure:
```json
{{
  "complexity_score": 5.5,
  "complexity_level": "medium",
  "recommended_swarm_size": 3,
  "recommended_pattern": "adaptive",
  "steps": [
    {{
      "description": "Analyze requirements",
      "estimated_time_minutes": 15,
      "complexity": 0.4
    }},
    {{
      "description": "Research background",
      "estimated_time_minutes": 30,
      "complexity": 0.6
    }}
  ],
  "domains": [
    {{
      "domain": "technical",
      "relevance": 0.8,
      "complexity": 0.7
    }},
    {{
      "domain": "analytical",
      "relevance": 0.6,
      "complexity": 0.5
    }}
  ],
  "reasoning": "This task involves technical analysis and data processing...",
  "recommendations": [
    "Use 3 agents with adaptive pattern",
    "Focus on technical expertise"
  ]
}}
```

Examples:

Example 1:
Task: "Write a simple Python function to calculate the factorial of a number"
Analysis:
```json
{{
  "complexity_score": 2.0,
  "complexity_level": "low",
  "recommended_swarm_size": 1,
  "recommended_pattern": "collaborative",
  "steps": [
    {{
      "description": "Understand factorial calculation",
      "estimated_time_minutes": 5,
      "complexity": 0.2
    }},
    {{
      "description": "Write Python function",
      "estimated_time_minutes": 10,
      "complexity": 0.3
    }},
    {{
      "description": "Test and validate",
      "estimated_time_minutes": 5,
      "complexity": 0.2
    }}
  ],
  "domains": [
    {{
      "domain": "technical",
      "relevance": 0.9,
      "complexity": 0.2
    }}
  ],
  "reasoning": "This is a simple programming task requiring basic Python knowledge. The factorial algorithm is straightforward and can be implemented with a simple recursive or iterative approach. No complex logic or domain knowledge is required.",
  "recommendations": [
    "Use 1 agent with collaborative pattern",
    "Focus on clean, efficient implementation",
    "Include both recursive and iterative approaches"
  ]
}}
```

Example 2:
Task: "Research and analyze the impact of AI on job markets across different industries, and provide policy recommendations"
Analysis:
```json
{{
  "complexity_score": 8.5,
  "complexity_level": "high",
  "recommended_swarm_size": 5,
  "recommended_pattern": "adaptive",
  "steps": [
    {{
      "description": "Define research scope and methodology",
      "estimated_time_minutes": 30,
      "complexity": 0.7
    }},
    {{
      "description": "Research AI impact across industries",
      "estimated_time_minutes": 90,
      "complexity": 0.9
    }},
    {{
      "description": "Analyze job market trends",
      "estimated_time_minutes": 60,
      "complexity": 0.8
    }},
    {{
      "description": "Identify policy implications",
      "estimated_time_minutes": 45,
      "complexity": 0.8
    }},
    {{
      "description": "Develop policy recommendations",
      "estimated_time_minutes": 60,
      "complexity": 0.9
    }},
    {{
      "description": "Synthesize findings and create report",
      "estimated_time_minutes": 45,
      "complexity": 0.7
    }}
  ],
  "domains": [
    {{
      "domain": "research",
      "relevance": 0.9,
      "complexity": 0.8
    }},
    {{
      "domain": "analytical",
      "relevance": 0.9,
      "complexity": 0.8
    }},
    {{
      "domain": "business",
      "relevance": 0.7,
      "complexity": 0.7
    }},
    {{
      "domain": "technical",
      "relevance": 0.6,
      "complexity": 0.7
    }}
  ],
  "reasoning": "This task requires extensive research across multiple domains, complex analysis of trends and impacts, and the development of nuanced policy recommendations. It involves understanding AI technology, labor economics, industry-specific factors, and policy frameworks. The breadth and depth of knowledge required, along with the need for synthesis across domains, makes this a high-complexity task.",
  "recommendations": [
    "Use 5 agents with adaptive pattern",
    "Assign specialized roles: researcher, analyst, policy expert, industry specialist, and synthesizer",
    "Implement collaborative research phase followed by competitive policy recommendation phase",
    "Allocate more resources to research and analysis steps"
  ]
}}
```

Now analyze the task provided and return your analysis in the specified JSON format.
"""

# Keyword extraction prompt
KEYWORD_EXTRACTION_PROMPT = """
You are an expert AI system specializing in keyword extraction and categorization.

Extract and categorize keywords from the following task:

Task: {task}

Provide a detailed analysis with the following:
1. List of all relevant keywords
2. Keywords grouped by category
3. Domain relevance scores

Your response should be in JSON format with the following structure:
```json
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "categories": {{
    "category1": ["keyword1", "keyword2"],
    "category2": ["keyword3"]
  }},
  "domain_relevance": {{
    "technical": 0.8,
    "analytical": 0.6,
    "creative": 0.2,
    "research": 0.7,
    "business": 0.3
  }}
}}
```

Example:

Task: "Analyze the performance of our e-commerce website and recommend optimizations to improve conversion rates"

```json
{{
  "keywords": ["analyze", "performance", "e-commerce", "website", "recommend", "optimizations", "improve", "conversion rates"],
  "categories": {{
    "analytical": ["analyze", "performance", "conversion rates"],
    "technical": ["e-commerce", "website", "optimizations"],
    "business": ["conversion rates", "e-commerce", "improve"]
  }},
  "domain_relevance": {{
    "technical": 0.7,
    "analytical": 0.9,
    "creative": 0.3,
    "research": 0.5,
    "business": 0.8
  }}
}}
```

Now extract and categorize keywords from the provided task and return your analysis in the specified JSON format.
"""

# Memory tier selection prompt
MEMORY_TIER_SELECTION_PROMPT = """
You are an expert AI system specializing in memory management and optimization.

Determine the optimal memory tier for the following information:

Key: {key}
Content: {content}
Access Pattern: {access_pattern}
Size: {size_bytes} bytes

Available memory tiers:
- local: Fast access, limited capacity, not shared between agents
- persistent: Medium access speed, larger capacity, shared between executions
- shared: Medium access speed, large capacity, shared between all agents
- archive: Slow access, unlimited capacity, long-term storage

Provide a detailed analysis with the following:
1. Selected memory tier
2. Access pattern score (0-1)
3. Importance score (0-1)
4. Reasoning behind your selection

Your response should be in JSON format with the following structure:
```json
{{
  "key": "{key}",
  "selected_tier": "persistent",
  "access_pattern_score": 0.7,
  "importance_score": 0.8,
  "size_bytes": {size_bytes},
  "reasoning": "This information is moderately important and accessed frequently..."
}}
```

Example:

Key: "execution_result:12345"
Content: "Detailed analysis of system performance with metrics and recommendations"
Access Pattern: "Created once, accessed 5 times in last hour, not accessed after execution"
Size: 15000 bytes

```json
{{
  "key": "execution_result:12345",
  "selected_tier": "persistent",
  "access_pattern_score": 0.7,
  "importance_score": 0.8,
  "size_bytes": 15000,
  "reasoning": "This execution result contains valuable analysis that may be referenced in future executions. The moderate size and frequent access during execution but limited access afterward suggests persistent storage is optimal. This allows future executions to reference the results while not consuming limited local memory."
}}
```

Now analyze the provided information and return your tier selection in the specified JSON format.
"""

# Class for accessing prompt templates
class PromptTemplates:
    """Class for accessing prompt templates."""
    
    @staticmethod
    def task_complexity_analysis(task: str) -> str:
        """Get task complexity analysis prompt."""
        return TASK_COMPLEXITY_ANALYSIS_PROMPT.format(task=task)
    
    @staticmethod
    def keyword_extraction(task: str) -> str:
        """Get keyword extraction prompt."""
        return KEYWORD_EXTRACTION_PROMPT.format(task=task)
    
    @staticmethod
    def memory_tier_selection(key: str, content: str, access_pattern: str, size_bytes: int) -> str:
        """Get memory tier selection prompt."""
        return MEMORY_TIER_SELECTION_PROMPT.format(
            key=key,
            content=content,
            access_pattern=access_pattern,
            size_bytes=size_bytes
        )
