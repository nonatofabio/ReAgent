"""Pydantic models for ReAgent."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator


class TaskStep(BaseModel):
    """A step in a task execution plan."""
    
    description: str = Field(..., description="Description of the step")
    estimated_time_minutes: int = Field(..., description="Estimated time to complete the step in minutes")
    complexity: Optional[float] = Field(None, description="Complexity score for this step (0-1)")
    dependencies: Optional[List[int]] = Field(None, description="Indices of steps this step depends on")
    
    @validator('complexity')
    def validate_complexity(cls, v):
        """Validate complexity is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Complexity must be between 0 and 1")
        return v


class DomainAssessment(BaseModel):
    """Assessment of a domain's relevance to a task."""
    
    domain: str = Field(..., description="Domain name")
    relevance: float = Field(..., description="Relevance score (0-1)")
    complexity: float = Field(..., description="Complexity score for this domain (0-1)")
    
    @validator('relevance', 'complexity')
    def validate_scores(cls, v):
        """Validate scores are between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Scores must be between 0 and 1")
        return v


class TaskComplexityAnalysis(BaseModel):
    """Analysis of task complexity for swarm configuration."""
    
    complexity_score: float = Field(..., description="Overall complexity score (1-10)")
    complexity_level: str = Field(..., description="Complexity level (low, medium, high)")
    recommended_swarm_size: int = Field(..., description="Recommended number of agents (1-8)")
    recommended_pattern: str = Field(..., description="Recommended coordination pattern")
    steps: List[TaskStep] = Field(..., description="Steps required to complete the task")
    domains: List[DomainAssessment] = Field(..., description="Domain assessments")
    reasoning: str = Field(..., description="Reasoning behind the analysis")
    recommendations: List[str] = Field(..., description="Recommendations for execution")
    
    @validator('complexity_score')
    def validate_complexity_score(cls, v):
        """Validate complexity score is between 1 and 10."""
        if v < 1 or v > 10:
            raise ValueError("Complexity score must be between 1 and 10")
        return v
    
    @validator('complexity_level')
    def validate_complexity_level(cls, v):
        """Validate complexity level is one of the allowed values."""
        allowed_levels = ["low", "medium", "high"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"Complexity level must be one of: {', '.join(allowed_levels)}")
        return v.lower()
    
    @validator('recommended_swarm_size')
    def validate_swarm_size(cls, v):
        """Validate recommended swarm size is between 1 and 8."""
        if v < 1 or v > 8:
            raise ValueError("Recommended swarm size must be between 1 and 8")
        return v
    
    @validator('recommended_pattern')
    def validate_pattern(cls, v):
        """Validate recommended pattern is one of the allowed values."""
        allowed_patterns = ["collaborative", "competitive", "hybrid", "adaptive"]
        if v.lower() not in allowed_patterns:
            raise ValueError(f"Recommended pattern must be one of: {', '.join(allowed_patterns)}")
        return v.lower()


class SwarmAdaptation(BaseModel):
    """Adaptation made to a swarm during execution."""
    
    timestamp: float = Field(..., description="Timestamp of the adaptation")
    trigger: str = Field(..., description="Trigger that caused the adaptation")
    previous_config: Dict[str, Any] = Field(..., description="Previous configuration")
    new_config: Dict[str, Any] = Field(..., description="New configuration")
    reasoning: str = Field(..., description="Reasoning behind the adaptation")


class KeywordExtractionResult(BaseModel):
    """Result of keyword extraction from a task."""
    
    keywords: List[str] = Field(..., description="Extracted keywords")
    categories: Dict[str, List[str]] = Field(..., description="Keywords grouped by category")
    domain_relevance: Dict[str, float] = Field(..., description="Relevance scores by domain")
    
    @validator('domain_relevance')
    def validate_domain_relevance(cls, v):
        """Validate domain relevance scores are between 0 and 1."""
        for domain, score in v.items():
            if score < 0 or score > 1:
                raise ValueError(f"Domain relevance score for {domain} must be between 0 and 1")
        return v


class MemoryTierSelection(BaseModel):
    """Selection of memory tier for a piece of information."""
    
    key: str = Field(..., description="Memory key")
    selected_tier: str = Field(..., description="Selected memory tier")
    access_pattern_score: float = Field(..., description="Access pattern score")
    importance_score: float = Field(..., description="Importance score")
    size_bytes: int = Field(..., description="Size in bytes")
    reasoning: str = Field(..., description="Reasoning behind the selection")
    
    @validator('selected_tier')
    def validate_tier(cls, v):
        """Validate tier is one of the allowed values."""
        allowed_tiers = ["local", "persistent", "shared", "archive"]
        if v.lower() not in allowed_tiers:
            raise ValueError(f"Tier must be one of: {', '.join(allowed_tiers)}")
        return v.lower()
    
    @validator('access_pattern_score', 'importance_score')
    def validate_scores(cls, v):
        """Validate scores are between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Scores must be between 0 and 1")
        return v
