"""
ReAgent - Reactive Agent Orchestration System

A reactive orchestration layer built on AWS Strands Agents SDK.
"""

__version__ = "0.1.0"
__author__ = "ReAgent Team"

from .core.orchestrator import ReactiveSwarmOrchestrator
from .core.memory import ReactiveSharedMemory
from .core.adaptation import AdaptationEngine

__all__ = [
    "ReactiveSwarmOrchestrator",
    "ReactiveSharedMemory", 
    "AdaptationEngine",
]
