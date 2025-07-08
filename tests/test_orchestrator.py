"""
Tests for ReactiveSwarmOrchestrator.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from reagent.core.orchestrator import (
    ReactiveSwarmOrchestrator, 
    SwarmConfiguration, 
    CoordinationPattern
)
from reagent.core.memory import ReactiveSharedMemory, MemoryTier
from reagent.core.adaptation import AdaptationEngine, AdaptationTrigger


class TestReactiveSwarmOrchestrator:
    """Test cases for ReactiveSwarmOrchestrator."""
    
    def test_swarm_configuration_creation(self):
        """Test SwarmConfiguration creation with defaults."""
        config = SwarmConfiguration()
        
        assert config.initial_size == 3
        assert config.max_size == 8
        assert config.min_size == 1
        assert config.coordination_pattern == CoordinationPattern.COLLABORATIVE
        assert config.timeout_seconds == 300
        assert config.enable_reactive_adaptation == True
    
    def test_swarm_configuration_custom(self):
        """Test SwarmConfiguration with custom values."""
        config = SwarmConfiguration(
            initial_size=5,
            max_size=10,
            coordination_pattern=CoordinationPattern.ADAPTIVE,
            timeout_seconds=600
        )
        
        assert config.initial_size == 5
        assert config.max_size == 10
        assert config.coordination_pattern == CoordinationPattern.ADAPTIVE
        assert config.timeout_seconds == 600
    
    @patch('reagent.core.orchestrator.swarm')
    def test_orchestrator_initialization_without_strands_tools(self, mock_swarm):
        """Test orchestrator initialization when strands_tools is not available."""
        mock_swarm = None
        
        with patch('reagent.core.orchestrator.SWARM_AVAILABLE', False):
            with pytest.raises(ImportError, match="strands_tools package not available"):
                ReactiveSwarmOrchestrator()
    
    @patch('reagent.core.orchestrator.SWARM_AVAILABLE', True)
    @patch('reagent.core.orchestrator.swarm')
    @patch('strands.Agent')
    def test_orchestrator_initialization_success(self, mock_agent, mock_swarm):
        """Test successful orchestrator initialization."""
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        orchestrator = ReactiveSwarmOrchestrator()
        
        assert orchestrator.base_agent == mock_agent_instance
        assert isinstance(orchestrator.shared_memory, ReactiveSharedMemory)
        assert isinstance(orchestrator.adaptation_engine, AdaptationEngine)
        assert orchestrator.execution_history == []
    
    def test_complexity_analysis_methods(self):
        """Test task complexity analysis helper methods."""
        with patch('reagent.core.orchestrator.SWARM_AVAILABLE', True), \
             patch('reagent.core.orchestrator.swarm'), \
             patch('strands.Agent'):
            
            orchestrator = ReactiveSwarmOrchestrator()
            
            # Test keyword counting
            task1 = "analyze and compare multiple different approaches"
            keywords1 = orchestrator._count_complexity_keywords(task1)
            assert keywords1 >= 3  # analyze, compare, multiple, different
            
            task2 = "simple task"
            keywords2 = orchestrator._count_complexity_keywords(task2)
            assert keywords2 == 0
            
            # Test step estimation
            task3 = "first do this, then do that, and finally finish"
            steps3 = orchestrator._estimate_task_steps(task3)
            assert steps3 >= 3  # first, then, finally
            
            # Test domain breadth
            task4 = "analyze code performance and create business strategy"
            domains4 = orchestrator._estimate_domain_breadth(task4)
            assert domains4 >= 2  # technical + business domains


class TestReactiveSharedMemory:
    """Test cases for ReactiveSharedMemory."""
    
    @pytest.fixture
    def memory(self):
        """Create a ReactiveSharedMemory instance for testing."""
        return ReactiveSharedMemory()
    
    @pytest.mark.asyncio
    async def test_memory_tier_determination(self, memory):
        """Test automatic memory tier determination."""
        # Large values should go to persistent
        large_value = "x" * 20000
        tier1 = memory._determine_optimal_tier("test1", large_value)
        assert tier1 == MemoryTier.PERSISTENT
        
        # Execution results should go to persistent
        tier2 = memory._determine_optimal_tier("execution:test", "result")
        assert tier2 == MemoryTier.PERSISTENT
        
        # Temporary data should go to local
        tier3 = memory._determine_optimal_tier("temp:test", "data")
        assert tier3 == MemoryTier.LOCAL
        
        # Default should be local
        tier4 = memory._determine_optimal_tier("regular_key", "small_value")
        assert tier4 == MemoryTier.LOCAL
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory):
        """Test basic store and retrieve operations."""
        # Store in local tier
        result = await memory.store_with_tier("test_key", "test_value", MemoryTier.LOCAL)
        assert result['success'] == True
        assert result['tier'] == 'local'
        
        # Retrieve the value
        retrieved = await memory.retrieve_with_context("test_key")
        assert retrieved['found'] == True
        assert retrieved['value'] == "test_value"
        assert retrieved['tier'] == 'local'
        assert retrieved['access_count'] == 1
    
    @pytest.mark.asyncio
    async def test_auto_tier_selection(self, memory):
        """Test automatic tier selection."""
        result = await memory.store_with_tier("auto_key", "auto_value", "auto")
        assert result['success'] == True
        assert result['tier'] in ['local', 'persistent', 'shared', 'archive']
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, memory):
        """Test memory statistics."""
        # Store some test data
        await memory.store_with_tier("local1", "value1", MemoryTier.LOCAL)
        await memory.store_with_tier("persistent1", "value2", MemoryTier.PERSISTENT)
        
        stats = await memory.get_memory_stats()
        
        assert 'tier_counts' in stats
        assert 'total_entries' in stats
        assert stats['tier_counts']['local'] >= 1
        assert stats['tier_counts']['persistent'] >= 1
        assert stats['total_entries'] >= 2


class TestAdaptationEngine:
    """Test cases for AdaptationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create an AdaptationEngine instance for testing."""
        return AdaptationEngine()
    
    def test_adaptation_rules_initialization(self, engine):
        """Test that adaptation rules are properly initialized."""
        assert len(engine.adaptation_rules) > 0
        
        # Check that rules have required attributes
        for rule in engine.adaptation_rules:
            assert hasattr(rule, 'trigger')
            assert hasattr(rule, 'condition')
            assert hasattr(rule, 'action')
            assert hasattr(rule, 'priority')
            assert hasattr(rule, 'threshold')
    
    def test_trigger_analysis(self, engine):
        """Test adaptation trigger analysis."""
        current_config = {
            'swarm_size': 3,
            'estimated_time': 60,
            'timeout_seconds': 300
        }
        
        intermediate_results = [
            {'success': True, 'quality_score': 0.8},
            {'success': False, 'quality_score': 0.4},
            {'success': True, 'quality_score': 0.9}
        ]
        
        performance_metrics = {
            'execution_time': 90,  # 50% longer than estimated
            'cpu_usage': 0.7,
            'memory_usage': 0.6
        }
        
        # This should trigger performance degradation rule
        triggered_rules = engine._analyze_triggers(
            current_config, intermediate_results, performance_metrics
        )
        
        # Should have at least one triggered rule
        assert len(triggered_rules) > 0
        
        # Check that rules are sorted by priority
        if len(triggered_rules) > 1:
            for i in range(len(triggered_rules) - 1):
                assert triggered_rules[i].priority >= triggered_rules[i + 1].priority
    
    def test_adaptation_generation(self, engine):
        """Test adaptation generation from triggered rules."""
        current_config = {
            'swarm_size': 3,
            'max_size': 8,
            'coordination_pattern': 'hybrid'
        }
        
        # Mock a triggered rule
        from reagent.core.adaptation import AdaptationRule
        mock_rule = AdaptationRule(
            trigger=AdaptationTrigger.COMPLEXITY_INCREASE,
            condition="Test condition",
            action="Test action",
            priority=8,
            threshold=0.7
        )
        
        adaptations = engine._generate_adaptations(current_config, [mock_rule])
        
        assert 'new_config' in adaptations
        assert 'adaptations_applied' in adaptations
        assert 'reasoning' in adaptations
        
        # Should have increased swarm size
        new_config = adaptations['new_config']
        assert new_config['swarm_size'] > current_config['swarm_size']
        assert new_config['coordination_pattern'] == 'collaborative'
    
    def test_adaptation_stats(self, engine):
        """Test adaptation statistics tracking."""
        # Initially no adaptations
        stats = engine.get_adaptation_stats()
        assert 'message' in stats
        
        # Mock some adaptation history
        engine.adaptation_history = [
            {
                'timestamp': 1234567890,
                'triggered_rules': ['complexity_increase'],
                'adaptations': {'test': 'adaptation'}
            }
        ]
        
        stats = engine.get_adaptation_stats()
        assert stats['total_adaptations'] == 1
        assert 'trigger_counts' in stats
        assert stats['trigger_counts']['complexity_increase'] == 1


if __name__ == "__main__":
    pytest.main([__file__])
