"""
AdaptationEngine - Reactive adaptation logic for swarm orchestration.

Analyzes execution patterns and adapts swarm configuration in real-time.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AdaptationTrigger(Enum):
    """Triggers that can cause swarm adaptation."""
    COMPLEXITY_INCREASE = "complexity_increase"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_INCREASE = "error_rate_increase"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    TASK_SCOPE_CHANGE = "task_scope_change"
    QUALITY_ISSUES = "quality_issues"
    TIME_PRESSURE = "time_pressure"


@dataclass
class AdaptationRule:
    """Rule for swarm adaptation."""
    trigger: AdaptationTrigger
    condition: str  # Description of when to trigger
    action: str     # Description of what to do
    priority: int   # Higher number = higher priority
    threshold: float  # Threshold value for triggering


class AdaptationEngine:
    """
    Engine for analyzing execution patterns and adapting swarm configuration.
    
    Features:
    - Real-time performance monitoring
    - Pattern recognition for optimization opportunities
    - Dynamic configuration adjustment
    - Learning from execution history
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AdaptationEngine")
        
        # Adaptation rules
        self.adaptation_rules = self._initialize_adaptation_rules()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.min_samples_for_adaptation = 3
        self.performance_window_size = 10
        self.adaptation_cooldown = 30  # seconds between adaptations
        self.last_adaptation_time = 0
        
        self.logger.info("AdaptationEngine initialized")
    
    def _initialize_adaptation_rules(self) -> List[AdaptationRule]:
        """Initialize default adaptation rules."""
        return [
            AdaptationRule(
                trigger=AdaptationTrigger.COMPLEXITY_INCREASE,
                condition="Task complexity exceeds initial estimate",
                action="Increase swarm size and switch to collaborative pattern",
                priority=8,
                threshold=0.7
            ),
            AdaptationRule(
                trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
                condition="Execution time significantly exceeds estimate",
                action="Increase swarm size or switch coordination pattern",
                priority=7,
                threshold=1.5  # 50% longer than expected
            ),
            AdaptationRule(
                trigger=AdaptationTrigger.ERROR_RATE_INCREASE,
                condition="Error rate above acceptable threshold",
                action="Switch to collaborative pattern and add validation agents",
                priority=9,
                threshold=0.2  # 20% error rate
            ),
            AdaptationRule(
                trigger=AdaptationTrigger.RESOURCE_CONSTRAINTS,
                condition="Resource usage approaching limits",
                action="Reduce swarm size or optimize coordination",
                priority=6,
                threshold=0.8  # 80% resource usage
            ),
            AdaptationRule(
                trigger=AdaptationTrigger.QUALITY_ISSUES,
                condition="Output quality below standards",
                action="Switch to collaborative pattern and add review agents",
                priority=8,
                threshold=0.6  # Quality score below 60%
            ),
            AdaptationRule(
                trigger=AdaptationTrigger.TIME_PRESSURE,
                condition="Execution time approaching timeout",
                action="Increase swarm size and switch to competitive pattern",
                priority=7,
                threshold=0.8  # 80% of timeout reached
            )
        ]
    
    def analyze_and_adapt(
        self,
        current_config: Dict[str, Any],
        intermediate_results: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze current execution and determine adaptations.
        
        Args:
            current_config: Current swarm configuration
            intermediate_results: Results from current execution phase
            performance_metrics: Performance measurements
            
        Returns:
            Adaptation recommendations and new configuration
        """
        self.logger.info("Starting adaptation analysis")
        
        # Record performance data
        self._record_performance(performance_metrics, intermediate_results)
        
        # Check adaptation cooldown
        current_time = time.time()
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return {
                'adaptations_needed': False,
                'reason': 'Adaptation cooldown active',
                'new_config': current_config
            }
        
        # Analyze for adaptation triggers
        triggered_rules = self._analyze_triggers(
            current_config, intermediate_results, performance_metrics
        )
        
        if not triggered_rules:
            return {
                'adaptations_needed': False,
                'reason': 'No adaptation triggers detected',
                'new_config': current_config
            }
        
        # Generate adaptations based on triggered rules
        adaptations = self._generate_adaptations(current_config, triggered_rules)
        
        # Record adaptation
        self._record_adaptation(triggered_rules, adaptations)
        self.last_adaptation_time = current_time
        
        return {
            'adaptations_needed': True,
            'triggered_rules': [rule.trigger.value for rule in triggered_rules],
            'adaptations': adaptations,
            'new_config': adaptations.get('new_config', current_config),
            'reasoning': adaptations.get('reasoning', 'Adaptation based on triggered rules')
        }
    
    def _analyze_triggers(
        self,
        current_config: Dict[str, Any],
        intermediate_results: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> List[AdaptationRule]:
        """Analyze current state for adaptation triggers."""
        triggered_rules = []
        
        for rule in self.adaptation_rules:
            if self._check_rule_trigger(rule, current_config, intermediate_results, performance_metrics):
                triggered_rules.append(rule)
        
        # Sort by priority (highest first)
        triggered_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return triggered_rules
    
    def _check_rule_trigger(
        self,
        rule: AdaptationRule,
        current_config: Dict[str, Any],
        intermediate_results: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> bool:
        """Check if a specific rule should trigger."""
        
        if rule.trigger == AdaptationTrigger.COMPLEXITY_INCREASE:
            # Check if task complexity has increased beyond initial estimate
            estimated_complexity = current_config.get('estimated_complexity', 0.5)
            actual_complexity = performance_metrics.get('actual_complexity', 0.5)
            return actual_complexity > estimated_complexity * (1 + rule.threshold)
        
        elif rule.trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            # Check if execution time exceeds estimates
            estimated_time = current_config.get('estimated_time', 60)
            actual_time = performance_metrics.get('execution_time', 0)
            return actual_time > estimated_time * rule.threshold
        
        elif rule.trigger == AdaptationTrigger.ERROR_RATE_INCREASE:
            # Check error rate in intermediate results
            total_results = len(intermediate_results)
            if total_results == 0:
                return False
            
            error_count = sum(1 for result in intermediate_results 
                            if not result.get('success', True))
            error_rate = error_count / total_results
            return error_rate > rule.threshold
        
        elif rule.trigger == AdaptationTrigger.RESOURCE_CONSTRAINTS:
            # Check resource usage
            cpu_usage = performance_metrics.get('cpu_usage', 0)
            memory_usage = performance_metrics.get('memory_usage', 0)
            max_usage = max(cpu_usage, memory_usage)
            return max_usage > rule.threshold
        
        elif rule.trigger == AdaptationTrigger.QUALITY_ISSUES:
            # Check output quality scores
            quality_scores = [result.get('quality_score', 1.0) 
                            for result in intermediate_results 
                            if 'quality_score' in result]
            if not quality_scores:
                return False
            
            avg_quality = sum(quality_scores) / len(quality_scores)
            return avg_quality < rule.threshold
        
        elif rule.trigger == AdaptationTrigger.TIME_PRESSURE:
            # Check if approaching timeout
            timeout = current_config.get('timeout_seconds', 300)
            elapsed_time = performance_metrics.get('execution_time', 0)
            return elapsed_time > timeout * rule.threshold
        
        return False
    
    def _generate_adaptations(
        self,
        current_config: Dict[str, Any],
        triggered_rules: List[AdaptationRule]
    ) -> Dict[str, Any]:
        """Generate specific adaptations based on triggered rules."""
        
        new_config = current_config.copy()
        adaptations_applied = []
        reasoning_parts = []
        
        for rule in triggered_rules:
            adaptation = self._apply_rule_adaptation(rule, new_config)
            if adaptation:
                adaptations_applied.append(adaptation)
                reasoning_parts.append(f"{rule.trigger.value}: {adaptation['description']}")
        
        return {
            'new_config': new_config,
            'adaptations_applied': adaptations_applied,
            'reasoning': '; '.join(reasoning_parts)
        }
    
    def _apply_rule_adaptation(
        self,
        rule: AdaptationRule,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply specific adaptation for a rule."""
        
        if rule.trigger == AdaptationTrigger.COMPLEXITY_INCREASE:
            # Increase swarm size and switch to collaborative
            current_size = config.get('swarm_size', 3)
            max_size = config.get('max_size', 8)
            new_size = min(max_size, current_size + 2)
            
            config['swarm_size'] = new_size
            config['coordination_pattern'] = 'collaborative'
            
            return {
                'type': 'swarm_size_increase',
                'description': f'Increased swarm size from {current_size} to {new_size}',
                'pattern_change': 'collaborative'
            }
        
        elif rule.trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            # Increase swarm size or change pattern
            current_size = config.get('swarm_size', 3)
            max_size = config.get('max_size', 8)
            
            if current_size < max_size:
                new_size = min(max_size, current_size + 1)
                config['swarm_size'] = new_size
                return {
                    'type': 'performance_optimization',
                    'description': f'Increased swarm size to {new_size} for better performance'
                }
            else:
                # Switch to competitive pattern for parallel processing
                config['coordination_pattern'] = 'competitive'
                return {
                    'type': 'pattern_optimization',
                    'description': 'Switched to competitive pattern for parallel processing'
                }
        
        elif rule.trigger == AdaptationTrigger.ERROR_RATE_INCREASE:
            # Switch to collaborative and add validation
            config['coordination_pattern'] = 'collaborative'
            config['enable_validation'] = True
            
            return {
                'type': 'error_mitigation',
                'description': 'Switched to collaborative pattern with validation enabled'
            }
        
        elif rule.trigger == AdaptationTrigger.RESOURCE_CONSTRAINTS:
            # Reduce swarm size
            current_size = config.get('swarm_size', 3)
            min_size = config.get('min_size', 1)
            new_size = max(min_size, current_size - 1)
            
            config['swarm_size'] = new_size
            
            return {
                'type': 'resource_optimization',
                'description': f'Reduced swarm size to {new_size} to manage resources'
            }
        
        elif rule.trigger == AdaptationTrigger.QUALITY_ISSUES:
            # Switch to collaborative with review
            config['coordination_pattern'] = 'collaborative'
            config['enable_quality_review'] = True
            
            return {
                'type': 'quality_improvement',
                'description': 'Enabled collaborative pattern with quality review'
            }
        
        elif rule.trigger == AdaptationTrigger.TIME_PRESSURE:
            # Increase swarm size and switch to competitive
            current_size = config.get('swarm_size', 3)
            max_size = config.get('max_size', 8)
            new_size = min(max_size, current_size + 2)
            
            config['swarm_size'] = new_size
            config['coordination_pattern'] = 'competitive'
            
            return {
                'type': 'urgency_optimization',
                'description': f'Increased swarm to {new_size} with competitive pattern for speed'
            }
        
        return None
    
    def _record_performance(
        self,
        performance_metrics: Dict[str, float],
        intermediate_results: List[Dict[str, Any]]
    ) -> None:
        """Record performance data for analysis."""
        performance_record = {
            'timestamp': time.time(),
            'metrics': performance_metrics,
            'result_count': len(intermediate_results),
            'success_rate': self._calculate_success_rate(intermediate_results)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window_size:
            self.performance_history = self.performance_history[-self.performance_window_size:]
    
    def _record_adaptation(
        self,
        triggered_rules: List[AdaptationRule],
        adaptations: Dict[str, Any]
    ) -> None:
        """Record adaptation for learning."""
        adaptation_record = {
            'timestamp': time.time(),
            'triggered_rules': [rule.trigger.value for rule in triggered_rules],
            'adaptations': adaptations,
            'rule_priorities': [rule.priority for rule in triggered_rules]
        }
        
        self.adaptation_history.append(adaptation_record)
    
    def _calculate_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate success rate from results."""
        if not results:
            return 1.0
        
        successful = sum(1 for result in results if result.get('success', True))
        return successful / len(results)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptations made."""
        if not self.adaptation_history:
            return {'message': 'No adaptations recorded yet'}
        
        # Count adaptations by trigger type
        trigger_counts = {}
        for record in self.adaptation_history:
            for trigger in record['triggered_rules']:
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'trigger_counts': trigger_counts,
            'recent_adaptations': self.adaptation_history[-5:],  # Last 5
            'performance_samples': len(self.performance_history)
        }
    
    def add_custom_rule(self, rule: AdaptationRule) -> None:
        """Add a custom adaptation rule."""
        self.adaptation_rules.append(rule)
        self.adaptation_rules.sort(key=lambda r: r.priority, reverse=True)
        self.logger.info(f"Added custom adaptation rule: trigger={rule.trigger.value}")
