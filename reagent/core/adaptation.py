"""
AdaptationEngine - Reactive adaptation logic for swarm orchestration.

Analyzes execution patterns and adapts swarm configuration in real-time.
"""

import time
import json
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
                'new_config': current_config,
                'triggered_rules': [],
                'adaptations': {},
                'reasoning': 'Adaptation cooldown active'
            }
        
        # Analyze for adaptation triggers
        triggered_rules = self._analyze_triggers(
            current_config, intermediate_results, performance_metrics
        )
        
        if not triggered_rules:
            return {
                'adaptations_needed': False,
                'reason': 'No adaptation triggers detected',
                'new_config': current_config,
                'triggered_rules': [],
                'adaptations': {},
                'reasoning': 'No adaptation triggers detected'
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
        """Check if a specific rule should trigger using LLM analysis."""
        
        try:
            from strands import Agent
            from strands.models import BedrockModel
            
            # Create agent for adaptation analysis
            model = BedrockModel(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name="us-west-2"
            )
            
            agent = Agent(
                model=model,
                system_prompt="You are a swarm adaptation analyzer. Evaluate execution data to determine if adaptations are needed."
            )
            
            analysis_prompt = f"""
            Analyze this swarm execution data to determine if adaptation is needed for: {rule.trigger.value}
            
            Rule condition: {rule.condition}
            Rule threshold: {rule.threshold}
            
            Current configuration:
            {json.dumps(current_config, indent=2)}
            
            Performance metrics:
            {json.dumps(performance_metrics, indent=2)}
            
            Intermediate results count: {len(intermediate_results)}
            Results summary: {self._summarize_results(intermediate_results)}
            
            Based on this data, should the adaptation rule "{rule.trigger.value}" be triggered?
            
            Consider:
            - Is the current performance meeting expectations?
            - Are there signs of the specific issue this rule addresses?
            - Would the suggested adaptation help?
            - Is the threshold condition met based on the data?
            
            Respond with only: YES or NO
            """
            
            response = str(agent(analysis_prompt)).strip().upper()
            return response == "YES"
            
        except Exception as e:
            self.logger.error(f"LLM adaptation analysis failed for {rule.trigger.value}: {str(e)}")
            # Fallback to conservative rule-based logic
            return self._fallback_rule_check(rule, current_config, intermediate_results, performance_metrics)
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Create a summary of intermediate results for LLM analysis."""
        if not results:
            return "No results available"
        
        success_count = sum(1 for r in results if r.get('success', True))
        total_count = len(results)
        
        summary = f"Success rate: {success_count}/{total_count}"
        
        # Add quality information if available
        quality_scores = [r.get('quality_score') for r in results if 'quality_score' in r]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            summary += f", Average quality: {avg_quality:.2f}"
        
        # Add error information if available
        errors = [r.get('error') for r in results if 'error' in r]
        if errors:
            summary += f", Errors present: {len(errors)}"
        
        return summary
    
    def _fallback_rule_check(
        self,
        rule: AdaptationRule,
        current_config: Dict[str, Any],
        intermediate_results: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> bool:
        """Conservative fallback rule checking when LLM analysis fails."""
        
        # Use very conservative thresholds to avoid unnecessary adaptations
        if rule.trigger == AdaptationTrigger.ERROR_RATE_INCREASE:
            total_results = len(intermediate_results)
            if total_results == 0:
                return False
            
            error_count = sum(1 for result in intermediate_results 
                            if not result.get('success', True))
            error_rate = error_count / total_results
            return error_rate > 0.5  # Only trigger on >50% error rate
        
        elif rule.trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            estimated_time = current_config.get('estimated_time', 60)
            actual_time = performance_metrics.get('execution_time', 0)
            return actual_time > estimated_time * 2.0  # Only trigger if 2x slower
        
        elif rule.trigger == AdaptationTrigger.RESOURCE_CONSTRAINTS:
            cpu_usage = performance_metrics.get('cpu_usage', 0)
            memory_usage = performance_metrics.get('memory_usage', 0)
            max_usage = max(cpu_usage, memory_usage)
            return max_usage > 0.9  # Only trigger at 90% usage
        
        # For other triggers, be very conservative
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
