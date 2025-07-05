"""
Reward function framework for AdBot reinforcement learning environments.

This module provides a comprehensive reward shaping system for optimizing
advertising campaigns across multiple objectives including ROI, ROAS, CTR,
conversion rates, and budget efficiency.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of reward functions available."""
    ROI = "roi"
    ROAS = "roas"
    CTR = "ctr"
    CONVERSION_RATE = "conversion_rate"
    CPA = "cpa"
    BUDGET_EFFICIENCY = "budget_efficiency"
    QUALITY_SCORE = "quality_score"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class MetricBounds:
    """Bounds for normalizing metrics."""
    min_value: float = 0.0
    max_value: float = 1.0
    target_value: Optional[float] = None


@dataclass
class RewardConfig:
    """Configuration for reward function calculation."""
    reward_type: RewardType = RewardType.MULTI_OBJECTIVE
    weights: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, MetricBounds] = field(default_factory=dict)
    penalty_threshold: float = 0.1
    bonus_multiplier: float = 1.5
    normalize_rewards: bool = True
    clip_rewards: bool = True
    reward_range: Tuple[float, float] = (-1.0, 1.0)


class BaseRewardFunction:
    """Base class for all reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.reward_history: List[float] = []
        self.metrics_history: List[Dict[str, float]] = []
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate reward based on metrics and actions."""
        raise NotImplementedError
    
    def normalize_metric(self, value: float, metric_name: str) -> float:
        """Normalize a metric to [0, 1] range."""
        if metric_name not in self.config.bounds:
            return value
        
        bounds = self.config.bounds[metric_name]
        if bounds.max_value <= bounds.min_value:
            return 0.0
        
        normalized = (value - bounds.min_value) / (bounds.max_value - bounds.min_value)
        return np.clip(normalized, 0.0, 1.0)
    
    def clip_reward(self, reward: float) -> float:
        """Clip reward to configured range."""
        if self.config.clip_rewards:
            return np.clip(reward, *self.config.reward_range)
        return reward
    
    def add_penalty(self, base_reward: float, penalty_factor: float) -> float:
        """Add penalty to base reward."""
        return base_reward * (1.0 - penalty_factor)
    
    def add_bonus(self, base_reward: float, bonus_factor: float) -> float:
        """Add bonus to base reward."""
        return base_reward * (1.0 + bonus_factor * self.config.bonus_multiplier)


class ROIRewardFunction(BaseRewardFunction):
    """Reward function optimizing for Return on Investment."""
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate ROI-based reward."""
        revenue = metrics.get('revenue', 0.0)
        cost = metrics.get('cost', 1.0)
        
        if cost <= 0:
            return self.config.reward_range[0]
        
        roi = (revenue - cost) / cost
        
        # Normalize ROI
        roi_bounds = self.config.bounds.get('roi', MetricBounds(-1.0, 5.0))
        normalized_roi = self.normalize_metric(roi, 'roi')
        
        # Apply penalties for low performance
        if roi < self.config.penalty_threshold:
            penalty = (self.config.penalty_threshold - roi) / self.config.penalty_threshold
            normalized_roi = self.add_penalty(normalized_roi, penalty)
        
        # Apply bonus for exceptional performance
        if roi_bounds.target_value and roi > roi_bounds.target_value:
            bonus = (roi - roi_bounds.target_value) / roi_bounds.target_value
            normalized_roi = self.add_bonus(normalized_roi, bonus)
        
        return self.clip_reward(normalized_roi)


class ROASRewardFunction(BaseRewardFunction):
    """Reward function optimizing for Return on Ad Spend."""
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate ROAS-based reward."""
        revenue = metrics.get('revenue', 0.0)
        ad_spend = metrics.get('ad_spend', 1.0)
        
        if ad_spend <= 0:
            return self.config.reward_range[0]
        
        roas = revenue / ad_spend
        
        # Normalize ROAS
        roas_bounds = self.config.bounds.get('roas', MetricBounds(0.0, 10.0))
        normalized_roas = self.normalize_metric(roas, 'roas')
        
        # Target ROAS threshold
        target_roas = roas_bounds.target_value or 2.0
        
        if roas < target_roas:
            penalty = (target_roas - roas) / target_roas
            normalized_roas = self.add_penalty(normalized_roas, penalty)
        else:
            bonus = (roas - target_roas) / target_roas
            normalized_roas = self.add_bonus(normalized_roas, bonus)
        
        return self.clip_reward(normalized_roas)


class CTRRewardFunction(BaseRewardFunction):
    """Reward function optimizing for Click-Through Rate."""
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate CTR-based reward."""
        clicks = metrics.get('clicks', 0.0)
        impressions = metrics.get('impressions', 1.0)
        
        if impressions <= 0:
            return self.config.reward_range[0]
        
        ctr = clicks / impressions
        
        # Normalize CTR
        ctr_bounds = self.config.bounds.get('ctr', MetricBounds(0.0, 0.2))
        normalized_ctr = self.normalize_metric(ctr, 'ctr')
        
        # Industry benchmarks
        industry_avg = ctr_bounds.target_value or 0.02
        
        if ctr >= industry_avg:
            bonus = (ctr - industry_avg) / industry_avg
            normalized_ctr = self.add_bonus(normalized_ctr, bonus)
        
        return self.clip_reward(normalized_ctr)


class ConversionRateRewardFunction(BaseRewardFunction):
    """Reward function optimizing for conversion rate."""
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate conversion rate-based reward."""
        conversions = metrics.get('conversions', 0.0)
        clicks = metrics.get('clicks', 1.0)
        
        if clicks <= 0:
            return self.config.reward_range[0]
        
        conversion_rate = conversions / clicks
        
        # Normalize conversion rate
        cr_bounds = self.config.bounds.get('conversion_rate', MetricBounds(0.0, 0.5))
        normalized_cr = self.normalize_metric(conversion_rate, 'conversion_rate')
        
        # Target conversion rate
        target_cr = cr_bounds.target_value or 0.05
        
        if conversion_rate >= target_cr:
            bonus = (conversion_rate - target_cr) / target_cr
            normalized_cr = self.add_bonus(normalized_cr, bonus)
        
        return self.clip_reward(normalized_cr)


class CPARewardFunction(BaseRewardFunction):
    """Reward function optimizing for Cost Per Acquisition."""
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate CPA-based reward (lower is better)."""
        cost = metrics.get('cost', 0.0)
        acquisitions = metrics.get('conversions', 1.0)
        
        if acquisitions <= 0:
            return self.config.reward_range[0]
        
        cpa = cost / acquisitions
        
        # Normalize CPA (inverted since lower is better)
        cpa_bounds = self.config.bounds.get('cpa', MetricBounds(0.0, 1000.0))
        normalized_cpa = 1.0 - self.normalize_metric(cpa, 'cpa')
        
        # Target CPA
        target_cpa = cpa_bounds.target_value or 50.0
        
        if cpa <= target_cpa:
            bonus = (target_cpa - cpa) / target_cpa
            normalized_cpa = self.add_bonus(normalized_cpa, bonus)
        
        return self.clip_reward(normalized_cpa)


class BudgetEfficiencyRewardFunction(BaseRewardFunction):
    """Reward function optimizing for budget utilization efficiency."""
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate budget efficiency reward."""
        spent = metrics.get('spent', 0.0)
        budget = metrics.get('budget', 1.0)
        performance = metrics.get('performance_score', 0.0)
        
        if budget <= 0:
            return self.config.reward_range[0]
        
        # Budget utilization
        utilization = spent / budget
        
        # Efficiency score (performance per dollar)
        efficiency = performance / max(spent, 1.0)
        
        # Optimal utilization range (80-95%)
        target_utilization = 0.9
        utilization_penalty = abs(utilization - target_utilization)
        
        # Combine efficiency and utilization
        base_reward = efficiency * (1.0 - utilization_penalty)
        
        return self.clip_reward(base_reward)


class MultiObjectiveRewardFunction(BaseRewardFunction):
    """Multi-objective reward function combining multiple metrics."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.sub_functions = self._initialize_sub_functions()
    
    def _initialize_sub_functions(self) -> Dict[str, BaseRewardFunction]:
        """Initialize sub-reward functions."""
        functions = {
            'roi': ROIRewardFunction(self.config),
            'roas': ROASRewardFunction(self.config),
            'ctr': CTRRewardFunction(self.config),
            'conversion_rate': ConversionRateRewardFunction(self.config),
            'cpa': CPARewardFunction(self.config),
            'budget_efficiency': BudgetEfficiencyRewardFunction(self.config)
        }
        return functions
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        actions: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate multi-objective reward."""
        weighted_rewards = []
        
        for metric_name, weight in self.config.weights.items():
            if metric_name in self.sub_functions and weight > 0:
                sub_reward = self.sub_functions[metric_name].calculate_reward(
                    metrics, actions, context
                )
                weighted_rewards.append(sub_reward * weight)
        
        if not weighted_rewards:
            logger.warning("No valid reward components found")
            return 0.0
        
        # Weighted average
        total_reward = sum(weighted_rewards) / sum(self.config.weights.values())
        
        return self.clip_reward(total_reward)


class RewardFunctionFactory:
    """Factory for creating reward functions."""
    
    @staticmethod
    def create_reward_function(config: RewardConfig) -> BaseRewardFunction:
        """Create a reward function based on configuration."""
        function_map = {
            RewardType.ROI: ROIRewardFunction,
            RewardType.ROAS: ROASRewardFunction,
            RewardType.CTR: CTRRewardFunction,
            RewardType.CONVERSION_RATE: ConversionRateRewardFunction,
            RewardType.CPA: CPARewardFunction,
            RewardType.BUDGET_EFFICIENCY: BudgetEfficiencyRewardFunction,
            RewardType.MULTI_OBJECTIVE: MultiObjectiveRewardFunction
        }
        
        if config.reward_type not in function_map:
            raise ValueError(f"Unknown reward type: {config.reward_type}")
        
        return function_map[config.reward_type](config)
    
    @staticmethod
    def create_default_config() -> RewardConfig:
        """Create default reward configuration."""
        return RewardConfig(
            reward_type=RewardType.MULTI_OBJECTIVE,
            weights={
                'roi': 0.3,
                'roas': 0.25,
                'ctr': 0.15,
                'conversion_rate': 0.2,
                'budget_efficiency': 0.1
            },
            bounds={
                'roi': MetricBounds(-1.0, 5.0, 1.0),
                'roas': MetricBounds(0.0, 10.0, 2.0),
                'ctr': MetricBounds(0.0, 0.2, 0.02),
                'conversion_rate': MetricBounds(0.0, 0.5, 0.05),
                'cpa': MetricBounds(0.0, 1000.0, 50.0)
            },
            penalty_threshold=0.1,
            bonus_multiplier=1.5,
            normalize_rewards=True,
            clip_rewards=True,
            reward_range=(-1.0, 1.0)
        )


class RewardShapingUtils:
    """Utilities for advanced reward shaping."""
    
    @staticmethod
    def add_temporal_bonus(
        reward: float,
        time_factor: float,
        decay_rate: float = 0.95
    ) -> float:
        """Add temporal bonus that decays over time."""
        return reward * (1.0 + time_factor * decay_rate)
    
    @staticmethod
    def add_exploration_bonus(
        reward: float,
        exploration_factor: float,
        exploration_weight: float = 0.1
    ) -> float:
        """Add exploration bonus for trying new actions."""
        return reward + exploration_factor * exploration_weight
    
    @staticmethod
    def add_consistency_bonus(
        reward: float,
        consistency_score: float,
        consistency_weight: float = 0.1
    ) -> float:
        """Add bonus for consistent performance."""
        return reward * (1.0 + consistency_score * consistency_weight)
    
    @staticmethod
    def calculate_performance_trend(
        metrics_history: List[Dict[str, float]],
        metric_name: str,
        window_size: int = 5
    ) -> float:
        """Calculate performance trend over time."""
        if len(metrics_history) < window_size:
            return 0.0
        
        recent_values = [
            m.get(metric_name, 0.0) 
            for m in metrics_history[-window_size:]
        ]
        
        if not recent_values:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        return slope / max(abs(np.mean(recent_values)), 1e-8)