"""
Campaign Optimization RL Environment - Minimal Viable Product

This is a simplified campaign optimization environment focused on basic
budget allocation and bid adjustments for testing RL algorithms.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .base import BaseAdEnvironment, AdEnvironmentConfig, AdEnvironmentMetrics
from ..reward_functions import RewardFunctionFactory, RewardType, RewardConfig

logger = logging.getLogger(__name__)


class SimpleCampaignEnv(BaseAdEnvironment):
    """
    Minimal campaign optimization environment for RL testing.
    
    Action Space: [budget_multiplier, bid_multiplier] 
    - budget_multiplier: 0.5 to 2.0 (adjust campaign budget)
    - bid_multiplier: 0.5 to 2.0 (adjust keyword bids)
    
    Observation Space: [current_roi, current_ctr, current_cost, budget_remaining, day_of_week]
    
    Reward: ROI-based with budget efficiency considerations
    """
    
    def __init__(self, config: Optional[AdEnvironmentConfig] = None):
        # Use default config if none provided
        if config is None:
            config = AdEnvironmentConfig(
                max_episode_steps=30,  # 30 days simulation
                observation_dim=5,
                action_dim=2,
                reward_config=RewardConfig(
                    reward_type=RewardType.ROI,
                    normalize_rewards=True
                )
            )
        
        super().__init__(config)
        
        # Campaign simulation parameters
        self.initial_budget = 1000.0
        self.daily_budget = self.initial_budget / config.max_episode_steps
        self.base_impressions = 1000
        self.base_ctr = 0.02  # 2% CTR
        self.base_conversion_rate = 0.05  # 5% conversion rate
        self.base_cpc = 1.0  # $1 cost per click
        self.conversion_value = 50.0  # $50 per conversion
        
        # Market dynamics (adds realism)
        self.market_competition = np.random.uniform(0.8, 1.2)  # Competition factor
        self.seasonal_factor = 1.0
        
        logger.info("Initialized SimpleCampaignEnv for RL testing")
    
    def _create_observation_space(self):
        """Define observation space: [roi, ctr, cost, budget_remaining, day]"""
        from gymnasium import spaces
        return spaces.Box(
            low=np.array([-2.0, 0.0, 0.0, 0.0, 0.0]),    # [roi, ctr, cost, budget, day]
            high=np.array([5.0, 0.2, 2000.0, 1000.0, 6.0]),
            dtype=np.float32
        )
    
    def _create_action_space(self):
        """Define action space: [budget_multiplier, bid_multiplier]"""
        from gymnasium import spaces
        return spaces.Box(
            low=np.array([0.5, 0.5]),    # Min 50% of current values
            high=np.array([2.0, 2.0]),   # Max 200% of current values
            dtype=np.float32
        )
    
    def _reset_environment_state(self):
        """Reset campaign state for new episode"""
        self.current_budget = self.initial_budget
        self.current_bid_multiplier = 1.0
        self.current_budget_multiplier = 1.0
        
        # Performance tracking
        self.total_cost = 0.0
        self.total_revenue = 0.0
        self.total_clicks = 0.0
        self.total_impressions = 0.0
        self.total_conversions = 0.0
        
        # Market dynamics
        self.market_competition = np.random.uniform(0.8, 1.2)
        
        logger.debug("Reset campaign environment state")
    
    def _execute_action(self, action: np.ndarray):
        """Execute budget and bid adjustments"""
        budget_mult, bid_mult = action
        
        # Update multipliers
        self.current_budget_multiplier = budget_mult
        self.current_bid_multiplier = bid_mult
        
        # Calculate daily spend based on budget multiplier
        daily_spend = self.daily_budget * budget_mult
        
        # Ensure we don't exceed remaining budget
        daily_spend = min(daily_spend, self.current_budget)
        
        # Simulate ad performance based on bid multiplier
        effective_cpc = self.base_cpc * self.market_competition / bid_mult
        
        # Higher bids = more impressions and better CTR
        impression_boost = min(bid_mult * 1.2, 2.0)
        impressions = self.base_impressions * impression_boost
        
        # CTR improves with higher bids (better ad positions)
        ctr_boost = min(1.0 + (bid_mult - 1.0) * 0.3, 1.5)
        ctr = self.base_ctr * ctr_boost
        
        # Calculate performance metrics
        clicks = impressions * ctr
        cost = min(clicks * effective_cpc, daily_spend)  # Can't spend more than budget
        actual_clicks = cost / effective_cpc if effective_cpc > 0 else 0
        
        conversions = actual_clicks * self.base_conversion_rate
        revenue = conversions * self.conversion_value
        
        # Update campaign state
        self.current_budget -= cost
        self.total_cost += cost
        self.total_revenue += revenue
        self.total_clicks += actual_clicks
        self.total_impressions += impressions
        self.total_conversions += conversions
        
        logger.debug(f"Day {self.current_step}: cost=${cost:.2f}, revenue=${revenue:.2f}, "
                    f"clicks={actual_clicks:.1f}, budget_left=${self.current_budget:.2f}")
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Calculate current ROI
        roi = AdEnvironmentMetrics.calculate_roi(self.total_revenue, self.total_cost)
        
        # Calculate current CTR
        ctr = AdEnvironmentMetrics.calculate_ctr(self.total_clicks, self.total_impressions)
        
        # Day of episode (normalized)
        day_progress = self.current_step / self.config.max_episode_steps
        
        observation = np.array([
            roi,
            ctr,
            self.total_cost,
            self.current_budget,
            day_progress
        ], dtype=np.float32)
        
        return observation
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics for reward calculation"""
        return {
            'revenue': self.total_revenue,
            'cost': self.total_cost,
            'roi': AdEnvironmentMetrics.calculate_roi(self.total_revenue, self.total_cost),
            'roas': AdEnvironmentMetrics.calculate_roas(self.total_revenue, self.total_cost),
            'ctr': AdEnvironmentMetrics.calculate_ctr(self.total_clicks, self.total_impressions),
            'conversions': self.total_conversions,
            'clicks': self.total_clicks,
            'impressions': self.total_impressions,
            'budget': self.current_budget,
            'spent': self.total_cost,
            'budget_utilization': self.total_cost / self.initial_budget,
            'performance_score': (self.total_revenue - self.total_cost) / max(self.total_cost, 1.0)
        }
    
    def _is_terminated(self) -> bool:
        """Episode terminates if budget is exhausted"""
        return self.current_budget <= 0
    
    def _action_to_dict(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert action array to dictionary"""
        return {
            'budget_multiplier': float(action[0]),
            'bid_multiplier': float(action[1])
        }
    
    def get_campaign_summary(self) -> Dict[str, Any]:
        """Get summary of campaign performance"""
        if self.total_cost == 0:
            return {"error": "No campaign data available"}
        
        return {
            "total_revenue": self.total_revenue,
            "total_cost": self.total_cost,
            "roi": (self.total_revenue - self.total_cost) / self.total_cost,
            "roas": self.total_revenue / self.total_cost,
            "total_clicks": self.total_clicks,
            "total_impressions": self.total_impressions,
            "total_conversions": self.total_conversions,
            "avg_cpc": self.total_cost / self.total_clicks if self.total_clicks > 0 else 0,
            "avg_ctr": self.total_clicks / self.total_impressions if self.total_impressions > 0 else 0,
            "conversion_rate": self.total_conversions / self.total_clicks if self.total_clicks > 0 else 0,
            "budget_utilization": self.total_cost / self.initial_budget,
            "days_run": self.current_step
        }


def create_test_config() -> AdEnvironmentConfig:
    """Create a test configuration for the campaign environment"""
    reward_config = RewardConfig(
        reward_type=RewardType.ROI,
        normalize_rewards=True,
        clip_rewards=True,
        reward_range=(-1.0, 1.0)
    )
    
    return AdEnvironmentConfig(
        max_episode_steps=10,  # Short episode for testing
        observation_dim=5,
        action_dim=2,
        reward_config=reward_config,
        random_seed=42,
        normalize_observations=True
    )


# Alias for backwards compatibility with existing imports
CampaignOptimizationEnv = SimpleCampaignEnv


if __name__ == "__main__":
    """Quick test of the environment"""
    print("Testing SimpleCampaignEnv...")
    
    # Create environment
    env = SimpleCampaignEnv(create_test_config())
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps
    for step in range(5):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Observation: {obs}")
        print(f"  Done: {terminated or truncated}")
        
        if terminated or truncated:
            break
    
    # Get final summary
    summary = env.get_campaign_summary()
    print(f"\nCampaign Summary: {summary}")
    
    print("✅ Environment test completed!")


# Alias for backwards compatibility with existing imports
CampaignOptimizationEnv = SimpleCampaignEnv
