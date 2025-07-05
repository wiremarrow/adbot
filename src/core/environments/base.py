"""
Base Ad Environment for AdBot RL training.

This module provides the foundation for all AdBot RL environments,
implementing the Gymnasium interface with advertising-specific
reward shaping, action spaces, and observation spaces.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

from ..reward_functions import RewardFunctionFactory, RewardConfig, RewardType


logger = logging.getLogger(__name__)


@dataclass
class AdEnvironmentConfig:
    """Configuration for AdBot RL environments."""
    max_episode_steps: int = 1000
    observation_dim: int = 100
    action_dim: int = 10
    reward_config: RewardConfig = None
    random_seed: int = 42
    normalize_observations: bool = True
    normalize_rewards: bool = True
    
    def __post_init__(self):
        if self.reward_config is None:
            self.reward_config = RewardFunctionFactory.create_default_config()


class BaseAdEnvironment(gym.Env):
    """
    Base environment for advertising optimization using RL.
    
    This class provides the foundation for all AdBot RL environments,
    implementing the Gymnasium interface with advertising-specific
    features like reward shaping, action validation, and metric tracking.
    """
    
    def __init__(self, config: AdEnvironmentConfig):
        super().__init__()
        self.config = config
        self.reward_function = RewardFunctionFactory.create_reward_function(
            config.reward_config
        )
        
        # Initialize spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # Environment state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_metrics = []
        self.done = False
        
        # Metrics tracking
        self.performance_history = []
        self.action_history = []
        
        # Random seed
        self.seed(config.random_seed)
        
        logger.info(f"Initialized {self.__class__.__name__} environment")
    
    def _create_observation_space(self) -> spaces.Space:
        """Create observation space for the environment."""
        # Default observation space - subclasses should override
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_dim,),
            dtype=np.float32
        )
    
    def _create_action_space(self) -> spaces.Space:
        """Create action space for the environment."""
        # Default action space - subclasses should override
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.action_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        self.episode_rewards = []
        self.episode_metrics = []
        self.done = False
        
        # Reset environment-specific state
        self._reset_environment_state()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug(f"Environment reset, initial observation shape: {observation.shape}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Environment is done, call reset() first")
        
        # Validate action
        if not self.action_space.contains(action):
            logger.debug(f"Action clipped: {action}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Execute action
        self._execute_action(action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        metrics = self._get_current_metrics()
        reward = self.reward_function.calculate_reward(
            metrics=metrics,
            actions=self._action_to_dict(action),
            context=self._get_context()
        )
        
        # Ensure reward is a Python float for SB3 compatibility
        reward = float(reward)
        
        # Check if episode is done
        terminated = bool(self._is_terminated())
        truncated = bool(self.current_step >= self.config.max_episode_steps)
        self.done = terminated or truncated
        
        # Update tracking
        self.episode_rewards.append(reward)
        self.episode_metrics.append(metrics)
        self.action_history.append(action.copy())
        self.current_step += 1
        
        # Get info
        info = self._get_info()
        
        logger.debug(f"Step {self.current_step}: reward={reward:.4f}, done={self.done}")
        
        return observation, reward, terminated, truncated, info
    
    def _reset_environment_state(self):
        """Reset environment-specific state. Override in subclasses."""
        pass
    
    def _execute_action(self, action: np.ndarray):
        """Execute the given action. Override in subclasses."""
        raise NotImplementedError
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation. Override in subclasses."""
        raise NotImplementedError
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics. Override in subclasses."""
        raise NotImplementedError
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate. Override in subclasses."""
        return False
    
    def _get_context(self) -> Dict[str, Any]:
        """Get additional context for reward calculation."""
        return {
            'step': self.current_step,
            'episode_progress': self.current_step / self.config.max_episode_steps,
            'action_history': self.action_history[-5:] if self.action_history else [],
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for step/reset."""
        return {
            'step': self.current_step,
            'episode_reward': sum(self.episode_rewards),
            'metrics': self.episode_metrics[-1] if self.episode_metrics else {},
            'done': self.done
        }
    
    def _action_to_dict(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert action array to dictionary. Override in subclasses."""
        return {'action': action.tolist()}
    
    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend."""
        if len(self.episode_rewards) < 5:
            return 0.0
        
        recent_rewards = self.episode_rewards[-5:]
        if len(recent_rewards) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_rewards))
        slope = np.polyfit(x, recent_rewards, 1)[0]
        
        return float(slope)
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Episode Reward: {sum(self.episode_rewards):.4f}")
            if self.episode_metrics:
                print(f"Latest Metrics: {self.episode_metrics[-1]}")
    
    def close(self):
        """Clean up environment resources."""
        logger.info(f"Closing {self.__class__.__name__} environment")
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'max_reward': max(self.episode_rewards),
            'episode_length': len(self.episode_rewards),
            'performance_trend': self._calculate_performance_trend()
        }
    
    def get_action_distribution(self) -> Dict[str, Any]:
        """Get statistics about action distribution."""
        if not self.action_history:
            return {}
        
        actions = np.array(self.action_history)
        return {
            'action_mean': np.mean(actions, axis=0).tolist(),
            'action_std': np.std(actions, axis=0).tolist(),
            'action_min': np.min(actions, axis=0).tolist(),
            'action_max': np.max(actions, axis=0).tolist()
        }


class AdEnvironmentMetrics:
    """Utility class for calculating advertising metrics."""
    
    @staticmethod
    def calculate_roi(revenue: float, cost: float) -> float:
        """Calculate Return on Investment."""
        if cost <= 0:
            return 0.0
        return (revenue - cost) / cost
    
    @staticmethod
    def calculate_roas(revenue: float, ad_spend: float) -> float:
        """Calculate Return on Ad Spend."""
        if ad_spend <= 0:
            return 0.0
        return revenue / ad_spend
    
    @staticmethod
    def calculate_ctr(clicks: float, impressions: float) -> float:
        """Calculate Click-Through Rate."""
        if impressions <= 0:
            return 0.0
        return clicks / impressions
    
    @staticmethod
    def calculate_conversion_rate(conversions: float, clicks: float) -> float:
        """Calculate Conversion Rate."""
        if clicks <= 0:
            return 0.0
        return conversions / clicks
    
    @staticmethod
    def calculate_cpa(cost: float, acquisitions: float) -> float:
        """Calculate Cost Per Acquisition."""
        if acquisitions <= 0:
            return float('inf')
        return cost / acquisitions
    
    @staticmethod
    def calculate_quality_score(
        ctr: float,
        relevance_score: float,
        landing_page_score: float
    ) -> float:
        """Calculate Quality Score (simplified)."""
        return (ctr * 0.4 + relevance_score * 0.3 + landing_page_score * 0.3) * 10
    
    @staticmethod
    def calculate_budget_efficiency(
        spent: float,
        budget: float,
        performance_score: float
    ) -> float:
        """Calculate budget utilization efficiency."""
        if budget <= 0:
            return 0.0
        
        utilization = spent / budget
        return performance_score * utilization


class ActionValidator:
    """Utility class for validating and constraining actions."""
    
    @staticmethod
    def validate_budget_allocation(allocations: np.ndarray) -> np.ndarray:
        """Ensure budget allocations sum to 1.0."""
        total = np.sum(allocations)
        if total <= 0:
            return np.ones_like(allocations) / len(allocations)
        return allocations / total
    
    @staticmethod
    def validate_bid_multipliers(multipliers: np.ndarray, min_val: float = 0.1, max_val: float = 5.0) -> np.ndarray:
        """Constrain bid multipliers to reasonable range."""
        return np.clip(multipliers, min_val, max_val)
    
    @staticmethod
    def validate_budget_changes(changes: np.ndarray, current_budgets: np.ndarray, max_change: float = 0.2) -> np.ndarray:
        """Limit budget changes to prevent extreme modifications."""
        max_delta = current_budgets * max_change
        return np.clip(changes, -max_delta, max_delta)