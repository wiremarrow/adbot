"""
Base RL Environment for AdBot advertising optimization
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

from ...utils.logger import get_logger


class BaseAdEnvironment(gym.Env, ABC):
    """
    Base class for all AdBot RL environments
    
    This provides the foundation for advertising optimization environments
    following the Gymnasium interface standard.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        platform: str = "simulation",
        time_horizon: int = 24,  # hours
        step_size: int = 1,  # hours
        seed: Optional[int] = None,
    ):
        """
        Initialize base advertising environment
        
        Args:
            config: Environment configuration
            platform: Platform name (google_ads, facebook, etc.)
            time_horizon: Total time horizon in hours
            step_size: Step size in hours
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.config = config
        self.platform = platform
        self.time_horizon = time_horizon
        self.step_size = step_size
        self.log = get_logger(f"adbot.env.{self.__class__.__name__.lower()}")
        
        # Environment state
        self.current_step = 0
        self.max_steps = time_horizon // step_size
        self.start_time = datetime.now()
        self.current_time = self.start_time
        
        # Performance tracking
        self.episode_return = 0.0
        self.episode_cost = 0.0
        self.episode_conversions = 0
        self.performance_history = []
        
        # Initialize spaces (to be defined by subclasses)
        self.observation_space = None
        self.action_space = None
        
        # Random state
        self.np_random = None
        if seed is not None:
            self.seed(seed)
        
        self.log.info(
            "Environment initialized",
            platform=platform,
            time_horizon=time_horizon,
            max_steps=self.max_steps
        )
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generator"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        pass
    
    @abstractmethod
    def _take_action(self, action: np.ndarray) -> Dict[str, float]:
        """Execute action and return metrics"""
        pass
    
    @abstractmethod
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward from metrics"""
        pass
    
    @abstractmethod
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        pass
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)"""
        return self.current_step >= self.max_steps
    
    def step(self, action: Union[np.ndarray, int, float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self.action_space.contains(action):
            self.log.warning("Invalid action", action=action)
            action = self.action_space.sample()  # Fallback to random action
        
        # Execute action and get metrics
        metrics = self._take_action(np.array(action))
        
        # Calculate reward
        reward = self._calculate_reward(metrics)
        
        # Update episode tracking
        self.episode_return += reward
        self.episode_cost += metrics.get('cost', 0)
        self.episode_conversions += metrics.get('conversions', 0)
        
        # Update time
        self.current_step += 1
        self.current_time += timedelta(hours=self.step_size)
        
        # Store performance
        step_data = {
            'step': self.current_step,
            'time': self.current_time,
            'action': action.copy() if isinstance(action, np.ndarray) else action,
            'reward': reward,
            'metrics': metrics.copy(),
        }
        self.performance_history.append(step_data)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Get new observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'time': self.current_time.isoformat(),
            'episode_return': self.episode_return,
            'episode_cost': self.episode_cost,
            'episode_conversions': self.episode_conversions,
            'metrics': metrics,
        }
        
        if terminated or truncated:
            info['episode'] = self._get_episode_summary()
        
        self.log.debug(
            "Step completed",
            step=self.current_step,
            reward=reward,
            terminated=terminated,
            truncated=truncated
        )
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.start_time = datetime.now()
        self.current_time = self.start_time
        
        # Reset episode tracking
        self.episode_return = 0.0
        self.episode_cost = 0.0
        self.episode_conversions = 0
        self.performance_history = []
        
        # Environment-specific reset
        self._reset_environment(options)
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'step': 0,
            'time': self.current_time.isoformat(),
            'episode_return': 0.0,
        }
        
        self.log.info("Environment reset")
        
        return observation, info
    
    @abstractmethod
    def _reset_environment(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Environment-specific reset logic"""
        pass
    
    def _get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode"""
        return {
            'total_steps': len(self.performance_history),
            'total_return': self.episode_return,
            'total_cost': self.episode_cost,
            'total_conversions': self.episode_conversions,
            'avg_reward': self.episode_return / max(len(self.performance_history), 1),
            'roi': (self.episode_conversions / max(self.episode_cost, 0.01)) if self.episode_cost > 0 else 0,
            'start_time': self.start_time.isoformat(),
            'end_time': self.current_time.isoformat(),
        }
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'current_time': self.current_time.isoformat(),
            'episode_return': self.episode_return,
            'episode_cost': self.episode_cost,
            'episode_conversions': self.episode_conversions,
            'performance_history_length': len(self.performance_history),
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Time: {self.current_time}")
            print(f"Episode Return: {self.episode_return:.4f}")
            print(f"Episode Cost: {self.episode_cost:.2f}")
            print(f"Episode Conversions: {self.episode_conversions}")
            print("-" * 40)
        elif mode == 'rgb_array':
            # Could implement visualization here
            return None
        
        return None
    
    def close(self) -> None:
        """Clean up environment"""
        self.log.info("Environment closed")


class ActionValidator:
    """Utility class for validating and constraining actions"""
    
    @staticmethod
    def clip_action(action: np.ndarray, action_space: spaces.Box) -> np.ndarray:
        """Clip action to valid range"""
        return np.clip(action, action_space.low, action_space.high)
    
    @staticmethod
    def normalize_budget_allocation(budget_allocation: np.ndarray) -> np.ndarray:
        """Normalize budget allocation to sum to 1"""
        total = np.sum(budget_allocation)
        if total > 0:
            return budget_allocation / total
        else:
            # Equal allocation if all zeros
            return np.ones_like(budget_allocation) / len(budget_allocation)
    
    @staticmethod
    def apply_bid_constraints(
        bids: np.ndarray,
        min_bid: float = 0.01,
        max_bid: float = 100.0,
        current_bids: Optional[np.ndarray] = None,
        max_change_pct: float = 0.5,
    ) -> np.ndarray:
        """Apply bid constraints and limits"""
        # Clip to absolute limits
        bids = np.clip(bids, min_bid, max_bid)
        
        # Apply change limits if current bids provided
        if current_bids is not None:
            max_change = current_bids * max_change_pct
            lower_bound = current_bids - max_change
            upper_bound = current_bids + max_change
            
            bids = np.clip(bids, 
                          np.maximum(lower_bound, min_bid),
                          np.minimum(upper_bound, max_bid))
        
        return bids


class RewardShaper:
    """Utility class for reward function components"""
    
    @staticmethod
    def roi_reward(conversions: float, cost: float, target_roi: float = 3.0) -> float:
        """Calculate ROI-based reward"""
        if cost <= 0:
            return 0.0
        
        roi = conversions / cost
        return np.tanh((roi - target_roi) / target_roi)
    
    @staticmethod
    def cost_efficiency_reward(cost: float, budget: float) -> float:
        """Reward for efficient budget usage"""
        if budget <= 0:
            return 0.0
        
        utilization = cost / budget
        # Reward high utilization but penalize overspend
        if utilization <= 1.0:
            return utilization
        else:
            return 1.0 - (utilization - 1.0)  # Penalty for overspend
    
    @staticmethod
    def conversion_rate_reward(
        conversions: float,
        clicks: float,
        target_cvr: float = 0.05
    ) -> float:
        """Reward for good conversion rates"""
        if clicks <= 0:
            return 0.0
        
        cvr = conversions / clicks
        return np.tanh((cvr - target_cvr) / target_cvr)
    
    @staticmethod
    def stability_penalty(
        current_action: np.ndarray,
        previous_action: Optional[np.ndarray] = None,
        penalty_weight: float = 0.1
    ) -> float:
        """Penalty for large action changes (promotes stability)"""
        if previous_action is None:
            return 0.0
        
        change = np.linalg.norm(current_action - previous_action)
        return -penalty_weight * change