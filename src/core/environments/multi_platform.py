"""
Multi-Platform RL Environment Wrapper
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

from .base import BaseAdEnvironment
from .campaign import CampaignOptimizationEnv
from .budget import BudgetAllocationEnv
from .bidding import BidOptimizationEnv


class MultiPlatformEnv(BaseAdEnvironment):
    """
    Multi-platform wrapper environment for coordinating optimization across platforms
    
    This environment manages:
    - Cross-platform budget allocation
    - Platform-specific campaign optimization
    - Unified performance tracking
    - Platform arbitrage opportunities
    """
    
    def __init__(
        self,
        platform_configs: Dict[str, Dict[str, Any]],
        total_budget: float,
        optimization_level: str = "campaign",  # campaign, budget, bidding
        time_horizon: int = 24,
        step_size: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize multi-platform environment
        
        Args:
            platform_configs: Configuration for each platform
            total_budget: Total budget across all platforms
            optimization_level: Level of optimization (campaign, budget, bidding)
            time_horizon: Time horizon in hours
            step_size: Step size in hours
            seed: Random seed
            config: Additional configuration
        """
        self.platform_configs = platform_configs
        self.total_budget = total_budget
        self.optimization_level = optimization_level
        self.platforms = list(platform_configs.keys())
        self.n_platforms = len(self.platforms)
        
        # Default config
        if config is None:
            config = {}
        
        super().__init__(config, "multi_platform", time_horizon, step_size, seed)
        
        # Initialize platform environments
        self.platform_envs = self._initialize_platform_environments()
        
        # Cross-platform state
        self.platform_allocations = {platform: 1.0 / self.n_platforms for platform in self.platforms}
        self.platform_performance = {platform: {} for platform in self.platforms}
        self.cross_platform_history = []
        
        # Define observation and action spaces
        self._setup_spaces()
        
        self.log.info(
            "Multi-platform environment initialized",
            platforms=self.platforms,
            optimization_level=optimization_level,
            total_budget=total_budget
        )
    
    def _initialize_platform_environments(self) -> Dict[str, BaseAdEnvironment]:
        """Initialize individual platform environments"""
        envs = {}
        
        for platform, config in self.platform_configs.items():
            platform_budget = self.total_budget / self.n_platforms  # Initial equal allocation
            
            if self.optimization_level == "campaign":
                # Campaign optimization for each platform
                campaigns = config.get('campaigns', [])
                env = CampaignOptimizationEnv(
                    campaigns=campaigns,
                    total_budget=platform_budget,
                    platform=platform,
                    time_horizon=self.time_horizon,
                    step_size=self.step_size,
                    seed=self.np_random.integers(0, 2**31) if self.np_random else None,
                    config=config.get('env_config', {})
                )
            
            elif self.optimization_level == "budget":
                # Budget allocation for each platform
                entities = config.get('entities', [])
                env = BudgetAllocationEnv(
                    entities=entities,
                    total_budget=platform_budget,
                    platform=platform,
                    time_horizon=self.time_horizon,
                    step_size=self.step_size,
                    seed=self.np_random.integers(0, 2**31) if self.np_random else None,
                    config=config.get('env_config', {})
                )
            
            elif self.optimization_level == "bidding":
                # Bid optimization for each platform
                keywords = config.get('keywords', [])
                budget_per_hour = platform_budget / self.time_horizon
                env = BidOptimizationEnv(
                    keywords=keywords,
                    budget_per_hour=budget_per_hour,
                    platform=platform,
                    time_horizon=self.time_horizon,
                    step_size=self.step_size,
                    seed=self.np_random.integers(0, 2**31) if self.np_random else None,
                    config=config.get('env_config', {})
                )
            
            else:
                raise ValueError(f"Unknown optimization level: {self.optimization_level}")
            
            envs[platform] = env
        
        return envs
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Get dimensions from platform environments
        platform_obs_dims = {}
        platform_action_dims = {}
        
        for platform, env in self.platform_envs.items():
            platform_obs_dims[platform] = env.observation_space.shape[0]
            platform_action_dims[platform] = env.action_space.shape[0]
        
        # Multi-platform observation space:
        # [platform_observations, cross_platform_features, allocation_features]
        total_platform_obs = sum(platform_obs_dims.values())
        cross_platform_features = 10  # relative_performance, correlations, arbitrage_opportunities, etc.
        allocation_features = self.n_platforms * 3  # [allocation, performance, efficiency] per platform
        
        obs_dim = total_platform_obs + cross_platform_features + allocation_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Multi-platform action space:
        # [platform_allocations, platform_actions]
        total_platform_actions = sum(platform_action_dims.values())
        allocation_actions = self.n_platforms  # Budget allocation across platforms
        
        action_dim = allocation_actions + total_platform_actions
        
        # Action bounds
        action_low = np.concatenate([
            np.zeros(allocation_actions),  # Platform allocations [0, 1]
            np.concatenate([env.action_space.low for env in self.platform_envs.values()])
        ])
        
        action_high = np.concatenate([
            np.ones(allocation_actions),   # Platform allocations [0, 1]
            np.concatenate([env.action_space.high for env in self.platform_envs.values()])
        ])
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Store dimensions for action parsing
        self.platform_action_dims = platform_action_dims
        self.allocation_dim = allocation_actions
    
    def _get_observation(self) -> np.ndarray:
        """Get current multi-platform observation"""
        obs = []
        
        # Get observations from each platform environment
        platform_obs = {}
        for platform, env in self.platform_envs.items():
            platform_observation = env._get_observation()
            platform_obs[platform] = platform_observation
            obs.extend(platform_observation)
        
        # Cross-platform features
        cross_platform_features = self._calculate_cross_platform_features()
        obs.extend(cross_platform_features)
        
        # Allocation features
        allocation_features = self._calculate_allocation_features()
        obs.extend(allocation_features)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_cross_platform_features(self) -> List[float]:
        """Calculate cross-platform comparative features"""
        features = []
        
        # Relative performance metrics
        platform_roas = {}
        platform_costs = {}
        platform_conversions = {}
        
        for platform in self.platforms:
            perf = self.platform_performance.get(platform, {})
            platform_roas[platform] = perf.get('roas', 1.0)
            platform_costs[platform] = perf.get('total_cost', 0.0)
            platform_conversions[platform] = perf.get('total_conversions', 0.0)
        
        # ROAS variance (opportunity for reallocation)
        roas_values = list(platform_roas.values())
        roas_mean = np.mean(roas_values) if roas_values else 1.0
        roas_std = np.std(roas_values) if len(roas_values) > 1 else 0.0
        roas_cv = roas_std / max(roas_mean, 0.01)  # Coefficient of variation
        
        features.extend([
            roas_mean / 10.0,  # Normalize
            roas_std / 10.0,
            roas_cv,
        ])
        
        # Best and worst performing platforms
        if roas_values:
            best_roas = max(roas_values)
            worst_roas = min(roas_values)
            roas_spread = (best_roas - worst_roas) / max(roas_mean, 0.01)
        else:
            best_roas = worst_roas = roas_spread = 0.0
        
        features.extend([
            best_roas / 10.0,
            worst_roas / 10.0,
            roas_spread,
        ])
        
        # Budget utilization variance
        total_budget = sum(platform_costs.values())
        if total_budget > 0:
            budget_distribution_entropy = -sum(
                (cost / total_budget) * np.log(max(cost / total_budget, 1e-8))
                for cost in platform_costs.values()
                if cost > 0
            )
            max_entropy = np.log(self.n_platforms)
            normalized_entropy = budget_distribution_entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0
        
        features.append(normalized_entropy)
        
        # Arbitrage opportunity score
        arbitrage_score = self._calculate_arbitrage_opportunity()
        features.append(arbitrage_score)
        
        # Time-based correlation (simplified)
        time_correlation = self._calculate_time_correlation()
        features.append(time_correlation)
        
        # Market saturation indicator
        saturation_score = self._calculate_market_saturation()
        features.append(saturation_score)
        
        return features
    
    def _calculate_allocation_features(self) -> List[float]:
        """Calculate platform allocation and efficiency features"""
        features = []
        
        for platform in self.platforms:
            allocation = self.platform_allocations[platform]
            perf = self.platform_performance.get(platform, {})
            
            # Current performance
            roas = perf.get('roas', 1.0)
            efficiency = perf.get('efficiency', 0.5)  # Cost per conversion relative to target
            
            platform_features = [
                allocation,
                roas / 10.0,  # Normalize
                efficiency,
            ]
            features.extend(platform_features)
        
        return features
    
    def _calculate_arbitrage_opportunity(self) -> float:
        """Calculate arbitrage opportunity score across platforms"""
        # Simplified: difference in ROAS suggests reallocation opportunity
        roas_values = []
        for platform in self.platforms:
            perf = self.platform_performance.get(platform, {})
            roas_values.append(perf.get('roas', 1.0))
        
        if len(roas_values) < 2:
            return 0.0
        
        roas_std = np.std(roas_values)
        roas_mean = np.mean(roas_values)
        
        # Higher variance = higher arbitrage opportunity
        return min(1.0, roas_std / max(roas_mean, 0.01))
    
    def _calculate_time_correlation(self) -> float:
        """Calculate time-based performance correlation"""
        # Simplified: how correlated are platform performances over time
        if len(self.cross_platform_history) < 5:
            return 0.0
        
        # Get recent performance trends
        recent_history = self.cross_platform_history[-5:]
        platform_trends = {platform: [] for platform in self.platforms}
        
        for record in recent_history:
            for platform in self.platforms:
                platform_perf = record['platform_performance'].get(platform, {})
                roas = platform_perf.get('roas', 1.0)
                platform_trends[platform].append(roas)
        
        # Calculate correlation between platforms (simplified)
        correlations = []
        platforms = list(platform_trends.keys())
        
        for i in range(len(platforms)):
            for j in range(i + 1, len(platforms)):
                trend1 = platform_trends[platforms[i]]
                trend2 = platform_trends[platforms[j]]
                
                if len(trend1) > 1 and len(trend2) > 1:
                    corr = np.corrcoef(trend1, trend2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_market_saturation(self) -> float:
        """Calculate market saturation indicator"""
        # Simplified: based on impression share and competition
        saturation_scores = []
        
        for platform, env in self.platform_envs.items():
            if hasattr(env, 'get_state_info'):
                state_info = env.get_state_info()
                # Approximate saturation based on performance trends
                # This would be more sophisticated in production
                saturation_scores.append(0.5)  # Placeholder
        
        return np.mean(saturation_scores) if saturation_scores else 0.5
    
    def _take_action(self, action: np.ndarray) -> Dict[str, float]:
        """Execute multi-platform action"""
        # Parse action
        allocation_actions = action[:self.allocation_dim]
        platform_actions_start = self.allocation_dim
        
        # Normalize platform allocations
        normalized_allocations = allocation_actions / max(np.sum(allocation_actions), 1e-8)
        
        # Update platform allocations
        for i, platform in enumerate(self.platforms):
            self.platform_allocations[platform] = normalized_allocations[i]
        
        # Update platform budgets based on new allocations
        for platform, env in self.platform_envs.items():
            new_budget = self.total_budget * self.platform_allocations[platform]
            
            # Update environment budget (method depends on environment type)
            if hasattr(env, 'total_budget'):
                env.total_budget = new_budget
            elif hasattr(env, 'budget_per_hour'):
                env.budget_per_hour = new_budget / self.time_horizon
        
        # Execute platform-specific actions
        platform_metrics = {}
        platform_actions_end = platform_actions_start
        
        for platform, env in self.platform_envs.items():
            action_dim = self.platform_action_dims[platform]
            platform_action = action[platform_actions_start:platform_actions_start + action_dim]
            platform_actions_start += action_dim
            
            # Execute action on platform environment
            metrics = env._take_action(platform_action)
            platform_metrics[platform] = metrics
            
            # Update platform performance tracking
            self.platform_performance[platform] = metrics
        
        # Calculate aggregated metrics
        aggregated_metrics = self._aggregate_platform_metrics(platform_metrics)
        
        # Store cross-platform history
        cross_platform_record = {
            'step': self.current_step,
            'time': self.current_time,
            'platform_allocations': self.platform_allocations.copy(),
            'platform_performance': platform_metrics.copy(),
            'aggregated_metrics': aggregated_metrics.copy(),
        }
        self.cross_platform_history.append(cross_platform_record)
        
        return aggregated_metrics
    
    def _aggregate_platform_metrics(self, platform_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across platforms"""
        aggregated = {
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_cost': 0.0,
            'total_revenue': 0.0,
        }
        
        # Sum metrics across platforms
        for platform, metrics in platform_metrics.items():
            for key in aggregated.keys():
                metric_key = key.replace('total_', '')
                if metric_key in metrics:
                    aggregated[key] += metrics[metric_key]
        
        # Calculate derived metrics
        aggregated['overall_ctr'] = (
            aggregated['total_clicks'] / max(aggregated['total_impressions'], 1)
        )
        aggregated['overall_cvr'] = (
            aggregated['total_conversions'] / max(aggregated['total_clicks'], 1)
        )
        aggregated['overall_cpc'] = (
            aggregated['total_cost'] / max(aggregated['total_clicks'], 1)
        )
        aggregated['overall_cpa'] = (
            aggregated['total_cost'] / max(aggregated['total_conversions'], 1)
        )
        aggregated['overall_roas'] = (
            aggregated['total_revenue'] / max(aggregated['total_cost'], 1)
        )
        
        # Platform diversification metrics
        platform_costs = [metrics.get('total_cost', 0) for metrics in platform_metrics.values()]
        total_cost = sum(platform_costs)
        
        if total_cost > 0:
            # Herfindahl index for budget concentration
            cost_shares = [cost / total_cost for cost in platform_costs]
            herfindahl_index = sum(share ** 2 for share in cost_shares)
            aggregated['budget_concentration'] = herfindahl_index
            
            # Effective number of platforms
            aggregated['effective_platforms'] = 1 / herfindahl_index if herfindahl_index > 0 else 0
        else:
            aggregated['budget_concentration'] = 1.0
            aggregated['effective_platforms'] = 1.0
        
        return aggregated
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate multi-platform reward"""
        # Primary reward: Overall ROAS
        roas_reward = np.tanh((metrics['overall_roas'] - 3.0) / 3.0)
        
        # Diversification reward (encourage platform diversification)
        diversification_reward = 1.0 - metrics['budget_concentration']
        
        # Efficiency reward (reward platforms that perform well together)
        platform_efficiency = []
        for platform, perf in self.platform_performance.items():
            allocation = self.platform_allocations[platform]
            roas = perf.get('roas', 1.0)
            # Efficiency = ROAS weighted by allocation
            efficiency = roas * allocation
            platform_efficiency.append(efficiency)
        
        efficiency_reward = np.mean(platform_efficiency) / 10.0  # Normalize
        
        # Arbitrage reward (reward capitalizing on platform differences)
        arbitrage_opportunity = self._calculate_arbitrage_opportunity()
        arbitrage_reward = 1.0 - arbitrage_opportunity  # Reward when differences are minimized
        
        # Volume reward
        volume_reward = np.tanh(metrics['total_conversions'] / 100.0)
        
        # Combine rewards
        total_reward = (
            0.4 * roas_reward +
            0.2 * diversification_reward +
            0.2 * efficiency_reward +
            0.1 * arbitrage_reward +
            0.1 * volume_reward
        )
        
        return total_reward
    
    def _is_terminated(self) -> bool:
        """Check if any platform environment should terminate"""
        for env in self.platform_envs.values():
            if env._is_terminated():
                return True
        return False
    
    def _reset_environment(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset multi-platform environment"""
        # Reset platform allocations
        self.platform_allocations = {platform: 1.0 / self.n_platforms for platform in self.platforms}
        self.platform_performance = {platform: {} for platform in self.platforms}
        self.cross_platform_history = []
        
        # Reset platform environments
        for platform, env in self.platform_envs.items():
            platform_options = None
            if options and 'platform_options' in options:
                platform_options = options['platform_options'].get(platform)
            
            env.reset(options=platform_options)
        
        # Apply reset options
        if options:
            if 'platform_configs' in options:
                self.platform_configs = options['platform_configs']
                self.platform_envs = self._initialize_platform_environments()
                self._setup_spaces()
            
            if 'total_budget' in options:
                self.total_budget = options['total_budget']
    
    def get_multi_platform_summary(self) -> Dict[str, Any]:
        """Get comprehensive multi-platform summary"""
        # Get summaries from each platform
        platform_summaries = {}
        for platform, env in self.platform_envs.items():
            if hasattr(env, 'get_campaign_summary'):
                platform_summaries[platform] = env.get_campaign_summary()
            elif hasattr(env, 'get_allocation_summary'):
                platform_summaries[platform] = env.get_allocation_summary()
            elif hasattr(env, 'get_bidding_summary'):
                platform_summaries[platform] = env.get_bidding_summary()
            else:
                platform_summaries[platform] = env.get_state_info()
        
        # Calculate cross-platform metrics
        total_cost = sum(
            perf.get('total_cost', 0) for perf in self.platform_performance.values()
        )
        total_revenue = sum(
            perf.get('total_revenue', 0) for perf in self.platform_performance.values()
        )
        total_conversions = sum(
            perf.get('total_conversions', 0) for perf in self.platform_performance.values()
        )
        
        return {
            'optimization_level': self.optimization_level,
            'total_budget': self.total_budget,
            'platform_allocations': self.platform_allocations,
            'total_spent': total_cost,
            'total_revenue': total_revenue,
            'total_conversions': total_conversions,
            'overall_roas': total_revenue / max(total_cost, 1),
            'overall_cpa': total_cost / max(total_conversions, 1),
            'budget_concentration': self._calculate_budget_concentration(),
            'arbitrage_opportunity': self._calculate_arbitrage_opportunity(),
            'platform_summaries': platform_summaries,
        }
    
    def _calculate_budget_concentration(self) -> float:
        """Calculate current budget concentration (Herfindahl index)"""
        allocations = list(self.platform_allocations.values())
        return sum(allocation ** 2 for allocation in allocations)
    
    def step(self, action: Union[np.ndarray, int, float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Override step to coordinate platform environments"""
        # Execute multi-platform step
        return super().step(action)
    
    def close(self) -> None:
        """Close all platform environments"""
        for env in self.platform_envs.values():
            env.close()
        super().close()