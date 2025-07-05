"""
Campaign Optimization RL Environment
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

from .base import BaseAdEnvironment, ActionValidator, RewardShaper


class CampaignOptimizationEnv(BaseAdEnvironment):
    """
    RL Environment for optimizing campaign-level parameters
    
    This environment focuses on high-level campaign decisions like:
    - Budget allocation across campaigns
    - Campaign status (active/paused)
    - Target audience adjustments
    - Bid strategy selection
    """
    
    def __init__(
        self,
        campaigns: List[Dict[str, Any]],
        total_budget: float,
        platform: str = "google_ads",
        time_horizon: int = 24,
        step_size: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize campaign optimization environment
        
        Args:
            campaigns: List of campaign configurations
            total_budget: Total budget available per time step
            platform: Advertising platform
            time_horizon: Total time horizon in hours
            step_size: Step size in hours
            seed: Random seed
            config: Additional configuration
        """
        self.campaigns = campaigns
        self.total_budget = total_budget
        self.n_campaigns = len(campaigns)
        
        # Default config
        if config is None:
            config = {}
        
        super().__init__(config, platform, time_horizon, step_size, seed)
        
        # Campaign state
        self.campaign_states = self._initialize_campaign_states()
        self.previous_action = None
        
        # Performance simulation parameters
        self.base_ctr = config.get('base_ctr', 0.02)
        self.base_cvr = config.get('base_cvr', 0.05)
        self.noise_std = config.get('noise_std', 0.1)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        self.log.info(
            "Campaign environment initialized",
            n_campaigns=self.n_campaigns,
            total_budget=total_budget
        )
    
    def _initialize_campaign_states(self) -> List[Dict[str, Any]]:
        """Initialize campaign states"""
        states = []
        for i, campaign in enumerate(self.campaigns):
            state = {
                'id': campaign.get('id', f'campaign_{i}'),
                'name': campaign.get('name', f'Campaign {i}'),
                'status': 'active',
                'budget_allocation': 1.0 / self.n_campaigns,  # Equal allocation initially
                'current_bid': campaign.get('initial_bid', 1.0),
                'target_cpa': campaign.get('target_cpa', 50.0),
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'cost': 0.0,
                'quality_score': campaign.get('quality_score', 7.0),
                'competition_level': campaign.get('competition_level', 0.5),
            }
            states.append(state)
        return states
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space: [campaign_metrics, time_features, budget_features]
        # Per campaign: [budget_allocation, bid, impressions, clicks, conversions, cost, ctr, cvr, cpa, quality_score]
        obs_dim_per_campaign = 10
        time_features = 4  # [hour_of_day, day_of_week, time_remaining, step_progress]
        budget_features = 3  # [remaining_budget, budget_utilization, total_spend]
        
        obs_dim = (obs_dim_per_campaign * self.n_campaigns) + time_features + budget_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: [budget_allocations (n_campaigns), bid_multipliers (n_campaigns), status_changes (n_campaigns)]
        # Budget allocations: [0, 1] and sum to 1
        # Bid multipliers: [0.5, 2.0] (50% to 200% of current bid)
        # Status: [0, 1] (0=pause, 1=active)
        action_dim = 3 * self.n_campaigns
        
        action_low = np.concatenate([
            np.zeros(self.n_campaigns),      # budget allocations
            np.full(self.n_campaigns, 0.5),  # bid multipliers
            np.zeros(self.n_campaigns),      # status (0=pause, 1=active)
        ])
        
        action_high = np.concatenate([
            np.ones(self.n_campaigns),       # budget allocations
            np.full(self.n_campaigns, 2.0),  # bid multipliers
            np.ones(self.n_campaigns),       # status
        ])
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Campaign metrics
        for state in self.campaign_states:
            # Calculate derived metrics
            ctr = state['clicks'] / max(state['impressions'], 1)
            cvr = state['conversions'] / max(state['clicks'], 1)
            cpa = state['cost'] / max(state['conversions'], 1)
            
            campaign_obs = [
                state['budget_allocation'],
                state['current_bid'] / 10.0,  # Normalize bid
                state['impressions'] / 10000.0,  # Normalize impressions
                state['clicks'] / 1000.0,  # Normalize clicks
                state['conversions'] / 100.0,  # Normalize conversions
                state['cost'] / 1000.0,  # Normalize cost
                ctr,
                cvr,
                cpa / 100.0,  # Normalize CPA
                state['quality_score'] / 10.0,  # Normalize quality score
            ]
            obs.extend(campaign_obs)
        
        # Time features
        hour_of_day = self.current_time.hour / 24.0
        day_of_week = self.current_time.weekday() / 7.0
        time_remaining = (self.max_steps - self.current_step) / self.max_steps
        step_progress = self.current_step / self.max_steps
        
        time_obs = [hour_of_day, day_of_week, time_remaining, step_progress]
        obs.extend(time_obs)
        
        # Budget features
        total_spend = sum(state['cost'] for state in self.campaign_states)
        budget_utilization = total_spend / max(self.total_budget * self.current_step, 1)
        remaining_budget = max(0, (self.total_budget * self.max_steps) - total_spend)
        
        budget_obs = [
            remaining_budget / (self.total_budget * self.max_steps),
            budget_utilization,
            total_spend / (self.total_budget * self.max_steps),
        ]
        obs.extend(budget_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _take_action(self, action: np.ndarray) -> Dict[str, float]:
        """Execute action and simulate campaign performance"""
        # Parse action
        budget_allocations = action[:self.n_campaigns]
        bid_multipliers = action[self.n_campaigns:2*self.n_campaigns]
        status_changes = action[2*self.n_campaigns:]
        
        # Normalize budget allocations
        budget_allocations = ActionValidator.normalize_budget_allocation(budget_allocations)
        
        # Apply actions to campaigns
        step_metrics = {
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_cost': 0.0,
        }
        
        for i, state in enumerate(self.campaign_states):
            # Update budget allocation
            state['budget_allocation'] = budget_allocations[i]
            
            # Update bid
            new_bid = state['current_bid'] * bid_multipliers[i]
            state['current_bid'] = np.clip(new_bid, 0.1, 10.0)
            
            # Update status
            state['status'] = 'active' if status_changes[i] > 0.5 else 'paused'
            
            # Simulate performance if campaign is active
            if state['status'] == 'active':
                step_budget = self.total_budget * state['budget_allocation']
                metrics = self._simulate_campaign_performance(state, step_budget)
                
                # Update cumulative metrics
                state['impressions'] += metrics['impressions']
                state['clicks'] += metrics['clicks']
                state['conversions'] += metrics['conversions']
                state['cost'] += metrics['cost']
                
                # Aggregate step metrics
                step_metrics['total_impressions'] += metrics['impressions']
                step_metrics['total_clicks'] += metrics['clicks']
                step_metrics['total_conversions'] += metrics['conversions']
                step_metrics['total_cost'] += metrics['cost']
        
        # Calculate aggregate metrics
        step_metrics['total_ctr'] = (
            step_metrics['total_clicks'] / max(step_metrics['total_impressions'], 1)
        )
        step_metrics['total_cvr'] = (
            step_metrics['total_conversions'] / max(step_metrics['total_clicks'], 1)
        )
        step_metrics['total_cpa'] = (
            step_metrics['total_cost'] / max(step_metrics['total_conversions'], 1)
        )
        step_metrics['budget_utilization'] = step_metrics['total_cost'] / self.total_budget
        
        return step_metrics
    
    def _simulate_campaign_performance(
        self,
        campaign_state: Dict[str, Any],
        step_budget: float
    ) -> Dict[str, float]:
        """Simulate campaign performance for one step"""
        if step_budget <= 0:
            return {'impressions': 0, 'clicks': 0, 'conversions': 0, 'cost': 0.0}
        
        # Base performance factors
        quality_factor = campaign_state['quality_score'] / 10.0
        competition_factor = 1.0 - campaign_state['competition_level']
        bid_factor = min(campaign_state['current_bid'] / 2.0, 2.0)  # Bid effectiveness
        
        # Time-of-day factor (simulate daily patterns)
        hour = self.current_time.hour
        time_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 2 PM
        
        # Calculate impressions based on budget and bid
        base_impressions = step_budget * bid_factor * quality_factor * time_factor * 100
        impressions = max(0, int(base_impressions * (1 + self.np_random.normal(0, self.noise_std))))
        
        # Calculate clicks
        ctr = self.base_ctr * quality_factor * bid_factor * (1 + self.np_random.normal(0, self.noise_std * 0.5))
        ctr = np.clip(ctr, 0, 0.5)
        clicks = int(impressions * ctr)
        
        # Calculate conversions
        cvr = self.base_cvr * quality_factor * (1 + self.np_random.normal(0, self.noise_std * 0.5))
        cvr = np.clip(cvr, 0, 0.5)
        conversions = int(clicks * cvr)
        
        # Calculate cost (based on bid and competition)
        avg_cpc = campaign_state['current_bid'] * competition_factor * (1 + self.np_random.normal(0, self.noise_std * 0.3))
        avg_cpc = max(0.05, avg_cpc)
        cost = min(clicks * avg_cpc, step_budget)  # Don't exceed budget
        
        return {
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'cost': cost,
        }
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward based on campaign performance"""
        # Primary reward: ROI
        roi_reward = RewardShaper.roi_reward(
            metrics['total_conversions'],
            metrics['total_cost'],
            target_roi=3.0
        )
        
        # Budget efficiency reward
        efficiency_reward = RewardShaper.cost_efficiency_reward(
            metrics['total_cost'],
            self.total_budget
        )
        
        # Conversion rate reward
        cvr_reward = RewardShaper.conversion_rate_reward(
            metrics['total_conversions'],
            metrics['total_clicks'],
            target_cvr=0.05
        )
        
        # Stability penalty (if we have previous action)
        stability_penalty = 0.0
        if self.previous_action is not None:
            current_action = np.concatenate([
                [state['budget_allocation'] for state in self.campaign_states],
                [state['current_bid'] for state in self.campaign_states],
            ])
            stability_penalty = RewardShaper.stability_penalty(
                current_action,
                self.previous_action,
                penalty_weight=0.05
            )
        
        # Combine rewards
        total_reward = (
            0.5 * roi_reward +
            0.3 * efficiency_reward +
            0.2 * cvr_reward +
            stability_penalty
        )
        
        # Store current action for next step
        self.previous_action = np.concatenate([
            [state['budget_allocation'] for state in self.campaign_states],
            [state['current_bid'] for state in self.campaign_states],
        ])
        
        return total_reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate early"""
        # Terminate if total spend exceeds budget significantly
        total_spend = sum(state['cost'] for state in self.campaign_states)
        budget_limit = self.total_budget * self.max_steps * 1.5  # 50% overspend tolerance
        
        if total_spend > budget_limit:
            self.log.warning("Episode terminated due to budget overspend", 
                           total_spend=total_spend, budget_limit=budget_limit)
            return True
        
        # Terminate if no active campaigns
        active_campaigns = sum(1 for state in self.campaign_states if state['status'] == 'active')
        if active_campaigns == 0:
            self.log.warning("Episode terminated due to no active campaigns")
            return True
        
        return False
    
    def _reset_environment(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset campaign-specific state"""
        # Reset campaign states
        self.campaign_states = self._initialize_campaign_states()
        self.previous_action = None
        
        # Apply any reset options
        if options:
            if 'campaigns' in options:
                self.campaigns = options['campaigns']
                self.n_campaigns = len(self.campaigns)
                self.campaign_states = self._initialize_campaign_states()
                self._setup_spaces()  # Reconfigure spaces if campaign count changed
            
            if 'total_budget' in options:
                self.total_budget = options['total_budget']
    
    def get_campaign_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all campaigns"""
        summaries = []
        for state in self.campaign_states:
            ctr = state['clicks'] / max(state['impressions'], 1)
            cvr = state['conversions'] / max(state['clicks'], 1)
            cpa = state['cost'] / max(state['conversions'], 1)
            roi = state['conversions'] / max(state['cost'], 0.01)
            
            summary = {
                'id': state['id'],
                'name': state['name'],
                'status': state['status'],
                'budget_allocation': state['budget_allocation'],
                'current_bid': state['current_bid'],
                'impressions': state['impressions'],
                'clicks': state['clicks'],
                'conversions': state['conversions'],
                'cost': state['cost'],
                'ctr': ctr,
                'cvr': cvr,
                'cpa': cpa,
                'roi': roi,
                'quality_score': state['quality_score'],
            }
            summaries.append(summary)
        
        return summaries