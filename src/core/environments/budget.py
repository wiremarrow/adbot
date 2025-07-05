"""
<<<<<<< HEAD
Budget Allocation RL Environment - Placeholder

This is a placeholder for the budget allocation environment.
Currently redirects to SimpleCampaignEnv for testing.
"""

from .campaign import SimpleCampaignEnv

# Placeholder - use SimpleCampaignEnv for now
BudgetAllocationEnv = SimpleCampaignEnv
=======
Budget Allocation RL Environment
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

from .base import BaseAdEnvironment, ActionValidator, RewardShaper


class BudgetAllocationEnv(BaseAdEnvironment):
    """
    RL Environment for dynamic budget allocation
    
    This environment focuses on:
    - Allocating budget across campaigns/ad groups
    - Adjusting budget based on performance
    - Managing daily/weekly budget pacing
    - Optimizing for ROI across different channels
    """
    
    def __init__(
        self,
        entities: List[Dict[str, Any]],  # Campaigns, ad groups, or channels
        total_budget: float,
        budget_period: str = "daily",  # daily, weekly, monthly
        allocation_constraints: Optional[Dict[str, Any]] = None,
        platform: str = "multi_platform",
        time_horizon: int = 24,
        step_size: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize budget allocation environment
        
        Args:
            entities: List of entities to allocate budget to
            total_budget: Total budget for the period
            budget_period: Budget period (daily, weekly, monthly)
            allocation_constraints: Min/max allocation constraints
            platform: Platform identifier
            time_horizon: Time horizon in hours
            step_size: Step size in hours
            seed: Random seed
            config: Additional configuration
        """
        self.entities = entities
        self.total_budget = total_budget
        self.budget_period = budget_period
        self.allocation_constraints = allocation_constraints or {}
        self.n_entities = len(entities)
        
        # Default config
        if config is None:
            config = {}
        
        super().__init__(config, platform, time_horizon, step_size, seed)
        
        # Budget state
        self.entity_states = self._initialize_entity_states()
        self.budget_history = []
        self.remaining_budget = total_budget
        
        # Performance tracking
        self.historical_performance = {entity['id']: [] for entity in entities}
        
        # Define observation and action spaces
        self._setup_spaces()
        
        self.log.info(
            "Budget allocation environment initialized",
            n_entities=self.n_entities,
            total_budget=total_budget,
            budget_period=budget_period
        )
    
    def _initialize_entity_states(self) -> List[Dict[str, Any]]:
        """Initialize entity states"""
        states = []
        for i, entity in enumerate(self.entities):
            # Default equal allocation
            default_allocation = 1.0 / self.n_entities
            
            # Apply constraints if provided
            constraints = self.allocation_constraints.get(entity.get('id', f'entity_{i}'), {})
            min_allocation = constraints.get('min_allocation', 0.0)
            max_allocation = constraints.get('max_allocation', 1.0)
            
            state = {
                'id': entity.get('id', f'entity_{i}'),
                'name': entity.get('name', f'Entity {i}'),
                'type': entity.get('type', 'campaign'),  # campaign, ad_group, channel
                'current_allocation': max(min_allocation, min(default_allocation, max_allocation)),
                'allocated_budget': 0.0,
                'spent_budget': 0.0,
                'remaining_budget': 0.0,
                
                # Performance metrics
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'revenue': 0.0,
                'cost': 0.0,
                
                # Historical performance
                'avg_cpc': entity.get('avg_cpc', 1.0),
                'avg_ctr': entity.get('avg_ctr', 0.02),
                'avg_cvr': entity.get('avg_cvr', 0.05),
                'avg_revenue_per_conversion': entity.get('avg_revenue_per_conversion', 100.0),
                
                # Constraints
                'min_allocation': min_allocation,
                'max_allocation': max_allocation,
                'priority': entity.get('priority', 1.0),
                
                # Seasonality and trends
                'performance_trend': entity.get('performance_trend', 1.0),
                'seasonality_factor': 1.0,
            }
            states.append(state)
        return states
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space per entity:
        # [allocation, spent_ratio, performance_metrics, constraints, trends]
        obs_dim_per_entity = 15  # allocation, spent_ratio, impressions, clicks, conversions, revenue, cost, ctr, cvr, roas, cpc, trend, seasonality, min_constraint, max_constraint
        
        # Global features
        global_features = 6  # total_spent_ratio, remaining_budget_ratio, time_progress, hour_of_day, day_of_week, budget_pace
        
        obs_dim = (obs_dim_per_entity * self.n_entities) + global_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: budget allocations for each entity [0, 1]
        # The allocations will be normalized to sum to 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_entities,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Entity-specific observations
        for state in self.entity_states:
            # Calculate derived metrics
            ctr = state['clicks'] / max(state['impressions'], 1)
            cvr = state['conversions'] / max(state['clicks'], 1)
            cpc = state['cost'] / max(state['clicks'], 1)
            roas = state['revenue'] / max(state['cost'], 1)
            spent_ratio = state['spent_budget'] / max(state['allocated_budget'], 1)
            
            # Update seasonality factor based on time
            state['seasonality_factor'] = self._calculate_seasonality_factor(state)
            
            entity_obs = [
                state['current_allocation'],
                spent_ratio,
                state['impressions'] / 10000.0,  # Normalize
                state['clicks'] / 1000.0,
                state['conversions'] / 100.0,
                state['revenue'] / 10000.0,
                state['cost'] / 1000.0,
                ctr,
                cvr,
                roas / 10.0,  # Normalize ROAS
                cpc / 10.0,   # Normalize CPC
                state['performance_trend'],
                state['seasonality_factor'],
                state['min_allocation'],
                state['max_allocation'],
            ]
            obs.extend(entity_obs)
        
        # Global features
        total_spent = sum(state['spent_budget'] for state in self.entity_states)
        total_allocated = sum(state['allocated_budget'] for state in self.entity_states)
        
        spent_ratio = total_spent / max(total_allocated, 1)
        remaining_budget_ratio = self.remaining_budget / self.total_budget
        time_progress = self.current_step / self.max_steps
        hour_of_day = self.current_time.hour / 24.0
        day_of_week = self.current_time.weekday() / 7.0
        
        # Calculate budget pacing (are we spending too fast/slow?)
        expected_spend_ratio = time_progress
        budget_pace = spent_ratio - expected_spend_ratio
        
        global_obs = [
            spent_ratio,
            remaining_budget_ratio,
            time_progress,
            hour_of_day,
            day_of_week,
            budget_pace,
        ]
        obs.extend(global_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_seasonality_factor(self, entity_state: Dict[str, Any]) -> float:
        """Calculate seasonality factor based on time"""
        hour = self.current_time.hour
        day_of_week = self.current_time.weekday()
        
        # Hour-of-day patterns (higher activity during business hours)
        if entity_state['type'] == 'B2B':
            hour_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 12)  # Peak at noon
        else:
            hour_factor = 0.3 + 0.7 * np.sin(2 * np.pi * (hour - 3) / 18)  # Peak at 9 PM
        
        # Day-of-week patterns
        if day_of_week < 5:  # Weekdays
            day_factor = 1.2 if entity_state['type'] == 'B2B' else 0.9
        else:  # Weekends
            day_factor = 0.8 if entity_state['type'] == 'B2B' else 1.3
        
        return hour_factor * day_factor
    
    def _take_action(self, action: np.ndarray) -> Dict[str, float]:
        """Execute budget allocation action"""
        # Normalize allocations to sum to 1
        allocations = ActionValidator.normalize_budget_allocation(action)
        
        # Apply allocation constraints
        for i, state in enumerate(self.entity_states):
            # Constrain allocation
            constrained_allocation = np.clip(
                allocations[i],
                state['min_allocation'],
                state['max_allocation']
            )
            allocations[i] = constrained_allocation
        
        # Re-normalize after constraints
        allocations = ActionValidator.normalize_budget_allocation(allocations)
        
        # Allocate budget
        step_budget = self.total_budget / self.max_steps  # Budget per step
        step_metrics = {
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'budget_allocated': step_budget,
            'budget_utilization': 0.0,
        }
        
        for i, state in enumerate(self.entity_states):
            # Update allocation
            state['current_allocation'] = allocations[i]
            
            # Allocate step budget
            entity_budget = step_budget * allocations[i]
            state['allocated_budget'] += entity_budget
            state['remaining_budget'] = state['allocated_budget'] - state['spent_budget']
            
            # Simulate performance for this step
            if entity_budget > 0:
                performance = self._simulate_entity_performance(state, entity_budget)
                
                # Update entity state
                state['impressions'] += performance['impressions']
                state['clicks'] += performance['clicks']
                state['conversions'] += performance['conversions']
                state['revenue'] += performance['revenue']
                state['cost'] += performance['cost']
                state['spent_budget'] += performance['cost']
                
                # Update remaining budget
                state['remaining_budget'] = state['allocated_budget'] - state['spent_budget']
                
                # Store performance history
                self.historical_performance[state['id']].append({
                    'step': self.current_step,
                    'time': self.current_time,
                    'allocation': allocations[i],
                    'budget': entity_budget,
                    'performance': performance,
                })
                
                # Aggregate metrics
                step_metrics['total_impressions'] += performance['impressions']
                step_metrics['total_clicks'] += performance['clicks']
                step_metrics['total_conversions'] += performance['conversions']
                step_metrics['total_revenue'] += performance['revenue']
                step_metrics['total_cost'] += performance['cost']
        
        # Update global budget tracking
        self.remaining_budget -= step_metrics['total_cost']
        step_metrics['budget_utilization'] = step_metrics['total_cost'] / step_budget
        
        # Store budget allocation history
        allocation_record = {
            'step': self.current_step,
            'time': self.current_time,
            'allocations': allocations.copy(),
            'step_budget': step_budget,
            'total_spent': step_metrics['total_cost'],
            'remaining_budget': self.remaining_budget,
        }
        self.budget_history.append(allocation_record)
        
        return step_metrics
    
    def _simulate_entity_performance(
        self,
        entity_state: Dict[str, Any],
        allocated_budget: float
    ) -> Dict[str, float]:
        """Simulate entity performance given allocated budget"""
        if allocated_budget <= 0:
            return {
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'revenue': 0.0,
                'cost': 0.0,
            }
        
        # Apply seasonality and trends
        performance_multiplier = (
            entity_state['performance_trend'] *
            entity_state['seasonality_factor']
        )
        
        # Calculate base metrics with noise
        base_cpc = entity_state['avg_cpc'] * (1 + self.np_random.normal(0, 0.1))
        base_ctr = entity_state['avg_ctr'] * performance_multiplier * (1 + self.np_random.normal(0, 0.1))
        base_cvr = entity_state['avg_cvr'] * performance_multiplier * (1 + self.np_random.normal(0, 0.1))
        
        # Ensure positive values
        base_cpc = max(0.01, base_cpc)
        base_ctr = max(0.001, min(0.5, base_ctr))
        base_cvr = max(0.001, min(0.5, base_cvr))
        
        # Calculate performance
        max_clicks = allocated_budget / base_cpc
        impressions = int(max_clicks / base_ctr)
        clicks = int(impressions * base_ctr)
        conversions = int(clicks * base_cvr)
        
        # Actual cost (might be less than budget due to competition)
        actual_cost = min(clicks * base_cpc, allocated_budget)
        
        # Revenue calculation
        revenue_per_conversion = entity_state['avg_revenue_per_conversion']
        revenue = conversions * revenue_per_conversion * (1 + self.np_random.normal(0, 0.1))
        revenue = max(0, revenue)
        
        return {
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'revenue': revenue,
            'cost': actual_cost,
        }
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward based on budget allocation performance"""
        # Primary reward: Total ROAS
        total_roas = metrics['total_revenue'] / max(metrics['total_cost'], 0.01)
        roas_reward = np.tanh((total_roas - 3.0) / 3.0)  # Target ROAS of 3.0
        
        # Budget utilization reward (penalize under/over-utilization)
        utilization = metrics['budget_utilization']
        if 0.8 <= utilization <= 1.2:  # Optimal range
            utilization_reward = 1.0
        elif utilization < 0.8:  # Under-utilization
            utilization_reward = utilization / 0.8
        else:  # Over-utilization
            utilization_reward = max(0, 2.0 - utilization)
        
        # Diversity reward (encourage balanced allocation)
        allocations = np.array([state['current_allocation'] for state in self.entity_states])
        entropy = -np.sum(allocations * np.log(allocations + 1e-8))
        max_entropy = np.log(self.n_entities)
        diversity_reward = entropy / max_entropy
        
        # Performance consistency reward
        entity_roas = []
        for state in self.entity_states:
            if state['cost'] > 0:
                entity_roas.append(state['revenue'] / state['cost'])
        
        if entity_roas:
            roas_std = np.std(entity_roas)
            consistency_reward = 1.0 / (1.0 + roas_std)  # Lower std = higher reward
        else:
            consistency_reward = 0.0
        
        # Constraint satisfaction reward
        constraint_violations = 0
        for state in self.entity_states:
            if state['current_allocation'] < state['min_allocation'] - 1e-6:
                constraint_violations += 1
            if state['current_allocation'] > state['max_allocation'] + 1e-6:
                constraint_violations += 1
        
        constraint_reward = 1.0 - (constraint_violations / (2 * self.n_entities))
        
        # Combine rewards
        total_reward = (
            0.4 * roas_reward +
            0.2 * utilization_reward +
            0.15 * diversity_reward +
            0.15 * consistency_reward +
            0.1 * constraint_reward
        )
        
        return total_reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate early"""
        # Terminate if budget is exhausted
        if self.remaining_budget <= 0:
            self.log.warning("Episode terminated: budget exhausted")
            return True
        
        # Terminate if all entities have zero allocation
        total_allocation = sum(state['current_allocation'] for state in self.entity_states)
        if total_allocation <= 1e-6:
            self.log.warning("Episode terminated: no budget allocated")
            return True
        
        return False
    
    def _reset_environment(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset budget allocation environment"""
        # Reset entity states
        self.entity_states = self._initialize_entity_states()
        self.budget_history = []
        self.remaining_budget = self.total_budget
        self.historical_performance = {entity['id']: [] for entity in self.entities}
        
        # Apply reset options
        if options:
            if 'entities' in options:
                self.entities = options['entities']
                self.n_entities = len(self.entities)
                self.entity_states = self._initialize_entity_states()
                self._setup_spaces()
            
            if 'total_budget' in options:
                self.total_budget = options['total_budget']
                self.remaining_budget = self.total_budget
            
            if 'allocation_constraints' in options:
                self.allocation_constraints = options['allocation_constraints']
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get current allocation summary"""
        total_allocated = sum(state['allocated_budget'] for state in self.entity_states)
        total_spent = sum(state['spent_budget'] for state in self.entity_states)
        total_revenue = sum(state['revenue'] for state in self.entity_states)
        
        entity_summaries = []
        for state in self.entity_states:
            roas = state['revenue'] / max(state['cost'], 0.01)
            ctr = state['clicks'] / max(state['impressions'], 1)
            cvr = state['conversions'] / max(state['clicks'], 1)
            
            summary = {
                'id': state['id'],
                'name': state['name'],
                'type': state['type'],
                'allocation': state['current_allocation'],
                'allocated_budget': state['allocated_budget'],
                'spent_budget': state['spent_budget'],
                'remaining_budget': state['remaining_budget'],
                'revenue': state['revenue'],
                'roas': roas,
                'conversions': state['conversions'],
                'ctr': ctr,
                'cvr': cvr,
            }
            entity_summaries.append(summary)
        
        return {
            'total_budget': self.total_budget,
            'total_allocated': total_allocated,
            'total_spent': total_spent,
            'total_revenue': total_revenue,
            'remaining_budget': self.remaining_budget,
            'overall_roas': total_revenue / max(total_spent, 0.01),
            'budget_utilization': total_spent / total_allocated if total_allocated > 0 else 0,
            'entities': entity_summaries,
        }
>>>>>>> origin/main
