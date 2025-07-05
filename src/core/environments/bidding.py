"""
Bid Optimization RL Environment
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

from .base import BaseAdEnvironment, ActionValidator, RewardShaper


class BidOptimizationEnv(BaseAdEnvironment):
    """
    RL Environment for keyword/placement bid optimization
    
    This environment focuses on:
    - Real-time bid adjustments for keywords/placements
    - CPC/CPM bid optimization based on performance
    - Auction dynamics and competition modeling
    - Quality Score and Ad Rank optimization
    """
    
    def __init__(
        self,
        keywords: List[Dict[str, Any]],
        budget_per_hour: float,
        bid_strategy: str = "target_cpa",  # target_cpa, target_roas, maximize_clicks
        platform: str = "google_ads",
        time_horizon: int = 24,
        step_size: int = 1,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize bid optimization environment
        
        Args:
            keywords: List of keyword/placement configurations
            budget_per_hour: Available budget per hour
            bid_strategy: Bidding strategy type
            platform: Advertising platform
            time_horizon: Time horizon in hours
            step_size: Step size in hours
            seed: Random seed
            config: Additional configuration
        """
        self.keywords = keywords
        self.budget_per_hour = budget_per_hour
        self.bid_strategy = bid_strategy
        self.n_keywords = len(keywords)
        
        # Default config
        if config is None:
            config = {}
        
        super().__init__(config, platform, time_horizon, step_size, seed)
        
        # Keyword state
        self.keyword_states = self._initialize_keyword_states()
        self.auction_history = []
        self.bid_history = []
        
        # Market dynamics
        self.market_volatility = config.get('market_volatility', 0.1)
        self.competition_intensity = config.get('competition_intensity', 0.5)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        self.log.info(
            "Bid optimization environment initialized",
            n_keywords=self.n_keywords,
            budget_per_hour=budget_per_hour,
            bid_strategy=bid_strategy
        )
    
    def _initialize_keyword_states(self) -> List[Dict[str, Any]]:
        """Initialize keyword states"""
        states = []
        for i, keyword in enumerate(self.keywords):
            state = {
                'id': keyword.get('id', f'keyword_{i}'),
                'text': keyword.get('text', f'keyword_{i}'),
                'match_type': keyword.get('match_type', 'exact'),
                'current_bid': keyword.get('initial_bid', 1.0),
                'max_bid': keyword.get('max_bid', 10.0),
                'min_bid': keyword.get('min_bid', 0.1),
                
                # Performance metrics
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'cost': 0.0,
                'revenue': 0.0,
                
                # Quality and relevance
                'quality_score': keyword.get('quality_score', 7.0),
                'expected_ctr': keyword.get('expected_ctr', 0.02),
                'ad_relevance': keyword.get('ad_relevance', 'average'),
                'landing_page_experience': keyword.get('landing_page_experience', 'average'),
                
                # Historical performance
                'avg_cpc': keyword.get('avg_cpc', 1.0),
                'avg_position': keyword.get('avg_position', 3.0),
                'conversion_rate': keyword.get('conversion_rate', 0.05),
                'avg_revenue_per_conversion': keyword.get('avg_revenue_per_conversion', 100.0),
                
                # Competition metrics
                'search_volume': keyword.get('search_volume', 1000),
                'competition_level': keyword.get('competition_level', 0.5),
                'top_of_page_bid_low': keyword.get('top_of_page_bid_low', 0.5),
                'top_of_page_bid_high': keyword.get('top_of_page_bid_high', 2.0),
                
                # Auction dynamics
                'impression_share': 0.0,
                'lost_impression_share_rank': 0.0,
                'lost_impression_share_budget': 0.0,
                'avg_ad_rank': 0.0,
                
                # Temporal factors
                'time_performance_factor': 1.0,
                'day_performance_factor': 1.0,
            }
            states.append(state)
        return states
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space per keyword:
        # [bid, performance_metrics, quality_metrics, competition_metrics, auction_metrics, temporal_factors]
        obs_dim_per_keyword = 20
        
        # Global features
        global_features = 8  # budget_utilization, hour_of_day, day_of_week, market_volatility, etc.
        
        obs_dim = (obs_dim_per_keyword * self.n_keywords) + global_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: bid multipliers for each keyword
        # Multipliers range from 0.5 to 2.0 (50% to 200% of current bid)
        self.action_space = spaces.Box(
            low=0.5,
            high=2.0,
            shape=(self.n_keywords,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Keyword-specific observations
        for state in self.keyword_states:
            # Calculate derived metrics
            ctr = state['clicks'] / max(state['impressions'], 1)
            cvr = state['conversions'] / max(state['clicks'], 1)
            cpc = state['cost'] / max(state['clicks'], 1)
            roas = state['revenue'] / max(state['cost'], 1)
            
            # Update temporal factors
            state['time_performance_factor'] = self._calculate_time_factor(state)
            state['day_performance_factor'] = self._calculate_day_factor(state)
            
            keyword_obs = [
                state['current_bid'] / 10.0,  # Normalize bid
                state['quality_score'] / 10.0,
                state['impressions'] / 10000.0,
                state['clicks'] / 1000.0,
                state['conversions'] / 100.0,
                state['cost'] / 1000.0,
                state['revenue'] / 10000.0,
                ctr,
                cvr,
                cpc / 10.0,
                roas / 10.0,
                state['avg_position'] / 10.0,
                state['impression_share'],
                state['lost_impression_share_rank'],
                state['lost_impression_share_budget'],
                state['competition_level'],
                state['search_volume'] / 10000.0,
                state['top_of_page_bid_low'] / 10.0,
                state['time_performance_factor'],
                state['day_performance_factor'],
            ]
            obs.extend(keyword_obs)
        
        # Global features
        total_cost = sum(state['cost'] for state in self.keyword_states)
        budget_utilization = total_cost / (self.budget_per_hour * self.current_step + 1)
        
        hour_of_day = self.current_time.hour / 24.0
        day_of_week = self.current_time.weekday() / 7.0
        time_progress = self.current_step / self.max_steps
        
        # Market volatility (simulated)
        market_factor = 1.0 + self.market_volatility * np.sin(2 * np.pi * self.current_step / 24)
        
        # Competition intensity
        competition_factor = self.competition_intensity
        
        # Auction pressure (how competitive the current time is)
        auction_pressure = self._calculate_auction_pressure()
        
        global_obs = [
            budget_utilization,
            hour_of_day,
            day_of_week,
            time_progress,
            market_factor,
            competition_factor,
            auction_pressure,
            self.market_volatility,
        ]
        obs.extend(global_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_time_factor(self, keyword_state: Dict[str, Any]) -> float:
        """Calculate time-of-day performance factor"""
        hour = self.current_time.hour
        
        # Different patterns for different keyword types
        if 'business' in keyword_state['text'].lower():
            # Business keywords peak during work hours
            if 9 <= hour <= 17:
                return 1.2
            elif 18 <= hour <= 22:
                return 0.8
            else:
                return 0.5
        else:
            # Consumer keywords peak in evenings
            if 18 <= hour <= 23:
                return 1.3
            elif 9 <= hour <= 17:
                return 0.9
            else:
                return 0.6
    
    def _calculate_day_factor(self, keyword_state: Dict[str, Any]) -> float:
        """Calculate day-of-week performance factor"""
        day = self.current_time.weekday()  # 0=Monday, 6=Sunday
        
        if 'business' in keyword_state['text'].lower():
            # Business keywords stronger on weekdays
            return 1.2 if day < 5 else 0.7
        else:
            # Consumer keywords stronger on weekends
            return 0.9 if day < 5 else 1.4
    
    def _calculate_auction_pressure(self) -> float:
        """Calculate current auction pressure/competition"""
        hour = self.current_time.hour
        day = self.current_time.weekday()
        
        # Higher pressure during peak hours
        if 9 <= hour <= 11 or 18 <= hour <= 20:
            time_pressure = 1.3
        elif 12 <= hour <= 17:
            time_pressure = 1.1
        else:
            time_pressure = 0.8
        
        # Higher pressure on weekdays
        day_pressure = 1.2 if day < 5 else 0.9
        
        return min(2.0, time_pressure * day_pressure)
    
    def _take_action(self, action: np.ndarray) -> Dict[str, float]:
        """Execute bid adjustments and simulate auction results"""
        # Apply bid multipliers
        new_bids = []
        for i, state in enumerate(self.keyword_states):
            # Calculate new bid
            new_bid = state['current_bid'] * action[i]
            
            # Apply constraints
            new_bid = ActionValidator.apply_bid_constraints(
                np.array([new_bid]),
                min_bid=state['min_bid'],
                max_bid=state['max_bid'],
                current_bids=np.array([state['current_bid']]),
                max_change_pct=0.5  # Max 50% change per step
            )[0]
            
            new_bids.append(new_bid)
            
            # Store bid history
            bid_change = {
                'keyword_id': state['id'],
                'step': self.current_step,
                'time': self.current_time,
                'old_bid': state['current_bid'],
                'new_bid': new_bid,
                'multiplier': action[i],
                'reason': 'rl_optimization',
            }
            self.bid_history.append(bid_change)
            
            # Update bid
            state['current_bid'] = new_bid
        
        # Simulate auction for each keyword
        step_metrics = {
            'total_impressions': 0,
            'total_clicks': 0,
            'total_conversions': 0,
            'total_cost': 0.0,
            'total_revenue': 0.0,
            'avg_position': 0.0,
            'avg_impression_share': 0.0,
        }
        
        auction_pressure = self._calculate_auction_pressure()
        
        for i, state in enumerate(self.keyword_states):
            # Simulate auction for this keyword
            auction_result = self._simulate_auction(state, auction_pressure)
            
            # Update keyword state
            state['impressions'] += auction_result['impressions']
            state['clicks'] += auction_result['clicks']
            state['conversions'] += auction_result['conversions']
            state['cost'] += auction_result['cost']
            state['revenue'] += auction_result['revenue']
            state['impression_share'] = auction_result['impression_share']
            state['avg_position'] = auction_result['avg_position']
            state['avg_ad_rank'] = auction_result['avg_ad_rank']
            state['lost_impression_share_rank'] = auction_result['lost_is_rank']
            state['lost_impression_share_budget'] = auction_result['lost_is_budget']
            
            # Aggregate metrics
            step_metrics['total_impressions'] += auction_result['impressions']
            step_metrics['total_clicks'] += auction_result['clicks']
            step_metrics['total_conversions'] += auction_result['conversions']
            step_metrics['total_cost'] += auction_result['cost']
            step_metrics['total_revenue'] += auction_result['revenue']
            
            # Store auction history
            auction_record = {
                'keyword_id': state['id'],
                'step': self.current_step,
                'time': self.current_time,
                'bid': state['current_bid'],
                'auction_result': auction_result,
            }
            self.auction_history.append(auction_record)
        
        # Calculate aggregated metrics
        if self.n_keywords > 0:
            step_metrics['avg_position'] = np.mean([
                self.auction_history[-self.n_keywords + i]['auction_result']['avg_position']
                for i in range(self.n_keywords)
            ])
            step_metrics['avg_impression_share'] = np.mean([
                self.auction_history[-self.n_keywords + i]['auction_result']['impression_share']
                for i in range(self.n_keywords)
            ])
        
        step_metrics['avg_cpc'] = (
            step_metrics['total_cost'] / max(step_metrics['total_clicks'], 1)
        )
        step_metrics['avg_ctr'] = (
            step_metrics['total_clicks'] / max(step_metrics['total_impressions'], 1)
        )
        step_metrics['conversion_rate'] = (
            step_metrics['total_conversions'] / max(step_metrics['total_clicks'], 1)
        )
        step_metrics['roas'] = (
            step_metrics['total_revenue'] / max(step_metrics['total_cost'], 1)
        )
        
        return step_metrics
    
    def _simulate_auction(
        self,
        keyword_state: Dict[str, Any],
        auction_pressure: float
    ) -> Dict[str, Union[int, float]]:
        """Simulate ad auction for a keyword"""
        # Calculate Ad Rank based on bid, quality score, and extensions
        quality_factor = keyword_state['quality_score'] / 10.0
        ad_rank = keyword_state['current_bid'] * quality_factor
        
        # Simulate competitor bids (simplified)
        n_competitors = max(1, int(keyword_state['search_volume'] / 1000 * auction_pressure))
        competitor_bids = []
        
        for _ in range(n_competitors):
            # Competitors bid around top-of-page estimates with some randomness
            base_bid = np.random.uniform(
                keyword_state['top_of_page_bid_low'],
                keyword_state['top_of_page_bid_high']
            )
            competitor_bid = base_bid * (1 + self.np_random.normal(0, 0.2))
            competitor_quality = self.np_random.uniform(5.0, 9.0) / 10.0
            competitor_ad_rank = max(0, competitor_bid * competitor_quality)
            competitor_bids.append(competitor_ad_rank)
        
        competitor_bids.sort(reverse=True)
        
        # Determine position
        position = 1
        for competitor_ad_rank in competitor_bids:
            if ad_rank < competitor_ad_rank:
                position += 1
            else:
                break
        
        # Calculate impression share based on position and budget
        if position <= 4:  # Top positions
            impression_share = max(0.1, 1.0 - (position - 1) * 0.2)
        else:  # Lower positions
            impression_share = max(0.01, 0.4 - (position - 4) * 0.05)
        
        # Budget constraints
        hourly_budget = self.budget_per_hour
        spent_this_hour = keyword_state['cost']  # Simplified
        
        if spent_this_hour >= hourly_budget:
            impression_share *= 0.1  # Severely limited
        elif spent_this_hour >= hourly_budget * 0.8:
            impression_share *= 0.5  # Partially limited
        
        # Calculate impressions based on search volume and impression share
        time_factor = keyword_state['time_performance_factor']
        day_factor = keyword_state['day_performance_factor']
        
        potential_impressions = (
            keyword_state['search_volume'] / 24 *  # Hourly volume
            time_factor * day_factor
        )
        impressions = int(potential_impressions * impression_share)
        
        # Calculate clicks
        position_ctr_factor = max(0.1, 1.5 - (position - 1) * 0.2)
        expected_ctr = keyword_state['expected_ctr'] * position_ctr_factor * quality_factor
        clicks = int(impressions * expected_ctr * (1 + self.np_random.normal(0, 0.1)))
        
        # Calculate conversions
        cvr = keyword_state['conversion_rate'] * (1 + self.np_random.normal(0, 0.1))
        cvr = max(0.001, min(0.5, cvr))
        conversions = int(clicks * cvr)
        
        # Calculate cost (second-price auction)
        if competitor_bids:
            # Pay just above the next highest ad rank
            next_ad_rank = competitor_bids[min(position - 1, len(competitor_bids) - 1)]
            actual_cpc = (next_ad_rank / quality_factor) + 0.01
            actual_cpc = min(actual_cpc, keyword_state['current_bid'])
        else:
            actual_cpc = keyword_state['current_bid'] * 0.5  # No competition
        
        cost = clicks * actual_cpc
        
        # Calculate revenue
        revenue = conversions * keyword_state['avg_revenue_per_conversion']
        revenue *= (1 + self.np_random.normal(0, 0.1))  # Add noise
        revenue = max(0, revenue)
        
        # Calculate lost impression share
        total_possible_impressions = potential_impressions
        lost_impressions_rank = max(0, total_possible_impressions * (1 - impression_share))
        lost_impressions_budget = 0  # Simplified for now
        
        lost_is_rank = lost_impressions_rank / max(total_possible_impressions, 1)
        lost_is_budget = lost_impressions_budget / max(total_possible_impressions, 1)
        
        return {
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'cost': cost,
            'revenue': revenue,
            'avg_position': float(position),
            'avg_ad_rank': ad_rank,
            'impression_share': impression_share,
            'lost_is_rank': lost_is_rank,
            'lost_is_budget': lost_is_budget,
            'actual_cpc': actual_cpc,
        }
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward based on bidding strategy and performance"""
        if self.bid_strategy == "target_cpa":
            return self._calculate_target_cpa_reward(metrics)
        elif self.bid_strategy == "target_roas":
            return self._calculate_target_roas_reward(metrics)
        elif self.bid_strategy == "maximize_clicks":
            return self._calculate_maximize_clicks_reward(metrics)
        else:
            return self._calculate_general_reward(metrics)
    
    def _calculate_target_cpa_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for target CPA strategy"""
        target_cpa = self.config.get('target_cpa', 50.0)
        
        if metrics['total_conversions'] > 0:
            actual_cpa = metrics['total_cost'] / metrics['total_conversions']
            cpa_ratio = target_cpa / actual_cpa
            
            # Reward achieving target CPA
            if 0.8 <= cpa_ratio <= 1.2:  # Within 20% of target
                cpa_reward = 1.0
            elif cpa_ratio > 1.2:  # Below target CPA (good)
                cpa_reward = min(2.0, cpa_ratio)
            else:  # Above target CPA (bad)
                cpa_reward = cpa_ratio
        else:
            cpa_reward = 0.0
        
        # Volume reward (encourage conversions)
        volume_reward = np.tanh(metrics['total_conversions'] / 10.0)
        
        # Quality reward (impression share and position)
        quality_reward = (
            metrics['avg_impression_share'] * 0.5 +
            max(0, (5 - metrics['avg_position']) / 5) * 0.5
        )
        
        return 0.6 * cpa_reward + 0.3 * volume_reward + 0.1 * quality_reward
    
    def _calculate_target_roas_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for target ROAS strategy"""
        target_roas = self.config.get('target_roas', 3.0)
        
        actual_roas = metrics['roas']
        roas_ratio = actual_roas / target_roas
        
        # Reward achieving target ROAS
        if 0.8 <= roas_ratio <= 1.2:  # Within 20% of target
            roas_reward = 1.0
        elif roas_ratio > 1.2:  # Above target ROAS (good)
            roas_reward = min(2.0, roas_ratio)
        else:  # Below target ROAS (bad)
            roas_reward = roas_ratio
        
        # Volume reward
        volume_reward = np.tanh(metrics['total_revenue'] / 1000.0)
        
        # Efficiency reward
        efficiency_reward = metrics['avg_impression_share']
        
        return 0.7 * roas_reward + 0.2 * volume_reward + 0.1 * efficiency_reward
    
    def _calculate_maximize_clicks_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for maximize clicks strategy"""
        # Primary: Click volume
        click_reward = np.tanh(metrics['total_clicks'] / 100.0)
        
        # Secondary: Cost efficiency
        efficiency_reward = metrics['total_clicks'] / max(metrics['total_cost'], 1)
        efficiency_reward = np.tanh(efficiency_reward / 10.0)
        
        # Quality reward
        quality_reward = metrics['avg_ctr']
        
        return 0.6 * click_reward + 0.3 * efficiency_reward + 0.1 * quality_reward
    
    def _calculate_general_reward(self, metrics: Dict[str, float]) -> float:
        """General reward function balancing multiple objectives"""
        # ROI reward
        roi_reward = RewardShaper.roi_reward(
            metrics['total_conversions'],
            metrics['total_cost'],
            target_roi=3.0
        )
        
        # Impression share reward
        is_reward = metrics['avg_impression_share']
        
        # Position reward
        position_reward = max(0, (5 - metrics['avg_position']) / 5)
        
        # CTR reward
        ctr_reward = np.tanh(metrics['avg_ctr'] / 0.05)  # Target 5% CTR
        
        return 0.4 * roi_reward + 0.2 * is_reward + 0.2 * position_reward + 0.2 * ctr_reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate early"""
        # Terminate if budget is severely overspent
        total_cost = sum(state['cost'] for state in self.keyword_states)
        budget_limit = self.budget_per_hour * self.max_steps * 2.0  # 100% overspend tolerance
        
        if total_cost > budget_limit:
            self.log.warning("Episode terminated due to budget overspend")
            return True
        
        return False
    
    def _reset_environment(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Reset bid optimization environment"""
        # Reset keyword states
        self.keyword_states = self._initialize_keyword_states()
        self.auction_history = []
        self.bid_history = []
        
        # Apply reset options
        if options:
            if 'keywords' in options:
                self.keywords = options['keywords']
                self.n_keywords = len(self.keywords)
                self.keyword_states = self._initialize_keyword_states()
                self._setup_spaces()
            
            if 'budget_per_hour' in options:
                self.budget_per_hour = options['budget_per_hour']
            
            if 'bid_strategy' in options:
                self.bid_strategy = options['bid_strategy']
    
    def get_bidding_summary(self) -> Dict[str, Any]:
        """Get summary of bidding performance"""
        total_cost = sum(state['cost'] for state in self.keyword_states)
        total_revenue = sum(state['revenue'] for state in self.keyword_states)
        total_conversions = sum(state['conversions'] for state in self.keyword_states)
        
        keyword_summaries = []
        for state in self.keyword_states:
            ctr = state['clicks'] / max(state['impressions'], 1)
            cvr = state['conversions'] / max(state['clicks'], 1)
            cpc = state['cost'] / max(state['clicks'], 1)
            roas = state['revenue'] / max(state['cost'], 1)
            
            summary = {
                'id': state['id'],
                'text': state['text'],
                'current_bid': state['current_bid'],
                'impressions': state['impressions'],
                'clicks': state['clicks'],
                'conversions': state['conversions'],
                'cost': state['cost'],
                'revenue': state['revenue'],
                'ctr': ctr,
                'cvr': cvr,
                'cpc': cpc,
                'roas': roas,
                'avg_position': state['avg_position'],
                'impression_share': state['impression_share'],
                'quality_score': state['quality_score'],
            }
            keyword_summaries.append(summary)
        
        return {
            'bid_strategy': self.bid_strategy,
            'total_budget': self.budget_per_hour * self.max_steps,
            'total_spent': total_cost,
            'total_revenue': total_revenue,
            'total_conversions': total_conversions,
            'overall_roas': total_revenue / max(total_cost, 1),
            'overall_cpa': total_cost / max(total_conversions, 1),
            'avg_position': np.mean([s['avg_position'] for s in self.keyword_states]),
            'avg_impression_share': np.mean([s['impression_share'] for s in self.keyword_states]),
            'keywords': keyword_summaries,
        }