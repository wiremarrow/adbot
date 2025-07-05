"""
Experiment and A/B testing models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, Index, Enum, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from .base import BaseModel, MetadataMixin
import enum


class ExperimentStatus(enum.Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(enum.Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    BANDIT = "bandit"
    SEQUENTIAL = "sequential"


class Experiment(BaseModel, MetadataMixin):
    """Experiment model for A/B testing and optimization"""
    
    __tablename__ = "experiments"
    
    # Basic experiment info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    hypothesis = Column(Text, nullable=True)
    
    # Experiment configuration
    experiment_type = Column(Enum(ExperimentType), default=ExperimentType.AB_TEST, nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.DRAFT, nullable=False)
    
    # Targeting
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False)
    campaign_ids = Column(JSON, default=list, nullable=False)  # List of campaign IDs
    platform = Column(String(50), nullable=False)
    
    # Experiment settings
    traffic_allocation = Column(Float, default=1.0, nullable=False)  # 0.0 to 1.0
    variants = Column(JSON, nullable=False)  # Experiment variants configuration
    
    # Statistical settings
    primary_metric = Column(String(100), nullable=False)
    secondary_metrics = Column(JSON, default=list, nullable=False)
    significance_level = Column(Float, default=0.05, nullable=False)
    minimum_detectable_effect = Column(Float, default=0.1, nullable=False)
    statistical_power = Column(Float, default=0.8, nullable=False)
    
    # Timing
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    duration_days = Column(Integer, nullable=True)
    
    # Sample size
    required_sample_size = Column(Integer, nullable=True)
    actual_sample_size = Column(Integer, default=0, nullable=False)
    
    # Results
    winner_variant = Column(String(100), nullable=True)
    confidence_level = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=True)
    
    # Relationships
    account = relationship("Account")
    results = relationship("ExperimentResult", back_populates="experiment", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_account_id", "account_id"),
        Index("idx_experiment_status", "status"),
        Index("idx_experiment_platform", "platform"),
        Index("idx_experiment_dates", "start_date", "end_date"),
    )
    
    def is_running(self) -> bool:
        """Check if experiment is currently running"""
        return self.status == ExperimentStatus.RUNNING
    
    def is_complete(self) -> bool:
        """Check if experiment is complete"""
        return self.status == ExperimentStatus.COMPLETED
    
    def has_significant_result(self) -> bool:
        """Check if experiment has statistically significant results"""
        return self.p_value is not None and self.p_value < self.significance_level
    
    def get_variant_config(self, variant_name: str) -> Dict[str, Any]:
        """Get configuration for a specific variant"""
        return self.variants.get(variant_name, {})
    
    def calculate_progress(self) -> float:
        """Calculate experiment progress as percentage"""
        if not self.required_sample_size:
            return 0.0
        return min(self.actual_sample_size / self.required_sample_size, 1.0)


class ExperimentResult(BaseModel):
    """Experiment results by variant and metric"""
    
    __tablename__ = "experiment_results"
    
    # Identifiers
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    variant_name = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Sample data
    sample_size = Column(Integer, nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    standard_error = Column(Float, nullable=True)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    
    # Statistical comparisons
    control_value = Column(Float, nullable=True)
    lift = Column(Float, nullable=True)  # Percentage improvement over control
    p_value = Column(Float, nullable=True)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_result_exp_id", "experiment_id"),
        Index("idx_experiment_result_variant", "variant_name"),
        Index("idx_experiment_result_metric", "metric_name"),
        Index("idx_experiment_result_date", "date"),
    )


class BanditArm(BaseModel):
    """Multi-armed bandit arm configuration"""
    
    __tablename__ = "bandit_arms"
    
    # Identifiers
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    arm_name = Column(String(100), nullable=False)
    
    # Configuration
    arm_config = Column(JSON, nullable=False)
    
    # Bandit statistics
    pulls = Column(Integer, default=0, nullable=False)
    rewards = Column(Float, default=0.0, nullable=False)
    
    # Thompson Sampling parameters
    alpha = Column(Float, default=1.0, nullable=False)
    beta = Column(Float, default=1.0, nullable=False)
    
    # UCB parameters
    confidence_bound = Column(Float, default=0.0, nullable=False)
    
    # Performance metrics
    conversion_rate = Column(Float, default=0.0, nullable=False)
    average_reward = Column(Float, default=0.0, nullable=False)
    
    # Relationships
    experiment = relationship("Experiment")
    
    # Indexes
    __table_args__ = (
        Index("idx_bandit_arm_experiment_id", "experiment_id"),
        Index("idx_bandit_arm_name", "arm_name"),
    )
    
    def update_statistics(self, reward: float) -> None:
        """Update bandit arm statistics with new reward"""
        self.pulls += 1
        self.rewards += reward
        self.average_reward = self.rewards / self.pulls
        
        # Update Thompson Sampling parameters
        if reward > 0:
            self.alpha += 1
        else:
            self.beta += 1
    
    def get_thompson_sample(self) -> float:
        """Get Thompson sampling probability"""
        import numpy as np
        return np.random.beta(self.alpha, self.beta)
    
    def get_ucb_score(self, total_pulls: int, c: float = 1.0) -> float:
        """Get Upper Confidence Bound score"""
        import math
        
        if self.pulls == 0:
            return float('inf')
        
        exploration_bonus = c * math.sqrt(math.log(total_pulls) / self.pulls)
        return self.average_reward + exploration_bonus


class ExperimentAssignment(BaseModel):
    """Track user/entity assignments to experiment variants"""
    
    __tablename__ = "experiment_assignments"
    
    # Identifiers
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    entity_id = Column(String(255), nullable=False)  # User ID, campaign ID, etc.
    entity_type = Column(String(50), nullable=False)  # user, campaign, etc.
    
    # Assignment details
    variant_name = Column(String(100), nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Tracking
    first_exposure = Column(DateTime, nullable=True)
    last_exposure = Column(DateTime, nullable=True)
    exposure_count = Column(Integer, default=0, nullable=False)
    
    # Conversion tracking
    converted = Column(Boolean, default=False, nullable=False)
    conversion_value = Column(Float, default=0.0, nullable=False)
    conversion_timestamp = Column(DateTime, nullable=True)
    
    # Relationships
    experiment = relationship("Experiment")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_assignment_exp_id", "experiment_id"),
        Index("idx_experiment_assignment_entity", "entity_id", "entity_type"),
        Index("idx_experiment_assignment_variant", "variant_name"),
    )
    
    def record_exposure(self) -> None:
        """Record an exposure event"""
        now = datetime.utcnow()
        
        if self.first_exposure is None:
            self.first_exposure = now
        
        self.last_exposure = now
        self.exposure_count += 1
    
    def record_conversion(self, value: float = 0.0) -> None:
        """Record a conversion event"""
        if not self.converted:
            self.converted = True
            self.conversion_value = value
            self.conversion_timestamp = datetime.utcnow()