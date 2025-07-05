"""
RL Agent and training models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, Index, Enum, JSON, LargeBinary
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from .base import BaseModel, MetadataMixin
import enum


class AgentStatus(enum.Enum):
    TRAINING = "training"
    DEPLOYED = "deployed"
    PAUSED = "paused"
    ARCHIVED = "archived"


class TrainingStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Agent(BaseModel, MetadataMixin):
    """RL Agent model"""
    
    __tablename__ = "agents"
    
    # Basic agent info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    agent_type = Column(String(50), nullable=False)  # PPO, SAC, TD3, etc.
    
    # Configuration
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False)
    status = Column(Enum(AgentStatus), default=AgentStatus.TRAINING, nullable=False)
    
    # Environment configuration
    environment_type = Column(String(100), nullable=False)
    environment_config = Column(JSON, nullable=False)
    
    # Agent configuration
    agent_config = Column(JSON, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    
    # Model storage
    model_path = Column(String(500), nullable=True)
    model_version = Column(String(50), nullable=True)
    model_size_mb = Column(Float, nullable=True)
    
    # Performance metrics
    best_reward = Column(Float, nullable=True)
    best_episode = Column(Integer, nullable=True)
    total_episodes = Column(Integer, default=0, nullable=False)
    total_timesteps = Column(Integer, default=0, nullable=False)
    
    # Training history
    training_start = Column(DateTime, nullable=True)
    training_end = Column(DateTime, nullable=True)
    training_duration_seconds = Column(Integer, nullable=True)
    
    # Deployment info
    deployed_at = Column(DateTime, nullable=True)
    deployment_version = Column(String(50), nullable=True)
    
    # Relationships
    account = relationship("Account")
    training_runs = relationship("TrainingRun", back_populates="agent", cascade="all, delete-orphan")
    configs = relationship("AgentConfig", back_populates="agent", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_agent_account_id", "account_id"),
        Index("idx_agent_status", "status"),
        Index("idx_agent_type", "agent_type"),
        Index("idx_agent_deployed", "deployed_at"),
    )
    
    def is_deployed(self) -> bool:
        """Check if agent is currently deployed"""
        return self.status == AgentStatus.DEPLOYED
    
    def is_training(self) -> bool:
        """Check if agent is currently training"""
        return self.status == AgentStatus.TRAINING
    
    def get_latest_training_run(self) -> Optional["TrainingRun"]:
        """Get the latest training run"""
        return max(self.training_runs, key=lambda x: x.created_at) if self.training_runs else None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "total_episodes": self.total_episodes,
            "total_timesteps": self.total_timesteps,
            "training_duration": self.training_duration_seconds,
            "is_deployed": self.is_deployed(),
        }


class AgentConfig(BaseModel):
    """Agent configuration snapshots"""
    
    __tablename__ = "agent_configs"
    
    # Identifiers
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    config_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Configuration data
    config_data = Column(JSON, nullable=False)
    config_hash = Column(String(64), nullable=False)  # SHA256 hash
    
    # Metadata
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    
    # Performance tracking
    performance_metrics = Column(JSON, nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="configs")
    
    # Indexes
    __table_args__ = (
        Index("idx_agent_config_agent_id", "agent_id"),
        Index("idx_agent_config_hash", "config_hash"),
        Index("idx_agent_config_active", "is_active"),
    )


class TrainingRun(BaseModel, MetadataMixin):
    """Training run tracking"""
    
    __tablename__ = "training_runs"
    
    # Identifiers
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    run_name = Column(String(255), nullable=False)
    
    # Training configuration
    algorithm = Column(String(50), nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    environment_config = Column(JSON, nullable=False)
    
    # Training progress
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING, nullable=False)
    current_episode = Column(Integer, default=0, nullable=False)
    current_timestep = Column(Integer, default=0, nullable=False)
    target_timesteps = Column(Integer, nullable=False)
    
    # Performance metrics
    episode_rewards = Column(JSON, default=list, nullable=False)
    episode_lengths = Column(JSON, default=list, nullable=False)
    loss_values = Column(JSON, default=list, nullable=False)
    
    # Current performance
    current_reward = Column(Float, nullable=True)
    best_reward = Column(Float, nullable=True)
    moving_average_reward = Column(Float, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Resources
    gpu_used = Column(Boolean, default=False, nullable=False)
    cpu_cores = Column(Integer, nullable=True)
    memory_gb = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Model artifacts
    checkpoint_path = Column(String(500), nullable=True)
    tensorboard_path = Column(String(500), nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="training_runs")
    
    # Indexes
    __table_args__ = (
        Index("idx_training_run_agent_id", "agent_id"),
        Index("idx_training_run_status", "status"),
        Index("idx_training_run_started", "started_at"),
    )
    
    def is_running(self) -> bool:
        """Check if training run is currently running"""
        return self.status == TrainingStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if training run is completed"""
        return self.status == TrainingStatus.COMPLETED
    
    def get_progress_percentage(self) -> float:
        """Get training progress as percentage"""
        if self.target_timesteps == 0:
            return 0.0
        return min(self.current_timestep / self.target_timesteps * 100, 100.0)
    
    def add_episode_data(self, reward: float, length: int, loss: float = None) -> None:
        """Add episode data to the training run"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if loss is not None:
            self.loss_values.append(loss)
        
        self.current_reward = reward
        
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
        
        # Calculate moving average (last 100 episodes)
        recent_rewards = self.episode_rewards[-100:]
        self.moving_average_reward = sum(recent_rewards) / len(recent_rewards)
        
        self.current_episode += 1
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training run summary"""
        return {
            "run_name": self.run_name,
            "status": self.status.value,
            "progress": self.get_progress_percentage(),
            "current_episode": self.current_episode,
            "current_timestep": self.current_timestep,
            "target_timesteps": self.target_timesteps,
            "best_reward": self.best_reward,
            "current_reward": self.current_reward,
            "moving_average_reward": self.moving_average_reward,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ModelCheckpoint(BaseModel):
    """Model checkpoint storage"""
    
    __tablename__ = "model_checkpoints"
    
    # Identifiers
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    training_run_id = Column(UUID(as_uuid=True), ForeignKey("training_runs.id"), nullable=False)
    
    # Checkpoint info
    checkpoint_name = Column(String(255), nullable=False)
    episode = Column(Integer, nullable=False)
    timestep = Column(Integer, nullable=False)
    
    # Performance at checkpoint
    reward = Column(Float, nullable=False)
    loss = Column(Float, nullable=True)
    
    # Storage
    file_path = Column(String(500), nullable=False)
    file_size_mb = Column(Float, nullable=False)
    
    # Metadata
    is_best = Column(Boolean, default=False, nullable=False)
    notes = Column(Text, nullable=True)
    
    # Relationships
    agent = relationship("Agent")
    training_run = relationship("TrainingRun")
    
    # Indexes
    __table_args__ = (
        Index("idx_checkpoint_agent_id", "agent_id"),
        Index("idx_checkpoint_training_run_id", "training_run_id"),
        Index("idx_checkpoint_episode", "episode"),
        Index("idx_checkpoint_is_best", "is_best"),
    )