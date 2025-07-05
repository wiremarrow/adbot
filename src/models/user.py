"""
User and account management models
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, 
    Text, Index, Enum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSON

from .base import BaseModel, MetadataMixin
import enum


class UserRole(enum.Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class AccountStatus(enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"


class PlatformStatus(enum.Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PENDING = "pending"


class User(BaseModel, MetadataMixin):
    """User model for authentication and authorization"""
    
    __tablename__ = "users"
    
    # Basic user info
    email = Column(String(255), nullable=False, unique=True)
    username = Column(String(100), nullable=False, unique=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Authorization
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    permissions = Column(JSON, default=list, nullable=False)
    
    # Profile
    avatar_url = Column(String(500), nullable=True)
    timezone = Column(String(50), default="UTC", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    
    # Preferences
    notification_settings = Column(JSON, default=dict, nullable=False)
    dashboard_settings = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_username", "username"),
        Index("idx_user_role", "role"),
        Index("idx_user_active", "is_active"),
    )
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    def can_access_account(self, account_id: str) -> bool:
        """Check if user can access specific account"""
        return any(acc.id == account_id for acc in self.accounts)


class Account(BaseModel, MetadataMixin):
    """Account model for organizing campaigns and platforms"""
    
    __tablename__ = "accounts"
    
    # Basic account info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Account settings
    status = Column(Enum(AccountStatus), default=AccountStatus.ACTIVE, nullable=False)
    subscription_tier = Column(String(50), default="basic", nullable=False)
    
    # Billing
    billing_email = Column(String(255), nullable=True)
    billing_address = Column(JSON, nullable=True)
    
    # Limits and quotas
    monthly_budget_limit = Column(String(20), nullable=True)  # Using string for flexibility
    api_rate_limit = Column(String(20), default="standard", nullable=False)
    
    # Tracking
    campaigns_count = Column(String(20), default="0", nullable=False)
    last_activity = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="accounts")
    platforms = relationship("Platform", back_populates="account", cascade="all, delete-orphan")
    campaigns = relationship("Campaign", back_populates="account")
    
    # Indexes
    __table_args__ = (
        Index("idx_account_user_id", "user_id"),
        Index("idx_account_status", "status"),
        Index("idx_account_name", "name"),
    )
    
    def get_total_spend(self) -> float:
        """Get total spend across all campaigns"""
        # This would be calculated from performance metrics
        return 0.0  # Placeholder
    
    def is_over_budget(self) -> bool:
        """Check if account is over monthly budget"""
        if not self.monthly_budget_limit:
            return False
        # Implementation would check against actual spend
        return False  # Placeholder


class Platform(BaseModel, MetadataMixin):
    """Platform integration model"""
    
    __tablename__ = "platforms"
    
    # Basic platform info
    name = Column(String(100), nullable=False)  # google_ads, facebook, tiktok, etc.
    display_name = Column(String(100), nullable=False)
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False)
    
    # Integration settings
    status = Column(Enum(PlatformStatus), default=PlatformStatus.PENDING, nullable=False)
    platform_account_id = Column(String(255), nullable=True)
    platform_account_name = Column(String(255), nullable=True)
    
    # API credentials (encrypted)
    api_credentials = Column(JSON, nullable=True)
    
    # Connection details
    connected_at = Column(DateTime, nullable=True)
    last_sync = Column(DateTime, nullable=True)
    sync_frequency = Column(String(20), default="hourly", nullable=False)
    
    # Status and health
    health_score = Column(String(20), default="100", nullable=False)
    last_error = Column(Text, nullable=True)
    error_count = Column(String(20), default="0", nullable=False)
    
    # Configuration
    sync_settings = Column(JSON, default=dict, nullable=False)
    webhook_settings = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    account = relationship("Account", back_populates="platforms")
    
    # Indexes
    __table_args__ = (
        Index("idx_platform_account_id", "account_id"),
        Index("idx_platform_name", "name"),
        Index("idx_platform_status", "status"),
        Index("idx_platform_last_sync", "last_sync"),
    )
    
    def is_healthy(self) -> bool:
        """Check if platform integration is healthy"""
        return self.status == PlatformStatus.CONNECTED and self.health_score == "100"
    
    def needs_sync(self) -> bool:
        """Check if platform needs synchronization"""
        if not self.last_sync:
            return True
        
        # Check based on sync frequency
        if self.sync_frequency == "realtime":
            return True
        elif self.sync_frequency == "hourly":
            return (datetime.utcnow() - self.last_sync).total_seconds() > 3600
        elif self.sync_frequency == "daily":
            return (datetime.utcnow() - self.last_sync).total_seconds() > 86400
        
        return False


class APIKey(BaseModel):
    """API key model for external access"""
    
    __tablename__ = "api_keys"
    
    # Basic key info
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Key settings
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Permissions and limits
    permissions = Column(JSON, default=list, nullable=False)
    rate_limit = Column(String(20), default="1000/hour", nullable=False)
    
    # Usage tracking
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(String(20), default="0", nullable=False)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("idx_api_key_user_id", "user_id"),
        Index("idx_api_key_active", "is_active"),
        Index("idx_api_key_expires", "expires_at"),
    )
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if not self.is_active:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True