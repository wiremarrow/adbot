"""
Campaign-related database models
"""

from datetime import datetime
from typing import List, Optional
from decimal import Decimal

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, Enum, Numeric, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from .base import BaseModel, MetadataMixin
import enum


class CampaignStatus(enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    DRAFT = "draft"


class AdGroupStatus(enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    REMOVED = "removed"


class AdStatus(enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    REMOVED = "removed"
    PENDING_REVIEW = "pending_review"
    DISAPPROVED = "disapproved"


class KeywordMatchType(enum.Enum):
    EXACT = "exact"
    PHRASE = "phrase"
    BROAD = "broad"
    BROAD_MATCH_MODIFIER = "broad_match_modifier"


class Campaign(BaseModel, MetadataMixin):
    """Campaign model"""
    
    __tablename__ = "campaigns"
    
    # Basic campaign info
    name = Column(String(255), nullable=False)
    platform = Column(String(50), nullable=False)  # google_ads, facebook, etc.
    platform_id = Column(String(255), nullable=False)  # Platform-specific ID
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.id"), nullable=False)
    
    # Campaign settings
    status = Column(Enum(CampaignStatus), default=CampaignStatus.ACTIVE, nullable=False)
    budget_type = Column(String(50), nullable=False)  # daily, lifetime, etc.
    budget_amount = Column(Numeric(10, 2), nullable=False)
    target_cpa = Column(Numeric(8, 2), nullable=True)
    target_roas = Column(Numeric(5, 2), nullable=True)
    
    # Scheduling
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    
    # Targeting
    geo_targets = Column(ARRAY(String), nullable=True)
    language_targets = Column(ARRAY(String), nullable=True)
    device_targets = Column(ARRAY(String), nullable=True)
    
    # Optimization
    optimization_goal = Column(String(100), nullable=True)
    bid_strategy = Column(String(100), nullable=True)
    
    # Relationships
    ad_groups = relationship("AdGroup", back_populates="campaign", cascade="all, delete-orphan")
    performance_metrics = relationship("PerformanceMetric", back_populates="campaign")
    
    # Indexes
    __table_args__ = (
        Index("idx_campaign_platform_id", "platform", "platform_id"),
        Index("idx_campaign_account_id", "account_id"),
        Index("idx_campaign_status", "status"),
    )


class AdGroup(BaseModel, MetadataMixin):
    """Ad Group model"""
    
    __tablename__ = "ad_groups"
    
    # Basic ad group info
    name = Column(String(255), nullable=False)
    platform_id = Column(String(255), nullable=False)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    
    # Ad group settings
    status = Column(Enum(AdGroupStatus), default=AdGroupStatus.ACTIVE, nullable=False)
    default_cpc = Column(Numeric(8, 2), nullable=True)
    default_cpm = Column(Numeric(8, 2), nullable=True)
    
    # Targeting
    audience_targets = Column(ARRAY(String), nullable=True)
    placement_targets = Column(ARRAY(String), nullable=True)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="ad_groups")
    ads = relationship("Ad", back_populates="ad_group", cascade="all, delete-orphan")
    keywords = relationship("Keyword", back_populates="ad_group", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_adgroup_campaign_id", "campaign_id"),
        Index("idx_adgroup_platform_id", "platform_id"),
    )


class Ad(BaseModel, MetadataMixin):
    """Ad model"""
    
    __tablename__ = "ads"
    
    # Basic ad info
    name = Column(String(255), nullable=False)
    platform_id = Column(String(255), nullable=False)
    ad_group_id = Column(UUID(as_uuid=True), ForeignKey("ad_groups.id"), nullable=False)
    
    # Ad settings
    status = Column(Enum(AdStatus), default=AdStatus.ACTIVE, nullable=False)
    ad_type = Column(String(50), nullable=False)  # text, image, video, etc.
    
    # Ad content
    headline = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    display_url = Column(String(500), nullable=True)
    final_url = Column(String(1000), nullable=False)
    
    # Creative assets
    image_urls = Column(ARRAY(String), nullable=True)
    video_url = Column(String(1000), nullable=True)
    
    # Performance tracking
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    cost = Column(Numeric(10, 2), default=0)
    
    # Relationships
    ad_group = relationship("AdGroup", back_populates="ads")
    
    # Indexes
    __table_args__ = (
        Index("idx_ad_adgroup_id", "ad_group_id"),
        Index("idx_ad_platform_id", "platform_id"),
        Index("idx_ad_status", "status"),
    )


class Keyword(BaseModel, MetadataMixin):
    """Keyword model"""
    
    __tablename__ = "keywords"
    
    # Basic keyword info
    text = Column(String(500), nullable=False)
    platform_id = Column(String(255), nullable=False)
    ad_group_id = Column(UUID(as_uuid=True), ForeignKey("ad_groups.id"), nullable=False)
    
    # Keyword settings
    match_type = Column(Enum(KeywordMatchType), nullable=False)
    status = Column(String(50), default="active", nullable=False)
    max_cpc = Column(Numeric(8, 2), nullable=True)
    
    # Performance tracking
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    cost = Column(Numeric(10, 2), default=0)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    first_page_cpc = Column(Numeric(8, 2), nullable=True)
    
    # Relationships
    ad_group = relationship("AdGroup", back_populates="keywords")
    
    # Indexes
    __table_args__ = (
        Index("idx_keyword_adgroup_id", "ad_group_id"),
        Index("idx_keyword_text", "text"),
        Index("idx_keyword_match_type", "match_type"),
    )