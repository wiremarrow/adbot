"""
Performance tracking database models
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Date, 
    ForeignKey, Numeric, Index, Boolean
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSON

from .base import BaseModel, MetadataMixin


class PerformanceMetric(BaseModel, MetadataMixin):
    """Performance metrics for campaigns, ad groups, ads, and keywords"""
    
    __tablename__ = "performance_metrics"
    
    # Identifiers
    entity_type = Column(String(50), nullable=False)  # campaign, ad_group, ad, keyword
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    platform = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    
    # Basic metrics
    impressions = Column(Integer, default=0, nullable=False)
    clicks = Column(Integer, default=0, nullable=False)
    conversions = Column(Integer, default=0, nullable=False)
    cost = Column(Numeric(12, 4), default=0, nullable=False)
    revenue = Column(Numeric(12, 4), default=0, nullable=False)
    
    # Calculated metrics
    ctr = Column(Float, default=0, nullable=False)  # Click-through rate
    cpc = Column(Numeric(8, 4), default=0, nullable=False)  # Cost per click
    cpm = Column(Numeric(8, 4), default=0, nullable=False)  # Cost per mille
    cpa = Column(Numeric(8, 4), default=0, nullable=False)  # Cost per acquisition
    roas = Column(Numeric(8, 4), default=0, nullable=False)  # Return on ad spend
    
    # Engagement metrics
    video_views = Column(Integer, default=0, nullable=False)
    video_completions = Column(Integer, default=0, nullable=False)
    likes = Column(Integer, default=0, nullable=False)
    shares = Column(Integer, default=0, nullable=False)
    comments = Column(Integer, default=0, nullable=False)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    
    # Relationships
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    campaign = relationship("Campaign", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index("idx_performance_entity", "entity_type", "entity_id"),
        Index("idx_performance_date", "date"),
        Index("idx_performance_platform", "platform"),
        Index("idx_performance_campaign_date", "campaign_id", "date"),
    )
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics"""
        if self.impressions > 0:
            self.ctr = (self.clicks / self.impressions) * 100
            self.cpm = (self.cost / self.impressions) * 1000
        
        if self.clicks > 0:
            self.cpc = self.cost / self.clicks
        
        if self.conversions > 0:
            self.cpa = self.cost / self.conversions
        
        if self.cost > 0:
            self.roas = self.revenue / self.cost


class ConversionData(BaseModel, MetadataMixin):
    """Detailed conversion tracking data"""
    
    __tablename__ = "conversion_data"
    
    # Identifiers
    conversion_id = Column(String(255), nullable=False, unique=True)
    platform = Column(String(50), nullable=False)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    
    # Conversion details
    conversion_type = Column(String(100), nullable=False)  # purchase, signup, etc.
    conversion_value = Column(Numeric(12, 4), default=0, nullable=False)
    conversion_timestamp = Column(DateTime, nullable=False)
    
    # Attribution
    click_timestamp = Column(DateTime, nullable=True)
    view_timestamp = Column(DateTime, nullable=True)
    attribution_model = Column(String(50), nullable=False)  # first_click, last_click, etc.
    
    # User data (anonymized)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    device_type = Column(String(50), nullable=True)
    geo_location = Column(String(100), nullable=True)
    
    # Product/service details
    product_id = Column(String(255), nullable=True)
    product_category = Column(String(100), nullable=True)
    quantity = Column(Integer, default=1, nullable=False)
    
    # Relationships
    campaign = relationship("Campaign")
    
    # Indexes
    __table_args__ = (
        Index("idx_conversion_campaign_id", "campaign_id"),
        Index("idx_conversion_timestamp", "conversion_timestamp"),
        Index("idx_conversion_type", "conversion_type"),
        Index("idx_conversion_platform", "platform"),
    )


class BidHistory(BaseModel):
    """Historical bid data for optimization tracking"""
    
    __tablename__ = "bid_history"
    
    # Identifiers
    entity_type = Column(String(50), nullable=False)  # campaign, ad_group, keyword
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    platform = Column(String(50), nullable=False)
    
    # Bid data
    old_bid = Column(Numeric(8, 4), nullable=True)
    new_bid = Column(Numeric(8, 4), nullable=False)
    bid_type = Column(String(50), nullable=False)  # cpc, cpm, cpa, etc.
    
    # Context
    reason = Column(String(500), nullable=True)
    optimizer_used = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Performance before/after
    previous_performance = Column(JSON, nullable=True)
    expected_performance = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_bid_history_entity", "entity_type", "entity_id"),
        Index("idx_bid_history_created", "created_at"),
        Index("idx_bid_history_platform", "platform"),
    )


class AnomalyDetection(BaseModel):
    """Anomaly detection results for performance monitoring"""
    
    __tablename__ = "anomaly_detection"
    
    # Identifiers
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    platform = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    
    # Anomaly details
    metric_name = Column(String(100), nullable=False)
    expected_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    
    # Detection method
    detection_method = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Status
    acknowledged = Column(Boolean, default=False, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_notes = Column(String(1000), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_anomaly_entity", "entity_type", "entity_id"),
        Index("idx_anomaly_date", "date"),
        Index("idx_anomaly_severity", "severity"),
        Index("idx_anomaly_unresolved", "resolved", "acknowledged"),
    )