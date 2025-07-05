"""
Google Ads data models for type safety and validation
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum


class CampaignStatus(Enum):
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"


class AdGroupStatus(Enum):
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"


class KeywordMatchType(Enum):
    EXACT = "EXACT"
    PHRASE = "PHRASE"
    BROAD = "BROAD"
    BROAD_MATCH_MODIFIER = "BROAD_MATCH_MODIFIER"


class BiddingStrategyType(Enum):
    MANUAL_CPC = "MANUAL_CPC"
    ENHANCED_CPC = "ENHANCED_CPC"
    TARGET_CPA = "TARGET_CPA"
    TARGET_ROAS = "TARGET_ROAS"
    MAXIMIZE_CLICKS = "MAXIMIZE_CLICKS"
    MAXIMIZE_CONVERSIONS = "MAXIMIZE_CONVERSIONS"
    MAXIMIZE_CONVERSION_VALUE = "MAXIMIZE_CONVERSION_VALUE"


@dataclass
class GoogleAdsCampaign:
    """Google Ads Campaign model"""
    id: str
    name: str
    status: CampaignStatus
    advertising_channel_type: str
    bidding_strategy_type: BiddingStrategyType
    budget_amount_micros: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_cpa_micros: Optional[int] = None
    target_roas: Optional[float] = None
    
    @property
    def budget_amount(self) -> float:
        """Get budget amount in currency units"""
        return self.budget_amount_micros / 1_000_000
    
    @property
    def target_cpa(self) -> Optional[float]:
        """Get target CPA in currency units"""
        return self.target_cpa_micros / 1_000_000 if self.target_cpa_micros else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'platform_id': self.id,
            'name': self.name,
            'status': self.status.value.lower(),
            'channel_type': self.advertising_channel_type.lower(),
            'bid_strategy': self.bidding_strategy_type.value.lower(),
            'budget_amount': self.budget_amount,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'target_cpa': self.target_cpa,
            'target_roas': self.target_roas,
        }
    
    @classmethod
    def from_google_ads_response(cls, row: Any) -> 'GoogleAdsCampaign':
        """Create from Google Ads API response"""
        campaign = row.campaign
        budget = row.campaign_budget
        
        return cls(
            id=str(campaign.id),
            name=campaign.name,
            status=CampaignStatus(campaign.status.name),
            advertising_channel_type=campaign.advertising_channel_type.name,
            bidding_strategy_type=BiddingStrategyType(campaign.bidding_strategy_type.name),
            budget_amount_micros=budget.amount_micros if budget.amount_micros else 0,
            start_date=campaign.start_date,
            end_date=campaign.end_date,
            target_cpa_micros=getattr(campaign.target_cpa, 'target_cpa_micros', None),
            target_roas=getattr(campaign.target_roas, 'target_roas', None),
        )


@dataclass
class GoogleAdsAdGroup:
    """Google Ads Ad Group model"""
    id: str
    name: str
    campaign_id: str
    status: AdGroupStatus
    cpc_bid_micros: Optional[int] = None
    target_cpa_micros: Optional[int] = None
    
    @property
    def cpc_bid(self) -> Optional[float]:
        """Get CPC bid in currency units"""
        return self.cpc_bid_micros / 1_000_000 if self.cpc_bid_micros else None
    
    @property
    def target_cpa(self) -> Optional[float]:
        """Get target CPA in currency units"""
        return self.target_cpa_micros / 1_000_000 if self.target_cpa_micros else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'platform_id': self.id,
            'name': self.name,
            'campaign_id': self.campaign_id,
            'status': self.status.value.lower(),
            'cpc_bid': self.cpc_bid,
            'target_cpa': self.target_cpa,
        }
    
    @classmethod
    def from_google_ads_response(cls, row: Any) -> 'GoogleAdsAdGroup':
        """Create from Google Ads API response"""
        ad_group = row.ad_group
        
        # Extract campaign ID from resource name
        campaign_resource = ad_group.campaign
        campaign_id = campaign_resource.split('/')[-1] if campaign_resource else None
        
        return cls(
            id=str(ad_group.id),
            name=ad_group.name,
            campaign_id=campaign_id,
            status=AdGroupStatus(ad_group.status.name),
            cpc_bid_micros=getattr(ad_group, 'cpc_bid_micros', None),
            target_cpa_micros=getattr(ad_group, 'target_cpa_micros', None),
        )


@dataclass
class GoogleAdsKeyword:
    """Google Ads Keyword model"""
    id: str
    text: str
    match_type: KeywordMatchType
    ad_group_id: str
    status: str
    cpc_bid_micros: Optional[int] = None
    quality_score: Optional[int] = None
    
    @property
    def cpc_bid(self) -> Optional[float]:
        """Get CPC bid in currency units"""
        return self.cpc_bid_micros / 1_000_000 if self.cpc_bid_micros else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'platform_id': self.id,
            'text': self.text,
            'match_type': self.match_type.value.lower(),
            'ad_group_id': self.ad_group_id,
            'status': self.status.lower(),
            'cpc_bid': self.cpc_bid,
            'quality_score': self.quality_score,
        }
    
    @classmethod
    def from_google_ads_response(cls, row: Any) -> 'GoogleAdsKeyword':
        """Create from Google Ads API response"""
        criterion = row.ad_group_criterion
        keyword = criterion.keyword
        
        # Extract ad group ID from resource name
        ad_group_resource = criterion.ad_group
        ad_group_id = ad_group_resource.split('/')[-1] if ad_group_resource else None
        
        return cls(
            id=str(criterion.criterion_id),
            text=keyword.text,
            match_type=KeywordMatchType(keyword.match_type.name),
            ad_group_id=ad_group_id,
            status=criterion.status.name,
            cpc_bid_micros=getattr(criterion, 'cpc_bid_micros', None),
            quality_score=getattr(criterion.quality_info, 'quality_score', None) if hasattr(criterion, 'quality_info') else None,
        )


@dataclass
class GoogleAdsAd:
    """Google Ads Ad model"""
    id: str
    ad_group_id: str
    status: str
    ad_type: str
    headlines: List[str]
    descriptions: List[str]
    final_urls: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'platform_id': self.id,
            'ad_group_id': self.ad_group_id,
            'status': self.status.lower(),
            'ad_type': self.ad_type.lower(),
            'headlines': self.headlines,
            'descriptions': self.descriptions,
            'final_urls': self.final_urls,
        }
    
    @classmethod
    def from_google_ads_response(cls, row: Any) -> 'GoogleAdsAd':
        """Create from Google Ads API response"""
        ad_group_ad = row.ad_group_ad
        ad = ad_group_ad.ad
        
        # Extract ad group ID from resource name
        ad_group_resource = ad_group_ad.ad_group
        ad_group_id = ad_group_resource.split('/')[-1] if ad_group_resource else None
        
        # Extract headlines and descriptions based on ad type
        headlines = []
        descriptions = []
        
        if hasattr(ad, 'responsive_search_ad'):
            rsa = ad.responsive_search_ad
            headlines = [headline.text for headline in rsa.headlines]
            descriptions = [description.text for description in rsa.descriptions]
        elif hasattr(ad, 'expanded_text_ad'):
            eta = ad.expanded_text_ad
            headlines = [eta.headline_part1, eta.headline_part2, getattr(eta, 'headline_part3', '')]
            descriptions = [eta.description, getattr(eta, 'description2', '')]
        
        # Clean empty strings
        headlines = [h for h in headlines if h]
        descriptions = [d for d in descriptions if d]
        
        return cls(
            id=str(ad.id),
            ad_group_id=ad_group_id,
            status=ad_group_ad.status.name,
            ad_type=ad.type_.name,
            headlines=headlines,
            descriptions=descriptions,
            final_urls=list(ad.final_urls) if ad.final_urls else [],
        )


@dataclass
class GoogleAdsPerformanceData:
    """Google Ads Performance Data model"""
    entity_id: str
    entity_type: str  # campaign, ad_group, keyword, ad
    date: date
    impressions: int
    clicks: int
    conversions: float
    cost_micros: int
    conversions_value_micros: int
    ctr: float
    average_cpc_micros: int
    average_cpm_micros: int
    
    @property
    def cost(self) -> float:
        """Get cost in currency units"""
        return self.cost_micros / 1_000_000
    
    @property
    def conversions_value(self) -> float:
        """Get conversions value in currency units"""
        return self.conversions_value_micros / 1_000_000
    
    @property
    def average_cpc(self) -> float:
        """Get average CPC in currency units"""
        return self.average_cpc_micros / 1_000_000
    
    @property
    def average_cpm(self) -> float:
        """Get average CPM in currency units"""
        return self.average_cpm_micros / 1_000_000
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate"""
        return self.conversions / self.clicks if self.clicks > 0 else 0.0
    
    @property
    def cost_per_conversion(self) -> float:
        """Calculate cost per conversion"""
        return self.cost / self.conversions if self.conversions > 0 else 0.0
    
    @property
    def roas(self) -> float:
        """Calculate return on ad spend"""
        return self.conversions_value / self.cost if self.cost > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'date': self.date.isoformat(),
            'impressions': self.impressions,
            'clicks': self.clicks,
            'conversions': self.conversions,
            'cost': self.cost,
            'revenue': self.conversions_value,
            'ctr': self.ctr,
            'cpc': self.average_cpc,
            'cpm': self.average_cpm,
            'conversion_rate': self.conversion_rate,
            'cost_per_conversion': self.cost_per_conversion,
            'roas': self.roas,
        }
    
    @classmethod
    def from_google_ads_response(
        cls,
        row: Any,
        entity_type: str,
        entity_id: str,
        report_date: date
    ) -> 'GoogleAdsPerformanceData':
        """Create from Google Ads API response"""
        metrics = row.metrics
        
        return cls(
            entity_id=entity_id,
            entity_type=entity_type,
            date=report_date,
            impressions=metrics.impressions,
            clicks=metrics.clicks,
            conversions=metrics.conversions,
            cost_micros=metrics.cost_micros,
            conversions_value_micros=metrics.conversions_value,
            ctr=metrics.ctr,
            average_cpc_micros=metrics.average_cpc,
            average_cpm_micros=metrics.average_cpm,
        )