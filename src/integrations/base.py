"""
Base platform client for all advertising platform integrations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date

from ..utils.logger import get_logger


class BasePlatformClient(ABC):
    """
    Abstract base class for all platform clients
    
    Provides common interface for all advertising platforms
    """
    
    def __init__(self, config: Dict[str, Any], platform_name: str):
        """
        Initialize base platform client
        
        Args:
            config: Platform configuration
            platform_name: Name of the platform
        """
        self.config = config
        self.platform_name = platform_name
        self.log = get_logger(f"adbot.integration.{platform_name}")
        self._authenticated = False
        
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the platform"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to platform"""
        pass
    
    # Campaign Management
    @abstractmethod
    async def get_campaigns(
        self,
        account_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get campaigns from platform"""
        pass
    
    @abstractmethod
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new campaign"""
        pass
    
    @abstractmethod
    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing campaign"""
        pass
    
    @abstractmethod
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause campaign"""
        pass
    
    @abstractmethod
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume campaign"""
        pass
    
    # Ad Group Management
    @abstractmethod
    async def get_ad_groups(
        self,
        campaign_id: str,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get ad groups from campaign"""
        pass
    
    @abstractmethod
    async def create_ad_group(self, ad_group_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new ad group"""
        pass
    
    @abstractmethod
    async def update_ad_group(
        self,
        ad_group_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing ad group"""
        pass
    
    # Keyword Management
    @abstractmethod
    async def get_keywords(
        self,
        ad_group_id: str,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get keywords from ad group"""
        pass
    
    @abstractmethod
    async def create_keywords(
        self,
        ad_group_id: str,
        keywords: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create new keywords"""
        pass
    
    @abstractmethod
    async def update_keyword_bids(
        self,
        keyword_updates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update keyword bids"""
        pass
    
    # Performance Data
    @abstractmethod
    async def get_campaign_performance(
        self,
        campaign_ids: List[str],
        start_date: date,
        end_date: date,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get campaign performance data"""
        pass
    
    @abstractmethod
    async def get_keyword_performance(
        self,
        keyword_ids: List[str],
        start_date: date,
        end_date: date,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get keyword performance data"""
        pass
    
    # Budget Management
    @abstractmethod
    async def update_campaign_budget(
        self,
        campaign_id: str,
        budget_amount: float,
        budget_type: str = "daily"
    ) -> Dict[str, Any]:
        """Update campaign budget"""
        pass
    
    @abstractmethod
    async def get_budget_recommendations(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        """Get budget recommendations"""
        pass
    
    # Bid Management
    @abstractmethod
    async def update_bid_strategy(
        self,
        campaign_id: str,
        bid_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update campaign bid strategy"""
        pass
    
    @abstractmethod
    async def get_bid_recommendations(
        self,
        keyword_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get bid recommendations"""
        pass
    
    # Audience and Targeting
    @abstractmethod
    async def get_audience_targets(
        self,
        campaign_id: str
    ) -> List[Dict[str, Any]]:
        """Get audience targeting"""
        pass
    
    @abstractmethod
    async def update_audience_targets(
        self,
        campaign_id: str,
        audiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update audience targeting"""
        pass
    
    # Geographic Targeting
    @abstractmethod
    async def get_geo_targets(
        self,
        campaign_id: str
    ) -> List[Dict[str, Any]]:
        """Get geographic targeting"""
        pass
    
    @abstractmethod
    async def update_geo_targets(
        self,
        campaign_id: str,
        geo_targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update geographic targeting"""
        pass
    
    # Account Information
    @abstractmethod
    async def get_account_info(self, account_id: str) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_account_performance(
        self,
        account_id: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Get account-level performance"""
        pass
    
    # Conversion Tracking
    @abstractmethod
    async def get_conversions(
        self,
        account_id: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get conversion data"""
        pass
    
    @abstractmethod
    async def create_conversion_action(
        self,
        account_id: str,
        conversion_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create conversion action"""
        pass
    
    # Utility Methods
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self._authenticated
    
    def get_platform_name(self) -> str:
        """Get platform name"""
        return self.platform_name
    
    @abstractmethod
    def format_error(self, error: Exception) -> Dict[str, Any]:
        """Format platform-specific error"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate platform configuration"""
        pass


class PlatformError(Exception):
    """Base exception for platform errors"""
    
    def __init__(self, message: str, platform: str, error_code: Optional[str] = None):
        self.message = message
        self.platform = platform
        self.error_code = error_code
        super().__init__(f"[{platform}] {message}")


class AuthenticationError(PlatformError):
    """Authentication failed"""
    pass


class RateLimitError(PlatformError):
    """Rate limit exceeded"""
    pass


class QuotaExceededError(PlatformError):
    """API quota exceeded"""
    pass


class InvalidRequestError(PlatformError):
    """Invalid request parameters"""
    pass


class ResourceNotFoundError(PlatformError):
    """Requested resource not found"""
    pass