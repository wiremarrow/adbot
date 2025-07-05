"""
Google Ads API client implementation
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date, timedelta
import json

try:
    from google.ads.googleads.client import GoogleAdsClient as BaseGoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    GOOGLE_ADS_AVAILABLE = True
except ImportError:
    GOOGLE_ADS_AVAILABLE = False

from ..base import (
    BasePlatformClient,
    PlatformError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    InvalidRequestError,
    ResourceNotFoundError,
)
from .utils import GoogleAdsUtils


class GoogleAdsClient(BasePlatformClient):
    """
    Google Ads API client implementation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google Ads client
        
        Args:
            config: Google Ads configuration containing:
                - developer_token: Developer token
                - client_id: OAuth2 client ID
                - client_secret: OAuth2 client secret
                - refresh_token: OAuth2 refresh token
                - customer_id: Customer ID (optional)
        """
        super().__init__(config, "google_ads")
        
        if not GOOGLE_ADS_AVAILABLE:
            raise ImportError(
                "Google Ads Python client library not installed. "
                "Install with: pip install google-ads"
            )
        
        self.client = None
        self.customer_id = config.get('customer_id')
        self.utils = GoogleAdsUtils()
        
        # Validate required config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate Google Ads configuration"""
        required_fields = [
            'developer_token',
            'client_id', 
            'client_secret',
            'refresh_token'
        ]
        
        missing_fields = [
            field for field in required_fields 
            if not self.config.get(field)
        ]
        
        if missing_fields:
            raise InvalidRequestError(
                f"Missing required config fields: {missing_fields}",
                "google_ads"
            )
    
    async def authenticate(self) -> bool:
        """Authenticate with Google Ads API"""
        try:
            # Create credentials
            credentials = Credentials(
                token=None,
                refresh_token=self.config['refresh_token'],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.config['client_id'],
                client_secret=self.config['client_secret']
            )
            
            # Refresh token if needed
            if not credentials.valid:
                credentials.refresh(Request())
            
            # Initialize Google Ads client
            self.client = BaseGoogleAdsClient(
                credentials=credentials,
                developer_token=self.config['developer_token'],
                version='v15'  # Latest version
            )
            
            self._authenticated = True
            self.log.info("Successfully authenticated with Google Ads API")
            return True
            
        except Exception as e:
            self.log.error("Authentication failed", error=str(e))
            raise AuthenticationError(str(e), "google_ads")
    
    async def test_connection(self) -> bool:
        """Test connection to Google Ads API"""
        try:
            if not self._authenticated:
                await self.authenticate()
            
            # Test with a simple query
            customer_service = self.client.get_service("CustomerService")
            customer_id = self.customer_id or self.config.get('customer_id')
            
            if customer_id:
                # Remove dashes from customer ID if present
                customer_id = customer_id.replace('-', '')
                customer = customer_service.get_customer(
                    customer_id=customer_id
                )
                self.log.info(
                    "Connection test successful",
                    customer_id=customer_id,
                    customer_name=customer.descriptive_name
                )
            else:
                # List accessible customers
                customer_service.list_accessible_customers()
                self.log.info("Connection test successful - can list customers")
            
            return True
            
        except Exception as e:
            self.log.error("Connection test failed", error=str(e))
            return False
    
    async def get_campaigns(
        self,
        account_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get campaigns from Google Ads"""
        try:
            customer_id = account_id or self.customer_id
            if not customer_id:
                raise InvalidRequestError("Customer ID required", "google_ads")
            
            customer_id = customer_id.replace('-', '')
            
            # Build query
            query = """
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign.bidding_strategy_type,
                    campaign_budget.amount_micros,
                    campaign_budget.delivery_method,
                    campaign.start_date,
                    campaign.end_date,
                    campaign.target_cpa.target_cpa_micros,
                    campaign.target_roas.target_roas,
                    campaign.maximize_conversions.target_cpa_micros
                FROM campaign
            """
            
            # Add status filter if provided
            if status_filter:
                status_conditions = [f"campaign.status = '{status.upper()}'" for status in status_filter]
                query += f" WHERE {' OR '.join(status_conditions)}"
            
            ga_service = self.client.get_service("GoogleAdsService")
            
            # Execute query
            response = ga_service.search(
                customer_id=customer_id,
                query=query
            )
            
            campaigns = []
            for row in response:
                campaign = row.campaign
                budget = row.campaign_budget
                
                # Convert to standard format
                campaign_data = {
                    'id': str(campaign.id),
                    'platform_id': str(campaign.id),
                    'name': campaign.name,
                    'status': campaign.status.name.lower(),
                    'channel_type': campaign.advertising_channel_type.name.lower(),
                    'bid_strategy': campaign.bidding_strategy_type.name.lower(),
                    'budget_amount': budget.amount_micros / 1_000_000 if budget.amount_micros else 0,
                    'budget_delivery': budget.delivery_method.name.lower() if budget.delivery_method else 'standard',
                    'start_date': campaign.start_date,
                    'end_date': campaign.end_date,
                    'target_cpa': None,
                    'target_roas': None,
                }
                
                # Extract target CPA/ROAS based on bid strategy
                if hasattr(campaign, 'target_cpa') and campaign.target_cpa.target_cpa_micros:
                    campaign_data['target_cpa'] = campaign.target_cpa.target_cpa_micros / 1_000_000
                elif hasattr(campaign, 'maximize_conversions') and campaign.maximize_conversions.target_cpa_micros:
                    campaign_data['target_cpa'] = campaign.maximize_conversions.target_cpa_micros / 1_000_000
                
                if hasattr(campaign, 'target_roas') and campaign.target_roas.target_roas:
                    campaign_data['target_roas'] = campaign.target_roas.target_roas
                
                campaigns.append(campaign_data)
            
            self.log.info(f"Retrieved {len(campaigns)} campaigns")
            return campaigns
            
        except GoogleAdsException as e:
            self.log.error("Google Ads API error", error=str(e))
            raise self._convert_google_ads_error(e)
        except Exception as e:
            self.log.error("Unexpected error getting campaigns", error=str(e))
            raise PlatformError(str(e), "google_ads")
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new campaign in Google Ads"""
        try:
            customer_id = campaign_data.get('customer_id') or self.customer_id
            if not customer_id:
                raise InvalidRequestError("Customer ID required", "google_ads")
            
            customer_id = customer_id.replace('-', '')
            
            # Create campaign budget first
            budget_operation = self.client.get_type("CampaignBudgetOperation")
            budget = budget_operation.create
            
            budget.name = f"{campaign_data['name']} Budget"
            budget.amount_micros = int(campaign_data['budget_amount'] * 1_000_000)
            budget.delivery_method = self.client.enums.BudgetDeliveryMethodEnum.STANDARD
            
            budget_service = self.client.get_service("CampaignBudgetService")
            budget_response = budget_service.mutate_campaign_budgets(
                customer_id=customer_id,
                operations=[budget_operation]
            )
            
            budget_resource_name = budget_response.results[0].resource_name
            
            # Create campaign
            campaign_operation = self.client.get_type("CampaignOperation")
            campaign = campaign_operation.create
            
            campaign.name = campaign_data['name']
            campaign.advertising_channel_type = getattr(
                self.client.enums.AdvertisingChannelTypeEnum,
                campaign_data.get('channel_type', 'SEARCH').upper()
            )
            campaign.status = getattr(
                self.client.enums.CampaignStatusEnum,
                campaign_data.get('status', 'PAUSED').upper()
            )
            campaign.campaign_budget = budget_resource_name
            
            # Set bid strategy
            bid_strategy = campaign_data.get('bid_strategy', 'manual_cpc')
            if bid_strategy == 'target_cpa':
                campaign.target_cpa.target_cpa_micros = int(
                    campaign_data.get('target_cpa', 50) * 1_000_000
                )
            elif bid_strategy == 'target_roas':
                campaign.target_roas.target_roas = campaign_data.get('target_roas', 3.0)
            elif bid_strategy == 'maximize_conversions':
                if 'target_cpa' in campaign_data:
                    campaign.maximize_conversions.target_cpa_micros = int(
                        campaign_data['target_cpa'] * 1_000_000
                    )
            else:
                # Manual CPC
                campaign.manual_cpc = self.client.get_type("ManualCpc")
            
            # Set dates
            if 'start_date' in campaign_data:
                campaign.start_date = campaign_data['start_date']
            if 'end_date' in campaign_data:
                campaign.end_date = campaign_data['end_date']
            
            campaign_service = self.client.get_service("CampaignService")
            response = campaign_service.mutate_campaigns(
                customer_id=customer_id,
                operations=[campaign_operation]
            )
            
            campaign_id = response.results[0].resource_name.split('/')[-1]
            
            self.log.info(f"Created campaign: {campaign_id}")
            
            return {
                'id': campaign_id,
                'platform_id': campaign_id,
                'name': campaign_data['name'],
                'status': 'created',
                'budget_resource_name': budget_resource_name,
            }
            
        except GoogleAdsException as e:
            self.log.error("Failed to create campaign", error=str(e))
            raise self._convert_google_ads_error(e)
        except Exception as e:
            self.log.error("Unexpected error creating campaign", error=str(e))
            raise PlatformError(str(e), "google_ads")
    
    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing campaign"""
        try:
            customer_id = updates.get('customer_id') or self.customer_id
            if not customer_id:
                raise InvalidRequestError("Customer ID required", "google_ads")
            
            customer_id = customer_id.replace('-', '')
            
            campaign_operation = self.client.get_type("CampaignOperation")
            campaign = campaign_operation.update
            
            campaign.resource_name = self.client.get_service(
                "CampaignService"
            ).campaign_path(customer_id, campaign_id)
            
            # Update fields
            field_mask = []
            
            if 'name' in updates:
                campaign.name = updates['name']
                field_mask.append('name')
            
            if 'status' in updates:
                campaign.status = getattr(
                    self.client.enums.CampaignStatusEnum,
                    updates['status'].upper()
                )
                field_mask.append('status')
            
            if 'target_cpa' in updates:
                campaign.target_cpa.target_cpa_micros = int(
                    updates['target_cpa'] * 1_000_000
                )
                field_mask.append('target_cpa.target_cpa_micros')
            
            if 'target_roas' in updates:
                campaign.target_roas.target_roas = updates['target_roas']
                field_mask.append('target_roas.target_roas')
            
            campaign_operation.update_mask.CopyFrom(
                self.client.get_type("FieldMask", paths=field_mask)
            )
            
            campaign_service = self.client.get_service("CampaignService")
            response = campaign_service.mutate_campaigns(
                customer_id=customer_id,
                operations=[campaign_operation]
            )
            
            self.log.info(f"Updated campaign: {campaign_id}")
            
            return {
                'id': campaign_id,
                'platform_id': campaign_id,
                'updated_fields': field_mask,
                'status': 'updated',
            }
            
        except GoogleAdsException as e:
            self.log.error("Failed to update campaign", error=str(e))
            raise self._convert_google_ads_error(e)
        except Exception as e:
            self.log.error("Unexpected error updating campaign", error=str(e))
            raise PlatformError(str(e), "google_ads")
    
    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause campaign"""
        try:
            await self.update_campaign(campaign_id, {'status': 'paused'})
            return True
        except Exception:
            return False
    
    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume campaign"""
        try:
            await self.update_campaign(campaign_id, {'status': 'enabled'})
            return True
        except Exception:
            return False
    
    async def update_campaign_budget(
        self,
        campaign_id: str,
        budget_amount: float,
        budget_type: str = "daily"
    ) -> Dict[str, Any]:
        """Update campaign budget"""
        try:
            customer_id = self.customer_id
            if not customer_id:
                raise InvalidRequestError("Customer ID required", "google_ads")
            
            customer_id = customer_id.replace('-', '')
            
            # First get the campaign's budget resource name
            query = f"""
                SELECT campaign_budget.resource_name
                FROM campaign
                WHERE campaign.id = {campaign_id}
            """
            
            ga_service = self.client.get_service("GoogleAdsService")
            response = ga_service.search(
                customer_id=customer_id,
                query=query
            )
            
            budget_resource_name = None
            for row in response:
                budget_resource_name = row.campaign_budget.resource_name
                break
            
            if not budget_resource_name:
                raise ResourceNotFoundError(f"Budget not found for campaign {campaign_id}", "google_ads")
            
            # Update budget
            budget_operation = self.client.get_type("CampaignBudgetOperation")
            budget = budget_operation.update
            
            budget.resource_name = budget_resource_name
            budget.amount_micros = int(budget_amount * 1_000_000)
            
            budget_operation.update_mask.CopyFrom(
                self.client.get_type("FieldMask", paths=['amount_micros'])
            )
            
            budget_service = self.client.get_service("CampaignBudgetService")
            budget_service.mutate_campaign_budgets(
                customer_id=customer_id,
                operations=[budget_operation]
            )
            
            self.log.info(f"Updated budget for campaign {campaign_id}: ${budget_amount}")
            
            return {
                'campaign_id': campaign_id,
                'budget_amount': budget_amount,
                'budget_type': budget_type,
                'status': 'updated',
            }
            
        except GoogleAdsException as e:
            self.log.error("Failed to update campaign budget", error=str(e))
            raise self._convert_google_ads_error(e)
        except Exception as e:
            self.log.error("Unexpected error updating budget", error=str(e))
            raise PlatformError(str(e), "google_ads")
    
    def _convert_google_ads_error(self, error: GoogleAdsException) -> PlatformError:
        """Convert Google Ads exception to platform error"""
        error_code = None
        error_message = str(error)
        
        if hasattr(error, 'error') and hasattr(error.error, 'code'):
            error_code = error.error.code
        
        # Check for specific error types
        if 'AUTHENTICATION_ERROR' in error_message:
            return AuthenticationError(error_message, "google_ads", error_code)
        elif 'RATE_EXCEEDED' in error_message:
            return RateLimitError(error_message, "google_ads", error_code)
        elif 'QUOTA_EXCEEDED' in error_message:
            return QuotaExceededError(error_message, "google_ads", error_code)
        elif 'INVALID_ARGUMENT' in error_message:
            return InvalidRequestError(error_message, "google_ads", error_code)
        elif 'NOT_FOUND' in error_message:
            return ResourceNotFoundError(error_message, "google_ads", error_code)
        else:
            return PlatformError(error_message, "google_ads", error_code)
    
    def format_error(self, error: Exception) -> Dict[str, Any]:
        """Format Google Ads error for logging"""
        return {
            'platform': 'google_ads',
            'error_type': type(error).__name__,
            'message': str(error),
            'error_code': getattr(error, 'error_code', None),
        }
    
    def validate_config(self) -> bool:
        """Validate Google Ads configuration"""
        try:
            self._validate_config()
            return True
        except Exception:
            return False
    
    # Placeholder implementations for remaining abstract methods
    # These would be implemented similarly to the above methods
    
    async def get_ad_groups(
        self,
        campaign_id: str,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get ad groups from campaign - placeholder"""
        # Implementation would be similar to get_campaigns
        return []
    
    async def create_ad_group(self, ad_group_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new ad group - placeholder"""
        return {}
    
    async def update_ad_group(
        self,
        ad_group_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing ad group - placeholder"""
        return {}
    
    async def get_keywords(
        self,
        ad_group_id: str,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get keywords from ad group - placeholder"""
        return []
    
    async def create_keywords(
        self,
        ad_group_id: str,
        keywords: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create new keywords - placeholder"""
        return []
    
    async def update_keyword_bids(
        self,
        keyword_updates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update keyword bids - placeholder"""
        return []
    
    async def get_campaign_performance(
        self,
        campaign_ids: List[str],
        start_date: date,
        end_date: date,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get campaign performance data - placeholder"""
        return []
    
    async def get_keyword_performance(
        self,
        keyword_ids: List[str],
        start_date: date,
        end_date: date,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get keyword performance data - placeholder"""
        return []
    
    async def get_budget_recommendations(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        """Get budget recommendations - placeholder"""
        return {}
    
    async def update_bid_strategy(
        self,
        campaign_id: str,
        bid_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update campaign bid strategy - placeholder"""
        return {}
    
    async def get_bid_recommendations(
        self,
        keyword_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get bid recommendations - placeholder"""
        return []
    
    async def get_audience_targets(
        self,
        campaign_id: str
    ) -> List[Dict[str, Any]]:
        """Get audience targeting - placeholder"""
        return []
    
    async def update_audience_targets(
        self,
        campaign_id: str,
        audiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update audience targeting - placeholder"""
        return []
    
    async def get_geo_targets(
        self,
        campaign_id: str
    ) -> List[Dict[str, Any]]:
        """Get geographic targeting - placeholder"""
        return []
    
    async def update_geo_targets(
        self,
        campaign_id: str,
        geo_targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update geographic targeting - placeholder"""
        return []
    
    async def get_account_info(self, account_id: str) -> Dict[str, Any]:
        """Get account information - placeholder"""
        return {}
    
    async def get_account_performance(
        self,
        account_id: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Get account-level performance - placeholder"""
        return {}
    
    async def get_conversions(
        self,
        account_id: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get conversion data - placeholder"""
        return []
    
    async def create_conversion_action(
        self,
        account_id: str,
        conversion_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create conversion action - placeholder"""
        return {}