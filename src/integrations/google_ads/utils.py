"""
Google Ads utility functions
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, date


class GoogleAdsUtils:
    """Utility functions for Google Ads integration"""
    
    @staticmethod
    def format_customer_id(customer_id: str) -> str:
        """Format customer ID by removing dashes"""
        return customer_id.replace('-', '') if customer_id else ''
    
    @staticmethod
    def add_customer_id_dashes(customer_id: str) -> str:
        """Add dashes to customer ID for display"""
        if not customer_id or len(customer_id) != 10:
            return customer_id
        
        return f"{customer_id[:3]}-{customer_id[3:6]}-{customer_id[6:]}"
    
    @staticmethod
    def micros_to_currency(micros: int) -> float:
        """Convert micros to currency amount"""
        return micros / 1_000_000 if micros else 0.0
    
    @staticmethod
    def currency_to_micros(amount: float) -> int:
        """Convert currency amount to micros"""
        return int(amount * 1_000_000) if amount else 0
    
    @staticmethod
    def format_date_range(start_date: date, end_date: date) -> str:
        """Format date range for Google Ads queries"""
        return f"segments.date BETWEEN '{start_date}' AND '{end_date}'"
    
    @staticmethod
    def parse_resource_name(resource_name: str) -> Dict[str, str]:
        """Parse Google Ads resource name into components"""
        # Example: customers/1234567890/campaigns/987654321
        parts = resource_name.split('/')
        
        result = {}
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                result[parts[i]] = parts[i + 1]
        
        return result
    
    @staticmethod
    def build_campaign_resource_name(customer_id: str, campaign_id: str) -> str:
        """Build campaign resource name"""
        customer_id = GoogleAdsUtils.format_customer_id(customer_id)
        return f"customers/{customer_id}/campaigns/{campaign_id}"
    
    @staticmethod
    def build_ad_group_resource_name(customer_id: str, ad_group_id: str) -> str:
        """Build ad group resource name"""
        customer_id = GoogleAdsUtils.format_customer_id(customer_id)
        return f"customers/{customer_id}/adGroups/{ad_group_id}"
    
    @staticmethod
    def build_keyword_resource_name(customer_id: str, keyword_id: str) -> str:
        """Build keyword resource name"""
        customer_id = GoogleAdsUtils.format_customer_id(customer_id)
        return f"customers/{customer_id}/adGroupCriteria/{keyword_id}"
    
    @staticmethod
    def normalize_status(status: str) -> str:
        """Normalize Google Ads status to standard format"""
        status_mapping = {
            'ENABLED': 'active',
            'PAUSED': 'paused',
            'REMOVED': 'removed',
            'PENDING': 'pending',
            'UNKNOWN': 'unknown',
        }
        return status_mapping.get(status.upper(), status.lower())
    
    @staticmethod
    def google_ads_status(status: str) -> str:
        """Convert standard status to Google Ads format"""
        status_mapping = {
            'active': 'ENABLED',
            'paused': 'PAUSED',
            'removed': 'REMOVED',
            'pending': 'PENDING',
        }
        return status_mapping.get(status.lower(), status.upper())
    
    @staticmethod
    def normalize_match_type(match_type: str) -> str:
        """Normalize Google Ads match type"""
        match_type_mapping = {
            'EXACT': 'exact',
            'PHRASE': 'phrase', 
            'BROAD': 'broad',
            'BROAD_MATCH_MODIFIER': 'broad_match_modifier',
        }
        return match_type_mapping.get(match_type.upper(), match_type.lower())
    
    @staticmethod
    def google_ads_match_type(match_type: str) -> str:
        """Convert standard match type to Google Ads format"""
        match_type_mapping = {
            'exact': 'EXACT',
            'phrase': 'PHRASE',
            'broad': 'BROAD',
            'broad_match_modifier': 'BROAD_MATCH_MODIFIER',
        }
        return match_type_mapping.get(match_type.lower(), match_type.upper())
    
    @staticmethod
    def build_gaql_query(
        resource: str,
        fields: List[str],
        conditions: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """Build Google Ads Query Language (GAQL) query"""
        query = f"SELECT {', '.join(fields)} FROM {resource}"
        
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    @staticmethod
    def extract_id_from_resource_name(resource_name: str) -> str:
        """Extract ID from resource name"""
        # Resource names are in format: customers/123/campaigns/456
        # Return the last part (456)
        return resource_name.split('/')[-1] if resource_name else ''
    
    @staticmethod
    def get_standard_metrics() -> List[str]:
        """Get list of standard performance metrics"""
        return [
            'metrics.impressions',
            'metrics.clicks',
            'metrics.conversions',
            'metrics.cost_micros',
            'metrics.ctr',
            'metrics.average_cpc',
            'metrics.average_cpm',
            'metrics.conversions_value',
            'metrics.cost_per_conversion',
            'metrics.value_per_conversion',
        ]
    
    @staticmethod
    def get_campaign_fields() -> List[str]:
        """Get standard campaign fields for queries"""
        return [
            'campaign.id',
            'campaign.name',
            'campaign.status',
            'campaign.advertising_channel_type',
            'campaign.bidding_strategy_type',
            'campaign_budget.amount_micros',
            'campaign.start_date',
            'campaign.end_date',
        ]
    
    @staticmethod
    def get_ad_group_fields() -> List[str]:
        """Get standard ad group fields for queries"""
        return [
            'ad_group.id',
            'ad_group.name', 
            'ad_group.status',
            'ad_group.campaign',
            'ad_group.cpc_bid_micros',
            'ad_group.target_cpa_micros',
        ]
    
    @staticmethod
    def get_keyword_fields() -> List[str]:
        """Get standard keyword fields for queries"""
        return [
            'ad_group_criterion.criterion_id',
            'ad_group_criterion.keyword.text',
            'ad_group_criterion.keyword.match_type',
            'ad_group_criterion.status',
            'ad_group_criterion.cpc_bid_micros',
            'ad_group_criterion.quality_info.quality_score',
        ]
    
    @staticmethod
    def validate_customer_id(customer_id: str) -> bool:
        """Validate Google Ads customer ID format"""
        if not customer_id:
            return False
        
        # Remove dashes and check if it's 10 digits
        clean_id = customer_id.replace('-', '')
        return len(clean_id) == 10 and clean_id.isdigit()
    
    @staticmethod
    def format_performance_data(row: Any) -> Dict[str, Any]:
        """Format Google Ads API response row to standard performance format"""
        metrics = row.metrics
        
        return {
            'impressions': metrics.impressions,
            'clicks': metrics.clicks,
            'conversions': metrics.conversions,
            'cost': GoogleAdsUtils.micros_to_currency(metrics.cost_micros),
            'revenue': GoogleAdsUtils.micros_to_currency(metrics.conversions_value),
            'ctr': metrics.ctr,
            'cpc': GoogleAdsUtils.micros_to_currency(metrics.average_cpc),
            'cpm': GoogleAdsUtils.micros_to_currency(metrics.average_cpm),
            'cpa': GoogleAdsUtils.micros_to_currency(metrics.cost_per_conversion),
            'roas': metrics.value_per_conversion / max(metrics.cost_per_conversion, 1) if metrics.cost_per_conversion else 0,
        }
    
    @staticmethod
    def create_field_mask(fields: List[str]) -> str:
        """Create field mask for update operations"""
        return ','.join(fields)
    
    @staticmethod
    def batch_operations(operations: List[Any], batch_size: int = 100) -> List[List[Any]]:
        """Batch operations for API calls"""
        batches = []
        for i in range(0, len(operations), batch_size):
            batches.append(operations[i:i + batch_size])
        return batches