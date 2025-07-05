"""
Platform integrations for AdBot
"""

from .google_ads import GoogleAdsClient
from .base import BasePlatformClient

__all__ = ["GoogleAdsClient", "BasePlatformClient"]