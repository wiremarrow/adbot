"""
AdBot RL Environment Package

This package contains all reinforcement learning environments for AdBot:
- BaseAdEnvironment: Foundation class with Gymnasium interface  
- SimpleCampaignEnv: Working simplified campaign environment (SB3 compatible)
- CampaignOptimizationEnv: Advanced campaign management (planned)
- BudgetAllocationEnv: Dynamic budget distribution (planned)
- BidOptimizationEnv: Real-time keyword bidding (planned)
- MultiPlatformEnv: Cross-platform coordination (planned)

Note: Currently using SimpleCampaignEnv as the primary working environment.
Advanced environments are placeholders for future development.
"""

from .base import BaseAdEnvironment
from .campaign import SimpleCampaignEnv, CampaignOptimizationEnv

# For now, use SimpleCampaignEnv for all environments until advanced ones are tested
BudgetAllocationEnv = SimpleCampaignEnv
BidOptimizationEnv = SimpleCampaignEnv  
MultiPlatformEnv = SimpleCampaignEnv

__all__ = [
    'BaseAdEnvironment',
    'SimpleCampaignEnv',
    'CampaignOptimizationEnv',
    'BudgetAllocationEnv', 
    'BidOptimizationEnv',
    'MultiPlatformEnv'
]