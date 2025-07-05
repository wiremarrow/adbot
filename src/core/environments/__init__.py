"""
AdBot RL Environment Package

This package contains all reinforcement learning environments for AdBot:
- BaseAdEnvironment: Foundation class with Gymnasium interface
- CampaignOptimizationEnv: High-level campaign management
- BudgetAllocationEnv: Dynamic budget distribution
- BidOptimizationEnv: Real-time keyword bidding
- MultiPlatformEnv: Cross-platform coordination
"""

from .base import BaseAdEnvironment
from .campaign import CampaignOptimizationEnv
from .budget import BudgetAllocationEnv
from .bidding import BidOptimizationEnv
from .multi_platform import MultiPlatformEnv

__all__ = [
    'BaseAdEnvironment',
    'CampaignOptimizationEnv',
    'BudgetAllocationEnv',
    'BidOptimizationEnv',
    'MultiPlatformEnv'
]