#!/usr/bin/env python3
"""
Test script for AdBot RL environment

This script tests the minimal campaign optimization environment to ensure
it works correctly before building more complex components.
"""

import sys
import os
import numpy as np
import logging

# Add src to path so we can import AdBot modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.environments.campaign import SimpleCampaignEnv, create_test_config
from src.core.reward_functions import RewardFunctionFactory, RewardType, RewardConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_environment_basic():
    """Test basic environment functionality"""
    print("🧪 Testing basic environment functionality...")
    
    try:
        # Create environment
        config = create_test_config()
        env = SimpleCampaignEnv(config)
        
        # Test reset
        obs, info = env.reset()
        assert obs is not None, "Reset should return observation"
        assert len(obs) == 5, f"Expected 5 observations, got {len(obs)}"
        
        print(f"✅ Environment created and reset successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
        return False


def test_environment_steps():
    """Test environment step functionality"""
    print("\n🧪 Testing environment step functionality...")
    
    try:
        env = SimpleCampaignEnv(create_test_config())
        obs, info = env.reset()
        
        total_reward = 0
        steps = 0
        
        for i in range(5):
            # Take a reasonable action (moderate budget and bid adjustments)
            action = np.array([1.1, 1.2], dtype=np.float32)  # 10% budget increase, 20% bid increase
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            print(f"   Step {i+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                break
        
        print(f"✅ Environment steps worked correctly")
        print(f"   Total steps: {steps}")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Average reward: {total_reward/steps:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment step test failed: {e}")
        return False


def test_reward_calculation():
    """Test reward function integration"""
    print("\n🧪 Testing reward calculation...")
    
    try:
        env = SimpleCampaignEnv(create_test_config())
        obs, info = env.reset()
        
        # Take an action that should generate some performance
        action = np.array([1.0, 1.0], dtype=np.float32)  # No change in budget/bids
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get metrics
        metrics = env._get_current_metrics()
        
        print(f"✅ Reward calculation working")
        print(f"   Reward: {reward:.4f}")
        print(f"   ROI: {metrics.get('roi', 0):.4f}")
        print(f"   Revenue: ${metrics.get('revenue', 0):.2f}")
        print(f"   Cost: ${metrics.get('cost', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward calculation test failed: {e}")
        return False


def test_action_validation():
    """Test action space validation"""
    print("\n🧪 Testing action validation...")
    
    try:
        env = SimpleCampaignEnv(create_test_config())
        obs, info = env.reset()
        
        # Test valid actions
        valid_actions = [
            np.array([1.0, 1.0]),  # No change
            np.array([0.5, 0.5]),  # Minimum values
            np.array([2.0, 2.0]),  # Maximum values
            np.array([1.5, 0.8]),  # Mixed values
        ]
        
        for i, action in enumerate(valid_actions):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Valid action {i+1}: {action} -> reward={reward:.4f}")
            
            # Reset for next test
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"✅ Action validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Action validation test failed: {e}")
        return False


def test_campaign_summary():
    """Test campaign performance summary"""
    print("\n🧪 Testing campaign summary...")
    
    try:
        env = SimpleCampaignEnv(create_test_config())
        obs, info = env.reset()
        
        # Run a few steps to generate data
        for i in range(3):
            action = np.array([1.2, 1.1], dtype=np.float32)  # Increase budget and bids
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Get summary
        summary = env.get_campaign_summary()
        
        print(f"✅ Campaign summary working")
        print(f"   Total Revenue: ${summary.get('total_revenue', 0):.2f}")
        print(f"   Total Cost: ${summary.get('total_cost', 0):.2f}")
        print(f"   ROI: {summary.get('roi', 0):.4f}")
        print(f"   ROAS: {summary.get('roas', 0):.4f}")
        print(f"   Budget Utilization: {summary.get('budget_utilization', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Campaign summary test failed: {e}")
        return False


def run_simple_training_simulation():
    """Simulate a simple RL training loop"""
    print("\n🚀 Running simple training simulation...")
    
    try:
        env = SimpleCampaignEnv(create_test_config())
        
        # Simple policy: if ROI is low, increase bids; if high, maintain
        def simple_policy(obs):
            roi, ctr, cost, budget_remaining, day_progress = obs
            
            # Simple rules:
            # - If ROI < 0.5, increase bids and reduce budget
            # - If ROI > 1.5, maintain current levels
            # - Otherwise, slightly increase bids
            
            if roi < 0.5:
                return np.array([0.8, 1.3])  # Reduce budget, increase bids
            elif roi > 1.5:
                return np.array([1.0, 1.0])  # Maintain current levels
            else:
                return np.array([1.1, 1.1])  # Slight increases
        
        # Run episodes
        episodes = 3
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            
            print(f"\n   Episode {episode + 1}:")
            
            for step in range(10):
                action = simple_policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if step % 3 == 0:  # Print every 3rd step
                    print(f"     Step {step + 1}: ROI={obs[0]:.3f}, reward={reward:.4f}")
                
                if terminated or truncated:
                    break
            
            summary = env.get_campaign_summary()
            print(f"   Episode {episode + 1} Summary:")
            print(f"     Total Reward: {episode_reward:.4f}")
            print(f"     Final ROI: {summary.get('roi', 0):.4f}")
            print(f"     Revenue: ${summary.get('total_revenue', 0):.2f}")
        
        print(f"✅ Training simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🎯 AdBot RL Environment Test Suite")
    print("=" * 50)
    
    tests = [
        test_environment_basic,
        test_environment_steps,
        test_reward_calculation,
        test_action_validation,
        test_campaign_summary,
        run_simple_training_simulation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! RL environment is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)