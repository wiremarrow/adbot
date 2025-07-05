#!/usr/bin/env python3
"""
Test RL training with Stable Baselines 3

This script tests that our AdBot RL environment works with actual RL algorithms
from Stable Baselines 3, proving the environment is production-ready.
"""

import sys
import os
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.environments.campaign import SimpleCampaignEnv, create_test_config

# Suppress gymnasium warnings for cleaner output
logging.getLogger('gymnasium').setLevel(logging.ERROR)

def test_stable_baselines3_integration():
    """Test that our environment works with Stable Baselines 3"""
    print("🤖 Testing Stable Baselines 3 Integration...")
    
    try:
        # Try to import SB3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        
        print("✅ Stable Baselines 3 imported successfully")
        
        # Create environment
        env = SimpleCampaignEnv(create_test_config())
        
        # Check environment compatibility
        print("🔍 Checking environment compatibility with SB3...")
        check_env(env, warn=True)
        print("✅ Environment passed SB3 compatibility check")
        
        # Create and train a simple PPO agent
        print("🚀 Training PPO agent for 1000 steps...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0,  # Reduce output
            learning_rate=0.001,
            n_steps=128,
            batch_size=32,
            n_epochs=3
        )
        
        # Train for a short time
        model.learn(total_timesteps=1000)
        print("✅ PPO training completed successfully")
        
        # Test the trained agent
        print("🎯 Testing trained agent...")
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(5):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"✅ Trained agent test completed")
        print(f"   Average reward: {total_reward / (step + 1):.4f}")
        
        # Get campaign performance
        summary = env.get_campaign_summary()
        print(f"   Final ROI: {summary.get('roi', 0):.4f}")
        print(f"   Revenue: ${summary.get('total_revenue', 0):.2f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Stable Baselines 3 not installed: {e}")
        print("   Install with: pip install stable-baselines3")
        return False
        
    except Exception as e:
        print(f"❌ SB3 integration test failed: {e}")
        return False


def test_custom_policy():
    """Test a simple custom policy to show the environment flexibility"""
    print("\n📊 Testing custom policy implementation...")
    
    try:
        env = SimpleCampaignEnv(create_test_config())
        
        class AdaptivePolicy:
            """Simple adaptive policy that adjusts based on performance"""
            
            def __init__(self):
                self.performance_history = []
            
            def predict(self, obs):
                roi, ctr, cost, budget_remaining, day_progress = obs
                
                # Track performance
                self.performance_history.append(roi)
                
                # Adaptive strategy based on ROI
                if roi < 0.5:
                    # Poor performance: reduce budget, increase bids
                    budget_mult = 0.7
                    bid_mult = 1.4
                elif roi > 2.0:
                    # Excellent performance: increase budget, maintain bids
                    budget_mult = 1.3
                    bid_mult = 1.1
                elif roi > 1.5:
                    # Good performance: slight increases
                    budget_mult = 1.1
                    bid_mult = 1.05
                else:
                    # Average performance: modest adjustments
                    budget_mult = 1.0
                    bid_mult = 1.2
                
                # Consider budget remaining
                if budget_remaining < 200:  # Low budget remaining
                    budget_mult *= 0.8  # Be more conservative
                
                return np.array([budget_mult, bid_mult], dtype=np.float32)
        
        # Test the custom policy
        policy = AdaptivePolicy()
        obs, info = env.reset()
        total_reward = 0
        
        print("   Running custom policy for 8 steps...")
        for step in range(8):
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 3 == 0:
                print(f"     Step {step + 1}: ROI={obs[0]:.3f}, action={action}, reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        summary = env.get_campaign_summary()
        print(f"✅ Custom policy test completed")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Final ROI: {summary.get('roi', 0):.4f}")
        print(f"   Budget utilization: {summary.get('budget_utilization', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom policy test failed: {e}")
        return False


def benchmark_environment():
    """Benchmark environment performance"""
    print("\n⚡ Benchmarking environment performance...")
    
    try:
        import time
        
        env = SimpleCampaignEnv(create_test_config())
        
        # Benchmark reset
        start_time = time.time()
        for _ in range(100):
            env.reset()
        reset_time = (time.time() - start_time) / 100
        
        # Benchmark steps
        env.reset()
        start_time = time.time()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
        step_time = (time.time() - start_time) / 1000
        
        print(f"✅ Environment performance:")
        print(f"   Reset time: {reset_time*1000:.2f}ms")
        print(f"   Step time: {step_time*1000:.2f}ms")
        print(f"   Steps per second: {1/step_time:.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def main():
    """Run all RL integration tests"""
    print("🤖 AdBot RL Training Integration Test")
    print("=" * 50)
    
    tests = [
        test_stable_baselines3_integration,
        test_custom_policy,
        benchmark_environment
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
    print(f"🏁 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Environment is RL-ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)