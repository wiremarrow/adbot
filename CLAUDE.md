# Claude Code Instructions for AdBot

> **Project**: AdBot - AI-Powered Advertising Optimization Platform  
> **Last Updated**: January 2025  
> **Status**: Development phase with working RL environment

## <� Project Overview

AdBot is a reinforcement learning-based advertising optimization platform that uses AI to maximize ROI across multiple advertising platforms (Google Ads, Facebook, TikTok, etc.).

### Current Architecture
- **RL Framework**: Stable Baselines 3 + custom environments
- **Database**: PostgreSQL (port 7500) + TimescaleDB
- **Caching**: Redis (port 7501)
- **API**: FastAPI (will run on port 7507)
- **ML Pipeline**: Custom reward functions + Bayesian optimization

##  Current Working State

### Infrastructure 
- PostgreSQL database with 21 tables (campaigns, performance, ML models, etc.)
- Alembic migration system fully operational
- Docker Compose orchestration with sequential port allocation (7500-7507)
- All port conflicts resolved

### RL Environment 
- `SimpleCampaignEnv` working with Gymnasium interface
- Stable Baselines 3 integration verified (PPO agent training successful)
- Multi-objective reward functions implemented
- Type safety fixes applied for SB3 compatibility

### Files Structure
```
adbot/
   src/core/environments/     # RL environments (working)
      base.py               # Base environment with SB3 fixes
      campaign.py           # Campaign optimization environment
   src/core/reward_functions.py  # Reward engineering (working)
   docs/                     # Comprehensive documentation
   test_rl_training.py      # RL integration tests (all passing)
   test_rl_env.py           # Environment-specific tests
   docker-compose.yml       # Service orchestration
```

## <� Key Implementation Details

### Port Configuration
- **PostgreSQL**: 7500 (was 1030, conflicts resolved)
- **Redis**: 7501 (was 6379, conflicts resolved)
- **MLflow**: 7502
- **Prometheus**: 7503
- **Grafana**: 7504
- **Kafka**: 7505
- **Zookeeper**: 7506
- **AdBot API**: 7507

### Database Schema
21 tables organized into:
- **Campaign Management**: campaigns, ad_groups, ads, keywords
- **Performance Analytics**: performance_metrics, conversion_data, bid_history
- **ML Infrastructure**: agents, training_runs, model_checkpoints
- **Experimentation**: experiments, experiment_results, bandit_arms
- **User Management**: users, accounts, platforms, api_keys

### RL Environment Details
- **Action Space**: Budget multipliers + bid adjustments [0.5, 2.0]
- **Observation Space**: ROI, CTR, cost, budget remaining, day progress
- **Reward Function**: Multi-objective (ROI + efficiency + stability)
- **Episode Length**: Configurable (default 30 days simulation)

## =' Development Commands

### Environment Setup
```bash
conda activate adbot
pip install stable-baselines3  # Required for RL
```

### Database Operations
```bash
# Start services
docker-compose up -d postgres redis

# Run migrations
alembic upgrade head

# Test database connection
python -c "import psycopg2; psycopg2.connect('postgresql://adbot:adbot_password@localhost:7500/adbot_dev'); print(' Connected')"
```

### RL Testing
```bash
# Test RL integration (should show 3 passed, 0 failed)
python test_rl_training.py

# Test specific environment
python test_rl_env.py
```

### Development Workflow
```bash
# Format code
black src/ tests/
ruff check src/ --fix

# Run tests
pytest tests/
```

## =� Known Issues & Solutions

### Fixed Issues 
1. **Port conflicts**: Resolved by moving to 7500-7507 range
2. **Database migrations**: Alembic working correctly
3. **RL integration**: Type conversion fixes applied
4. **Environment variables**: Proper .env setup

### Watch Out For
- Always use `float(reward)` and `bool(terminated)` in RL environments
- PostgreSQL connection string: `postgresql://adbot:adbot_password@localhost:7500/adbot_dev`
- Start Docker services before running any database operations

## <� Next Development Priorities

1. **API Development**: Build FastAPI endpoints for campaign management
2. **Platform Integration**: Complete Google Ads API client implementation
3. **Frontend**: Dashboard for campaign monitoring and RL agent performance
4. **Deployment**: Kubernetes configuration and CI/CD pipeline

## =� Documentation Reference

- **Primary**: `DEVELOPMENT.md` (3000+ lines, comprehensive guide)
- **Quick Ref**: `docs/QUICK_REFERENCE.md` (essential commands)
- **Database**: `docs/DATABASE_AND_PORTS.md` (detailed setup guide)
- **Troubleshooting**: `docs/MIGRATION_TROUBLESHOOTING.md`

## = Code Style & Standards

- **Type hints**: Required for all functions
- **Docstrings**: Google style for all public functions
- **Error handling**: Comprehensive with custom exception classes
- **Testing**: Pytest with >80% coverage target
- **RL Environments**: Must pass `stable_baselines3.common.env_checker.check_env()`

## =� Helpful Context

- The project uses a sophisticated reward engineering system with configurable objectives
- All RL environments inherit from `BaseAdEnvironment` which handles SB3 compatibility
- Database models use SQLAlchemy with Alembic for migrations
- Configuration is managed through Pydantic with fallbacks for development
- The system is designed for production deployment with monitoring and scaling capabilities