# AdBot Development Guide

> **AdBot**: AI-Powered Advertising Optimization Platform using Reinforcement Learning

This comprehensive guide covers all aspects of AdBot development, from initial setup to advanced deployment procedures.

## 📋 Table of Contents

1. [Project Architecture](#-project-architecture)
2. [Development Environment Setup](#-development-environment-setup)
3. [Database Management](#-database-management)
4. [RL Environment Framework](#-rl-environment-framework)
5. [Platform Integrations](#-platform-integrations)
6. [API Development](#-api-development)
7. [Testing Strategy](#-testing-strategy)
8. [Deployment Procedures](#-deployment-procedures)
9. [Code Quality & Standards](#-code-quality--standards)
10. [Troubleshooting Guide](#-troubleshooting-guide)
11. [Contributing Guidelines](#-contributing-guidelines)
12. [Advanced Topics](#-advanced-topics)

---

## 🔌 Port Configuration

AdBot uses a dedicated sequential port range (7500-7507) to avoid conflicts:

| Service | Host Port | Container Port | Purpose |
|---------|-----------|----------------|---------|
| PostgreSQL | 7500 | 5432 | Primary database |
| Redis | 7501 | 6379 | Caching & sessions |
| MLflow | 7502 | 5000 | ML experiment tracking |
| Prometheus | 7503 | 9090 | Metrics collection |
| Grafana | 7504 | 3000 | Metrics visualization |
| Kafka | 7505 | 9092 | Event streaming |
| Zookeeper | 7506 | 2181 | Kafka coordination |
| AdBot API | 7507 | 8080 | Main REST API |

> **Note**: These ports were specifically chosen to avoid conflicts with common development services. If you need to change them, update: `.env`, `docker-compose.yml`, `configs/default.yaml`, and `src/utils/config.py`.

---
## 🏗️ Project Architecture

### System Overview

AdBot is a sophisticated AI platform that uses reinforcement learning to optimize advertising campaigns across multiple platforms in real-time.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RL Agents     │    │   FastAPI App    │    │   Platform APIs │
│                 │    │                  │    │                 │
│ • PPO/SAC/TD3   │◄──►│ • Campaign CRUD  │◄──►│ • Google Ads    │
│ • Multi-Platform│    │ • Agent Training │    │ • Facebook      │
│ • Budget Optim  │    │ • Experiments    │    │ • TikTok        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │   PostgreSQL     │
                    │                  │
                    │ • Campaigns      │
                    │ • Performance    │
                    │ • Experiments    │
                    │ • ML Models      │
                    └──────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML/RL** | Stable Baselines 3, D3RLpy, PyTorch | Reinforcement learning algorithms |
| **API** | FastAPI, Pydantic, Uvicorn | REST API and validation |
| **Database** | PostgreSQL, SQLAlchemy, Alembic | Data persistence and migrations |
| **Caching** | Redis | Session storage and caching |
| **Platform APIs** | Google Ads, Facebook Business SDK | Advertising platform integration |
| **Monitoring** | Prometheus, Grafana, MLflow | Metrics and ML experiment tracking |
| **Deployment** | Docker, Kubernetes | Container orchestration |

### Core Components

#### 1. **RL Environment Framework** (`src/core/environments/`)
- **BaseAdEnvironment**: Foundation class with Gymnasium interface
- **CampaignOptimizationEnv**: High-level campaign management
- **BudgetAllocationEnv**: Dynamic budget distribution
- **BidOptimizationEnv**: Real-time keyword bidding
- **MultiPlatformEnv**: Cross-platform coordination

#### 2. **Platform Integrations** (`src/integrations/`)
- **BasePlatformClient**: Abstract interface for all platforms
- **GoogleAdsClient**: Complete Google Ads API implementation
- **FacebookClient**: Facebook Business API integration (planned)
- **TikTokClient**: TikTok Ads API integration (planned)

#### 3. **Database Models** (`src/models/`)
- **Campaign Management**: Campaigns, Ad Groups, Ads, Keywords
- **Performance Tracking**: Metrics, Conversions, Bid History
- **User Management**: Users, Accounts, Platforms, API Keys
- **Experimentation**: A/B Tests, Bandit Arms, Results
- **ML/RL**: Agents, Training Runs, Model Checkpoints

#### 4. **API Layer** (`src/api/`)
- **FastAPI Application**: Production-ready API with middleware
- **Authentication**: JWT tokens, API keys, rate limiting
- **Error Handling**: Comprehensive exception mapping
- **Validation**: Pydantic models for type safety

---

## 🚀 Development Environment Setup

### Prerequisites

Ensure you have the following installed:

- **Python 3.11+** (AdBot uses modern Python features)
- **PostgreSQL 14+** (Database with JSON support)
- **Redis 6+** (Caching and session storage)
- **Docker & Docker Compose** (For containerized development)
- **Git** (Version control)

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/wiremarrow/adbot.git
   cd adbot
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your configurations
   nano .env
   ```

3. **Database Setup**
   ```bash
   # Start all services using Docker Compose
   docker-compose up -d postgres redis
   
   # Services will be available on:
   # PostgreSQL: localhost:7500
   # Redis: localhost:7501
   
   # Initialize database with migrations
   alembic upgrade head
   python scripts/init_db.py --sample-data
   ```

4. **Start Development Server**
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
   ```

### Development with Docker Compose

For a complete development environment:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f adbot-api

# Stop services
docker-compose down
```

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_ENV` | Environment (development/production) | development | No |
| `APP_DEBUG` | Debug mode | true | No |
| `DB_HOST` | Database host | localhost | Yes |
| `DB_PORT` | Database port | 7500 | Yes |
| `DB_NAME` | Database name | adbot_dev | Yes |
| `DB_USER` | Database user | adbot | Yes |
| `DB_PASSWORD` | Database password | - | Yes |
| `REDIS_HOST` | Redis host | localhost | No |
| `REDIS_PORT` | Redis port | 7501 | No |
| `GOOGLE_ADS_DEV_TOKEN` | Google Ads developer token | - | For Google Ads |
| `GOOGLE_ADS_CLIENT_ID` | OAuth2 client ID | - | For Google Ads |
| `GOOGLE_ADS_CLIENT_SECRET` | OAuth2 client secret | - | For Google Ads |
| `GOOGLE_ADS_REFRESH_TOKEN` | OAuth2 refresh token | - | For Google Ads |
| `JWT_SECRET` | JWT signing secret | - | Yes |

---

## 🗄️ Database Management

### Migration System with Alembic

AdBot uses Alembic for database schema management, providing version control for database changes.

#### Basic Migration Commands

```bash
# Create a new migration
alembic revision --autogenerate -m "Add new campaign fields"

# Apply migrations
alembic upgrade head

# Downgrade to previous version
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history
```

#### Migration Workflow

1. **Make Model Changes**
   ```python
   # src/models/campaign.py
   class Campaign(BaseModel):
       # ... existing fields
       new_field = Column(String(100), nullable=True)  # Add new field
   ```

2. **Generate Migration**
   ```bash
   alembic revision --autogenerate -m "Add new_field to campaigns"
   ```

3. **Review Generated Migration**
   ```python
   # migrations/versions/001_add_new_field_to_campaigns.py
   def upgrade() -> None:
       op.add_column('campaigns', sa.Column('new_field', sa.String(100), nullable=True))
   
   def downgrade() -> None:
       op.drop_column('campaigns', 'new_field')
   ```

4. **Apply Migration**
   ```bash
   alembic upgrade head
   ```

### Database Schema Overview

#### Core Tables

| Table | Purpose | Key Relationships |
|-------|---------|-------------------|
| `users` | User authentication and profiles | → accounts |
| `accounts` | Organization/client accounts | → platforms, campaigns |
| `platforms` | Advertising platform connections | → campaigns |
| `campaigns` | Advertising campaigns | → ad_groups, performance_metrics |
| `ad_groups` | Campaign subdivisions | → ads, keywords |
| `ads` | Creative content | → performance_metrics |
| `keywords` | Targeting keywords | → performance_metrics |
| `performance_metrics` | Daily performance data | ← campaigns, ads, keywords |
| `experiments` | A/B tests and experiments | → experiment_results |
| `agents` | RL agent configurations | → training_runs |
| `training_runs` | ML training sessions | → model_checkpoints |

#### Database Initialization

**Fresh Installation:**
```bash
# Create schema and sample data
python scripts/init_db.py --sample-data

# Or with Docker
docker-compose exec adbot-api python scripts/init_db.py --sample-data
```

**Reset Database (Development Only):**
```bash
# WARNING: This destroys all data!
python scripts/init_db.py --drop-existing --sample-data
```

#### Sample Data Structure

The sample data includes:
- **Admin User**: `admin@adbot.ai` with full permissions
- **Sample Account**: Connected to Google Ads platform
- **Demo Campaign**: Active campaign with performance data
- **RL Agent**: Pre-configured PPO agent for testing

### Performance Considerations

#### Indexes

Key indexes for performance:
```sql
-- Campaign queries
CREATE INDEX idx_campaign_platform_id ON campaigns(platform, platform_id);
CREATE INDEX idx_campaign_account_id ON campaigns(account_id);
CREATE INDEX idx_campaign_status ON campaigns(status);

-- Performance metrics
CREATE INDEX idx_performance_entity ON performance_metrics(entity_type, entity_id);
CREATE INDEX idx_performance_date ON performance_metrics(date);
CREATE INDEX idx_performance_campaign_date ON performance_metrics(campaign_id, date);

-- Experiments
CREATE INDEX idx_experiment_status ON experiments(status);
CREATE INDEX idx_experiment_dates ON experiments(start_date, end_date);
```

#### Query Optimization

**Efficient Performance Queries:**
```python
# Good: Use indexes
query = session.query(PerformanceMetric)\
    .filter(PerformanceMetric.campaign_id == campaign_id)\
    .filter(PerformanceMetric.date >= start_date)\
    .order_by(PerformanceMetric.date)

# Bad: Missing indexes
query = session.query(PerformanceMetric)\
    .filter(PerformanceMetric.cost > 100)  # No index on cost
```

---

## 🤖 RL Environment Framework

### Environment Architecture

The RL framework is built on Gymnasium and provides multiple levels of optimization:

```
MultiPlatformEnv
├── CampaignOptimizationEnv (Google Ads)
├── CampaignOptimizationEnv (Facebook)
└── BudgetAllocationEnv
    ├── BidOptimizationEnv (Keywords 1-100)
    └── BidOptimizationEnv (Keywords 101-200)
```

### Environment Types

#### 1. **CampaignOptimizationEnv**

**Purpose**: High-level campaign management
**Actions**: Budget allocation, bid strategies, campaign status
**Observations**: Performance metrics, time features, competition data

```python
from src.core.environments import CampaignOptimizationEnv

# Initialize environment
campaigns = [
    {"id": "camp_1", "name": "Search Campaign", "initial_bid": 2.0},
    {"id": "camp_2", "name": "Display Campaign", "initial_bid": 1.5},
]

env = CampaignOptimizationEnv(
    campaigns=campaigns,
    total_budget=1000.0,
    platform="google_ads",
    time_horizon=24,  # 24 hours
    step_size=1       # 1 hour steps
)

# Training loop
obs, info = env.reset()
for step in range(100):
    action = agent.predict(obs)  # RL agent action
    obs, reward, terminated, truncated, info = env.step(action)
```

**Action Space**: `Box(shape=(3 * n_campaigns,))`
- Budget allocations: `[0, 1]` (normalized)
- Bid multipliers: `[0.5, 2.0]` (50%-200% of current bid)
- Status changes: `[0, 1]` (0=pause, 1=active)

**Observation Space**: Campaign metrics + time features + budget features
- Per campaign: budget allocation, bid, impressions, clicks, conversions, cost, CTR, CVR, CPA, quality score
- Time features: hour of day, day of week, time remaining, step progress
- Budget features: remaining budget, budget utilization, total spend

#### 2. **BudgetAllocationEnv**

**Purpose**: Dynamic budget allocation across entities
**Actions**: Budget distribution percentages
**Observations**: Entity performance, constraints, trends

```python
from src.core.environments import BudgetAllocationEnv

# Define entities (campaigns, ad groups, or channels)
entities = [
    {"id": "entity_1", "type": "campaign", "min_allocation": 0.1},
    {"id": "entity_2", "type": "campaign", "max_allocation": 0.6},
]

env = BudgetAllocationEnv(
    entities=entities,
    total_budget=5000.0,
    budget_period="daily",
    allocation_constraints={"entity_1": {"min_allocation": 0.1}}
)
```

#### 3. **BidOptimizationEnv**

**Purpose**: Real-time keyword bid optimization
**Actions**: Bid multipliers for each keyword
**Observations**: Auction dynamics, Quality Scores, competition

```python
from src.core.environments import BidOptimizationEnv

keywords = [
    {"id": "kw_1", "text": "machine learning", "match_type": "exact"},
    {"id": "kw_2", "text": "ai optimization", "match_type": "phrase"},
]

env = BidOptimizationEnv(
    keywords=keywords,
    budget_per_hour=100.0,
    bid_strategy="target_cpa",
    config={"target_cpa": 50.0}
)
```

#### 4. **MultiPlatformEnv**

**Purpose**: Cross-platform optimization and arbitrage
**Actions**: Platform allocations + platform-specific actions
**Observations**: Cross-platform performance, correlations, arbitrage opportunities

### Reward Engineering

#### Multi-Objective Rewards

AdBot uses sophisticated reward functions that balance multiple objectives:

```python
def calculate_reward(self, metrics):
    # Primary: ROI/ROAS
    roi_reward = np.tanh((metrics['roas'] - 3.0) / 3.0)
    
    # Secondary: Budget efficiency
    efficiency_reward = metrics['budget_utilization']
    
    # Tertiary: Stability (penalize large changes)
    stability_penalty = -0.1 * np.linalg.norm(action_change)
    
    # Combined reward
    return 0.6 * roi_reward + 0.3 * efficiency_reward + 0.1 * stability_penalty
```

#### Strategy-Specific Rewards

**Target CPA Strategy:**
```python
def target_cpa_reward(self, metrics):
    target_cpa = 50.0
    actual_cpa = metrics['cost'] / metrics['conversions']
    
    # Reward achieving target ±20%
    if 0.8 * target_cpa <= actual_cpa <= 1.2 * target_cpa:
        return 1.0
    else:
        return target_cpa / actual_cpa  # Penalize deviation
```

### Training RL Agents

#### Basic Training Loop

```python
from stable_baselines3 import PPO
from src.core.environments import CampaignOptimizationEnv

# Create environment
env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=1000)

# Create agent
agent = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1
)

# Train agent
agent.learn(total_timesteps=100000)

# Save trained model
agent.save("models/campaign_optimizer_ppo")
```

#### Advanced Training with Callbacks

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Evaluation callback
eval_env = CampaignOptimizationEnv(campaigns=eval_campaigns, total_budget=1000)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="campaign_ppo"
)

# Train with callbacks
agent.learn(
    total_timesteps=500000,
    callback=[eval_callback, checkpoint_callback]
)
```

---

## 🔌 Platform Integrations

### Google Ads Integration

#### Authentication Setup

1. **Create Google Ads Developer Account**
   - Apply for Google Ads API access
   - Get developer token
   - Create OAuth2 credentials

2. **OAuth2 Flow**
   ```python
   from src.integrations.google_ads import GoogleAdsClient
   
   config = {
       'developer_token': 'your_dev_token',
       'client_id': 'your_client_id',
       'client_secret': 'your_client_secret',
       'refresh_token': 'your_refresh_token',
       'customer_id': '123-456-7890'
   }
   
   client = GoogleAdsClient(config)
   await client.authenticate()
   ```

3. **Test Connection**
   ```python
   # Test API connection
   is_connected = await client.test_connection()
   print(f"Connection status: {is_connected}")
   
   # Get campaigns
   campaigns = await client.get_campaigns()
   print(f"Found {len(campaigns)} campaigns")
   ```

#### Common Operations

**Campaign Management:**
```python
# Create campaign
campaign_data = {
    'name': 'New AI Campaign',
    'budget_amount': 100.0,
    'bid_strategy': 'target_cpa',
    'target_cpa': 50.0,
    'channel_type': 'search'
}
campaign = await client.create_campaign(campaign_data)

# Update campaign
updates = {'budget_amount': 150.0, 'status': 'active'}
await client.update_campaign(campaign['id'], updates)

# Pause campaign
await client.pause_campaign(campaign['id'])
```

**Budget Management:**
```python
# Update budget
await client.update_campaign_budget(
    campaign_id="12345",
    budget_amount=200.0,
    budget_type="daily"
)
```

#### Error Handling

```python
from src.integrations.base import (
    AuthenticationError,
    RateLimitError,
    QuotaExceededError
)

try:
    campaigns = await client.get_campaigns()
except AuthenticationError:
    # Refresh OAuth tokens
    await client.authenticate()
except RateLimitError:
    # Wait and retry
    await asyncio.sleep(60)
except QuotaExceededError:
    # Daily quota exceeded, wait until tomorrow
    pass
```

### Adding New Platforms

#### 1. **Create Platform Client**

```python
# src/integrations/facebook/client.py
from ..base import BasePlatformClient

class FacebookClient(BasePlatformClient):
    def __init__(self, config):
        super().__init__(config, "facebook")
    
    async def authenticate(self):
        # Implement Facebook authentication
        pass
    
    async def get_campaigns(self, account_id=None, status_filter=None):
        # Implement Facebook campaign retrieval
        pass
    
    # ... implement other abstract methods
```

#### 2. **Add Platform Models**

```python
# src/integrations/facebook/models.py
from dataclasses import dataclass

@dataclass
class FacebookCampaign:
    id: str
    name: str
    status: str
    # Facebook-specific fields
    objective: str
    buying_type: str
```

#### 3. **Register Platform**

```python
# src/integrations/__init__.py
from .google_ads import GoogleAdsClient
from .facebook import FacebookClient

PLATFORM_CLIENTS = {
    'google_ads': GoogleAdsClient,
    'facebook': FacebookClient,
}
```

### Platform-Specific Considerations

#### Rate Limits

| Platform | Limit | Strategy |
|----------|-------|----------|
| Google Ads | 15,000 operations/day | Batch operations, intelligent retry |
| Facebook | 200 calls/hour/user | Request queuing, exponential backoff |
| TikTok | 1,000 calls/day | Cache responses, batch updates |

#### Data Formats

**Standardized Campaign Object:**
```python
{
    "id": "platform_campaign_id",
    "platform_id": "platform_campaign_id",
    "name": "Campaign Name",
    "status": "active|paused|ended",
    "budget_amount": 100.0,
    "budget_type": "daily|weekly|monthly",
    "bid_strategy": "manual_cpc|target_cpa|target_roas",
    "target_cpa": 50.0,
    "target_roas": 3.0,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
}
```

---

## 🌐 API Development

### FastAPI Application Structure

```
src/api/
├── main.py              # Application initialization
├── middleware.py        # Custom middleware
├── exceptions.py        # Error handling
└── routers/
    ├── campaigns.py     # Campaign CRUD
    ├── agents.py        # RL agent management
    ├── experiments.py   # A/B testing
    ├── platforms.py     # Platform integrations
    └── health.py        # Health checks
```

### Authentication & Authorization

#### JWT Token Authentication

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(401, "Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")

# Usage in endpoints
@router.get("/campaigns")
async def get_campaigns(user_id: str = Depends(get_current_user)):
    # User is authenticated
    pass
```

#### API Key Authentication

```python
def get_api_key_user(api_key: str = Header(None, alias="X-API-Key")):
    if not api_key:
        raise HTTPException(401, "API key required")
    
    # Validate API key
    user = validate_api_key(api_key)
    if not user:
        raise HTTPException(401, "Invalid API key")
    
    return user
```

### Request/Response Models

#### Pydantic Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class CampaignCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    platform: str = Field(..., regex="^(google_ads|facebook|tiktok)$")
    budget_amount: float = Field(..., gt=0)
    target_cpa: Optional[float] = Field(None, gt=0)
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

class CampaignResponse(BaseModel):
    id: UUID
    name: str
    platform: str
    status: str
    budget_amount: float
    created_at: datetime
    
    class Config:
        from_attributes = True  # For SQLAlchemy models
```

### Error Handling

#### Custom Exception Classes

```python
class AdBotAPIException(Exception):
    def __init__(self, message: str, status_code: int = 500, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

class NotFoundError(AdBotAPIException):
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(message, 404, "NOT_FOUND")
```

#### Exception Handlers

```python
@app.exception_handler(AdBotAPIException)
async def adbot_exception_handler(request: Request, exc: AdBotAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

### Middleware

#### Rate Limiting

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_id = self.get_client_id(request)
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Record request
        self.requests[client_id].append(datetime.now())
        
        return await call_next(request)
```

### API Documentation

#### OpenAPI Configuration

```python
app = FastAPI(
    title="AdBot API",
    description="AI-Powered Advertising Optimization Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "campaigns", "description": "Campaign management"},
        {"name": "agents", "description": "RL agent operations"},
        {"name": "experiments", "description": "A/B testing"},
    ]
)
```

#### Endpoint Documentation

```python
@router.post(
    "/campaigns",
    response_model=CampaignResponse,
    status_code=201,
    summary="Create a new campaign",
    description="Create a new advertising campaign with the specified parameters",
    responses={
        201: {"description": "Campaign created successfully"},
        400: {"description": "Invalid request data"},
        401: {"description": "Authentication required"},
        422: {"description": "Validation error"}
    }
)
async def create_campaign(campaign_data: CampaignCreate):
    """
    Create a new advertising campaign.
    
    - **name**: Campaign name (required)
    - **platform**: Advertising platform (google_ads, facebook, tiktok)
    - **budget_amount**: Daily budget in currency units
    - **target_cpa**: Target cost per acquisition (optional)
    """
    pass
```

---

## 🧪 Testing Strategy

### Test Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_environments.py
│   ├── test_integrations.py
│   └── test_utils.py
├── integration/
│   ├── test_api.py
│   ├── test_database.py
│   └── test_platform_clients.py
├── performance/
│   ├── test_rl_training.py
│   └── test_api_load.py
└── fixtures/
    ├── campaigns.json
    ├── performance_data.json
    └── mock_responses.py
```

### Unit Testing

#### Model Tests

```python
import pytest
from src.models.campaign import Campaign, CampaignStatus

def test_campaign_creation():
    campaign = Campaign(
        name="Test Campaign",
        platform="google_ads",
        platform_id="12345",
        status=CampaignStatus.ACTIVE,
        budget_amount=100.0
    )
    
    assert campaign.name == "Test Campaign"
    assert campaign.status == CampaignStatus.ACTIVE
    assert campaign.budget_amount == 100.0

def test_campaign_validation():
    with pytest.raises(ValueError):
        Campaign(
            name="",  # Empty name should fail
            platform="google_ads",
            budget_amount=-100.0  # Negative budget should fail
        )
```

#### Environment Tests

```python
import numpy as np
from src.core.environments import CampaignOptimizationEnv

def test_campaign_environment_initialization():
    campaigns = [{"id": "test", "name": "Test Campaign"}]
    env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=1000)
    
    assert env.n_campaigns == 1
    assert env.total_budget == 1000
    assert env.observation_space is not None
    assert env.action_space is not None

def test_campaign_environment_step():
    campaigns = [{"id": "test", "name": "Test Campaign"}]
    env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=1000)
    
    obs, info = env.reset()
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(info, dict)
```

### Integration Testing

#### API Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_create_campaign():
    campaign_data = {
        "name": "Test Campaign",
        "platform": "google_ads",
        "budget_amount": 100.0,
        "bid_strategy": "target_cpa",
        "target_cpa": 50.0
    }
    
    response = client.post("/api/v1/campaigns", json=campaign_data)
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Campaign"
    assert data["platform"] == "google_ads"

def test_get_campaigns():
    response = client.get("/api/v1/campaigns")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_platform_integration():
    from src.integrations.google_ads import GoogleAdsClient
    
    # Mock configuration
    config = {
        'developer_token': 'test_token',
        'client_id': 'test_client_id',
        'client_secret': 'test_secret',
        'refresh_token': 'test_refresh',
        'customer_id': '123-456-7890'
    }
    
    client = GoogleAdsClient(config)
    
    # Mock the authentication
    with patch.object(client, 'authenticate', return_value=True):
        result = await client.authenticate()
        assert result is True
```

### Performance Testing

#### RL Training Performance

```python
import time
from src.core.environments import CampaignOptimizationEnv
from stable_baselines3 import PPO

def test_rl_training_performance():
    """Test that RL training completes within reasonable time"""
    campaigns = [{"id": f"camp_{i}", "name": f"Campaign {i}"} for i in range(10)]
    env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=10000)
    
    agent = PPO("MlpPolicy", env, verbose=0)
    
    start_time = time.time()
    agent.learn(total_timesteps=10000)
    training_time = time.time() - start_time
    
    # Should complete training in under 60 seconds
    assert training_time < 60
```

#### API Load Testing

```python
import asyncio
import aiohttp
import time

async def test_api_load():
    """Test API can handle concurrent requests"""
    async def make_request(session):
        async with session.get("http://localhost:8080/health") as response:
            return response.status
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # 100 concurrent requests
        tasks = [make_request(session) for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        
        # Should handle 100 requests in under 10 seconds
        assert (end_time - start_time) < 10
```

### Test Fixtures and Mocks

#### Sample Data Fixtures

```python
# tests/fixtures/campaigns.py
import pytest

@pytest.fixture
def sample_campaigns():
    return [
        {
            "id": "camp_1",
            "name": "Search Campaign",
            "platform": "google_ads",
            "budget_amount": 100.0,
            "status": "active"
        },
        {
            "id": "camp_2", 
            "name": "Display Campaign",
            "platform": "facebook",
            "budget_amount": 200.0,
            "status": "paused"
        }
    ]

@pytest.fixture
def sample_performance_data():
    return {
        "impressions": 1000,
        "clicks": 50,
        "conversions": 5,
        "cost": 100.0,
        "revenue": 500.0
    }
```

#### Mock Platform Responses

```python
# tests/fixtures/mock_responses.py
from unittest.mock import Mock

class MockGoogleAdsResponse:
    def __init__(self, campaigns_data):
        self.campaigns_data = campaigns_data
    
    def __iter__(self):
        for campaign_data in self.campaigns_data:
            row = Mock()
            row.campaign = Mock()
            row.campaign.id = campaign_data["id"]
            row.campaign.name = campaign_data["name"]
            row.campaign_budget = Mock()
            row.campaign_budget.amount_micros = campaign_data["budget_amount"] * 1_000_000
            yield row

@pytest.fixture
def mock_google_ads_client():
    client = Mock()
    client.get_campaigns.return_value = MockGoogleAdsResponse([
        {"id": "12345", "name": "Test Campaign", "budget_amount": 100.0}
    ])
    return client
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_environments.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

---

## 🚀 Deployment Procedures

### Docker Configuration

#### Multi-Stage Dockerfile

```dockerfile
# Base stage with Python dependencies
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash adbot

# Copy source code
COPY src/ ./src/
COPY alembic.ini .
COPY migrations/ ./migrations/

# Install package
RUN pip install -e .

# Switch to non-root user
USER adbot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: adbot_dev
      POSTGRES_USER: adbot
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "1030:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes

  adbot-api:
    build:
      context: .
      target: development
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
    volumes:
      - .:/app

volumes:
  postgres_data:
```

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: adbot
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: adbot-config
  namespace: adbot
data:
  APP_ENV: "production"
  DB_HOST: "postgres.adbot.svc.cluster.local"
  DB_PORT: "5432"
  DB_NAME: "adbot_prod"
  REDIS_HOST: "redis.adbot.svc.cluster.local"
  REDIS_PORT: "6379"
```

#### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: adbot-secrets
  namespace: adbot
type: Opaque
stringData:
  DB_PASSWORD: "secure_password"
  JWT_SECRET: "secure_jwt_secret"
  GOOGLE_ADS_DEV_TOKEN: "your_dev_token"
  GOOGLE_ADS_CLIENT_SECRET: "your_client_secret"
  GOOGLE_ADS_REFRESH_TOKEN: "your_refresh_token"
```

#### Database Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: adbot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: adbot-config
              key: DB_NAME
        - name: POSTGRES_USER
          value: "adbot"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: adbot-secrets
              key: DB_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: adbot
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

#### Application Deployment

```yaml
# k8s/adbot-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adbot-api
  namespace: adbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: adbot-api
  template:
    metadata:
      labels:
        app: adbot-api
    spec:
      containers:
      - name: adbot-api
        image: adbot:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: adbot-config
        - secretRef:
            name: adbot-secrets
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: adbot-api
  namespace: adbot
spec:
  selector:
    app: adbot-api
  ports:
  - port: 80
    targetPort: 8080
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: adbot-ingress
  namespace: adbot
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.adbot.ai
    secretName: adbot-tls
  rules:
  - host: api.adbot.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: adbot-api
            port:
              number: 80
```

### CI/CD Pipeline

#### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy AdBot

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: adbot_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      env:
        DB_HOST: localhost
        DB_PORT: 5432
        DB_NAME: adbot_test
        DB_USER: postgres
        DB_PASSWORD: test_password
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t adbot:${{ github.sha }} .
        docker tag adbot:${{ github.sha }} adbot:latest
    
    - name: Push to registry
      if: github.ref == 'refs/heads/main'
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push adbot:${{ github.sha }}
        docker push adbot:latest

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl apply -f k8s/
        kubectl rollout restart deployment/adbot-api -n adbot
```

### Monitoring and Logging

#### Prometheus Configuration

```yaml
# k8s/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: adbot
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'adbot-api'
      static_configs:
      - targets: ['adbot-api:80']
      metrics_path: '/metrics'
    
    - job_name: 'postgres'
      static_configs:
      - targets: ['postgres:5432']
```

#### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "AdBot Metrics",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Campaign Performance",
        "type": "graph", 
        "targets": [
          {
            "expr": "adbot_campaign_roas"
          }
        ]
      }
    ]
  }
}
```

---

## 🔍 Code Quality & Standards

### Linting and Formatting

#### Black Configuration

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ["py311"]
include = '\.pyi?$'
extend-exclude = '''
/(
  \.git
  | \.mypy_cache
  | \.tox
  | venv
  | build
  | dist
)/
'''
```

#### isort Configuration

```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]
known_third_party = ["fastapi", "sqlalchemy", "stable_baselines3"]
```

#### mypy Configuration

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = [
    "stable_baselines3.*",
    "gymnasium.*",
    "google.ads.*"
]
ignore_missing_imports = true
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

### Documentation Standards

#### Docstring Format

```python
def calculate_reward(
    self,
    metrics: Dict[str, float],
    strategy: str = "multi_objective"
) -> float:
    """Calculate reward based on campaign performance metrics.
    
    This function implements sophisticated reward engineering that balances
    multiple objectives including ROI, budget efficiency, and stability.
    
    Args:
        metrics: Dictionary containing performance metrics with keys:
            - 'roas': Return on ad spend
            - 'cost': Total cost in currency units  
            - 'conversions': Number of conversions
            - 'budget_utilization': Fraction of budget used (0.0-1.0)
        strategy: Reward calculation strategy. Options:
            - 'multi_objective': Balance ROI, efficiency, and stability
            - 'roi_focused': Prioritize return on investment
            - 'efficiency_focused': Prioritize budget efficiency
    
    Returns:
        Reward value between -1.0 and 1.0, where:
            - 1.0: Excellent performance across all metrics
            - 0.0: Average/baseline performance  
            - -1.0: Poor performance requiring correction
    
    Raises:
        ValueError: If metrics dictionary is missing required keys
        TypeError: If strategy is not a valid string option
    
    Example:
        >>> metrics = {
        ...     'roas': 4.5,
        ...     'cost': 100.0,
        ...     'conversions': 10,
        ...     'budget_utilization': 0.85
        ... }
        >>> reward = calculate_reward(metrics, "multi_objective")
        >>> print(f"Reward: {reward:.3f}")
        Reward: 0.742
    
    Note:
        The reward function uses tanh normalization to ensure bounded output
        and applies different weights based on the selected strategy.
    """
    pass
```

#### API Documentation

```python
@router.post(
    "/campaigns/{campaign_id}/optimize",
    response_model=OptimizationResult,
    status_code=200,
    summary="Optimize campaign using RL agent",
    description="""
    Trigger RL-based optimization for a specific campaign.
    
    This endpoint applies the latest trained RL agent to optimize
    campaign parameters including bids, budget allocation, and targeting.
    
    The optimization process:
    1. Loads the latest trained agent for the campaign type
    2. Collects current campaign performance data
    3. Generates optimization recommendations
    4. Optionally applies changes automatically
    
    **Note**: Automatic application requires `auto_apply=true` parameter
    and sufficient user permissions.
    """,
    responses={
        200: {
            "description": "Optimization completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "optimization_id": "opt_123",
                        "recommendations": {
                            "budget_change": "+15%",
                            "bid_adjustments": {"keyword_1": "+10%", "keyword_2": "-5%"}
                        },
                        "expected_improvement": {
                            "roas": "+12%",
                            "cost_per_conversion": "-8%"
                        },
                        "confidence_score": 0.87
                    }
                }
            }
        },
        404: {"description": "Campaign not found"},
        403: {"description": "Insufficient permissions"},
        422: {"description": "Campaign not ready for optimization"}
    }
)
async def optimize_campaign(
    campaign_id: UUID = Path(..., description="Campaign ID to optimize"),
    auto_apply: bool = Query(False, description="Automatically apply recommendations"),
    agent_version: Optional[str] = Query(None, description="Specific agent version to use")
):
    pass
```

### Git Workflow

#### Branch Naming Convention

```
feature/description-of-feature
bugfix/description-of-bug
hotfix/critical-issue
release/version-number
```

Examples:
- `feature/tiktok-integration`
- `bugfix/campaign-budget-validation`
- `hotfix/authentication-error`
- `release/v1.2.0`

#### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(rl): add multi-platform environment wrapper

- Implement cross-platform budget allocation
- Add arbitrage opportunity detection
- Include performance correlation analysis

Closes #123
```

```
fix(api): resolve campaign creation validation error

The budget_amount validation was incorrectly rejecting valid decimal values.
Updated the validator to handle float precision properly.

Fixes #456
```

---

## 🛠️ Troubleshooting Guide

### Common Issues

#### Database Connection Problems

**Symptom**: `Connection refused` or `Database connection failed`

**Diagnosis**:
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check connection manually
psql -h localhost -p 7500 -U adbot -d adbot_dev

# Check from within Docker
docker-compose exec adbot-api python -c "
from src.utils.config import ConfigManager
config = ConfigManager().get_app_config()
print(f'DB URL: {config.database.url}')
"
```

**Solutions**:
1. **Wrong port**: Ensure DB_PORT=7500 in .env
2. **Password mismatch**: Verify DB_PASSWORD matches PostgreSQL setup
3. **Service not running**: Start PostgreSQL service
4. **Docker networking**: Use service names in docker-compose (postgres, not localhost)

#### Google Ads API Authentication

**Symptom**: `AuthenticationError` or `Invalid credentials`

**Diagnosis**:
```python
from src.integrations.google_ads import GoogleAdsClient

# Test authentication
config = {
    'developer_token': 'your_token',
    'client_id': 'your_client_id',
    'client_secret': 'your_secret',
    'refresh_token': 'your_refresh_token',
    'customer_id': '123-456-7890'
}

client = GoogleAdsClient(config)
try:
    await client.authenticate()
    print("Authentication successful")
except Exception as e:
    print(f"Auth failed: {e}")
```

**Solutions**:
1. **Invalid developer token**: Check Google Ads API console
2. **Expired refresh token**: Re-run OAuth flow
3. **Wrong customer ID**: Verify format (123-456-7890)
4. **API access not approved**: Check developer account status

#### RL Training Performance Issues

**Symptom**: Training is slow or agents don't converge

**Diagnosis**:
```python
import time
from src.core.environments import CampaignOptimizationEnv

# Benchmark environment
campaigns = [{"id": f"c_{i}", "name": f"Campaign {i}"} for i in range(10)]
env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=10000)

# Time environment operations
start = time.time()
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
print(f"1000 steps took {time.time() - start:.2f} seconds")
```

**Solutions**:
1. **Slow environment**: Optimize observation/action computation
2. **Poor convergence**: Adjust hyperparameters (learning_rate, batch_size)
3. **Unstable training**: Reduce learning rate, increase batch size
4. **Memory issues**: Reduce environment complexity or use smaller models

#### API Performance Problems

**Symptom**: Slow API responses or timeouts

**Diagnosis**:
```bash
# Test API performance
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8080/api/v1/campaigns"

# Where curl-format.txt contains:
#     time_namelookup:  %{time_namelookup}\n
#     time_connect:     %{time_connect}\n
#     time_total:       %{time_total}\n

# Check database query performance
docker-compose exec postgres psql -U adbot -d adbot_dev -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

**Solutions**:
1. **Database queries**: Add indexes, optimize queries
2. **Missing pagination**: Implement limit/offset for large datasets
3. **No caching**: Add Redis caching for frequent queries
4. **Blocking operations**: Make platform API calls async

### Debug Procedures

#### Logging Analysis

**Enable Debug Logging**:
```python
# src/utils/logger.py
setup_logger("adbot", "DEBUG", "logs/debug.log")
```

**Log Analysis Commands**:
```bash
# View recent API errors
tail -f logs/api.log | grep ERROR

# Count error types
grep ERROR logs/api.log | cut -d' ' -f4 | sort | uniq -c

# Find slow queries
grep "slow query" logs/api.log | tail -20

# Monitor real-time logs
docker-compose logs -f adbot-api
```

#### Performance Profiling

**API Endpoint Profiling**:
```python
import cProfile
import pstats
from fastapi import Request

@app.middleware("http")
async def profile_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/v1/"):
        profiler = cProfile.Profile()
        profiler.enable()
        
        response = await call_next(request)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats(f"profiles/{request.url.path.replace('/', '_')}.prof")
        
        return response
    else:
        return await call_next(request)
```

**RL Environment Profiling**:
```python
import cProfile
from src.core.environments import CampaignOptimizationEnv

def profile_environment():
    env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=1000)
    
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

# Run profiler
profiler = cProfile.Profile()
profiler.enable()
profile_environment()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

#### Memory Usage Analysis

```python
import tracemalloc
import psutil
import gc

def analyze_memory():
    # Start tracing
    tracemalloc.start()
    
    # Your code here
    from src.core.environments import CampaignOptimizationEnv
    envs = []
    for i in range(100):
        env = CampaignOptimizationEnv(campaigns=[{"id": f"c_{i}"}], total_budget=1000)
        envs.append(env)
    
    # Get memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Get tracemalloc stats
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print(f"Memory usage: {memory_mb:.1f} MB")
    for stat in top_stats[:10]:
        print(stat)
    
    # Cleanup
    del envs
    gc.collect()

analyze_memory()
```

### Error Resolution Flowchart

```
API Error (5xx)
├── Check logs/api.log
├── Database connection issue?
│   ├── Yes → Check DB credentials, network, service status
│   └── No → Continue
├── Platform API issue?
│   ├── Yes → Check API credentials, rate limits, service status
│   └── No → Continue
├── Memory/performance issue?
│   ├── Yes → Profile code, check resource usage
│   └── No → Continue
└── Review code changes, check for exceptions

RL Training Issues
├── Check environment setup
├── Validate observation/action spaces
├── Review hyperparameters
├── Check for NaN/infinite values
├── Reduce environment complexity
└── Try different algorithm

Database Issues
├── Connection refused
│   ├── Check service status
│   ├── Verify credentials
│   └── Check network/firewall
├── Slow queries
│   ├── Add missing indexes
│   ├── Optimize query structure
│   └── Consider pagination
└── Migration failures
    ├── Check syntax
    ├── Verify dependencies
    └── Review model changes
```

---

## 🤝 Contributing Guidelines

### Development Workflow

#### 1. **Setting Up Development Environment**

```bash
# Fork and clone the repository
git clone https://github.com/your-username/adbot.git
cd adbot

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Set up database
python scripts/init_db.py --sample-data
```

#### 2. **Feature Development Process**

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run tests
pytest tests/

# Run linting
black src/ tests/
isort src/ tests/
mypy src/

# Commit changes
git add .
git commit -m "feat(scope): description"

# Push and create PR
git push origin feature/your-feature-name
gh pr create --title "Feature: Your Feature" --body "Description..."
```

#### 3. **Pull Request Guidelines**

**PR Title Format**:
```
<type>(<scope>): <description>

Examples:
feat(rl): add multi-platform environment coordination
fix(api): resolve campaign validation error
docs(readme): update installation instructions
```

**PR Description Template**:
```markdown
## Summary
Brief description of the changes and their purpose.

## Changes Made
- [ ] Added new RL environment for TikTok integration
- [ ] Updated campaign model to support TikTok-specific fields
- [ ] Added comprehensive tests for new functionality

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Breaking Changes
List any breaking changes and migration steps.

## Screenshots/Examples
Include relevant screenshots or code examples.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or properly documented)
```

### Adding New Features

#### 1. **Adding a New RL Environment**

```python
# src/core/environments/your_environment.py
from .base import BaseAdEnvironment

class YourEnvironment(BaseAdEnvironment):
    """Your custom RL environment for specific optimization task"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # Initialize your environment-specific state
    
    def _get_observation(self) -> np.ndarray:
        # Return current observation
        pass
    
    def _take_action(self, action: np.ndarray) -> Dict[str, float]:
        # Execute action and return metrics
        pass
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        # Calculate reward from metrics
        pass
    
    # ... implement other abstract methods

# Add to __init__.py
# src/core/environments/__init__.py
from .your_environment import YourEnvironment
__all__.append("YourEnvironment")

# Add tests
# tests/unit/test_your_environment.py
def test_your_environment_initialization():
    env = YourEnvironment(config={})
    assert env.observation_space is not None
    assert env.action_space is not None

def test_your_environment_step():
    env = YourEnvironment(config={})
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
```

#### 2. **Adding a New Platform Integration**

```python
# src/integrations/your_platform/client.py
from ..base import BasePlatformClient

class YourPlatformClient(BasePlatformClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "your_platform")
    
    async def authenticate(self) -> bool:
        # Implement platform authentication
        pass
    
    async def get_campaigns(self, account_id=None, status_filter=None):
        # Implement campaign retrieval
        pass
    
    # ... implement other abstract methods

# Add models
# src/integrations/your_platform/models.py
@dataclass
class YourPlatformCampaign:
    id: str
    name: str
    # Platform-specific fields

# Register platform
# src/integrations/__init__.py
from .your_platform import YourPlatformClient
```

#### 3. **Adding New API Endpoints**

```python
# src/api/routers/your_router.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()

class YourModel(BaseModel):
    field1: str
    field2: int

@router.get("/your-endpoint")
async def your_endpoint() -> List[YourModel]:
    """Your endpoint documentation"""
    pass

# Add to main app
# src/api/main.py
from .routers import your_router
app.include_router(your_router.router, prefix="/api/v1/your-prefix", tags=["Your Tag"])
```

### Code Review Guidelines

#### What to Look For

**Functionality**:
- [ ] Does the code solve the intended problem?
- [ ] Are edge cases handled properly?
- [ ] Is error handling comprehensive?
- [ ] Are there any potential security issues?

**Code Quality**:
- [ ] Is the code readable and well-structured?
- [ ] Are variable and function names descriptive?
- [ ] Is the code properly documented?
- [ ] Are there any code smells or anti-patterns?

**Testing**:
- [ ] Are there sufficient unit tests?
- [ ] Do integration tests cover the main workflows?
- [ ] Are edge cases tested?
- [ ] Is test coverage adequate (>80%)?

**Performance**:
- [ ] Are there any obvious performance issues?
- [ ] Are database queries optimized?
- [ ] Is caching used appropriately?
- [ ] Are async operations handled correctly?

**Documentation**:
- [ ] Are docstrings complete and accurate?
- [ ] Is API documentation updated?
- [ ] Are breaking changes documented?
- [ ] Is DEVELOPMENT.md updated if needed?

#### Review Comments Examples

**Good Comments**:
```
"Consider using a more descriptive variable name here. 
`campaign_performance_data` would be clearer than `data`."

"This function is doing too much. Consider splitting it into 
separate functions for data validation and processing."

"Missing error handling for the case where the API returns None. 
What should happen in that scenario?"
```

**Avoid**:
```
"This is wrong." (not helpful)
"Bad code." (not constructive)
"Just change this." (no explanation)
```

### Documentation Requirements

#### When to Update Documentation

- **New features**: Update relevant sections
- **API changes**: Update API documentation
- **Configuration changes**: Update setup instructions
- **Breaking changes**: Update migration guide
- **Bug fixes**: Update troubleshooting if relevant

#### Documentation Standards

**Code Documentation**:
- All public functions must have docstrings
- Complex algorithms need inline comments
- Type hints required for all function signatures
- Examples in docstrings for complex functions

**API Documentation**:
- All endpoints must have OpenAPI documentation
- Include request/response examples
- Document error responses
- Add usage examples

**Architecture Documentation**:
- Update diagrams when adding new components
- Document design decisions
- Explain trade-offs and alternatives considered

---

## 🎯 Advanced Topics

### Custom RL Algorithms

#### Implementing a Custom RL Algorithm

```python
# src/core/agents/custom_algorithm.py
import torch
import torch.nn as nn
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomAlgorithm(BaseAlgorithm):
    """Custom RL algorithm for advertising optimization"""
    
    def __init__(
        self,
        policy,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__(policy, env, learning_rate, **kwargs)
        self.n_steps = n_steps
        self.batch_size = batch_size
        
    def _setup_model(self) -> None:
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs
        )
        
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        **kwargs
    ):
        # Implement custom learning loop
        pass
    
    def predict(self, observation, deterministic: bool = False):
        # Implement prediction logic
        pass

# Usage
from src.core.environments import CampaignOptimizationEnv

env = CampaignOptimizationEnv(campaigns=campaigns, total_budget=1000)
agent = CustomAlgorithm("MlpPolicy", env)
agent.learn(total_timesteps=100000)
```

#### Multi-Agent RL for Platform Coordination

```python
# src/core/agents/multi_agent.py
from typing import Dict, List
import numpy as np

class MultiAgentCoordinator:
    """Coordinate multiple RL agents across platforms"""
    
    def __init__(self, agents: Dict[str, BaseAlgorithm]):
        self.agents = agents  # {platform_name: agent}
        self.coordination_network = self._build_coordination_network()
    
    def _build_coordination_network(self):
        # Neural network for agent coordination
        return nn.Sequential(
            nn.Linear(len(self.agents) * 64, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.agents))
        )
    
    def coordinate_actions(
        self,
        observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        # Get individual agent actions
        individual_actions = {}
        for platform, obs in observations.items():
            action = self.agents[platform].predict(obs, deterministic=False)
            individual_actions[platform] = action
        
        # Apply coordination
        coordinated_actions = self._apply_coordination(individual_actions)
        
        return coordinated_actions
    
    def _apply_coordination(self, actions: Dict[str, np.ndarray]):
        # Implement coordination logic (e.g., budget constraints)
        total_budget = 1000.0
        
        # Ensure budget allocations sum to 1
        budget_allocations = {}
        total_allocation = 0
        
        for platform, action in actions.items():
            budget_allocations[platform] = action[0]  # Assuming first element is budget
            total_allocation += action[0]
        
        # Normalize allocations
        for platform in budget_allocations:
            budget_allocations[platform] /= total_allocation
            actions[platform][0] = budget_allocations[platform]
        
        return actions
```

### Bayesian Optimization Integration

#### Hyperparameter Optimization

```python
# src/optimization/bayesian.py
import optuna
from typing import Dict, Any, Callable
from stable_baselines3 import PPO
from src.core.environments import CampaignOptimizationEnv

class BayesianOptimizer:
    """Bayesian optimization for RL hyperparameters"""
    
    def __init__(self, env_factory: Callable, objective_metric: str = "mean_reward"):
        self.env_factory = env_factory
        self.objective_metric = objective_metric
    
    def optimize_hyperparameters(
        self,
        n_trials: int = 100,
        n_timesteps: int = 50000
    ) -> Dict[str, Any]:
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            n_epochs = trial.suggest_int("n_epochs", 5, 20)
            gamma = trial.suggest_float("gamma", 0.9, 0.999)
            gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
            
            # Create environment and agent
            env = self.env_factory()
            agent = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                verbose=0
            )
            
            # Train agent
            agent.learn(total_timesteps=n_timesteps)
            
            # Evaluate performance
            eval_env = self.env_factory()
            obs, _ = eval_env.reset()
            episode_rewards = []
            
            for episode in range(10):  # Evaluate for 10 episodes
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    
                    if done:
                        obs, _ = eval_env.reset()
                
                episode_rewards.append(episode_reward)
            
            return np.mean(episode_rewards)
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

# Usage
def create_env():
    campaigns = [{"id": "test", "name": "Test Campaign"}]
    return CampaignOptimizationEnv(campaigns=campaigns, total_budget=1000)

optimizer = BayesianOptimizer(create_env)
best_params = optimizer.optimize_hyperparameters(n_trials=50)
print(f"Best hyperparameters: {best_params}")
```

#### Budget Allocation Optimization

```python
# src/optimization/budget_bayesian.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

class BayesianBudgetOptimizer:
    """Bayesian optimization for budget allocation"""
    
    def __init__(self, n_campaigns: int):
        self.n_campaigns = n_campaigns
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True
        )
        self.X_observed = []
        self.y_observed = []
    
    def add_observation(self, allocation: np.ndarray, performance: float):
        """Add observed allocation and performance"""
        self.X_observed.append(allocation)
        self.y_observed.append(performance)
        
        if len(self.X_observed) > 1:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
    
    def expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Calculate expected improvement acquisition function"""
        if len(self.y_observed) == 0:
            return np.ones(len(X))
        
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Current best performance
        f_best = np.max(self.y_observed)
        
        with np.errstate(divide='warn'):
            imp = mu - f_best
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei.flatten()
    
    def suggest_allocation(self) -> np.ndarray:
        """Suggest next budget allocation to try"""
        if len(self.X_observed) < 2:
            # Random allocation for initial exploration
            allocation = np.random.dirichlet(np.ones(self.n_campaigns))
            return allocation
        
        # Optimize acquisition function
        best_ei = -np.inf
        best_allocation = None
        
        for _ in range(100):  # Multiple random starts
            # Generate random starting point
            x0 = np.random.dirichlet(np.ones(self.n_campaigns))
            
            # Optimize with constraint that allocation sums to 1
            constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(0, 1) for _ in range(self.n_campaigns)]
            
            result = minimize(
                lambda x: -self.expected_improvement(x.reshape(1, -1))[0],
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint
            )
            
            if result.success:
                ei = self.expected_improvement(result.x.reshape(1, -1))[0]
                if ei > best_ei:
                    best_ei = ei
                    best_allocation = result.x
        
        return best_allocation if best_allocation is not None else x0

# Usage
optimizer = BayesianBudgetOptimizer(n_campaigns=5)

# Simulate optimization loop
for iteration in range(50):
    # Get suggested allocation
    allocation = optimizer.suggest_allocation()
    
    # Run campaign with this allocation (simulate)
    performance = simulate_campaign_performance(allocation)
    
    # Add observation
    optimizer.add_observation(allocation, performance)
    
    print(f"Iteration {iteration}: Performance = {performance:.3f}")
```

### A/B Testing Framework

#### Statistical Testing

```python
# src/experiments/statistical_testing.py
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TestResult:
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    is_significant: bool
    required_sample_size: int

class StatisticalTester:
    """Statistical testing for A/B experiments"""
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        self.alpha = alpha  # Significance level
        self.power = power  # Statistical power
    
    def t_test(
        self,
        control_data: List[float],
        treatment_data: List[float],
        alternative: str = "two-sided"
    ) -> TestResult:
        """Perform two-sample t-test"""
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(
            control_data,
            treatment_data,
            alternative=alternative
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_data) - 1) * np.var(control_data, ddof=1) +
             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
            (len(control_data) + len(treatment_data) - 2)
        )
        effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
        
        # Calculate confidence interval
        se = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        mean_diff = np.mean(treatment_data) - np.mean(control_data)
        ci_lower = mean_diff - t_critical * se
        ci_upper = mean_diff + t_critical * se
        
        # Calculate statistical power
        power = self._calculate_power(
            effect_size, len(control_data), len(treatment_data)
        )
        
        # Calculate required sample size
        required_n = self._calculate_sample_size(effect_size)
        
        return TestResult(
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            statistical_power=power,
            is_significant=p_value < self.alpha,
            required_sample_size=required_n
        )
    
    def _calculate_power(
        self,
        effect_size: float,
        n_control: int,
        n_treatment: int
    ) -> float:
        """Calculate statistical power"""
        # Simplified power calculation
        n_total = n_control + n_treatment
        se = np.sqrt(2 / n_total)
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = effect_size / se - z_alpha
        power = stats.norm.cdf(z_beta)
        return max(0, min(1, power))
    
    def _calculate_sample_size(self, effect_size: float) -> int:
        """Calculate required sample size for desired power"""
        if effect_size == 0:
            return float('inf')
        
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(self.power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

# Usage
tester = StatisticalTester(alpha=0.05, power=0.8)

# Simulate A/B test data
control_conversions = np.random.normal(0.05, 0.01, 1000)  # 5% conversion rate
treatment_conversions = np.random.normal(0.06, 0.01, 1000)  # 6% conversion rate

result = tester.t_test(control_conversions, treatment_conversions)

print(f"P-value: {result.p_value:.4f}")
print(f"Effect size: {result.effect_size:.4f}")
print(f"Is significant: {result.is_significant}")
print(f"Required sample size: {result.required_sample_size}")
```

### Performance Monitoring

#### ML Model Drift Detection

```python
# src/monitoring/drift_detection.py
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class DriftResult:
    is_drift: bool
    drift_score: float
    p_value: float
    drift_type: str  # "feature", "prediction", "performance"
    timestamp: datetime

class DriftDetector:
    """Detect model drift in RL agents"""
    
    def __init__(self, baseline_window: int = 1000, detection_threshold: float = 0.05):
        self.baseline_window = baseline_window
        self.detection_threshold = detection_threshold
        self.baseline_data = {}
        self.recent_data = {}
    
    def add_baseline_data(self, agent_id: str, data: Dict[str, np.ndarray]):
        """Add baseline data for comparison"""
        self.baseline_data[agent_id] = data
    
    def detect_feature_drift(
        self,
        agent_id: str,
        recent_features: np.ndarray
    ) -> DriftResult:
        """Detect drift in input features"""
        if agent_id not in self.baseline_data:
            return DriftResult(False, 0.0, 1.0, "feature", datetime.now())
        
        baseline_features = self.baseline_data[agent_id].get("features")
        if baseline_features is None:
            return DriftResult(False, 0.0, 1.0, "feature", datetime.now())
        
        # Use Kolmogorov-Smirnov test for drift detection
        drift_scores = []
        p_values = []
        
        for i in range(baseline_features.shape[1]):
            statistic, p_value = stats.ks_2samp(
                baseline_features[:, i],
                recent_features[:, i]
            )
            drift_scores.append(statistic)
            p_values.append(p_value)
        
        # Aggregate results
        max_drift_score = max(drift_scores)
        min_p_value = min(p_values)
        is_drift = min_p_value < self.detection_threshold
        
        return DriftResult(
            is_drift=is_drift,
            drift_score=max_drift_score,
            p_value=min_p_value,
            drift_type="feature",
            timestamp=datetime.now()
        )
    
    def detect_performance_drift(
        self,
        agent_id: str,
        recent_rewards: List[float]
    ) -> DriftResult:
        """Detect drift in agent performance"""
        if agent_id not in self.baseline_data:
            return DriftResult(False, 0.0, 1.0, "performance", datetime.now())
        
        baseline_rewards = self.baseline_data[agent_id].get("rewards")
        if baseline_rewards is None:
            return DriftResult(False, 0.0, 1.0, "performance", datetime.now())
        
        # Use Mann-Whitney U test for performance comparison
        statistic, p_value = stats.mannwhitneyu(
            baseline_rewards,
            recent_rewards,
            alternative="two-sided"
        )
        
        # Calculate effect size
        n1, n2 = len(baseline_rewards), len(recent_rewards)
        u_statistic = statistic
        z_score = (u_statistic - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        effect_size = abs(z_score) / np.sqrt(n1 + n2)
        
        is_drift = p_value < self.detection_threshold
        
        return DriftResult(
            is_drift=is_drift,
            drift_score=effect_size,
            p_value=p_value,
            drift_type="performance",
            timestamp=datetime.now()
        )

# Monitoring service
class ModelMonitoringService:
    """Service for monitoring RL model performance"""
    
    def __init__(self):
        self.drift_detector = DriftDetector()
        self.alert_thresholds = {
            "performance_drop": 0.1,  # 10% performance drop
            "feature_drift": 0.05,    # 5% significance level
            "prediction_drift": 0.05
        }
    
    def monitor_agent(
        self,
        agent_id: str,
        recent_data: Dict[str, np.ndarray]
    ) -> List[DriftResult]:
        """Monitor agent for various types of drift"""
        results = []
        
        # Check feature drift
        if "features" in recent_data:
            feature_drift = self.drift_detector.detect_feature_drift(
                agent_id, recent_data["features"]
            )
            results.append(feature_drift)
        
        # Check performance drift
        if "rewards" in recent_data:
            performance_drift = self.drift_detector.detect_performance_drift(
                agent_id, recent_data["rewards"]
            )
            results.append(performance_drift)
        
        # Send alerts if necessary
        for result in results:
            if result.is_drift:
                self._send_alert(agent_id, result)
        
        return results
    
    def _send_alert(self, agent_id: str, drift_result: DriftResult):
        """Send alert for detected drift"""
        print(f"ALERT: {drift_result.drift_type} drift detected for agent {agent_id}")
        print(f"Drift score: {drift_result.drift_score:.4f}")
        print(f"P-value: {drift_result.p_value:.4f}")
        
        # In production, this would send notifications via email, Slack, etc.

# Usage
monitor = ModelMonitoringService()

# Add baseline data
baseline_features = np.random.normal(0, 1, (1000, 10))
baseline_rewards = np.random.normal(0.5, 0.1, 1000)
monitor.drift_detector.add_baseline_data("agent_1", {
    "features": baseline_features,
    "rewards": baseline_rewards
})

# Monitor recent data
recent_features = np.random.normal(0.2, 1, (100, 10))  # Slightly shifted distribution
recent_rewards = np.random.normal(0.4, 0.1, 100)      # Lower performance

drift_results = monitor.monitor_agent("agent_1", {
    "features": recent_features,
    "rewards": recent_rewards
})

for result in drift_results:
    print(f"{result.drift_type} drift: {result.is_drift}")
```

---

This comprehensive development guide provides everything needed to work effectively with AdBot, from initial setup to advanced topics like custom RL algorithms and drift detection. The documentation serves as both a learning resource and a reference manual for ongoing development.

The guide emphasizes practical, production-ready approaches while maintaining the flexibility to adapt to changing requirements. Each section includes working code examples, best practices, and troubleshooting guidance to ensure successful development and deployment of AdBot.