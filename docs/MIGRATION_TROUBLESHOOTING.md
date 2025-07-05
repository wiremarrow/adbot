# AdBot Database Migration Troubleshooting Guide

> **Context**: This guide documents the specific migration issues encountered during AdBot development and their solutions.

## 🔍 Issue: Port 1030 Connection Refused

### Problem Description
```
psycopg2.OperationalError: connection to server at "localhost" (::1), port 1030 failed: Connection refused
Is the server running on that host and accepting TCP/IP connections?
```

### Root Cause Analysis

1. **Initial Configuration**: The project was configured to use port 1030 for PostgreSQL
2. **Port Conflicts**: Multiple projects were using standard database ports (5432, 5433, 6379)
3. **Docker Mapping**: Docker was mapping external port 1030 to internal port 5432
4. **Service Not Running**: PostgreSQL container wasn't started

### Solution Implemented

1. **Port Reallocation**: Moved to sequential port range 7500-7507
2. **Configuration Sync**: Updated all configuration files to use new ports
3. **Docker Services**: Properly started PostgreSQL and Redis containers

### Step-by-Step Fix

```bash
# 1. Check existing port usage
lsof -i -P -n | grep LISTEN | awk '{print $9}' | cut -d: -f2 | sort -n | uniq

# 2. Update docker-compose.yml
# Changed from:
#   ports:
#     - "1030:5432"
# To:
#   ports:
#     - "7500:5432"

# 3. Update all configuration files
# - .env (DB_PORT=7500)
# - configs/default.yaml
# - src/utils/config.py
# - alembic.ini

# 4. Start services
docker-compose up -d postgres redis

# 5. Verify connection
python -c "import psycopg2; psycopg2.connect('postgresql://adbot:adbot_password@localhost:7500/adbot_dev'); print('Success!')"
```

## 🔍 Issue: Alembic Async Driver Error

### Problem Description
```
sqlalchemy.exc.InvalidRequestError: The asyncio extension requires an async driver to be used. 
The loaded 'psycopg2' is not async.
```

### Root Cause

The `migrations/env.py` file was trying to use async SQLAlchemy with the sync psycopg2 driver. Alembic migrations should use synchronous connections even if the application uses async.

### Solution

Modified `migrations/env.py` to use synchronous connections:

```python
# Before (incorrect):
from sqlalchemy.ext.asyncio import async_engine_from_config
# ... async configuration

# After (correct):
from sqlalchemy import pool, engine_from_config
# ... sync configuration
```

## 🔍 Issue: Environment Variables Not Loading

### Problem Description
```
ValidationError: 1 validation error for DatabaseConfig
DB_PASSWORD
  Field required [type=missing, input_value={}, input_type=dict]
```

### Root Causes

1. **Missing .env file**: The `.env` file wasn't created
2. **Pydantic v2 changes**: BaseSettings moved to pydantic-settings package
3. **Import path issues**: Circular imports when loading configuration

### Solutions Applied

1. **Created .env file**:
   ```bash
   cp .env.example .env
   # Edited with actual values
   ```

2. **Updated imports**:
   ```python
   # Before:
   from pydantic import BaseSettings
   
   # After:
   from pydantic_settings import BaseSettings
   ```

3. **Fixed Alembic fallback**:
   ```python
   # Added fallback values in migrations/env.py
   db_port = os.getenv("DB_PORT", "7500")
   db_password = os.getenv("DB_PASSWORD", "adbot_password")
   ```

## 🔍 Issue: Multiple Import Errors During Migration

### Problem Description
```
ModuleNotFoundError: No module named 'yaml'
ModuleNotFoundError: No module named 'structlog'
ModuleNotFoundError: No module named 'psycopg2'
```

### Root Cause

Conda environment was missing required dependencies.

### Solution

```bash
# Installed all missing dependencies
conda activate adbot
pip install pyyaml pydantic pydantic-settings asyncpg psycopg2-binary structlog
```

## 📋 Migration Checklist

Before running migrations, ensure:

- [ ] Docker services are running: `docker-compose ps`
- [ ] PostgreSQL is accessible: `nc -zv localhost 7500`
- [ ] Environment variables are set: `cat .env`
- [ ] Dependencies are installed: `pip list | grep -E "alembic|psycopg2|sqlalchemy"`
- [ ] Current directory is project root: `pwd` should show `/path/to/adbot`

## 🛠️ Common Migration Commands

### Debug Commands

```bash
# Test database connection
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://adbot:adbot_password@localhost:7500/adbot_dev')
print('Connected successfully')
conn.close()
"

# Check Alembic configuration
alembic show

# Verify current migration state
alembic current

# Show migration history
alembic history --verbose
```

### Recovery Commands

```bash
# If migrations are out of sync
alembic stamp head  # Mark current state as latest

# If you need to recreate from scratch
alembic downgrade base
alembic upgrade head

# If migration files are corrupted
rm -rf migrations/versions/*
alembic revision --autogenerate -m "Recreate initial schema"
alembic upgrade head
```

## 🚨 Critical Configuration Files

### 1. **alembic.ini**
```ini
# Key setting - must match your database URL
sqlalchemy.url = postgresql://adbot:adbot_password@localhost:7500/adbot_dev
```

### 2. **migrations/env.py**
```python
# Fallback configuration if env loading fails
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "7500")
db_name = os.getenv("DB_NAME", "adbot_dev")
db_user = os.getenv("DB_USER", "adbot")
db_password = os.getenv("DB_PASSWORD", "adbot_password")
```

### 3. **.env**
```bash
# Critical database settings
DB_HOST=localhost
DB_PORT=7500
DB_NAME=adbot_dev
DB_USER=adbot
DB_PASSWORD=adbot_password
```

## 🔄 Migration State Recovery

If migrations get into a bad state:

### Option 1: Reset Everything
```bash
# WARNING: This will delete all data!
docker-compose down -v  # Remove volumes
docker-compose up -d postgres
alembic upgrade head
```

### Option 2: Fix Migration State
```bash
# Connect to database
psql -h localhost -p 7500 -U adbot -d adbot_dev

# Check alembic version
SELECT * FROM alembic_version;

# Manually update if needed
UPDATE alembic_version SET version_num = 'your_migration_id';
```

## 📝 Lessons Learned

1. **Always use standard ports internally**: Docker should map custom external ports to standard internal ports
2. **Check port availability first**: Use `lsof` or `netstat` before allocating ports
3. **Keep configurations synchronized**: Changes must be reflected in all config files
4. **Use clear error messages**: PostgreSQL connection errors often hide the real issue
5. **Test incrementally**: Verify each service starts before moving to the next

## 🎯 Quick Diagnosis Flow

```mermaid
graph TD
    A[Migration Error] --> B{Can connect to DB?}
    B -->|No| C[Check Docker services]
    B -->|Yes| D{Alembic configured?}
    C --> E[docker-compose up -d]
    D -->|No| F[Check alembic.ini]
    D -->|Yes| G{Env vars loaded?}
    F --> H[Update DB URL]
    G -->|No| I[Create .env file]
    G -->|Yes| J[Run migration]
    I --> J
    H --> J
    E --> B
```

## 🚀 Future Improvements

1. **Add health checks** to docker-compose.yml
2. **Create init script** that validates environment before migrations
3. **Add connection retry logic** to handle startup race conditions
4. **Implement migration testing** in CI/CD pipeline
5. **Add automatic port detection** to avoid conflicts

This troubleshooting guide is based on actual issues encountered during AdBot development. Each solution has been tested and verified to work with the current codebase.