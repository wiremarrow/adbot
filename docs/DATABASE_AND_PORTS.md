# AdBot Database & Port Configuration Guide

> **Last Updated**: January 2025  
> **Status**: ✅ **RESOLVED** - All issues documented have been fixed  
> **Critical Reference**: This document contains essential information for database setup, port configuration, and migration management.

## ✅ Current Status

**All systems operational and configured correctly:**
- ✅ PostgreSQL running on port 7500
- ✅ Redis running on port 7501  
- ✅ All 21 database tables created successfully
- ✅ Alembic migrations working
- ✅ Port conflicts resolved
- ✅ Configuration synchronized across all files

## 🚀 Quick Start

```bash
# 1. Create conda environment
conda create -n adbot python=3.11 -y
conda activate adbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start database services
docker-compose up -d postgres redis

# 4. Run migrations
alembic upgrade head

# 5. (Optional) Create sample data
python scripts/init_db.py --sample-data
```

## 🔌 Port Configuration

### Complete Port Mapping

AdBot uses a **sequential port range (7500-7507)** to avoid conflicts with other projects:

| Service | Host Port | Container Port | Purpose | Access URL |
|---------|-----------|----------------|---------|------------|
| PostgreSQL | 7500 | 5432 | Primary database | `postgresql://localhost:7500/adbot_dev` |
| Redis | 7501 | 6379 | Caching & sessions | `redis://localhost:7501` |
| MLflow | 7502 | 5000 | ML experiment tracking | `http://localhost:7502` |
| Prometheus | 7503 | 9090 | Metrics collection | `http://localhost:7503` |
| Grafana | 7504 | 3000 | Metrics visualization | `http://localhost:7504` |
| Kafka | 7505 | 9092 | Event streaming | `localhost:7505` |
| Zookeeper | 7506 | 2181 | Kafka coordination | `localhost:7506` |
| AdBot API | 7507 | 8080 | Main REST API | `http://localhost:7507` |

### Why These Ports?

- **Sequential allocation**: Easy to remember and manage
- **High port range**: Avoids conflicts with common services
- **7500-7507**: Unlikely to be used by other applications
- **Previous conflicts resolved**: ✅
  - PostgreSQL (5432, 5433 were taken) → Now using 7500
  - Redis (6379 was taken) → Now using 7501
  - Standard web ports (3000, 8000, 8080 were taken) → Now using 7507

### Port Configuration Files

The port configuration is synchronized across multiple files:

1. **`.env`** - Primary configuration source
2. **`docker-compose.yml`** - Docker service definitions
3. **`configs/default.yaml`** - Application defaults
4. **`src/utils/config.py`** - Python configuration classes
5. **`alembic.ini`** - Database migration configuration

## 🗄️ Database Migration System

### Understanding Alembic

Alembic is our database migration tool that:
- Tracks schema changes over time
- Enables rollbacks if needed
- Supports team collaboration
- Maintains database version history

### Migration Commands

#### 1. **Create a New Migration**

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Create empty migration for manual editing
alembic revision -m "Manual migration description"
```

#### 2. **Apply Migrations**

```bash
# Upgrade to latest version
alembic upgrade head

# Upgrade to specific revision
alembic upgrade +1  # Next version
alembic upgrade ae1027a853ef  # Specific revision

# View current version
alembic current
```

#### 3. **Rollback Migrations**

```bash
# Downgrade one version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade ae1027a853ef

# Downgrade to base (remove all migrations)
alembic downgrade base
```

#### 4. **Migration History**

```bash
# Show migration history
alembic history

# Show detailed history
alembic history --verbose
```

### Database Schema Overview

The AdBot database contains **21 tables** organized into functional groups:

#### Campaign Management
- `campaigns` - Marketing campaigns
- `ad_groups` - Ad group hierarchies  
- `ads` - Individual advertisements
- `keywords` - Keyword targeting

#### Performance Tracking
- `performance_metrics` - Daily performance data
- `conversion_data` - Conversion tracking
- `bid_history` - Bid adjustment history

#### User & Account Management
- `users` - System users
- `accounts` - Business accounts
- `platforms` - Connected ad platforms
- `api_keys` - API authentication

#### Machine Learning
- `agents` - RL agent configurations
- `training_runs` - Training sessions
- `model_checkpoints` - Saved model states
- `agent_configs` - Agent hyperparameters

#### Experimentation
- `experiments` - A/B test definitions
- `experiment_assignments` - Test group assignments
- `experiment_results` - Test outcomes
- `bandit_arms` - Multi-armed bandit variants

#### Monitoring
- `anomaly_detection` - Detected anomalies

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. **Port Already in Use**

**Error**: `Bind for 0.0.0.0:XXXX failed: port is already allocated`

**Solution**:
```bash
# Check what's using the port
lsof -i :7500  # Replace with actual port

# Stop conflicting service or change AdBot ports
docker-compose down
# Edit docker-compose.yml and .env to use different ports
docker-compose up -d
```

#### 2. **Database Connection Refused**

**Error**: `connection to server at "localhost", port 7500 failed: Connection refused`

**Common Causes & Solutions**:

1. **Docker not running**:
   ```bash
   docker ps | grep adbot-postgres
   # If not running:
   docker-compose up -d postgres
   ```

2. **Wrong port configuration**:
   ```bash
   # Verify .env file
   cat .env | grep DB_PORT
   # Should show: DB_PORT=7500
   ```

3. **Firewall blocking connection**:
   ```bash
   # macOS: Check firewall settings
   sudo pfctl -s rules | grep 7500
   ```

#### 3. **Alembic Migration Errors**

**Error**: `Target database is not up to date`

**Solution**:
```bash
# Check current version
alembic current

# Force upgrade if needed
alembic stamp head
alembic upgrade head
```

**Error**: `Can't locate revision identified by 'XXXXX'`

**Solution**:
```bash
# Check migration files exist
ls migrations/versions/

# If missing, recreate from database
alembic revision --autogenerate -m "Recreate schema"
```

#### 4. **Environment Variable Issues**

**Error**: `Field required DB_PASSWORD`

**Solutions**:

1. **Create .env file**:
   ```bash
   cp .env.example .env
   # Edit .env with actual values
   ```

2. **Export variables for session**:
   ```bash
   export DB_PASSWORD=adbot_password
   export JWT_SECRET=your_secret_here
   ```

3. **Check variable loading**:
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('DB_PASSWORD'))"
   ```

### Database Connection Testing

#### Quick Connection Test

```python
# test_db_connection.py
import psycopg2

try:
    conn = psycopg2.connect(
        host='localhost',
        port=7500,
        database='adbot_dev',
        user='adbot',
        password='adbot_password'
    )
    print("✅ Database connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

#### Verify Tables Created

```bash
# Using psql (if installed)
psql -h localhost -p 7500 -U adbot -d adbot_dev -c "\dt"

# Using Python
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://adbot:adbot_password@localhost:7500/adbot_dev')
cur = conn.cursor()
cur.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema='public'\")
tables = cur.fetchall()
print(f'Found {len(tables)} tables')
for t in tables: print(f'  - {t[0]}')
"
```

## 🛠️ Development Workflows

### Initial Setup Workflow

```bash
# 1. Clone repository
git clone https://github.com/your-org/adbot.git
cd adbot

# 2. Create conda environment
conda create -n adbot python=3.11 -y
conda activate adbot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
cp .env.example .env
# Edit .env with your configuration

# 5. Start infrastructure
docker-compose up -d postgres redis

# 6. Run migrations
alembic upgrade head

# 7. Verify setup
python scripts/test_db_connection.py
```

### Daily Development Workflow

```bash
# Start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs if needed
docker-compose logs -f postgres

# Run application
python -m src.api.main  # Or your entry point

# Stop services when done
docker-compose down
```

### Schema Change Workflow

```bash
# 1. Make model changes in src/models/

# 2. Generate migration
alembic revision --autogenerate -m "Add new field to campaigns"

# 3. Review generated migration
# Edit migrations/versions/XXXX_add_new_field_to_campaigns.py

# 4. Apply migration
alembic upgrade head

# 5. Commit changes
git add migrations/versions/
git commit -m "Add migration for new campaign field"
```

## 📊 Database Administration

### Backup and Restore

```bash
# Backup database
docker exec adbot-postgres pg_dump -U adbot adbot_dev > backup_$(date +%Y%m%d).sql

# Restore database
docker exec -i adbot-postgres psql -U adbot adbot_dev < backup_20250105.sql
```

### Performance Monitoring

```sql
-- Check table sizes
SELECT 
    schemaname AS table_schema,
    tablename AS table_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    min_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## 🔐 Security Considerations

### Production Deployment

1. **Change default passwords** in `.env`
2. **Use SSL connections** for PostgreSQL
3. **Restrict port access** with firewall rules
4. **Use secrets management** (e.g., AWS Secrets Manager)
5. **Enable PostgreSQL SSL**:
   ```yaml
   # docker-compose.yml
   postgres:
     command: postgres -c ssl=on -c ssl_cert_file=/var/lib/postgresql/server.crt
   ```

### Connection String Security

```python
# Use environment variables
import os
from urllib.parse import quote_plus

# Escape special characters in password
password = quote_plus(os.getenv('DB_PASSWORD'))
db_url = f"postgresql://adbot:{password}@localhost:7500/adbot_dev"
```

## 🆘 Emergency Procedures

### Database Corruption Recovery

```bash
# 1. Stop services
docker-compose down

# 2. Backup corrupted data (if possible)
docker cp adbot-postgres:/var/lib/postgresql/data ./corrupted_backup

# 3. Remove volume
docker volume rm adbot_postgres_data

# 4. Restart and restore from backup
docker-compose up -d postgres
docker exec -i adbot-postgres psql -U adbot adbot_dev < last_good_backup.sql
```

### Port Conflict Resolution

```bash
# Find all AdBot services
docker ps --filter "name=adbot"

# Force remove if needed
docker rm -f $(docker ps -aq --filter "name=adbot")

# Clean up networks
docker network prune

# Restart fresh
docker-compose up -d
```

## 📚 Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Docker Image](https://hub.docker.com/_/postgres)
- [Docker Compose Networking](https://docs.docker.com/compose/networking/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)

## 🏁 Conclusion

This guide covers the essential aspects of AdBot's database and port configuration. The sequential port allocation (7500-7507) provides a clean, conflict-free setup for all services. The Alembic migration system ensures database schema changes are tracked and reversible.

For questions or issues not covered here, please:
1. Check the troubleshooting section
2. Review error logs with `docker-compose logs`
3. Consult the team or create an issue

Remember: **Always backup before major migrations!**