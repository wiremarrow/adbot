# AdBot Quick Reference Card

## 🚀 Essential Commands

### Environment Setup
```bash
# Create and activate environment
conda create -n adbot python=3.11 -y
conda activate adbot

# Install dependencies
pip install -r requirements.txt
pip install stable-baselines3
```

### Docker Services
```bash
# Start all services
docker-compose up -d

# Start only database services
docker-compose up -d postgres redis

# View service status
docker-compose ps

# View logs
docker-compose logs -f postgres

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Check current version
alembic current

# View history
alembic history

# Rollback one version
alembic downgrade -1
```

### Database Connection Test
```bash
# Quick Python test (should print ✅ Connected)
python -c "import psycopg2; psycopg2.connect('postgresql://adbot:adbot_password@localhost:7500/adbot_dev'); print('✅ Connected')"

# Check tables
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://adbot:adbot_password@localhost:7500/adbot_dev')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s', ('public',))
print(f'Tables: {cur.fetchone()[0]}')
"
```

### Development Workflow
```bash
# 1. Start services
docker-compose up -d postgres redis

# 2. Run migrations
alembic upgrade head

# 3. Test RL environment (should show 3 passed, 0 failed)
python test_rl_training.py

# 4. Start API server (when ready)
uvicorn src.api.main:app --reload --port 7507

# 5. Run tests
pytest tests/

# 6. Format code
black src/ tests/
ruff check src/ --fix
```

## 🔌 Port Reference

| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 7500 | `postgresql://localhost:7500/adbot_dev` |
| Redis | 7501 | `redis://localhost:7501` |
| MLflow | 7502 | `http://localhost:7502` |
| Prometheus | 7503 | `http://localhost:7503` |
| Grafana | 7504 | `http://localhost:7504` |
| Kafka | 7505 | `localhost:7505` |
| Zookeeper | 7506 | `localhost:7506` |
| API | 7507 | `http://localhost:7507` |

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Find what's using a port
lsof -i :7500

# Kill process using port
kill -9 $(lsof -t -i:7500)
```

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check PostgreSQL logs
docker logs adbot-postgres --tail 50

# Test connection
nc -zv localhost 7500
```

### Migration Errors
```bash
# Reset migration state
alembic stamp head

# Recreate from scratch
alembic downgrade base
alembic upgrade head
```

### Environment Variables
```bash
# Check if .env exists
ls -la .env

# Create from example
cp .env.example .env

# Source manually if needed
export $(cat .env | xargs)
```

## 📁 Key Files

```
adbot/
├── .env                    # Environment variables (create from .env.example)
├── docker-compose.yml      # Service definitions
├── alembic.ini            # Migration configuration
├── configs/default.yaml    # Application config
├── migrations/
│   ├── env.py             # Migration environment
│   └── versions/          # Migration files
└── src/
    ├── models/            # Database models
    ├── api/               # FastAPI application
    └── utils/config.py    # Configuration classes
```

## 🎯 Common Tasks

### Add New Database Model
1. Create model in `src/models/`
2. Import in `src/models/__init__.py`
3. Generate migration: `alembic revision --autogenerate -m "Add model"`
4. Review and apply: `alembic upgrade head`

### Change Port Configuration
1. Update `.env`
2. Update `docker-compose.yml`
3. Update `configs/default.yaml`
4. Update `src/utils/config.py` defaults
5. Restart services: `docker-compose down && docker-compose up -d`

### Debug Database Issues
```bash
# 1. Check service is running
docker-compose ps

# 2. Test connection
python scripts/test_db_connection.py

# 3. Check tables exist
psql -h localhost -p 7500 -U adbot -d adbot_dev -c "\dt"

# 4. Review logs
docker-compose logs postgres --tail 100
```

## 💡 Pro Tips

1. **Always backup before migrations**: 
   ```bash
   docker exec adbot-postgres pg_dump -U adbot adbot_dev > backup.sql
   ```

2. **Use transactions for data changes**:
   ```python
   with session.begin():
       # Your changes here
       session.commit()
   ```

3. **Check ports before starting**:
   ```bash
   for port in 7500 7501 7502 7503 7504 7505 7506 7507; do
       echo -n "Port $port: "
       lsof -i :$port >/dev/null 2>&1 && echo "IN USE" || echo "FREE"
   done
   ```

4. **Monitor Docker resources**:
   ```bash
   docker stats --no-stream
   ```

5. **Clean up unused Docker resources**:
   ```bash
   docker system prune -a --volumes
   ```

---
*Keep this reference handy for quick access to common AdBot commands!*