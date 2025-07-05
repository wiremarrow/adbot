"""
Health check endpoints
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import platform

from ...utils.logger import get_logger
from ...utils.config import ConfigManager

router = APIRouter()
logger = get_logger("adbot.api.health")


def get_config_manager() -> ConfigManager:
    """Dependency to get config manager"""
    return ConfigManager()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AdBot API"
    }


@router.get("/detailed")
async def detailed_health_check(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """Detailed health check with system information"""
    app_config = config_manager.get_app_config()
    
    # System information
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        }
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AdBot API",
        "version": app_config.version,
        "environment": app_config.environment,
        "debug": app_config.debug,
        "system": system_info,
        "components": {
            "database": "unknown",  # Would check actual DB connection
            "redis": "unknown",     # Would check actual Redis connection
            "ml_tracking": "unknown"  # Would check MLflow connection
        }
    }


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes"""
    # Check if all dependencies are ready
    checks = {
        "database": await _check_database(),
        "redis": await _check_redis(),
        "ml_tracking": await _check_ml_tracking(),
    }
    
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for Kubernetes"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _check_database() -> bool:
    """Check database connection"""
    try:
        # Would implement actual database connection check
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False


async def _check_redis() -> bool:
    """Check Redis connection"""
    try:
        # Would implement actual Redis connection check
        return True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        return False


async def _check_ml_tracking() -> bool:
    """Check ML tracking service"""
    try:
        # Would implement actual MLflow connection check
        return True
    except Exception as e:
        logger.error("ML tracking health check failed", error=str(e))
        return False