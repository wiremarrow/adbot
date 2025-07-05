"""
Main FastAPI application
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import structlog

from ..utils.config import ConfigManager
from ..utils.logger import setup_logger, get_logger
from .routers import campaigns, agents, experiments, platforms, health
from .middleware import LoggingMiddleware, RateLimitMiddleware
from .exceptions import setup_exception_handlers

# Initialize configuration and logging
config_manager = ConfigManager()
app_config = config_manager.get_app_config()

# Setup structured logging
setup_logger(
    name="adbot.api",
    level="INFO" if not app_config.debug else "DEBUG",
    log_file="logs/api.log"
)

logger = get_logger("adbot.api")

# Create FastAPI app
app = FastAPI(
    title="AdBot API",
    description="AI-Powered Advertising Optimization Platform API",
    version=app_config.version,
    debug=app_config.debug,
    docs_url="/docs" if app_config.debug else None,
    redoc_url="/redoc" if app_config.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.adbot.ai"]
)

# Custom middleware
app.add_middleware(LoggingMiddleware)

if app_config.api.rate_limit_enabled:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=app_config.api.rate_limit_per_minute
    )

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(campaigns.router, prefix="/api/v1/campaigns", tags=["Campaigns"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["RL Agents"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["Experiments"])
app.include_router(platforms.router, prefix="/api/v1/platforms", tags=["Platforms"])


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(
        "AdBot API starting up",
        version=app_config.version,
        environment=app_config.environment,
        debug=app_config.debug
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("AdBot API shutting down")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "AdBot API",
        "version": app_config.version,
        "documentation": "/docs" if app_config.debug else "Contact support for API documentation",
        "status": "operational"
    }


@app.get("/api/v1/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "AdBot API",
        "version": app_config.version,
        "environment": app_config.environment,
        "features": {
            "reinforcement_learning": True,
            "multi_platform": True,
            "real_time_optimization": True,
            "bayesian_optimization": True,
            "a_b_testing": True,
        },
        "supported_platforms": [
            "google_ads",
            "facebook",
            "tiktok", 
            "linkedin",
            "twitter",
            "instagram",
        ],
        "rl_algorithms": [
            "PPO",
            "SAC", 
            "TD3",
            "A2C",
            "DQN",
        ]
    }