"""
Exception handlers for FastAPI application
"""

from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

from ..utils.logger import get_logger
from ..integrations.base import (
    PlatformError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    InvalidRequestError,
    ResourceNotFoundError,
)

logger = get_logger("adbot.api.exceptions")


class AdBotAPIException(Exception):
    """Base exception for AdBot API"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = None,
        details: Dict[str, Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(AdBotAPIException):
    """Validation error"""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})}
        )


class NotFoundError(AdBotAPIException):
    """Resource not found error"""
    
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "resource_id": resource_id}
        )


class ConflictError(AdBotAPIException):
    """Resource conflict error"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT",
            details=details
        )


class UnauthorizedError(AdBotAPIException):
    """Unauthorized access error"""
    
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="UNAUTHORIZED"
        )


class ForbiddenError(AdBotAPIException):
    """Forbidden access error"""
    
    def __init__(self, message: str = "Forbidden"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="FORBIDDEN"
        )


class InternalServerError(AdBotAPIException):
    """Internal server error"""
    
    def __init__(self, message: str = "Internal server error", details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="INTERNAL_ERROR",
            details=details
        )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for the FastAPI app"""
    
    @app.exception_handler(AdBotAPIException)
    async def adbot_api_exception_handler(request: Request, exc: AdBotAPIException):
        """Handle AdBot API exceptions"""
        logger.error(
            "AdBot API exception",
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code or "API_ERROR",
                "message": exc.message,
                "details": exc.details,
                "path": request.url.path,
                "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
            }
        )
    
    @app.exception_handler(PlatformError)
    async def platform_error_handler(request: Request, exc: PlatformError):
        """Handle platform integration errors"""
        status_code = 500
        error_code = "PLATFORM_ERROR"
        
        # Map platform errors to HTTP status codes
        if isinstance(exc, AuthenticationError):
            status_code = 401
            error_code = "PLATFORM_AUTH_ERROR"
        elif isinstance(exc, RateLimitError):
            status_code = 429
            error_code = "PLATFORM_RATE_LIMIT"
        elif isinstance(exc, QuotaExceededError):
            status_code = 429
            error_code = "PLATFORM_QUOTA_EXCEEDED"
        elif isinstance(exc, InvalidRequestError):
            status_code = 400
            error_code = "PLATFORM_INVALID_REQUEST"
        elif isinstance(exc, ResourceNotFoundError):
            status_code = 404
            error_code = "PLATFORM_RESOURCE_NOT_FOUND"
        
        logger.error(
            "Platform error",
            platform=exc.platform,
            error_code=exc.error_code,
            message=exc.message,
            status_code=status_code,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_code,
                "message": exc.message,
                "platform": exc.platform,
                "platform_error_code": exc.error_code,
                "path": request.url.path,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(
            "Request validation error",
            errors=exc.errors(),
            path=request.url.path,
            method=request.method
        )
        
        # Format validation errors
        formatted_errors = []
        for error in exc.errors():
            formatted_errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "errors": formatted_errors,
                "path": request.url.path,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "path": request.url.path,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions"""
        logger.warning(
            "Starlette HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "path": request.url.path,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        logger.error(
            "Unhandled exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            path=request.url.path,
            method=request.method
        )
        
        # Don't expose internal error details in production
        message = "Internal server error"
        details = {}
        
        # In debug mode, include more details
        if app.debug:
            message = str(exc)
            details = {
                "type": type(exc).__name__,
                "traceback": traceback.format_exc().split("\n")
            }
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": message,
                "details": details,
                "path": request.url.path,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )