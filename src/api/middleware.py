"""
Custom middleware for FastAPI application
"""

import time
import json
from typing import Callable, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..utils.logger import get_logger

logger = get_logger("adbot.api.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            client=request.client.host if request.client else None,
        )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
            )
            
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Clean old requests
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=1)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                requests_count=len(self.requests[client_id]),
                limit=self.requests_per_minute
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.requests[client_id].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.requests[client_id]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int((current_time + timedelta(minutes=1)).timestamp()))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Try to get user ID from auth
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_host = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_host}"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
        self.error_count = defaultdict(int)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            endpoint = f"{method} {path}"
            self.request_count[endpoint] += 1
            self.request_duration[endpoint].append(duration)
            
            # Keep only recent durations (last 1000 requests)
            if len(self.request_duration[endpoint]) > 1000:
                self.request_duration[endpoint] = self.request_duration[endpoint][-1000:]
            
            # Record errors
            if response.status_code >= 400:
                self.error_count[endpoint] += 1
            
            # Add metrics headers
            response.headers["X-Request-Count"] = str(self.request_count[endpoint])
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            endpoint = f"{method} {path}"
            self.error_count[endpoint] += 1
            
            logger.error(
                "Request exception in metrics middleware",
                endpoint=endpoint,
                duration=duration,
                error=str(e)
            )
            
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = {
            "request_counts": dict(self.request_count),
            "error_counts": dict(self.error_count),
            "avg_durations": {},
        }
        
        for endpoint, durations in self.request_duration.items():
            if durations:
                metrics["avg_durations"][endpoint] = sum(durations) / len(durations)
        
        return metrics


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Add server header
        response.headers["Server"] = "AdBot-API"
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Simple response compression middleware"""
    
    def __init__(self, app, minimum_size: int = 1024):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if compression is supported
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding
        
        if not supports_gzip:
            return response
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        compressible_types = [
            "application/json",
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript",
            "application/javascript",
        ]
        
        is_compressible = any(ct in content_type for ct in compressible_types)
        
        if not is_compressible:
            return response
        
        # Check response size
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < self.minimum_size:
            return response
        
        # Add compression hint (actual compression would be handled by reverse proxy)
        response.headers["X-Compression-Hint"] = "gzip"
        
        return response