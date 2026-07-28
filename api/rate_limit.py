"""
Rate limiting and request ID tracking middleware.
"""

import time
import uuid
import logging
from collections import defaultdict
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Limits requests per API key or IP address.
    """
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Requests allowed per minute
            requests_per_hour: Requests allowed per hour
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests: key -> [(timestamp, count)]
        self.request_history: Dict[str, list] = defaultdict(list)
        
        logger.info(f"RateLimiter initialized: {requests_per_minute}/min, {requests_per_hour}/hour")
    
    def is_allowed(self, key: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed.
        
        Args:
            key: Identifier (API key or IP)
            
        Returns:
            Tuple of (allowed, error_message)
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        # Clean old requests
        self.request_history[key] = [
            ts for ts in self.request_history[key] if ts > hour_ago
        ]
        
        # Check minute limit
        minute_requests = sum(1 for ts in self.request_history[key] if ts > minute_ago)
        if minute_requests >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
        
        # Check hour limit
        hour_requests = len(self.request_history[key])
        if hour_requests >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
        
        # Record this request
        self.request_history[key].append(now)
        
        return True, None
    
    def get_remaining(self, key: str) -> Dict[str, int]:
        """
        Get remaining requests for a key.
        
        Args:
            key: Identifier
            
        Returns:
            Dictionary with remaining requests
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        minute_requests = sum(1 for ts in self.request_history[key] if ts > minute_ago)
        hour_requests = sum(1 for ts in self.request_history[key] if ts > hour_ago)
        
        return {
            "remaining_per_minute": max(0, self.requests_per_minute - minute_requests),
            "remaining_per_hour": max(0, self.requests_per_hour - hour_requests),
            "total_requests": len(self.request_history[key])
        }


class RequestIDMiddleware:
    """
    Middleware to add unique request IDs to all requests.
    """
    
    async def __call__(self, request: Request, call_next):
        """
        Add request ID and process request.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response with request ID header
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add to request state for access in routes
        request.state.start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Add processing time
        process_time = time.time() - request.state.start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"
        
        return response


class RateLimitMiddleware:
    """
    Middleware for rate limiting.
    """
    
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize rate limit middleware.
        
        Args:
            rate_limiter: RateLimiter instance
        """
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        """
        Check rate limit and process request.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response or error if rate limited
        """
        # Get identifier (API key or IP)
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            identifier = f"api_key:{api_key}"
        else:
            identifier = f"ip:{request.client.host}"
        
        # Check rate limit
        allowed, error_message = self.rate_limiter.is_allowed(identifier)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {identifier}")
            
            # Get remaining info
            remaining = self.rate_limiter.get_remaining(identifier)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": error_message,
                    "retry_after": 60,
                    "remaining": remaining
                }
            )
        
        # Add rate limit info to headers
        remaining = self.rate_limiter.get_remaining(identifier)
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining["remaining_per_minute"])
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
        
        return response


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
