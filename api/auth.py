"""
Authentication middleware for API key validation.
"""

import logging
from typing import Optional, List
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from api.config import settings


logger = logging.getLogger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthManager:
    """
    Manager for API key authentication.
    
    Supports multiple API keys with different access levels.
    """
    
    def __init__(self):
        """Initialize authentication manager."""
        # In production, load from environment or database
        self.api_keys = {
            # Format: api_key -> {"name": "key_name", "access_level": "level"}
            "dev-key-12345": {"name": "Development", "access_level": "full"},
            "prod-key-67890": {"name": "Production", "access_level": "full"},
            "readonly-key-11111": {"name": "Read Only", "access_level": "readonly"},
        }
        
        # For demo purposes, allow empty key in development
        self.allow_empty_key = settings.debug
        
        logger.info(f"AuthManager initialized with {len(self.api_keys)} API keys")
    
    def validate_api_key(self, api_key: Optional[str] = Security(api_key_header)) -> str:
        """
        Validate API key and return key name.
        
        Args:
            api_key: API key from header
            
        Returns:
            Key name if valid
            
        Raises:
            HTTPException: If key is invalid
        """
        # Allow empty key in development mode
        if self.allow_empty_key and (api_key is None or api_key == ""):
            logger.debug("Allowing empty API key in development mode")
            return "development"
        
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key header missing",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        key_info = self.api_keys[api_key]
        logger.info(f"API key validated: {key_info['name']}")
        
        return key_info["name"]
    
    def check_access_level(self, key_name: str, required_level: str = "full") -> bool:
        """
        Check if key has required access level.
        
        Args:
            key_name: Name of the API key
            required_level: Required access level
            
        Returns:
            True if has access, False otherwise
        """
        if key_name == "development":
            return True  # Development has full access
        
        # Find key info
        for api_key, info in self.api_keys.items():
            if info["name"] == key_name:
                if info["access_level"] == "full":
                    return True
                elif info["access_level"] == "readonly" and required_level == "readonly":
                    return True
                return False
        
        return False
    
    def add_api_key(self, api_key: str, name: str, access_level: str = "full") -> None:
        """
        Add a new API key.
        
        Args:
            api_key: API key string
            name: Key name
            access_level: Access level (full/readonly)
        """
        self.api_keys[api_key] = {
            "name": name,
            "access_level": access_level
        }
        logger.info(f"Added API key: {name}")
    
    def remove_api_key(self, api_key: str) -> bool:
        """
        Remove an API key.
        
        Args:
            api_key: API key to remove
            
        Returns:
            True if removed, False if not found
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"Removed API key")
            return True
        return False


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Dependency for getting current API key.
    
    Args:
        api_key: API key from header
        
    Returns:
        Key name if valid
        
    Raises:
        HTTPException: If key is invalid
    """
    return auth_manager.validate_api_key(api_key)


async def require_full_access(api_key_name: str = Security(get_current_api_key)) -> str:
    """
    Dependency for requiring full access.
    
    Args:
        api_key_name: Validated API key name
        
    Returns:
        Key name if has full access
        
    Raises:
        HTTPException: If doesn't have full access
    """
    if not auth_manager.check_access_level(api_key_name, "full"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Full access required"
        )
    return api_key_name
