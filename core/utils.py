"""
Common utilities for GAJA Assistant Server.
Contains shared functions to reduce code duplication across modules.
"""

import json
import logging
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


logger = logging.getLogger(__name__)


class JSONUtils:
    """Utilities for JSON handling."""
    
    @staticmethod
    def safe_loads(data: str, default: Any = None) -> Any:
        """Safely parse JSON with default fallback."""
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return default
    
    @staticmethod
    def safe_dumps(data: Any, default: str = "{}") -> str:
        """Safely serialize to JSON with default fallback."""
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries with deep merging."""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = JSONUtils.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


class FileUtils:
    """Utilities for file operations."""
    
    @staticmethod
    async def safe_read_file(file_path: Union[str, Path], encoding: str = "utf-8") -> Optional[str]:
        """Safely read file content asynchronously."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, lambda: path.read_text(encoding=encoding)
            )
            return content
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    @staticmethod
    async def safe_write_file(file_path: Union[str, Path], content: str, encoding: str = "utf-8") -> bool:
        """Safely write content to file asynchronously."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: path.write_text(content, encoding=encoding)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False
    
    @staticmethod
    def ensure_directory(dir_path: Union[str, Path]) -> bool:
        """Ensure directory exists, create if necessary."""
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False


class ValidationUtils:
    """Utilities for data validation."""
    
    @staticmethod
    def validate_user_id(user_id: Any) -> Optional[str]:
        """Validate and normalize user ID."""
        if user_id is None:
            return None
        
        # Convert to string and strip whitespace
        user_id_str = str(user_id).strip()
        
        # Check if empty
        if not user_id_str:
            return None
        
        # Basic validation (alphanumeric, underscore, hyphen)
        if not user_id_str.replace("_", "").replace("-", "").isalnum():
            return None
        
        return user_id_str
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation."""
        if not email or "@" not in email:
            return False
        
        parts = email.split("@")
        if len(parts) != 2:
            return False
        
        local, domain = parts
        if not local or not domain or "." not in domain:
            return False
        
        return True
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes and control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in ["\n", "\r", "\t"])
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()


class APIKeyUtils:
    """Utilities for API key management."""
    
    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """Mask API key for logging purposes."""
        if not api_key:
            return "***EMPTY***"
        
        if len(api_key) <= 8:
            return "***SHORT***"
        
        return f"{api_key[:4]}***{api_key[-4:]}"
    
    @staticmethod
    def validate_api_key_format(api_key: str, provider: str) -> bool:
        """Validate API key format for different providers."""
        if not api_key:
            return False
        
        provider = provider.lower()
        
        if provider == "openai":
            return api_key.startswith("sk-") and len(api_key) > 20
        elif provider == "anthropic":
            return api_key.startswith("sk-ant-") and len(api_key) > 20
        elif provider == "weatherapi":
            # WeatherAPI keys vary in length but usually >= 10 chars alphanumeric
            return len(api_key) >= 10 and api_key.replace("_", "").isalnum()
        elif provider == "newsapi":
            return len(api_key) == 32 and api_key.isalnum()
        else:
            # Generic validation - at least 8 characters
            return len(api_key) >= 8


class ResponseUtils:
    """Utilities for standardized API responses."""
    
    @staticmethod
    def success_response(data: Any = None, message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create standardized success response."""
        response: Dict[str, Any] = {"success": True}
        
        if data is not None:
            response["data"] = data
        
        if message:
            response["message"] = message
        
        response.update(kwargs)
        return response
    
    @staticmethod
    def error_response(error: str, code: Optional[str] = None, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Create standardized error response."""
        response: Dict[str, Any] = {
            "success": False,
            "error": error
        }
        
        if code:
            response["error_code"] = code
        
        if data is not None:
            response["data"] = data
        
        response.update(kwargs)
        return response
    
    @staticmethod
    def paginated_response(items: List[Any], page: int = 1, per_page: int = 10, total: Optional[int] = None) -> Dict[str, Any]:
        """Create paginated response."""
        if total is None:
            total = len(items)
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_items = items[start_idx:end_idx]
        
        return ResponseUtils.success_response(
            data={
                "items": page_items,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "pages": (total + per_page - 1) // per_page,
                    "has_next": end_idx < total,
                    "has_prev": page > 1
                }
            }
        )


class DateTimeUtils:
    """Utilities for date and time operations."""
    
    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC datetime."""
        return datetime.utcnow()
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string."""
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
        """Parse datetime from string."""
        try:
            return datetime.strptime(dt_str, format_str)
        except ValueError:
            return None
    
    @staticmethod
    def timestamp_to_datetime(timestamp: float) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(timestamp)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> float:
        """Convert datetime to timestamp."""
        return dt.timestamp()


class CacheUtils:
    """Simple in-memory cache utilities."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if entry["expires_at"] < datetime.utcnow().timestamp():
            del self._cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        expires_at = datetime.utcnow().timestamp() + ttl
        
        self._cache[key] = {
            "value": value,
            "expires_at": expires_at
        }
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        current_time = datetime.utcnow().timestamp()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry["expires_at"] < current_time
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)


class EnvironmentUtils:
    """Utilities for environment variable handling."""
    
    @staticmethod
    def get_env_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, "").lower()
        return value in ("true", "1", "yes", "on")
    
    @staticmethod
    def get_env_int(key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def get_env_float(key: str, default: float = 0.0) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def get_env_list(key: str, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
        """Get list from environment variable."""
        if default is None:
            default = []
        
        value = os.getenv(key, "")
        if not value:
            return default
        
        return [item.strip() for item in value.split(separator) if item.strip()]


# Global cache instance
default_cache = CacheUtils()


def get_cache() -> CacheUtils:
    """Get default cache instance."""
    return default_cache
