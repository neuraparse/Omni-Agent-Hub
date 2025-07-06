"""
Redis manager for caching and session management.

This module provides async Redis operations for caching, session storage,
and real-time data management.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from ..core.logging import LoggerMixin
from ..core.exceptions import DatabaseError


class RedisManager(LoggerMixin):
    """Async Redis manager for caching and session operations."""
    
    def __init__(
        self,
        redis_url: str,
        max_connections: int = 20,
        decode_responses: bool = True
    ):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        self.redis_client: Optional[Redis] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self.logger.info("Initializing Redis connection")
            
            # Create connection pool
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            self.logger.info("Redis initialized successfully")
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_initialization"})
            raise DatabaseError(f"Failed to initialize Redis: {str(e)}")
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Redis connections closed")
    
    async def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            await self.redis_client.ping()
            self.logger.info("Redis connection test successful")
        except Exception as e:
            raise DatabaseError(f"Redis connection test failed: {str(e)}")
    
    def _ensure_initialized(self) -> None:
        """Ensure Redis is initialized."""
        if not self._initialized or not self.redis_client:
            raise DatabaseError("Redis not initialized")
    
    # Basic Operations
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set a key-value pair with optional expiration."""
        self._ensure_initialized()
        
        try:
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = await self.redis_client.set(key, value, ex=expire)
            return result
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_set", "key": key})
            raise DatabaseError(f"Redis SET failed: {str(e)}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        self._ensure_initialized()
        
        try:
            value = await self.redis_client.get(key)
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            self.log_error(e, {"operation": "redis_get", "key": key})
            raise DatabaseError(f"Redis GET failed: {str(e)}")
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        self._ensure_initialized()
        
        try:
            return await self.redis_client.delete(*keys)
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_delete", "keys": keys})
            raise DatabaseError(f"Redis DELETE failed: {str(e)}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        self._ensure_initialized()
        
        try:
            return bool(await self.redis_client.exists(key))
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_exists", "key": key})
            raise DatabaseError(f"Redis EXISTS failed: {str(e)}")
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key."""
        self._ensure_initialized()
        
        try:
            return await self.redis_client.expire(key, seconds)
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_expire", "key": key})
            raise DatabaseError(f"Redis EXPIRE failed: {str(e)}")
    
    # Hash Operations
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        self._ensure_initialized()
        
        try:
            # Serialize complex values
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized_mapping[k] = json.dumps(v)
                else:
                    serialized_mapping[k] = v
            
            return await self.redis_client.hset(name, mapping=serialized_mapping)
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_hset", "hash": name})
            raise DatabaseError(f"Redis HSET failed: {str(e)}")
    
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field value."""
        self._ensure_initialized()
        
        try:
            value = await self.redis_client.hget(name, key)
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            self.log_error(e, {"operation": "redis_hget", "hash": name, "key": key})
            raise DatabaseError(f"Redis HGET failed: {str(e)}")
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields."""
        self._ensure_initialized()
        
        try:
            hash_data = await self.redis_client.hgetall(name)
            
            # Deserialize JSON values
            result = {}
            for k, v in hash_data.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            
            return result
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_hgetall", "hash": name})
            raise DatabaseError(f"Redis HGETALL failed: {str(e)}")
    
    # List Operations
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to the left of a list."""
        self._ensure_initialized()
        
        try:
            # Serialize complex objects
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            
            return await self.redis_client.lpush(name, *serialized_values)
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_lpush", "list": name})
            raise DatabaseError(f"Redis LPUSH failed: {str(e)}")
    
    async def rpop(self, name: str) -> Optional[Any]:
        """Pop value from the right of a list."""
        self._ensure_initialized()
        
        try:
            value = await self.redis_client.rpop(name)
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            self.log_error(e, {"operation": "redis_rpop", "list": name})
            raise DatabaseError(f"Redis RPOP failed: {str(e)}")
    
    async def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """Get list range."""
        self._ensure_initialized()
        
        try:
            values = await self.redis_client.lrange(name, start, end)
            
            # Deserialize JSON values
            result = []
            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value)
            
            return result
            
        except Exception as e:
            self.log_error(e, {"operation": "redis_lrange", "list": name})
            raise DatabaseError(f"Redis LRANGE failed: {str(e)}")

    async def ltrim(self, name: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        self._ensure_initialized()

        try:
            result = await self.redis_client.ltrim(name, start, end)
            return result

        except Exception as e:
            self.log_error(e, {"operation": "redis_ltrim", "list": name})
            raise DatabaseError(f"Redis LTRIM failed: {str(e)}")

    async def llen(self, name: str) -> int:
        """Get list length."""
        self._ensure_initialized()

        try:
            return await self.redis_client.llen(name)

        except Exception as e:
            self.log_error(e, {"operation": "redis_llen", "list": name})
            raise DatabaseError(f"Redis LLEN failed: {str(e)}")

    async def ping(self) -> bool:
        """Ping Redis server to check connection."""
        self._ensure_initialized()

        try:
            result = await self.redis_client.ping()
            return result

        except Exception as e:
            self.log_error(e, {"operation": "redis_ping"})
            raise DatabaseError(f"Redis PING failed: {str(e)}")
    
    # Session Management
    async def create_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        expire_seconds: int = 3600
    ) -> bool:
        """Create a session with expiration."""
        session_key = f"session:{session_id}"
        
        try:
            await self.hset(session_key, session_data)
            await self.expire(session_key, expire_seconds)
            
            self.logger.info("Session created", session_id=session_id)
            return True
            
        except Exception as e:
            self.log_error(e, {"operation": "create_session", "session_id": session_id})
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        session_key = f"session:{session_id}"
        
        try:
            session_data = await self.hgetall(session_key)
            return session_data if session_data else None
            
        except Exception as e:
            self.log_error(e, {"operation": "get_session", "session_id": session_id})
            return None
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        session_key = f"session:{session_id}"
        
        try:
            if await self.exists(session_key):
                await self.hset(session_key, updates)
                return True
            return False
            
        except Exception as e:
            self.log_error(e, {"operation": "update_session", "session_id": session_id})
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_key = f"session:{session_id}"
        
        try:
            deleted = await self.delete(session_key)
            self.logger.info("Session deleted", session_id=session_id)
            return deleted > 0
            
        except Exception as e:
            self.log_error(e, {"operation": "delete_session", "session_id": session_id})
            return False
    
    # Cache Operations
    async def cache_set(
        self,
        cache_key: str,
        data: Any,
        expire_seconds: int = 300
    ) -> bool:
        """Set cache with expiration."""
        key = f"cache:{cache_key}"
        return await self.set(key, data, expire_seconds)
    
    async def cache_get(self, cache_key: str) -> Optional[Any]:
        """Get cached data."""
        key = f"cache:{cache_key}"
        return await self.get(key)
    
    async def cache_delete(self, cache_key: str) -> bool:
        """Delete cached data."""
        key = f"cache:{cache_key}"
        deleted = await self.delete(key)
        return deleted > 0
