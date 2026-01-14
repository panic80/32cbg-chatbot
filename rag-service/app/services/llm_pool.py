"""LLM Connection Pool Manager for reducing cold start latency."""

import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import threading
from contextlib import asynccontextmanager

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import Provider
from app.utils.langchain_utils import RetryableLLM

logger = get_logger(__name__)


class LLMConnection:
    """Wrapper for an LLM connection with health tracking."""
    
    def __init__(self, llm: Any, provider: Provider, model: str):
        self.llm = llm
        self.provider = provider
        self.model = model
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()
        self.in_use = False
        self.health_check_failures = 0
        self._lock = threading.Lock()
    
    @property
    def age(self) -> float:
        """Age of connection in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def idle_time(self) -> float:
        """Time since last use in seconds."""
        return (datetime.utcnow() - self.last_used).total_seconds()
    
    def acquire(self) -> bool:
        """Try to acquire this connection."""
        with self._lock:
            if not self.in_use:
                self.in_use = True
                self.last_used = datetime.utcnow()
                return True
            return False
    
    def release(self):
        """Release this connection."""
        with self._lock:
            self.in_use = False
            self.last_used = datetime.utcnow()


class LLMPool:
    """Connection pool for LLM instances."""
    
    def __init__(self, 
                 min_connections: int = 2,
                 max_connections: int = 10,
                 max_idle_time: int = 300,  # 5 minutes
                 max_age: int = 3600):  # 1 hour
        # Use settings for configuration
        self.min_connections = settings.llm_pool_min_connections if hasattr(settings, 'llm_pool_min_connections') else min_connections
        self.max_connections = settings.llm_pool_max_connections if hasattr(settings, 'llm_pool_max_connections') else max_connections
        self.max_idle_time = max_idle_time
        self.max_age = max_age

        if self.min_connections < 1:
            logger.info("LLM pool min_connections set below 1; forcing to 1 for warm cache support")
            self.min_connections = 1
        
        # Pool storage: provider:model -> list of connections
        self._pools: Dict[str, List[LLMConnection]] = {}
        self._pool_lock = threading.Lock()
        self._shutdown = False
        
        # Start background tasks
        self._health_check_task = None
        self._cleanup_task = None
    
    def _get_pool_key(self, provider: Provider, model: str) -> str:
        """Get pool key for provider and model."""
        return f"{provider.value}:{model}"
    
    async def initialize(self):
        """Initialize the pool and start background tasks."""
        logger.info("Initializing LLM connection pool")
        
        # Pre-warm connections for commonly used models (if min_connections > 0)
        if self.min_connections > 0:
            warm_configs: List[tuple[Provider, str]] = []

            default_openai = getattr(settings, 'openai_chat_model', 'gpt-4.1-mini')
            smart_openai = getattr(settings, 'openai_smart_model', 'gpt-5-mini')

            warm_configs.append((Provider.OPENAI, default_openai))
            if smart_openai and smart_openai != default_openai:
                warm_configs.append((Provider.OPENAI, smart_openai))

            anthropic_model = getattr(settings, 'anthropic_chat_model', None)
            if anthropic_model:
                warm_configs.append((Provider.ANTHROPIC, anthropic_model))

            # Remove duplicates while preserving order
            seen = set()
            unique_configs = []
            for provider, model in warm_configs:
                if not model:
                    continue
                key = (provider, model)
                if key not in seen:
                    seen.add(key)
                    unique_configs.append(key)

            for provider, model in unique_configs:
                if self._has_api_key(provider):
                    await self._ensure_min_connections(provider, model)
        else:
            logger.info("Skipping connection pre-warming (min_connections=0)")
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("LLM connection pool initialized")
    
    async def shutdown(self):
        """Shutdown the pool and cleanup resources."""
        logger.info("Shutting down LLM connection pool")
        self._shutdown = True
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Clear all connections
        with self._pool_lock:
            self._pools.clear()
    
    def _has_api_key(self, provider: Provider) -> bool:
        """Check if API key is configured for provider."""
        if provider == Provider.OPENAI:
            return bool(settings.openai_api_key)
        elif provider == Provider.GOOGLE:
            return bool(settings.google_api_key)
        elif provider == Provider.ANTHROPIC:
            return bool(settings.anthropic_api_key)
        elif provider == Provider.OPENROUTER:
            return bool(settings.openrouter_api_key)
        return False
    
    def _create_llm(self, provider: Provider, model: str) -> Any:
        """Create a new LLM instance (sync)."""
        if provider == Provider.OPENAI:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            # Check if it's an O-series reasoning model
            is_o_series = (model and (
                model.startswith('o3') or 
                model.startswith('o4') or
                model == 'o1' or
                model == 'o1-mini'
            ))
            
            # GPT-5 Mini/Nano and GPT-4.1 Mini don't support temperature parameter
            is_gpt5_mini = (model and model.strip().lower() == 'gpt-5-mini')
            is_gpt5_nano = (model and model.strip().lower() == 'gpt-5-nano')
            is_gpt41_mini = (model and model.strip().lower() == 'gpt-4.1-mini')
            
            # Models that don't support temperature parameter
            if is_o_series or is_gpt5_mini or is_gpt5_nano or is_gpt41_mini:
                llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=model,
                    max_tokens=8192
                )
            else:
                llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=model
                )
            return RetryableLLM(llm)

        elif provider == Provider.GOOGLE:
            if not settings.google_api_key:
                raise ValueError("Google API key not configured")

            # Check if it's a Gemini 3 model (supports thinking_level)
            is_gemini_3 = model and (
                model.startswith('gemini-3-') or
                model.startswith('gemini-3.')
            )

            if is_gemini_3:
                llm = ChatGoogleGenerativeAI(
                    google_api_key=settings.google_api_key,
                    model=model,
                    thinking_level="low"  # Use low for faster responses
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    google_api_key=settings.google_api_key,
                    model=model
                )
            return RetryableLLM(llm)

        elif provider == Provider.ANTHROPIC:
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")

            # Check if this is Claude Sonnet 4
            if model == "claude-sonnet-4-20250514":
                # Note: thinking mode requires direct Anthropic API usage
                # LangChain's ChatAnthropic doesn't support thinking parameter yet
                logger.info("Claude Sonnet 4 detected in pool - using default client configuration")

            llm = ChatAnthropic(
                api_key=settings.anthropic_api_key,
                model=model
            )
            return RetryableLLM(llm)

        elif provider == Provider.OPENROUTER:
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API key not configured")

            # OpenRouter uses OpenAI-compatible API
            import os
            llm = ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                model=model,
                default_headers={
                    "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),
                    "X-Title": "CF Travel Bot RAG"
                }
            )
            logger.info(f"Created OpenRouter LLM connection for model: {model}")
            return RetryableLLM(llm)

        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _warm_connection(self, conn: LLMConnection):
        """Warm up a connection with a test query."""
        # Skip warming if disabled
        if not settings.llm_pool_warmup:
            logger.debug(f"Connection warming disabled for {conn.provider.value}:{conn.model}")
            return
            
        try:
            # Simple test query to warm up the connection
            test_message = HumanMessage(content="Hello")
            if conn.provider == Provider.GOOGLE:
                await conn.llm.ainvoke([test_message])
            else:
                await conn.llm.ainvoke([test_message], max_tokens=10)
            logger.debug(f"Warmed up connection for {conn.provider.value}:{conn.model}")
        except Exception as e:
            logger.warning(f"Failed to warm connection: {e}")
            conn.health_check_failures += 1
    
    async def _ensure_min_connections(self, provider: Provider, model: str):
        """Ensure minimum connections exist for a provider/model."""
        pool_key = self._get_pool_key(provider, model)
        
        with self._pool_lock:
            pool = self._pools.get(pool_key, [])
            active_count = len([c for c in pool if c.health_check_failures < 3])
            
            needed = self.min_connections - active_count
            if needed > 0:
                logger.info(f"Creating {needed} connections for {pool_key}")
        
        # Create connections outside the lock
        for _ in range(needed):
            try:
                # Create in thread to avoid blocking
                llm = await asyncio.to_thread(self._create_llm, provider, model)
                conn = LLMConnection(llm, provider, model)
                
                # Warm up the connection
                await self._warm_connection(conn)
                
                # Add to pool
                with self._pool_lock:
                    if pool_key not in self._pools:
                        self._pools[pool_key] = []
                    self._pools[pool_key].append(conn)
                    
            except Exception as e:
                logger.error(f"Failed to create connection for {pool_key}: {e}")
    
    @asynccontextmanager
    async def acquire(self, provider: Provider, model: str):
        """Acquire a connection from the pool."""
        pool_key = self._get_pool_key(provider, model)
        conn = None
        
        try:
            # Try to get an existing connection
            with self._pool_lock:
                pool = self._pools.get(pool_key, [])
                for c in pool:
                    if c.acquire():
                        conn = c
                        logger.debug(f"Acquired existing connection for {pool_key}")
                        break
            
            # If no connection available, create a new one
            if not conn:
                # Check if we're at max connections
                with self._pool_lock:
                    total_connections = sum(len(p) for p in self._pools.values())
                    if total_connections >= self.max_connections:
                        # Try to evict an idle connection
                        evicted = self._evict_idle_connection()
                        if not evicted:
                            raise RuntimeError("Connection pool exhausted")
                
                # Create new connection
                logger.info(f"Creating new connection for {pool_key}")
                llm = await asyncio.to_thread(self._create_llm, provider, model)
                conn = LLMConnection(llm, provider, model)
                conn.acquire()
                
                # Add to pool
                with self._pool_lock:
                    if pool_key not in self._pools:
                        self._pools[pool_key] = []
                    self._pools[pool_key].append(conn)
            
            # Yield the LLM
            yield conn.llm
            
        finally:
            # Release the connection
            if conn:
                conn.release()
    
    def _evict_idle_connection(self) -> bool:
        """Evict the most idle connection. Must be called with lock held."""
        oldest_idle = None
        oldest_pool_key = None
        
        for pool_key, pool in self._pools.items():
            for conn in pool:
                if not conn.in_use and conn.idle_time > 60:  # At least 1 minute idle
                    if not oldest_idle or conn.idle_time > oldest_idle.idle_time:
                        oldest_idle = conn
                        oldest_pool_key = pool_key
        
        if oldest_idle and oldest_pool_key:
            self._pools[oldest_pool_key].remove(oldest_idle)
            logger.info(f"Evicted idle connection for {oldest_pool_key}")
            return True
        
        return False
    
    async def _health_check_loop(self):
        """Periodically check connection health."""
        # Skip health checks if disabled
        if not settings.llm_pool_health_checks:
            logger.info("LLM pool health checks disabled")
            return
            
        while not self._shutdown:
            try:
                health_check_interval = settings.llm_pool_health_check_interval if hasattr(settings, 'llm_pool_health_check_interval') else 60
                await asyncio.sleep(health_check_interval)
                
                with self._pool_lock:
                    all_connections = []
                    for pool in self._pools.values():
                        all_connections.extend(pool)
                
                # Test connections outside the lock
                for conn in all_connections:
                    if not conn.in_use and conn.idle_time > 30:
                        try:
                            test_message = HumanMessage(content="ping")
                            if conn.provider == Provider.GOOGLE:
                                await conn.llm.ainvoke([test_message])
                            else:
                                await conn.llm.ainvoke([test_message], max_tokens=5)
                            conn.health_check_failures = 0
                        except Exception as e:
                            logger.warning(f"Health check failed for {conn.provider.value}:{conn.model}: {e}")
                            conn.health_check_failures += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _cleanup_loop(self):
        """Periodically cleanup old/unhealthy connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
                with self._pool_lock:
                    for pool_key, pool in list(self._pools.items()):
                        # Remove unhealthy or old connections
                        healthy_connections = []
                        for conn in pool:
                            if (conn.health_check_failures >= 3 or 
                                conn.age > self.max_age or
                                (not conn.in_use and conn.idle_time > self.max_idle_time)):
                                logger.info(f"Removing connection for {pool_key}: "
                                          f"failures={conn.health_check_failures}, "
                                          f"age={conn.age}s, idle={conn.idle_time}s")
                            else:
                                healthy_connections.append(conn)
                        
                        self._pools[pool_key] = healthy_connections
                        
                        # Remove empty pools
                        if not healthy_connections:
                            del self._pools[pool_key]
                
                # Ensure minimum connections for active pools
                for pool_key in list(self._pools.keys()):
                    provider_str, model = pool_key.split(":", 1)
                    provider = Provider(provider_str)
                    await self._ensure_min_connections(provider, model)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._pool_lock:
            stats = {
                "total_connections": sum(len(p) for p in self._pools.values()),
                "pools": {}
            }
            
            for pool_key, pool in self._pools.items():
                in_use = sum(1 for c in pool if c.in_use)
                stats["pools"][pool_key] = {
                    "total": len(pool),
                    "in_use": in_use,
                    "available": len(pool) - in_use,
                    "healthy": sum(1 for c in pool if c.health_check_failures == 0)
                }
            
            return stats


# Global pool instance
_llm_pool: Optional[LLMPool] = None


def get_llm_pool() -> LLMPool:
    """Get the global LLM pool instance."""
    global _llm_pool
    if _llm_pool is None:
        _llm_pool = LLMPool()
    return _llm_pool


async def initialize_llm_pool():
    """Initialize the global LLM pool."""
    pool = get_llm_pool()
    await pool.initialize()


async def shutdown_llm_pool():
    """Shutdown the global LLM pool."""
    global _llm_pool
    if _llm_pool:
        await _llm_pool.shutdown()
        _llm_pool = None
# Export singleton instance
llm_pool = get_llm_pool()
