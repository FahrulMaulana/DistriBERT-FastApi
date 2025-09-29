import logging
import time
import functools
from typing import Callable, Any, Dict
from datetime import datetime
import asyncio

# Setup logging utility
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the service"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('distilbert-service.log')
        ]
    )
    return logging.getLogger(__name__)

# Performance monitoring decorator
def monitor_performance(func_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                logging.getLogger().debug(
                    f"✅ {name} completed in {execution_time:.2f}ms"
                )
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logging.getLogger().error(
                    f"❌ {name} failed after {execution_time:.2f}ms: {str(e)}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            name = func_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                logging.getLogger().debug(
                    f"✅ {name} completed in {execution_time:.2f}ms"
                )
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logging.getLogger().error(
                    f"❌ {name} failed after {execution_time:.2f}ms: {str(e)}"
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Request ID generator
import uuid

def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

# Text preprocessing utilities
import re

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep Indonesian characters
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text: str) -> list:
    """Extract meaningful keywords from text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove common stop words (simplified)
    stop_words = {
        'dan', 'atau', 'yang', 'untuk', 'dari', 'ke', 'di', 'pada', 'dengan',
        'adalah', 'akan', 'telah', 'sudah', 'bisa', 'dapat', 'ada', 'tidak',
        'saya', 'anda', 'kamu', 'dia', 'mereka', 'kami', 'kita'
    }
    
    # Split into words and filter
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    
    return words

# Caching utility
from functools import lru_cache
import hashlib

class SimpleCache:
    """Simple in-memory cache for classification results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, text: str) -> Dict | None:
        """Get cached result for text"""
        key = self._get_cache_key(text)
        
        if key in self.cache and not self._is_expired(key):
            return self.cache[key]
        
        return None
    
    def set(self, text: str, result: Dict):
        """Cache result for text"""
        # Cleanup expired entries
        self._cleanup_expired()
        
        # If cache is full, remove oldest entries
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.cache.pop(oldest_key, None)
            self.timestamps.pop(oldest_key, None)
        
        key = self._get_cache_key(text)
        self.cache[key] = result
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hit_rate': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
        }

# Global cache instance
classification_cache = SimpleCache()

# Validation utilities
def validate_intent(intent: str) -> bool:
    """Validate if intent is in allowed list"""
    from config import settings
    return intent in settings.intent_labels

def validate_confidence(confidence: float) -> bool:
    """Validate confidence score"""
    return 0.0 <= confidence <= 1.0

# Error handling utilities
class DistilBERTServiceError(Exception):
    """Base exception for DistilBERT service"""
    pass

class ModelNotLoadedError(DistilBERTServiceError):
    """Raised when model is not loaded"""
    pass

class ClassificationError(DistilBERTServiceError):
    """Raised when classification fails"""
    pass

class ValidationError(DistilBERTServiceError):
    """Raised when input validation fails"""
    pass

# Health check utilities
def get_system_info() -> Dict:
    """Get system information for health checks"""
    import psutil
    import torch
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'torch_version': torch.__version__,
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
    }

# Sample data generator
def generate_sample_texts(intent: str = None, count: int = 5) -> list:
    """Generate sample texts for testing"""
    from api.models import SAMPLE_TEST_DATA
    
    if intent and intent in SAMPLE_TEST_DATA:
        samples = SAMPLE_TEST_DATA[intent]
        return samples[:count] if len(samples) >= count else samples
    
    # Return random samples from all categories
    all_samples = []
    for samples in SAMPLE_TEST_DATA.values():
        all_samples.extend(samples)
    
    import random
    return random.sample(all_samples, min(count, len(all_samples)))
