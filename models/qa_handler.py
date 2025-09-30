import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class QAResult:
    """Result from QA extraction"""
    answer: str
    confidence: float
    start_pos: int
    end_pos: int
    context_used: str
    processing_time_ms: float
    model_confidence: float

@dataclass 
class CacheEntry:
    """Cache entry for QA results"""
    result: QAResult
    timestamp: datetime
    ttl: int

class QACache:
    """Simple in-memory cache for QA results"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, question: str, context: str) -> str:
        """Generate cache key from question and context"""
        combined = f"{question}||{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, question: str, context: str) -> Optional[QAResult]:
        """Get cached result if available and not expired"""
        key = self._generate_key(question, context)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry.timestamp < timedelta(seconds=entry.ttl):
                self.hits += 1
                return entry.result
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, question: str, context: str, result: QAResult, ttl: Optional[int] = None):
        """Cache QA result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        key = self._generate_key(question, context)
        self.cache[key] = CacheEntry(
            result=result,
            timestamp=datetime.now(),
            ttl=ttl or self.default_ttl
        )
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }

class QAHandler:
    """Question Answering handler using DistilBERT-Squad model"""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad", 
                 cache_enabled: bool = True, max_length: int = 512):
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self.cache = QACache() if cache_enabled else None
        self.is_loaded = False
        self.load_time = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self) -> bool:
        """Initialize the QA model asynchronously"""
        try:
            start_time = time.time()
            logger.info(f"🤖 Loading QA model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Create pipeline for easier usage
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"✅ QA model loaded successfully in {self.load_time:.2f}s on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load QA model: {str(e)}")
            return False
    
    def _preprocess_context(self, context: str) -> str:
        """Preprocess context text for better QA performance"""
        # Remove excessive whitespace
        context = ' '.join(context.split())
        
        # Truncate if too long (keep some buffer for question)
        max_context_length = self.max_length - 100  # Reserve space for question
        if len(context) > max_context_length:
            context = context[:max_context_length]
            # Try to cut at sentence boundary
            last_period = context.rfind('.')
            if last_period > max_context_length * 0.8:  # If we can cut at reasonable point
                context = context[:last_period + 1]
        
        return context.strip()
    
    def _validate_inputs(self, question: str, context: str) -> Tuple[bool, str]:
        """Validate question and context inputs"""
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if not context or not context.strip():
            return False, "Context cannot be empty"
        
        if len(question.strip()) < 3:
            return False, "Question too short (minimum 3 characters)"
        
        if len(context.strip()) < 10:
            return False, "Context too short (minimum 10 characters)"
        
        return True, ""
    
    async def extract_answer(self, question: str, context: str, 
                           confidence_threshold: float = 0.1) -> Optional[QAResult]:
        """Extract answer from context using QA model"""
        if not self.is_loaded:
            raise RuntimeError("QA model not loaded. Call initialize() first.")
        
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(question, context)
        if not is_valid:
            logger.warning(f"Invalid QA inputs: {error_msg}")
            return None
        
        # Check cache first
        if self.cache_enabled and self.cache:
            cached_result = self.cache.get(question, context)
            if cached_result:
                logger.debug(f"💾 QA cache hit for question: {question[:50]}...")
                return cached_result
        
        try:
            start_time = time.time()
            
            # Preprocess context
            processed_context = self._preprocess_context(context)
            
            # Run QA model
            result = self.qa_pipeline(
                question=question.strip(),
                context=processed_context,
                max_answer_len=200,
                handle_impossible_answer=True
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Extract result details
            answer = result.get('answer', '').strip()
            model_confidence = result.get('score', 0.0)
            start_pos = result.get('start', 0)
            end_pos = result.get('end', 0)
            
            # Check confidence threshold
            if model_confidence < confidence_threshold:
                logger.debug(f"QA confidence {model_confidence:.3f} below threshold {confidence_threshold}")
                return None
            
            # Create result object
            qa_result = QAResult(
                answer=answer,
                confidence=min(model_confidence, 1.0),  # Ensure confidence <= 1.0
                start_pos=start_pos,
                end_pos=end_pos,
                context_used=processed_context,
                processing_time_ms=processing_time,
                model_confidence=model_confidence
            )
            
            # Cache result if caching enabled and confidence is reasonable
            if self.cache_enabled and self.cache and model_confidence > 0.3:
                self.cache.set(question, context, qa_result)
            
            logger.debug(f"QA extracted: '{answer}' (confidence: {model_confidence:.3f}, time: {processing_time:.1f}ms)")
            return qa_result
            
        except Exception as e:
            logger.error(f"❌ QA extraction error: {str(e)}")
            return None
    
    async def extract_from_multiple(self, question: str, contexts: List[str],
                                  confidence_threshold: float = 0.1) -> List[QAResult]:
        """Extract answers from multiple contexts and rank by confidence"""
        if not contexts:
            return []
        
        results = []
        
        # Process each context
        for i, context in enumerate(contexts):
            if not context or not context.strip():
                continue
                
            try:
                result = await self.extract_answer(question, context, confidence_threshold)
                if result:
                    results.append(result)
                    logger.debug(f"QA result {i+1}/{len(contexts)}: {result.confidence:.3f}")
                    
            except Exception as e:
                logger.warning(f"Error processing context {i+1}: {str(e)}")
                continue
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"QA processed {len(contexts)} contexts, found {len(results)} valid answers")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded QA model"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "max_length": self.max_length,
            "load_time_seconds": self.load_time,
            "cache_enabled": self.cache_enabled,
            "cache_stats": self.cache.stats() if self.cache else None
        }
    
    def clear_cache(self):
        """Clear QA cache"""
        if self.cache:
            self.cache.clear()
            logger.info("QA cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.stats() if self.cache else {"cache_enabled": False}

# Global QA handler instance
qa_handler = QAHandler()
