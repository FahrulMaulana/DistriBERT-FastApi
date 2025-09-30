import logging
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Union
import uvicorn
from datetime import datetime

# Internal imports
from config import settings, RESPONSE_TEMPLATES
from models.distilbert_handler import DistilBERTHandler
from models.qa_handler import qa_handler
from models.response_generator import knowledge_base, HybridResponseGenerator
from api.models import (
    ClassificationRequest, ClassificationResponse,
    BatchClassificationRequest, BatchClassificationResponse,
    HealthResponse, ModelInfoResponse, IntentsListResponse,
    TestClassificationRequest, ErrorResponse, SAMPLE_TEST_DATA,
    ChatRequest, ChatResponse
)
from utils.helpers import (
    setup_logging, monitor_performance, generate_request_id,
    classification_cache, DistilBERTServiceError, ModelNotLoadedError,
    ClassificationError, get_system_info, generate_sample_texts
)

# Setup logging
logger = setup_logging(settings.log_level)

# Global variables
distilbert_handler = DistilBERTHandler()
hybrid_generator = HybridResponseGenerator(qa_handler, knowledge_base)
service_start_time = datetime.utcnow()

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("🚀 Starting DistilBERT Classification Service...")
    logger.info(f"📊 Configuration: {settings.model_name} with {len(settings.intent_labels)} intents")
    
    # Initialize DistilBERT model
    distilbert_success = await distilbert_handler.initialize()
    
    # Initialize hybrid system
    hybrid_generator.set_classification_handler(distilbert_handler)
    hybrid_success = await hybrid_generator.initialize()
    
    if distilbert_success and hybrid_success:
        logger.info("✅ Hybrid QA system startup completed successfully!")
    else:
        logger.error("❌ System startup failed - continuing with degraded functionality")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down DistilBERT Classification Service...")
    classification_cache.clear()
    logger.info("✅ Shutdown completed successfully")

# Create FastAPI application
app = FastAPI(
    title="Hybrid QA Chatbot Service",
    description="""
    🤖 **Advanced Hybrid Question-Answering Chatbot**
    
    This service combines DistilBERT classification with DistilBERT-Squad QA extraction to provide intelligent, context-aware responses from a comprehensive knowledge base.
    
    **Features:**
    - 🧠 Hybrid QA system (Classification + Knowledge Extraction)
    - 📚 Rich knowledge base with 25+ detailed contexts
    - 🎯 Dual processing modes: Knowledge-based and Conversational
    - ⚡ Sub-500ms response times with caching
    - 🔍 Advanced question answering with confidence scoring
    - 💾 Smart caching for optimal performance
    
    **Processing Modes:**
    - **Knowledge Mode**: Uses QA extraction from detailed contexts
    - **Conversational Mode**: Uses template responses for social interactions
    - **Fallback Mode**: Intelligent fallback for edge cases
    
    **Categories:**
    - Product (6 intents), Pricing (6 intents), Guides (6 intents)
    - Policies (5 intents), Account (4 intents), Company (3 intents)
    - Conversational (5 intents)
    """,
    version="1.0.0",
    contact={
        "name": "Campus IT Support",
        "email": "support@kampus.ac.id"
    },
    lifespan=lifespan
)

# Security setup (optional)
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API key (optional security)"""
    if not credentials:
        return True  # Allow requests without auth for now
    
    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    request_id = generate_request_id()
    
    logger.info(f"📥 {request_id} - {request.method} {request.url.path} from {request.client.host}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"📤 {request_id} - {response.status_code} in {process_time:.2f}ms")
    
    return response

# Exception handlers
@app.exception_handler(DistilBERTServiceError)
async def service_exception_handler(request: Request, exc: DistilBERTServiceError):
    """Handle service-specific exceptions"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            detail="DistilBERT service error",
            timestamp=datetime.utcnow().isoformat(),
            request_id=generate_request_id()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.log_level == "DEBUG" else None,
            timestamp=datetime.utcnow().isoformat(),
            request_id=generate_request_id()
        ).dict()
    )

# API Routes

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - service information"""
    return {
        "service": "Hybrid QA Chatbot Service",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": distilbert_handler.is_loaded,
        "uptime_seconds": (datetime.utcnow() - service_start_time).total_seconds(),
        "endpoints": {
            "chat": "/chat",
            "classify": "/classify",
            "batch_classify": "/batch-classify", 
            "health": "/health",
            "model_info": "/model-info",
            "intents": "/intents"
        }
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@monitor_performance("chat")
async def chat_conversation(
    request: ChatRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    💬 **Hybrid QA Chat Endpoint**
    
    Advanced conversational AI that combines intent classification with question-answering 
    to provide intelligent responses from a comprehensive knowledge base.
    
    **Processing Flow:**
    1. **Classification**: Identifies user intent using DistilBERT
    2. **Mode Selection**: Chooses knowledge, conversational, or fallback mode
    3. **Response Generation**: 
       - Knowledge mode: QA extraction from detailed contexts
       - Conversational mode: Template-based responses
       - Fallback mode: Intelligent fallback responses
    
    **Example Usage:**
    ```python
    {
        "message": "What are the key features of your chatbot platform?",
        "user_id": "user123",
        "include_debug": false
    }
    ```
    
    **Returns:**
    - Intelligent context-aware responses
    - Processing mode and source information
    - Confidence scores and metadata
    - Sub-500ms response times
    """
    try:
        # Generate hybrid response
        response = await hybrid_generator.generate_response(
            question=request.message,
            user_id=request.user_id,
            include_debug=request.include_debug
        )
        
        # Convert to API response format
        return ChatResponse(
            message=response.message,
            intent=response.intent,
            confidence=response.confidence,
            mode=response.mode,
            source=response.source,
            metadata=response.metadata,
            processing_time_ms=response.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"❌ Hybrid chat error: {str(e)}")
        # Return error response
        return ChatResponse(
            message="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            intent="error",
            confidence=0.0,
            mode="fallback",
            source="error",
            metadata={"error": str(e)},
            processing_time_ms=0.0
        )

@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
@monitor_performance("classify_single")
async def classify_intent(
    request: ClassificationRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    🎯 **Classify a single text for campus intent**
    
    Analyzes input text using hybrid DistilBERT + keyword matching approach to determine the most likely intent category.
    
    **Example Usage:**
    ```python
    {
        "text": "Kapan jadwal kuliah Informatika besok?",
        "include_debug": false
    }
    ```
    
    **Returns:**
    - Intent category and confidence score
    - Processing method used (neural network, keywords, or combined)
    - Processing time and optional debug information
    """
    try:
        # Check cache first
        if settings.enable_caching:
            cached_result = classification_cache.get(request.text)
            if cached_result:
                logger.debug(f"💾 Cache hit for text: {request.text[:50]}...")
                return ClassificationResponse(**cached_result)
        
        # Perform classification
        if not distilbert_handler.is_loaded:
            raise ModelNotLoadedError("DistilBERT model not loaded")
        
        result = await distilbert_handler.classify_intent(
            request.text, 
            request.include_debug
        )
        
        # Cache result if caching enabled
        if settings.enable_caching and result['confidence'] > 0.5:
            classification_cache.set(request.text, result)
        
        return ClassificationResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Classification error: {str(e)}")
        raise ClassificationError(f"Failed to classify text: {str(e)}")

@app.post("/batch-classify", response_model=BatchClassificationResponse, tags=["Classification"])
@monitor_performance("classify_batch")
async def batch_classify_intents(
    request: BatchClassificationRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    🚀 **Classify multiple texts in batch for better performance**
    
    Processes multiple texts simultaneously with optimized performance for bulk operations.
    
    **Example Usage:**
    ```python
    {
        "texts": [
            "Kapan jadwal kuliah besok?",
            "Bagaimana cara bayar UKT?", 
            "Saya lupa password"
        ],
        "include_debug": false
    }
    ```
    """
    try:
        start_time = time.time()
        
        if not distilbert_handler.is_loaded:
            raise ModelNotLoadedError("DistilBERT model not loaded")
        
        # Process batch
        results = await distilbert_handler.batch_classify(
            request.texts,
            request.include_debug
        )
        
        # Convert to response models
        classification_responses = [ClassificationResponse(**result) for result in results]
        
        total_time = (time.time() - start_time) * 1000
        average_time = total_time / len(request.texts)
        
        return BatchClassificationResponse(
            results=classification_responses,
            total_processed=len(request.texts),
            total_time_ms=round(total_time, 2),
            average_time_ms=round(average_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Batch classification error: {str(e)}")
        raise ClassificationError(f"Failed to process batch: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    💚 **Health check endpoint for monitoring and load balancing**
    
    Returns comprehensive service health information including model status, uptime, and system metrics.
    """
    try:
        model_info = distilbert_handler.get_model_info()
        health_status = distilbert_handler.get_health_status()
        
        uptime = (datetime.utcnow() - service_start_time).total_seconds()
        
        return HealthResponse(
            status="healthy" if distilbert_handler.is_loaded else "degraded",
            model_loaded=distilbert_handler.is_loaded,
            available_intents=settings.intent_labels,
            uptime_seconds=uptime,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            available_intents=[],
            uptime_seconds=0,
            model_info={"error": str(e)}
        )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Information"])
async def get_model_info():
    """
    📊 **Get detailed DistilBERT model information and statistics**
    
    Returns comprehensive information about the loaded model, configuration, and performance metrics.
    """
    try:
        model_info = distilbert_handler.get_model_info()
        return ModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"❌ Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/intents", response_model=IntentsListResponse, tags=["Information"])
async def get_available_intents():
    """
    📋 **Get list of available intent categories with descriptions**
    
    Returns all supported intent categories that the model can classify.
    """
    descriptions = {
        "greeting": "Greetings and salutations - hello messages, good morning, welcome interactions",
        "question": "General questions and inquiries - what, how, when, where, why questions", 
        "help_request": "Help and assistance requests - asking for help, support, guidance",
        "information": "Information seeking - requests for data, facts, explanations, details",
        "weather": "Weather related questions - temperature, forecast, climate inquiries",
        "food_recipe": "Food and recipe questions - cooking instructions, ingredients, dishes",
        "technology": "Technology related questions - computers, software, apps, gadgets",
        "smalltalk": "Casual conversation - how are you, thank you, general politeness",
        "goodbye": "Farewell messages - goodbye, see you later, take care",
        "unknown": "Unrecognized or ambiguous intents that don't fit other categories"
    }
    
    return IntentsListResponse(
        intents=settings.intent_labels,
        total=len(settings.intent_labels),
        descriptions=descriptions
    )

@app.post("/test-classification", tags=["Testing"])
async def test_classification_samples(
    request: TestClassificationRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    🧪 **Test classification with predefined sample data**
    
    Useful for validating model performance with known good examples.
    """
    try:
        if not distilbert_handler.is_loaded:
            raise ModelNotLoadedError("DistilBERT model not loaded")
        
        # Generate sample texts
        if request.sample_type == "specific" and request.intent_category:
            if request.intent_category not in settings.intent_labels:
                raise HTTPException(status_code=400, detail=f"Invalid intent: {request.intent_category}")
            samples = generate_sample_texts(request.intent_category, request.count)
        else:
            samples = generate_sample_texts(count=request.count)
        
        # Classify samples
        results = await distilbert_handler.batch_classify(samples, include_debug=True)
        
        # Add expected vs actual comparison
        test_results = []
        for i, (sample, result) in enumerate(zip(samples, results)):
            # Try to determine expected intent from sample data
            expected_intent = None
            for intent, intent_samples in SAMPLE_TEST_DATA.items():
                if sample in intent_samples:
                    expected_intent = intent
                    break
            
            test_result = {
                "sample_text": sample,
                "expected_intent": expected_intent,
                "predicted_intent": result["intent"],
                "confidence": result["confidence"],
                "correct": expected_intent == result["intent"] if expected_intent else None,
                "source": result["source"]
            }
            test_results.append(test_result)
        
        # Calculate accuracy if we have expected results
        correct_predictions = sum(1 for r in test_results if r["correct"] is True)
        total_with_expected = sum(1 for r in test_results if r["correct"] is not None)
        accuracy = correct_predictions / total_with_expected if total_with_expected > 0 else None
        
        return {
            "test_results": test_results,
            "summary": {
                "total_samples": len(samples),
                "samples_with_expected": total_with_expected,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "average_confidence": sum(r["confidence"] for r in results) / len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Test classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.get("/cache-stats", tags=["Monitoring"])
async def get_cache_statistics():
    """📊 Get caching statistics for performance monitoring"""
    if settings.enable_caching:
        return {
            "caching_enabled": True,
            "cache_stats": classification_cache.stats(),
            "cache_size": len(classification_cache.cache),
            "max_cache_size": classification_cache.max_size,
            "ttl_seconds": classification_cache.ttl_seconds
        }
    else:
        return {"caching_enabled": False}

@app.post("/clear-cache", tags=["Administration"])
async def clear_classification_cache(authenticated: bool = Depends(verify_api_key)):
    """🗑️ Clear the classification cache (admin only)"""
    if settings.enable_caching:
        classification_cache.clear()
        return {"message": "Cache cleared successfully"}
    else:
        return {"message": "Caching is disabled"}

@app.get("/hybrid-stats", tags=["Monitoring"])
async def get_hybrid_system_stats():
    """🔍 Get comprehensive hybrid QA system statistics"""
    try:
        return {
            "hybrid_system": hybrid_generator.get_stats(),
            "service_uptime": (datetime.utcnow() - service_start_time).total_seconds(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get hybrid stats: {str(e)}"}

@app.get("/system-info", tags=["Monitoring"])
async def get_system_information():
    """🖥️ Get system resource information"""
    try:
        system_info = get_system_info()
        return {
            "system": system_info,
            "service": {
                "uptime_seconds": (datetime.utcnow() - service_start_time).total_seconds(),
                "model_loaded": distilbert_handler.is_loaded,
                "cache_enabled": settings.enable_caching
            }
        }
    except Exception as e:
        return {"error": f"Failed to get system info: {str(e)}"}

# Run the application
if __name__ == "__main__":
    logger.info("🚀 Starting DistilBERT Classification Service...")
    
    uvicorn.run(
        app,
        host=settings.service_host,
        port=settings.service_port,
        log_level=settings.log_level.lower(),
        access_log=True,
        reload=False,  # Set to True for development
        workers=1  # Use 1 worker for model loading
    )
