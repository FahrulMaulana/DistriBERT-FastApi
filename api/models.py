from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat conversation"""
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=512, 
        description="User message for chat",
        example="Kapan jadwal kuliah Informatika besok?"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for session tracking"
    )
    include_debug: bool = Field(
        False, 
        description="Include classification debug information"
    )
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v.strip()

class ChatResponse(BaseModel):
    """Response model for hybrid chat conversation"""
    message: str = Field(..., description="Bot response message")
    intent: str = Field(..., description="Detected intent category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    mode: str = Field(..., description="Processing mode: knowledge, conversational, or fallback")
    source: str = Field(..., description="Response source: qa_extraction, template, or fallback")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ClassificationRequest(BaseModel):
    """Request model for single text classification"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=512, 
        description="Text to classify for campus intent",
        example="Kapan jadwal kuliah Informatika besok?"
    )
    include_debug: bool = Field(
        False, 
        description="Include detailed debug information in response"
    )
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()

class ClassificationResponse(BaseModel):
    """Response model for classification results"""
    intent: str = Field(..., description="Classified intent category")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    source: str = Field(..., description="Classification method used")
    text_length: int = Field(..., description="Length of input text")
    timestamp: str = Field(..., description="Processing timestamp (ISO format)")
    processing_time_ms: Optional[float] = Field(
        None, 
        description="Processing time in milliseconds"
    )
    debug_info: Optional[Dict[str, Any]] = Field(
        None, 
        description="Debug information (only if requested)"
    )

class BatchClassificationRequest(BaseModel):
    """Request model for batch text classification"""
    texts: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=20,  # Limit batch size
        description="List of texts to classify"
    )
    include_debug: bool = Field(
        False, 
        description="Include debug information for all results"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        # Validate each text
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 512:
                raise ValueError(f'Text at index {i} too long (max 512 characters)')
        
        return [text.strip() for text in v]

class BatchClassificationResponse(BaseModel):
    """Response model for batch classification"""
    results: List[ClassificationResponse] = Field(
        ..., 
        description="List of classification results"
    )
    total_processed: int = Field(..., description="Total number of texts processed")
    total_time_ms: float = Field(..., description="Total processing time")
    average_time_ms: float = Field(..., description="Average time per classification")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether DistilBERT model is loaded")
    available_intents: List[str] = Field(..., description="List of available intent categories")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    model_info: Dict[str, Any] = Field(..., description="Detailed model information")

class ModelInfoResponse(BaseModel):
    """Detailed model information response"""
    model_name: str = Field(..., description="Name of the DistilBERT model")
    model_cache_dir: str = Field(..., description="Model cache directory")
    is_loaded: bool = Field(..., description="Model load status")
    load_time_seconds: Optional[float] = Field(None, description="Time taken to load model")
    available_intents: List[str] = Field(..., description="Available intent categories")
    total_intents: int = Field(..., description="Total number of intent categories")
    confidence_threshold: float = Field(..., description="Confidence threshold setting")
    max_sequence_length: int = Field(..., description="Maximum input sequence length")
    total_keywords: int = Field(..., description="Total keywords across all categories")
    keyword_categories: Dict[str, int] = Field(
        ..., 
        description="Number of keywords per intent category"
    )
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    pytorch_version: str = Field(..., description="PyTorch version")

class IntentsListResponse(BaseModel):
    """Response model for available intents"""
    intents: List[str] = Field(..., description="List of available intent categories")
    total: int = Field(..., description="Total number of intents")
    descriptions: Dict[str, str] = Field(
        ...,
        description="Intent descriptions",
        example={
            "jadwal_kuliah": "Schedule and class information queries",
            "pembayaran": "Payment and tuition related questions", 
            "reset_password": "Account and authentication issues",
            "faq_informasi": "General campus information requests",
            "smalltalk": "Casual conversation and greetings",
            "unknown": "Unrecognized or ambiguous intents"
        }
    )

class TestClassificationRequest(BaseModel):
    """Request model for testing classification with sample data"""
    sample_type: str = Field(
        "random",
        description="Type of sample to test",
        pattern="^(random|all|specific)$"
    )
    intent_category: Optional[str] = Field(
        None,
        description="Specific intent category to test (if sample_type='specific')"
    )
    count: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of samples to test"
    )

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

# Sample test data for validation
SAMPLE_TEST_DATA = {
    # Knowledge-based samples
    "product_features": [
        "What are the main features of your chatbot platform?",
        "Tell me about your AI capabilities",
        "What can your chatbot do?",
        "What features does your platform offer?",
        "How does your chatbot work?"
    ],
    "pricing_basic": [
        "How much does the basic plan cost?",
        "What's included in the starter plan?",
        "Tell me about your cheapest option",
        "Basic plan pricing information",
        "What does the $29 plan include?"
    ],
    "setup_guide": [
        "How do I get started?",
        "Setup instructions please",
        "How to configure the chatbot?",
        "Getting started guide",
        "Initial setup steps"
    ],
    "privacy_policy": [
        "What's your privacy policy?",
        "How do you handle user data?",
        "Data privacy information",
        "What data do you collect?",
        "Privacy and data protection"
    ],
    "account_creation": [
        "How do I create an account?",
        "Sign up process",
        "How to register?",
        "Account creation steps",
        "New user registration"
    ],
    "company_about": [
        "Tell me about your company",
        "Who are you?",
        "Company information",
        "About your organization",
        "Your company background"
    ],
    
    # Conversational samples
    "conversational_greeting": [
        "Hello!",
        "Hi there!",
        "Good morning!",
        "Hey, how are you?",
        "Greetings!"
    ],
    "conversational_thanks": [
        "Thank you so much!",
        "Thanks for your help",
        "I appreciate it",
        "Thanks a lot",
        "Much appreciated"
    ],
    "conversational_goodbye": [
        "Goodbye!",
        "See you later",
        "Take care",
        "Farewell",
        "Bye bye"
    ],
    "conversational_chitchat": [
        "How are you doing?",
        "What's up?",
        "How's your day?",
        "Nice to meet you",
        "How are things?"
    ],
    "conversational_feedback": [
        "Great service!",
        "I have some feedback",
        "This is really helpful",
        "Excellent response",
        "Could be improved"
    ]
}
