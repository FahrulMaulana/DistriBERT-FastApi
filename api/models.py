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
    """Response model for chat conversation"""
    message: str = Field(..., description="Bot response message")
    intent: str = Field(..., description="Detected intent category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    response_type: str = Field(..., description="Type of response (template, generated, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information if requested")

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
    "jadwal_kuliah": [
        "Kapan jadwal kuliah Informatika besok?",
        "Jam berapa kuliah Algoritma hari ini?",
        "Jadwal kuliah hari Senin gimana?",
        "Ruang kuliah Pemrograman Web di mana?",
        "Siapa dosen pengampu mata kuliah Database?"
    ],
    "pembayaran": [
        "Bagaimana cara bayar UKT?", 
        "Berapa UKT untuk semester ini?",
        "Cara pembayaran semester bagaimana?",
        "Batas waktu pembayaran UKT kapan?",
        "Bisa cicilan pembayaran semester?"
    ],
    "reset_password": [
        "Saya lupa password akun saya",
        "Reset password mahasiswa gimana?",
        "Password saya tidak bisa login",
        "Akun terkunci karena salah password",
        "Bagaimana cara ganti password?"
    ],
    "faq_informasi": [
        "Dimana saya bisa dapat informasi beasiswa?",
        "Info pendaftaran PKL bagaimana?",
        "Informasi tentang wisuda",
        "Syarat mengajukan cuti akademik",
        "Cara mengurus transkrip nilai"
    ],
    "smalltalk": [
        "Halo, selamat pagi!",
        "Hai bot, apa kabar?",
        "Terima kasih atas bantuannya",
        "Selamat siang",
        "Halo, saya butuh bantuan"
    ]
}
