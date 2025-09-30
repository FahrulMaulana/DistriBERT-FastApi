import os
from typing import List, Dict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Service Configuration
    service_host: str = "0.0.0.0"
    service_port: int = 8000
    
    # Model Configuration
    model_name: str = "distilbert-base-uncased-distilled-squad"  # QA model
    model_cache_dir: str = "./models/cache"
    confidence_threshold: float = 0.3  # Lower for QA approach
    max_sequence_length: int = 512  # Longer for QA context
    batch_size: int = 16
    
    # Intent Labels for Hybrid QA System
    intent_labels: List[str] = [
        # Knowledge-based intents (will use QA extraction)
        "product_features", "product_technical_specs", "product_integration", 
        "product_ai_capabilities", "product_security", "product_customization",
        "pricing_basic", "pricing_professional", "pricing_enterprise", 
        "pricing_api", "pricing_add_ons", "pricing_comparison",
        "setup_guide", "integration_guide", "optimization_guide", 
        "troubleshooting_guide", "advanced_features", "deployment_strategies",
        "privacy_policy", "terms_of_service", "data_retention", 
        "security_policy", "compliance_standards",
        "account_creation", "account_management", "billing_support", "account_security",
        "company_about", "company_team", "company_careers",
        
        # Conversational intents (will use templates)
        "conversational_greeting", "conversational_thanks", "conversational_goodbye", 
        "conversational_chitchat", "conversational_feedback",
        
        "unknown"  # Fallback intent
    ]
    
    # QA Configuration
    qa_confidence_threshold: float = 0.3
    qa_model_name: str = "distilbert-base-cased-distilled-squad"
    qa_max_answer_length: int = 200
    qa_cache_enabled: bool = True
    
    # Security
    api_key: str = "distilbert-service-2024"
    cors_origins: List[str] = [
        "http://localhost:3002",
        "http://localhost:3003", 
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003"
    ]
    
    # Performance
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl: int = 300
    
    # Logging
    log_level: str = "INFO"
    
    model_config = {
        "env_file": ".env",
        "protected_namespaces": ()
    }

# Global settings instance
settings = Settings()

# Enhanced keyword patterns for hybrid QA system
INTENT_KEYWORDS: Dict[str, List[str]] = {
    # Product intents
    "product_features": ["features", "capabilities", "functionality", "what can", "platform", "ai", "chatbot"],
    "product_technical_specs": ["technical", "specs", "requirements", "architecture", "performance", "cpu", "memory"],
    "product_integration": ["integration", "api", "connect", "crm", "platforms", "third party", "webhook"],
    "product_ai_capabilities": ["ai", "artificial intelligence", "nlp", "machine learning", "sentiment", "language"],
    "product_security": ["security", "encryption", "privacy", "gdpr", "compliance", "authentication"],
    "product_customization": ["customize", "branding", "white label", "theme", "personality", "configuration"],
    
    # Pricing intents
    "pricing_basic": ["basic plan", "starter", "small business", "cheap", "affordable", "29", "39"],
    "pricing_professional": ["professional", "business", "99", "129", "growing", "advanced"],
    "pricing_enterprise": ["enterprise", "unlimited", "custom", "499", "large", "organization"],
    "pricing_api": ["api pricing", "pay per use", "developer", "calls", "usage based"],
    "pricing_add_ons": ["add-on", "additional", "extra", "premium", "voice", "analytics"],
    "pricing_comparison": ["compare", "difference", "plans", "which plan", "vs", "versus"],
    
    # Guide intents
    "setup_guide": ["setup", "getting started", "how to start", "configuration", "initial"],
    "integration_guide": ["how to integrate", "connect", "embed", "implementation"],
    "optimization_guide": ["optimize", "improve", "performance", "best practices"],
    "troubleshooting_guide": ["troubleshoot", "problem", "issue", "error", "not working"],
    "advanced_features": ["advanced", "custom", "sophisticated", "complex"],
    "deployment_strategies": ["deploy", "production", "rollout", "implementation"],
    
    # Policy intents  
    "privacy_policy": ["privacy", "data collection", "personal information"],
    "terms_of_service": ["terms", "conditions", "agreement", "usage policy"],
    "data_retention": ["data retention", "how long", "storage", "delete"],
    "security_policy": ["security policy", "protection", "safety"],
    "compliance_standards": ["compliance", "certification", "standards", "audit"],
    
    # Account intents
    "account_creation": ["create account", "sign up", "register", "new account"],
    "account_management": ["manage account", "profile", "settings", "update"],
    "billing_support": ["billing", "payment", "invoice", "refund", "subscription"],
    "account_security": ["account security", "password", "login", "authentication"],
    
    # Company intents
    "company_about": ["about", "company", "who are you", "background", "story"],
    "company_team": ["team", "employees", "staff", "founders", "leadership"],
    "company_careers": ["career", "job", "hiring", "work", "employment"],
    
    # Conversational intents
    "conversational_greeting": [
        "halo", "hai", "hello", "hi", "selamat pagi", "selamat siang", 
        "good morning", "good afternoon", "hey", "morning"
    ],
    "conversational_thanks": [
        "terima kasih", "thanks", "thank you", "appreciate", "grateful"
    ],
    "conversational_goodbye": [
        "selamat tinggal", "goodbye", "bye", "see you", "sampai jumpa",
        "take care", "farewell"
    ],
    "conversational_chitchat": [
        "bagaimana kabar", "apa kabar", "how are you", "what's up", "nice to meet"
    ],
    "conversational_feedback": [
        "feedback", "suggestion", "comment", "opinion", "review"
    ]
}

# Knowledge-based intents (use QA extraction)
KNOWLEDGE_INTENTS = [
    "product_features", "product_technical_specs", "product_integration", 
    "product_ai_capabilities", "product_security", "product_customization",
    "pricing_basic", "pricing_professional", "pricing_enterprise", 
    "pricing_api", "pricing_add_ons", "pricing_comparison",
    "setup_guide", "integration_guide", "optimization_guide", 
    "troubleshooting_guide", "advanced_features", "deployment_strategies",
    "privacy_policy", "terms_of_service", "data_retention", 
    "security_policy", "compliance_standards",
    "account_creation", "account_management", "billing_support", "account_security",
    "company_about", "company_team", "company_careers"
]

# Conversational intents (use templates)
CONVERSATIONAL_INTENTS = [
    "conversational_greeting", "conversational_thanks", "conversational_goodbye", 
    "conversational_chitchat", "conversational_feedback"
]

# Response templates for conversational intents
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "conversational_greeting": [
        "Hello! Welcome to our chatbot service. How can I assist you today?",
        "Hi there! I'm here to help you with any questions about our platform.",
        "Good day! I'm your AI assistant. What would you like to know?"
    ],
    "conversational_thanks": [
        "You're very welcome! I'm glad I could help.",
        "Happy to assist! Is there anything else you'd like to know?",
        "My pleasure! Feel free to ask if you have more questions."
    ],
    "conversational_goodbye": [
        "Goodbye! Thank you for using our service. Have a great day!",
        "See you later! Don't hesitate to return if you need more help.",
        "Take care! I'm always here when you need assistance."
    ],
    "conversational_chitchat": [
        "I'm doing great, thank you for asking! How can I help you today?",
        "I'm here and ready to assist you. What's on your mind?",
        "Thanks for the friendly greeting! What would you like to know about our service?"
    ],
    "conversational_feedback": [
        "Thank you for your feedback! We really value your input and use it to improve our service.",
        "I appreciate you taking the time to share your thoughts. Your feedback helps us get better.",
        "That's valuable feedback! We'll definitely consider it for future improvements."
    ]
}
