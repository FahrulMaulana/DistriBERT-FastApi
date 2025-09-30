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
    
    # Intent Labels for General Chatbot
    intent_labels: List[str] = [
        "greeting",         # 0 - Greetings and salutations
        "question",         # 1 - General questions and inquiries
        "help_request",     # 2 - Help and assistance requests
        "information",      # 3 - Information seeking
        "weather",          # 4 - Weather related questions
        "food_recipe",      # 5 - Food and recipe questions
        "technology",       # 6 - Technology related questions
        "smalltalk",        # 7 - Casual conversation
        "goodbye",          # 8 - Farewell messages
        "unknown"           # 9 - Unrecognized intents
    ]
    
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

# Enhanced keyword patterns for general chatbot
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "greeting": [
        "halo", "hai", "hello", "hi", "selamat pagi", "selamat siang", 
        "selamat sore", "selamat malam", "good morning", "good afternoon",
        "good evening", "good night", "hey", "morning", "evening"
    ],
    "question": [
        "apa", "bagaimana", "kapan", "dimana", "mengapa", "kenapa",
        "what", "how", "when", "where", "why", "who", "which",
        "siapa", "mana", "berapa", "bisakah", "dapatkah", "can", "could"
    ],
    "help_request": [
        "bantuan", "tolong", "help", "assist", "bantu", "mohon",
        "please", "bisa bantu", "could you help", "need help",
        "butuh bantuan", "minta tolong", "assistance"
    ],
    "information": [
        "informasi", "info", "berita", "news", "data", "fakta",
        "information", "details", "penjelasan", "explanation",
        "tahu", "know", "tentang", "about", "mengenai", "regarding"
    ],
    "weather": [
        "cuaca", "hujan", "panas", "dingin", "ujan", "weather",
        "temperature", "suhu", "iklim", "climate", "cerah", "mendung",
        "sunny", "cloudy", "rainy", "hot", "cold", "warm"
    ],
    "food_recipe": [
        "makanan", "food", "resep", "recipe", "masak", "cook",
        "cooking", "memasak", "bahan", "ingredient", "hidangan",
        "dish", "menu", "makan", "eat", "eating", "restoran", "restaurant"
    ],
    "technology": [
        "teknologi", "technology", "komputer", "computer", "software",
        "hardware", "internet", "programming", "coding", "aplikasi",
        "application", "app", "sistem", "system", "gadget", "smartphone"
    ],
    "smalltalk": [
        "bagaimana kabar", "apa kabar", "how are you", "terima kasih",
        "thanks", "thank you", "maaf", "sorry", "permisi", "excuse me",
        "senang", "happy", "sedih", "sad", "baik", "good", "fine"
    ],
    "goodbye": [
        "selamat tinggal", "goodbye", "bye", "see you", "sampai jumpa",
        "dadah", "farewell", "until next time", "take care",
        "selamat jalan", "good bye", "bye bye"
    ]
}

# Response templates for general chatbot
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "greeting": [
        "Halo! Selamat datang! Ada yang bisa saya bantu hari ini?",
        "Hai! Saya di sini untuk membantu Anda. Silakan tanyakan apa saja.",
        "Selamat pagi! Semoga hari Anda menyenangkan. Bagaimana saya bisa membantu?"
    ],
    "question": [
        "Itu pertanyaan yang menarik! Saya akan mencoba membantu menjawabnya.",
        "Terima kasih atas pertanyaannya. Biarkan saya membantu Anda mencari jawabannya.",
        "Saya akan berusaha memberikan informasi yang Anda butuhkan."
    ],
    "help_request": [
        "Tentu saja! Saya siap membantu Anda. Silakan jelaskan apa yang Anda butuhkan.",
        "Dengan senang hati saya akan membantu. Apa yang bisa saya lakukan untuk Anda?",
        "Saya di sini untuk membantu. Silakan beri tahu saya apa yang Anda perlukan."
    ],
    "information": [
        "Saya akan mencari informasi yang Anda butuhkan. Bisa tolong lebih spesifik?",
        "Untuk informasi yang lebih akurat, bisa Anda jelaskan lebih detail tentang apa yang ingin diketahui?",
        "Saya siap memberikan informasi. Topik apa yang ingin Anda ketahui lebih lanjut?"
    ],
    "weather": [
        "Untuk informasi cuaca yang akurat, saya sarankan Anda mengecek aplikasi cuaca atau situs BMKG.",
        "Saya tidak bisa memberikan prakiraan cuaca real-time, tapi Anda bisa mengecek aplikasi cuaca di ponsel Anda.",
        "Informasi cuaca terkini bisa Anda dapatkan dari BMKG atau aplikasi cuaca terpercaya."
    ],
    "food_recipe": [
        "Wah, menarik sekali! Untuk resep masakan, saya sarankan Anda mencari di website resep atau aplikasi memasak.",
        "Saya tidak memiliki database resep lengkap, tapi Anda bisa mencari resep di internet atau bertanya pada chef profesional.",
        "Untuk resep yang akurat dan detail, lebih baik konsultasi dengan ahli masak atau cari di sumber resep terpercaya."
    ],
    "technology": [
        "Teknologi memang berkembang pesat! Untuk informasi teknis yang spesifik, mungkin perlu konsultasi dengan ahli IT.",
        "Itu topik teknologi yang menarik. Untuk detail yang lebih teknis, saya sarankan mencari di dokumentasi resmi atau forum teknologi.",
        "Bidang teknologi sangat luas. Bisa tolong lebih spesifik tentang aspek teknologi yang ingin dibahas?"
    ],
    "smalltalk": [
        "Terima kasih sudah bertanya! Saya baik-baik saja dan siap membantu Anda.",
        "Saya senang bisa mengobrol dengan Anda! Ada hal lain yang ingin dibicarakan?",
        "Ngobrol dengan Anda menyenangkan! Bagaimana hari Anda berjalan?"
    ],
    "goodbye": [
        "Selamat tinggal! Terima kasih sudah menggunakan layanan saya. Sampai jumpa lagi!",
        "Sampai jumpa! Jangan ragu untuk kembali jika butuh bantuan lagi.",
        "Dadah! Semoga hari Anda menyenangkan. Take care!"
    ]
}
