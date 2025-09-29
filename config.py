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
    
    # Intent Labels for Campus Chatbot
    intent_labels: List[str] = [
        "jadwal_kuliah",    # 0 - Schedule and class information
        "pembayaran",       # 1 - Payment and tuition inquiries  
        "reset_password",   # 2 - Account and authentication help
        "faq_informasi",    # 3 - General campus information
        "smalltalk",        # 4 - Casual conversation and greetings
        "unknown"           # 5 - Unrecognized intents
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

# Enhanced keyword patterns based on campus dataset
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "jadwal_kuliah": [
        "jadwal", "kuliah", "kelas", "mata kuliah", "matkul", 
        "jam", "ruang", "dosen", "ujian", "uts", "uas", 
        "semester", "hari", "waktu", "schedule", "class",
        "course", "lecture", "exam", "midterm", "final"
    ],
    "pembayaran": [
        "bayar", "ukt", "pembayaran", "biaya", "cicilan", 
        "tagihan", "lunas", "transfer", "virtual account", 
        "va", "spp", "keuangan", "payment", "fee",
        "tuition", "installment", "billing", "finance"
    ],
    "reset_password": [
        "password", "kata sandi", "login", "akun", "reset",
        "lupa", "forgot", "masuk", "sign in", "account",
        "username", "user", "auth", "authentication",
        "access", "credentials", "recover"
    ],
    "faq_informasi": [
        "informasi", "info", "beasiswa", "wisuda", "pkl",
        "skripsi", "thesis", "magang", "cuti", "akademik",
        "transkrip", "nilai", "grade", "graduation",
        "scholarship", "internship", "academic", "transcript"
    ],
    "smalltalk": [
        "halo", "hai", "selamat", "terima kasih", "tolong",
        "bantuan", "hello", "hi", "thanks", "help",
        "morning", "pagi", "siang", "malam", "good",
        "please", "sorry", "excuse", "welcome"
    ]
}

# Response templates for chat
RESPONSE_TEMPLATES: Dict[str, List[str]] = {
    "jadwal_kuliah": [
        "Untuk melihat jadwal kuliah, silakan login ke portal akademik atau hubungi bagian akademik.",
        "Jadwal kuliah dapat dilihat di sistem informasi akademik. Apakah ada mata kuliah tertentu yang ingin ditanyakan?",
        "Silakan cek jadwal terbaru di portal mahasiswa atau hubungi koordinator program studi Anda."
    ],
    "pembayaran": [
        "Untuk pembayaran UKT, silakan login ke portal keuangan atau kunjungi bank mitra kampus.",
        "Pembayaran dapat dilakukan melalui virtual account yang tersedia di portal mahasiswa.",
        "Hubungi bagian keuangan untuk informasi lebih lanjut tentang pembayaran semester."
    ],
    "reset_password": [
        "Untuk reset password, silakan kunjungi halaman reset password di portal atau hubungi IT support.",
        "Anda dapat mereset password melalui email recovery atau hubungi admin sistem.",
        "Silakan hubungi helpdesk IT dengan membawa KTM untuk reset password akun Anda."
    ],
    "faq_informasi": [
        "Informasi lengkap tersedia di website resmi kampus atau hubungi bagian kemahasiswaan.",
        "Silakan kunjungi pusat informasi mahasiswa atau check website resmi untuk info terbaru.",
        "Untuk informasi lebih detail, hubungi bagian terkait atau kunjungi kantor kemahasiswaan."
    ],
    "smalltalk": [
        "Halo! Saya siap membantu Anda dengan informasi seputar kampus.",
        "Hai! Ada yang bisa saya bantu hari ini?",
        "Selamat datang! Silakan tanyakan apa saja yang ingin Anda ketahui."
    ]
}
