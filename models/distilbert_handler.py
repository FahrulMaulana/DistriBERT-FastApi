import torch
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from transformers import pipeline
from datetime import datetime
import re
from config import settings, INTENT_KEYWORDS

logger = logging.getLogger(__name__)

class DistilBERTHandler:
    """Enhanced DistilBERT handler for campus chatbot with direct question-answering capability"""
    
    def __init__(self):
        # Question-Answering Pipeline (Primary method)
        self.qa_pipeline = None
        
        # Intent Classification Pipeline (Fallback)
        self.intent_classifier = None
        
        # State tracking
        self.is_loaded: bool = False
        self.load_start_time: Optional[datetime] = None
        
        # Campus context data
        self.campus_contexts = self._load_campus_contexts()
        
    async def initialize(self) -> bool:
        """Initialize DistilBERT models for both QA and intent classification"""
        try:
            self.load_start_time = datetime.utcnow()
            logger.info("🤖 Loading DistilBERT models...")
            
            # Load Question-Answering Pipeline (Primary)
            logger.info("🎯 Loading DistilBERT Question-Answering pipeline...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad",
                tokenizer="distilbert-base-uncased-distilled-squad",
                device=-1,  # CPU inference
                cache_dir=settings.model_cache_dir
            )
            
            # Load Intent Classification Pipeline (Fallback)
            logger.info("🧠 Loading DistilBERT Intent Classification pipeline...")
            self.intent_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                tokenizer="distilbert-base-uncased",
                device=-1,
                return_all_scores=True,
                cache_dir=settings.model_cache_dir
            )
            
            self.is_loaded = True
            load_time = (datetime.utcnow() - self.load_start_time).total_seconds()
            logger.info(f"✅ DistilBERT models loaded successfully in {load_time:.2f} seconds!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load DistilBERT models: {e}")
            self.is_loaded = False
            return False
    
    async def classify_intent(self, text: str, include_debug: bool = False) -> Dict:
        """Process user message with direct QA approach and intent classification fallback"""
        start_time = datetime.utcnow()
        
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call initialize() first.")
        
        try:
            # Clean and prepare text
            cleaned_text = self._preprocess_text(text)
            
            # Step 1: Try direct question-answering approach
            qa_result = await self._question_answering_approach(cleaned_text)
            
            # Step 2: If QA confidence is high, return direct answer
            if qa_result['confidence'] >= 0.3:  # Lower threshold for QA
                final_result = {
                    'intent': qa_result['intent'],
                    'confidence': qa_result['confidence'],
                    'source': 'distilbert_qa',
                    'answer': qa_result['answer'],
                    'context_used': qa_result.get('context_used', ''),
                    'text_length': len(text),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                # Step 3: Fallback to hybrid intent classification
                intent_result = await self._hybrid_intent_classification(cleaned_text)
                
                final_result = {
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence'],
                    'source': intent_result['source'],
                    'answer': self._generate_template_response(intent_result['intent']),
                    'text_length': len(text),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Add processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            final_result['processing_time_ms'] = round(processing_time, 2)
            
            # Add debug info if requested
            if include_debug:
                final_result['debug_info'] = {
                    'qa_attempt': qa_result,
                    'original_text': text,
                    'processed_text': cleaned_text
                }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_classification(text, str(e))
    
    async def batch_classify(self, texts: List[str], include_debug: bool = False) -> List[Dict]:
        """Classify multiple texts efficiently"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        
        try:
            # Process texts concurrently
            tasks = [self.classify_intent(text, include_debug) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(self._fallback_classification(
                        texts[i], str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            return [self._fallback_classification(text, str(e)) for text in texts]
    
    def _load_campus_contexts(self) -> Dict[str, str]:
        """Load campus context data for question-answering"""
        return {
            "jadwal_kuliah": """
            Universitas memiliki sistem informasi akademik yang dapat diakses melalui portal mahasiswa. 
            Jadwal kuliah tersedia setiap semester dan dapat dilihat berdasarkan program studi. 
            Kuliah dimulai pukul 07.00 WIB hingga 21.00 WIB dari Senin sampai Sabtu. 
            Ujian Tengah Semester (UTS) biasanya dilaksanakan pada minggu ke-8, sedangkan Ujian Akhir Semester (UAS) pada minggu ke-16. 
            Ruang kuliah tersebar di berbagai gedung dengan kode A, B, C, dan D. 
            Setiap mata kuliah memiliki dosen pengampu yang dapat dihubungi melalui sistem akademik.
            """,
            
            "pembayaran": """
            Uang Kuliah Tunggal (UKT) dibayar setiap semester melalui virtual account yang tersedia di portal keuangan. 
            Pembayaran dapat dilakukan melalui bank mitra seperti BNI, BRI, Mandiri, dan BCA. 
            Batas waktu pembayaran UKT adalah tanggal 15 setiap bulannya. 
            Tersedia sistem cicilan untuk mahasiswa yang membutuhkan dengan syarat tertentu. 
            Biaya UKT bervariasi berdasarkan program studi dan golongan UKT mahasiswa. 
            Status pembayaran dapat dicek melalui portal keuangan mahasiswa.
            """,
            
            "reset_password": """
            Password akun mahasiswa dapat direset melalui halaman forgot password di portal akademik. 
            Mahasiswa perlu memasukkan NIM dan email yang terdaftar untuk mendapat link reset. 
            Alternatif lain adalah menghubungi IT support di lantai 1 gedung A dengan membawa KTM. 
            Password baru harus mengandung minimal 8 karakter dengan kombinasi huruf dan angka. 
            Akun akan terkunci otomatis setelah 5 kali salah password.
            """,
            
            "faq_informasi": """
            Informasi beasiswa tersedia di website resmi universitas dan bagian kemahasiswaan. 
            Pendaftaran PKL dibuka setiap semester dengan syarat minimal 100 SKS telah lulus. 
            Wisuda dilaksanakan 2 kali dalam setahun yaitu bulan Juli dan Desember. 
            Syarat wisuda meliputi IPK minimal 2.75 dan telah menyelesaikan semua mata kuliah. 
            Transkrip nilai dapat diurus di bagian akademik dengan membawa persyaratan lengkap.
            """,
            
            "smalltalk": """
            Universitas menyediakan berbagai layanan untuk mahasiswa termasuk konseling dan bimbingan. 
            Kampus buka dari pukul 07.00 hingga 21.00 WIB setiap hari kecuali hari libur nasional. 
            Tersedia berbagai fasilitas seperti perpustakaan, laboratorium, dan ruang diskusi. 
            Mahasiswa dapat bergabung dengan berbagai organisasi dan unit kegiatan mahasiswa.
            """,
            
            "default": """
            Universitas adalah institusi pendidikan tinggi yang menyediakan program sarjana dan pascasarjana. 
            Kampus dilengkapi dengan fasilitas modern untuk mendukung proses pembelajaran. 
            Mahasiswa dapat mengakses berbagai layanan melalui portal akademik dan sistem informasi. 
            Untuk informasi lebih lanjut, silakan hubungi bagian informasi atau kunjungi website resmi.
            """
        }
    
    async def _question_answering_approach(self, text: str) -> Dict:
        """Use DistilBERT QA to directly answer campus questions"""
        try:
            # Determine best context based on question
            best_context, context_intent = self._select_best_context(text)
            
            # Use DistilBERT QA pipeline
            qa_result = self.qa_pipeline(
                question=text,
                context=best_context
            )
            
            # Extract answer details
            answer = qa_result['answer'].strip()
            confidence = float(qa_result['score'])
            
            # Enhance answer if confidence is reasonable
            if confidence >= 0.2 and len(answer) > 3:
                # Create natural response
                natural_answer = self._create_natural_response(text, answer, context_intent)
                
                return {
                    'intent': context_intent,
                    'confidence': min(confidence * 1.2, 0.95),  # Boost confidence slightly
                    'answer': natural_answer,
                    'raw_answer': answer,
                    'context_used': best_context[:100] + "...",
                    'method': 'question_answering'
                }
            else:
                return {
                    'intent': 'unknown',
                    'confidence': confidence,
                    'answer': '',
                    'method': 'question_answering_low_confidence'
                }
                
        except Exception as e:
            logger.error(f"Question-answering error: {e}")
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'answer': '',
                'method': 'question_answering_error',
                'error': str(e)
            }
    
    def _select_best_context(self, text: str) -> Tuple[str, str]:
        """Select most relevant context based on question keywords"""
        text_lower = text.lower()
        
        # Analyze question to determine best context
        context_scores = {}
        
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context_scores[intent] = score
        
        # Select context with highest keyword match
        if context_scores:
            best_intent = max(context_scores, key=context_scores.get)
            return self.campus_contexts.get(best_intent, self.campus_contexts['default']), best_intent
        else:
            return self.campus_contexts['default'], 'faq_informasi'
    
    def _create_natural_response(self, question: str, answer: str, intent: str) -> str:
        """Create natural language response from QA result"""
        question_lower = question.lower()
        
        # Response templates based on question type
        if any(word in question_lower for word in ['kapan', 'when', 'jam', 'waktu']):
            return f"Untuk informasi waktu: {answer}. Silakan cek jadwal terbaru di portal akademik."
        
        elif any(word in question_lower for word in ['dimana', 'where', 'lokasi', 'tempat']):
            return f"Lokasi: {answer}. Untuk petunjuk lengkap, hubungi bagian informasi."
        
        elif any(word in question_lower for word in ['bagaimana', 'how', 'cara']):
            return f"Cara: {answer}. Jika memerlukan bantuan lebih lanjut, silakan hubungi admin."
        
        elif any(word in question_lower for word in ['berapa', 'how much', 'biaya']):
            return f"Informasi biaya: {answer}. Hubungi bagian keuangan untuk detail lengkap."
        
        else:
            # Generic response
            return f"{answer}. Untuk informasi lebih lengkap, silakan hubungi bagian terkait."
    
    async def _hybrid_intent_classification(self, text: str) -> Dict:
        """Fallback to hybrid intent classification when QA fails"""
        try:
            # Get intent classification
            intent_results = self.intent_classifier(text)
            
            # Get keyword matching
            keyword_result = await self._keyword_classification(text)
            
            if intent_results and len(intent_results) > 0:
                best_result = max(intent_results, key=lambda x: x['score'])
                neural_confidence = float(best_result['score'])
                
                # Map label to campus intent
                neural_intent = self._map_label_to_intent(best_result['label'], text)
                
                # Combine with keyword results
                if (neural_intent == keyword_result['intent'] and 
                    neural_confidence > 0.4 and 
                    keyword_result['confidence'] > 0.3):
                    
                    return {
                        'intent': neural_intent,
                        'confidence': min(neural_confidence * 1.1, 0.85),
                        'source': 'hybrid_classification'
                    }
                elif keyword_result['confidence'] > neural_confidence:
                    return {
                        'intent': keyword_result['intent'],
                        'confidence': keyword_result['confidence'],
                        'source': 'keyword_classification'
                    }
                else:
                    return {
                        'intent': neural_intent,
                        'confidence': neural_confidence,
                        'source': 'neural_classification'
                    }
            else:
                return keyword_result
                
        except Exception as e:
            logger.error(f"Hybrid classification error: {e}")
            return await self._keyword_classification(text)
    
    def _generate_template_response(self, intent: str) -> str:
        """Generate template response for given intent"""
        templates = {
            "jadwal_kuliah": "Untuk melihat jadwal kuliah terbaru, silakan login ke portal akademik atau hubungi bagian akademik.",
            "pembayaran": "Informasi pembayaran UKT dapat dilihat di portal keuangan atau hubungi bagian keuangan.",
            "reset_password": "Untuk reset password, kunjungi halaman forgot password di portal atau hubungi IT support.",
            "faq_informasi": "Informasi lengkap tersedia di website resmi atau hubungi bagian kemahasiswaan.",
            "smalltalk": "Halo! Saya siap membantu Anda dengan informasi seputar kampus."
        }
        
        return templates.get(intent, "Silakan hubungi bagian informasi untuk bantuan.")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize Indonesian abbreviations
        replacements = {
            'gmn': 'bagaimana',
            'dmn': 'dimana',
            'yg': 'yang',
            'tdk': 'tidak',
            'ga': 'tidak',
            'gak': 'tidak'
        }
        
        for old, new in replacements.items():
            cleaned = re.sub(rf'\b{old}\b', new, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _map_label_to_intent(self, label: str, text: str) -> str:
        """Map model labels to campus intents with context awareness"""
        # Extract index from LABEL_X format
        if label.startswith('LABEL_'):
            try:
                idx = int(label.split('_')[1])
                if 0 <= idx < len(settings.intent_labels):
                    return settings.intent_labels[idx]
            except (ValueError, IndexError):
                pass
        
        # Context-based mapping for generic labels
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in INTENT_KEYWORDS['jadwal_kuliah']):
            return 'jadwal_kuliah'
        elif any(kw in text_lower for kw in INTENT_KEYWORDS['pembayaran']):
            return 'pembayaran'
        elif any(kw in text_lower for kw in INTENT_KEYWORDS['reset_password']):
            return 'reset_password'
        elif any(kw in text_lower for kw in INTENT_KEYWORDS['faq_informasi']):
            return 'faq_informasi'
        elif any(kw in text_lower for kw in INTENT_KEYWORDS['smalltalk']):
            return 'smalltalk'
        else:
            return 'unknown'
    
    async def _keyword_classification(self, text: str) -> Dict:
        """Enhanced keyword-based classification"""
        intent_scores = {}
        matched_keywords = {}
        
        for intent, keywords in INTENT_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text.lower()]
            
            if matches:
                # Calculate weighted score
                base_score = len(matches) / len(keywords)
                
                # Bonus for exact word matches
                exact_matches = sum(1 for kw in matches 
                                  if f" {kw} " in f" {text.lower()} " or 
                                     text.lower().startswith(kw) or 
                                     text.lower().endswith(kw))
                word_bonus = exact_matches * 0.15
                
                final_score = min(base_score + word_bonus, 1.0)
                intent_scores[intent] = final_score
                matched_keywords[intent] = matches
            else:
                intent_scores[intent] = 0.0
        
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        return {
            'intent': best_intent if best_score > 0 else 'unknown',
            'confidence': min(best_score * 1.1, 0.9),
            'source': 'keyword_matching',
            'matched_keywords': matched_keywords.get(best_intent, [])
        }
    

    
    def _fallback_classification(self, text: str, error_msg: str = "") -> Dict:
        """Fallback classification when all methods fail"""
        return {
            'intent': 'unknown',
            'confidence': 0.1,
            'source': 'fallback_error',
            'text_length': len(text),
            'timestamp': datetime.utcnow().isoformat(),
            'error': error_msg,
            'processing_time_ms': 0.0
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        load_time = None
        if self.load_start_time and self.is_loaded:
            load_time = (datetime.utcnow() - self.load_start_time).total_seconds()
        
        return {
            'model_name': 'distilbert-base-uncased-distilled-squad (QA) + distilbert-base-uncased (Classification)',
            'model_cache_dir': settings.model_cache_dir,
            'is_loaded': self.is_loaded,
            'load_time_seconds': load_time,
            'available_intents': settings.intent_labels,
            'total_intents': len(settings.intent_labels),
            'confidence_threshold': settings.confidence_threshold,
            'max_sequence_length': settings.max_sequence_length,
            'qa_pipeline_ready': self.qa_pipeline is not None,
            'intent_classifier_ready': self.intent_classifier is not None,
            'total_keywords': sum(len(keywords) for keywords in INTENT_KEYWORDS.values()),
            'keyword_categories': {
                intent: len(keywords) 
                for intent, keywords in INTENT_KEYWORDS.items()
            },
            'device': 'cpu',
            'pytorch_version': torch.__version__ if torch else 'N/A'
        }
    
    def get_health_status(self) -> Dict:
        """Get health status for monitoring"""
        return {
            'status': 'healthy' if self.is_loaded else 'unhealthy',
            'model_loaded': self.is_loaded,
            'qa_pipeline_ready': self.qa_pipeline is not None,
            'intent_classifier_ready': self.intent_classifier is not None,
            'uptime_seconds': (
                (datetime.utcnow() - self.load_start_time).total_seconds() 
                if self.load_start_time else 0
            )
        }
