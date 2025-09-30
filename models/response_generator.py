import asyncio
import json
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

from models.qa_handler import QAHandler, QAResult
from config import settings, RESPONSE_TEMPLATES, KNOWLEDGE_INTENTS, CONVERSATIONAL_INTENTS

logger = logging.getLogger(__name__)

@dataclass
class HybridResponse:
    """Response from hybrid processing"""
    message: str
    intent: str
    confidence: float
    mode: str  # 'knowledge', 'conversational', 'fallback'
    source: str  # 'qa_extraction', 'template', 'fallback'
    metadata: Dict[str, Any]
    processing_time_ms: float

class KnowledgeBase:
    """Knowledge base manager for QA contexts"""
    
    def __init__(self, knowledge_file: str = "knowledge_base.json"):
        self.knowledge_file = knowledge_file
        self.knowledge: Dict[str, Dict[str, str]] = {}
        self.loaded = False
        
    async def load_knowledge(self) -> bool:
        """Load knowledge base from JSON file"""
        try:
            knowledge_path = Path(__file__).parent.parent / self.knowledge_file
            
            if not knowledge_path.exists():
                logger.error(f"Knowledge base file not found: {knowledge_path}")
                return False
            
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            
            self.loaded = True
            logger.info(f"✅ Knowledge base loaded: {len(self.knowledge)} intents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge base: {str(e)}")
            return False
    
    def get_context(self, intent: str) -> Optional[str]:
        """Get context for specific intent"""
        if not self.loaded:
            return None
            
        knowledge_item = self.knowledge.get(intent)
        if not knowledge_item:
            return None
            
        return knowledge_item.get('context')
    
    def get_category(self, intent: str) -> Optional[str]:
        """Get category for specific intent"""
        if not self.loaded:
            return None
            
        knowledge_item = self.knowledge.get(intent)
        if not knowledge_item:
            return None
            
        return knowledge_item.get('category')
    
    def search_related_contexts(self, intent: str, limit: int = 3) -> List[Tuple[str, str]]:
        """Search for related contexts by category"""
        if not self.loaded:
            return []
        
        target_category = self.get_category(intent)
        if not target_category:
            return []
        
        related = []
        for key, value in self.knowledge.items():
            if key != intent and value.get('category') == target_category:
                context = value.get('context')
                if context:
                    related.append((key, context))
        
        return related[:limit]
    
    def get_all_contexts(self) -> List[str]:
        """Get all available contexts"""
        contexts = []
        for value in self.knowledge.values():
            context = value.get('context')
            if context:
                contexts.append(context)
        return contexts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if not self.loaded:
            return {"loaded": False}
        
        categories = {}
        total_chars = 0
        
        for value in self.knowledge.values():
            category = value.get('category', 'Unknown')
            context = value.get('context', '')
            
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            total_chars += len(context)
        
        return {
            "loaded": True,
            "total_intents": len(self.knowledge),
            "categories": categories,
            "total_characters": total_chars,
            "average_context_length": total_chars // len(self.knowledge) if self.knowledge else 0
        }

class HybridResponseGenerator:
    """Hybrid response generator combining classification, QA, and templates"""
    
    def __init__(self, qa_handler: QAHandler, knowledge_base: KnowledgeBase):
        self.qa_handler = qa_handler
        self.knowledge_base = knowledge_base
        self.distilbert_handler = None  # Will be injected
        
    def set_classification_handler(self, distilbert_handler):
        """Set the DistilBERT classification handler"""
        self.distilbert_handler = distilbert_handler
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        success = True
        
        # Initialize QA handler
        if not self.qa_handler.is_loaded:
            qa_success = await self.qa_handler.initialize()
            if not qa_success:
                logger.warning("QA handler initialization failed")
                success = False
        
        # Load knowledge base
        if not self.knowledge_base.loaded:
            kb_success = await self.knowledge_base.load_knowledge()
            if not kb_success:
                logger.warning("Knowledge base loading failed")
                success = False
        
        logger.info(f"Hybrid response generator initialized: {'✅' if success else '⚠️'}")
        return success
    
    def _determine_mode(self, intent: str, confidence: float) -> str:
        """Determine processing mode based on intent and confidence"""
        if intent in CONVERSATIONAL_INTENTS:
            return "conversational"
        elif intent in KNOWLEDGE_INTENTS and confidence > settings.qa_confidence_threshold:
            return "knowledge"
        else:
            return "fallback"
    
    async def _generate_knowledge_response(self, question: str, intent: str, 
                                         classification_confidence: float) -> Optional[HybridResponse]:
        """Generate response using knowledge base and QA extraction"""
        start_time = time.time()
        
        # Get primary context
        primary_context = self.knowledge_base.get_context(intent)
        if not primary_context:
            logger.warning(f"No context found for intent: {intent}")
            return None
        
        # Try QA extraction on primary context
        qa_result = await self.qa_handler.extract_answer(question, primary_context)
        
        if qa_result and qa_result.confidence > settings.qa_confidence_threshold:
            # Success with primary context
            processing_time = (time.time() - start_time) * 1000
            
            return HybridResponse(
                message=qa_result.answer,
                intent=intent,
                confidence=min(classification_confidence, qa_result.confidence),
                mode="knowledge",
                source="qa_extraction",
                metadata={
                    "qa_confidence": qa_result.confidence,
                    "classification_confidence": classification_confidence,
                    "context_used": intent,
                    "answer_length": len(qa_result.answer),
                    "qa_processing_time_ms": qa_result.processing_time_ms
                },
                processing_time_ms=processing_time
            )
        
        # Try related contexts if primary failed
        related_contexts = self.knowledge_base.search_related_contexts(intent, limit=3)
        if related_contexts:
            contexts = [context for _, context in related_contexts]
            qa_results = await self.qa_handler.extract_from_multiple(question, contexts)
            
            if qa_results:
                best_result = qa_results[0]  # Already sorted by confidence
                
                if best_result.confidence > settings.qa_confidence_threshold:
                    processing_time = (time.time() - start_time) * 1000
                    
                    return HybridResponse(
                        message=best_result.answer,
                        intent=intent,
                        confidence=min(classification_confidence, best_result.confidence),
                        mode="knowledge",
                        source="qa_extraction",
                        metadata={
                            "qa_confidence": best_result.confidence,
                            "classification_confidence": classification_confidence,
                            "context_used": "related_contexts",
                            "contexts_searched": len(contexts),
                            "answer_length": len(best_result.answer),
                            "qa_processing_time_ms": best_result.processing_time_ms
                        },
                        processing_time_ms=processing_time
                    )
        
        return None
    
    def _generate_conversational_response(self, intent: str, 
                                        classification_confidence: float) -> HybridResponse:
        """Generate conversational response using templates"""
        start_time = time.time()
        
        # Get template responses
        templates = RESPONSE_TEMPLATES.get(intent, [])
        if not templates:
            # Fallback conversational response
            templates = ["I understand you're trying to communicate. How can I help you better?"]
        
        # Select random template
        message = random.choice(templates)
        
        processing_time = (time.time() - start_time) * 1000
        
        return HybridResponse(
            message=message,
            intent=intent,
            confidence=classification_confidence,
            mode="conversational",
            source="template",
            metadata={
                "classification_confidence": classification_confidence,
                "template_options": len(templates),
                "template_selected": templates.index(message) if message in templates else 0
            },
            processing_time_ms=processing_time
        )
    
    def _generate_fallback_response(self, intent: str, confidence: float, 
                                  question: str) -> HybridResponse:
        """Generate fallback response for low confidence or unknown intents"""
        start_time = time.time()
        
        fallback_messages = [
            "I'm not entirely sure about that. Could you please rephrase your question or provide more details?",
            "I didn't quite understand your question. Could you try asking it differently?",
            "I'm still learning about that topic. Can you be more specific about what you'd like to know?",
            "That's an interesting question, but I'm not confident in my answer. Could you provide more context?",
            "I want to make sure I give you accurate information. Could you clarify what you're looking for?"
        ]
        
        # Add some context-aware fallbacks
        if len(question.split()) < 3:
            fallback_messages.insert(0, "Your question seems quite brief. Could you provide more details about what you're looking for?")
        
        if "?" not in question:
            fallback_messages.insert(0, "I notice your message doesn't contain a question mark. Are you asking a question or making a statement?")
        
        message = random.choice(fallback_messages)
        processing_time = (time.time() - start_time) * 1000
        
        return HybridResponse(
            message=message,
            intent=intent,
            confidence=confidence,
            mode="fallback",
            source="fallback",
            metadata={
                "classification_confidence": confidence,
                "question_length": len(question),
                "question_words": len(question.split()),
                "fallback_reason": "low_confidence" if confidence < settings.qa_confidence_threshold else "unknown_intent"
            },
            processing_time_ms=processing_time
        )
    
    async def generate_response(self, question: str, user_id: Optional[str] = None, 
                              include_debug: bool = False) -> HybridResponse:
        """Generate hybrid response using classification -> knowledge/conversational -> fallback"""
        overall_start_time = time.time()
        
        if not self.distilbert_handler:
            raise RuntimeError("DistilBERT handler not set. Call set_classification_handler() first.")
        
        try:
            # Step 1: Classification
            classification_result = await self.distilbert_handler.classify_intent(question, include_debug)
            intent = classification_result['intent']
            classification_confidence = classification_result['confidence']
            
            # Step 2: Determine processing mode
            mode = self._determine_mode(intent, classification_confidence)
            
            logger.debug(f"Processing mode: {mode} for intent: {intent} (confidence: {classification_confidence:.3f})")
            
            # Step 3: Generate response based on mode
            response = None
            
            if mode == "knowledge":
                response = await self._generate_knowledge_response(question, intent, classification_confidence)
            
            if response is None and mode in ["knowledge", "conversational"]:
                # Fallback to conversational if knowledge failed
                if intent in CONVERSATIONAL_INTENTS or mode == "conversational":
                    response = self._generate_conversational_response(intent, classification_confidence)
            
            if response is None:
                # Final fallback
                response = self._generate_fallback_response(intent, classification_confidence, question)
            
            # Add debug information if requested
            if include_debug:
                response.metadata.update({
                    "debug": {
                        "classification_result": classification_result,
                        "determined_mode": mode,
                        "processing_steps": ["classification", mode, "response_generation"],
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            
            # Update total processing time
            total_processing_time = (time.time() - overall_start_time) * 1000
            response.processing_time_ms = total_processing_time
            
            logger.info(f"Generated {response.mode} response for '{question[:50]}...' in {total_processing_time:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating hybrid response: {str(e)}")
            
            # Emergency fallback
            processing_time = (time.time() - overall_start_time) * 1000
            return HybridResponse(
                message="I apologize, but I'm experiencing technical difficulties. Please try again later.",
                intent="error",
                confidence=0.0,
                mode="fallback",
                source="error",
                metadata={
                    "error": str(e),
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                processing_time_ms=processing_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "qa_handler": self.qa_handler.get_model_info(),
            "knowledge_base": self.knowledge_base.get_stats(),
            "configuration": {
                "qa_confidence_threshold": settings.qa_confidence_threshold,
                "knowledge_intents": len(KNOWLEDGE_INTENTS),
                "conversational_intents": len(CONVERSATIONAL_INTENTS),
                "response_templates": len(RESPONSE_TEMPLATES)
            }
        }

# Global instances
knowledge_base = KnowledgeBase()
hybrid_generator = None  # Will be initialized in main.py
