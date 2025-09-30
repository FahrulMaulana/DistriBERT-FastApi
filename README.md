# Hybrid QA Chatbot Service

Advanced conversational AI system that combines DistilBERT intent classification with DistilBERT-Squad question answering to provide intelligent, context-aware responses from a comprehensive knowledge base.

## 🌟 Features

- **🧠 Hybrid QA System**: Combines classification with knowledge extraction
- **📚 Rich Knowledge Base**: 25+ detailed contexts (250-500 words each)
- **🎯 Dual Processing Modes**: Knowledge-based and Conversational
- **⚡ High Performance**: Sub-500ms response times with caching
- **🔍 Advanced QA**: DistilBERT-Squad for accurate answer extraction
- **💾 Smart Caching**: Multi-level caching for optimal performance

## 🏗️ Architecture

### Processing Flow
1. **Classification**: DistilBERT classifies user intent
2. **Mode Selection**: Determines processing approach
3. **Response Generation**:
   - **Knowledge Mode**: QA extraction from detailed contexts
   - **Conversational Mode**: Template-based responses
   - **Fallback Mode**: Intelligent fallback handling

### Components

#### 1. Knowledge Base (`knowledge_base.json`)
- **30 intents** with detailed contexts (250-500 words each)
- **Categories**: Product (6), Pricing (6), Guides (6), Policies (5), Account (4), Company (3), Conversational (5)
- **Rich Content**: Numbers, specifications, examples, detailed explanations

#### 2. QA Handler (`models/qa_handler.py`)
- **Model**: `distilbert-base-cased-distilled-squad`
- **Methods**: `extract_answer()`, `extract_from_multiple()`
- **Features**: Caching, error handling, confidence validation
- **Performance**: GPU acceleration, batch processing

#### 3. Hybrid Response Generator (`models/response_generator.py`)
- **Orchestration**: Classification → Knowledge Retrieval → QA Extraction → Fallback
- **Mode Decision**: Knowledge-based vs Conversational
- **Error Handling**: Comprehensive error recovery
- **Metrics**: Processing time tracking, confidence scoring

## 📊 Intent Categories

### Knowledge-Based Intents (25 intents)
**Product Category (6 intents)**
- `product_features` - Platform capabilities and features
- `product_technical_specs` - Technical specifications
- `product_integration` - Integration capabilities
- `product_ai_capabilities` - AI and ML features
- `product_security` - Security features
- `product_customization` - Customization options

**Pricing Category (6 intents)**
- `pricing_basic` - Basic plan details
- `pricing_professional` - Professional plan features
- `pricing_enterprise` - Enterprise solutions
- `pricing_api` - API-only pricing
- `pricing_add_ons` - Additional features
- `pricing_comparison` - Plan comparisons

**Guides Category (6 intents)**
- `setup_guide` - Getting started
- `integration_guide` - Integration instructions
- `optimization_guide` - Performance optimization
- `troubleshooting_guide` - Problem resolution
- `advanced_features` - Advanced functionality
- `deployment_strategies` - Deployment approaches

**Policies Category (5 intents)**
- `privacy_policy` - Data privacy practices
- `terms_of_service` - Usage terms
- `data_retention` - Data storage policies
- `security_policy` - Security measures
- `compliance_standards` - Compliance information

**Account Category (4 intents)**
- `account_creation` - Account setup
- `account_management` - Profile management
- `billing_support` - Payment and billing
- `account_security` - Security settings

**Company Category (3 intents)**
- `company_about` - Company information
- `company_team` - Team details
- `company_careers` - Career opportunities

### Conversational Intents (5 intents)
- `conversational_greeting` - Greetings and welcome
- `conversational_thanks` - Gratitude expressions
- `conversational_goodbye` - Farewells
- `conversational_chitchat` - Casual conversation
- `conversational_feedback` - User feedback

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install additional QA dependencies
pip install transformers torch

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Configuration
```python
# config.py
QA_CONFIDENCE_THRESHOLD = 0.3
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
QA_CACHE_ENABLED = True
```

## 📡 API Endpoints

### Main Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "message": "What are the key features of your chatbot platform?",
  "user_id": "user123",
  "include_debug": false
}
```

**Response:**
```json
{
  "message": "Our flagship AI-powered chatbot platform offers enterprise-grade conversational capabilities...",
  "intent": "product_features",
  "confidence": 0.95,
  "mode": "knowledge",
  "source": "qa_extraction",
  "metadata": {
    "qa_confidence": 0.87,
    "processing_time_ms": 245.5,
    "context_used": "product_features"
  },
  "processing_time_ms": 312.8,
  "timestamp": "2025-09-30T10:30:00Z"
}
```

### Other Endpoints
- `GET /` - Service information
- `POST /classify` - Intent classification only
- `GET /health` - Health check
- `GET /intents` - Available intents
- `GET /hybrid-stats` - System statistics
- `GET /model-info` - Model information

## 🧪 Testing

### Sample Requests

**Knowledge-based Query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your pricing plans?"}'
```

**Conversational Query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! How are you today?"}'
```

### Test Cases
```python
# Knowledge extraction test
test_cases = [
    {
        "message": "What security features do you offer?",
        "expected_mode": "knowledge",
        "expected_intent": "product_security"
    },
    {
        "message": "How much does the enterprise plan cost?",
        "expected_mode": "knowledge", 
        "expected_intent": "pricing_enterprise"
    },
    {
        "message": "Thank you for your help!",
        "expected_mode": "conversational",
        "expected_intent": "conversational_thanks"
    }
]
```

## ⚡ Performance

### Benchmarks
- **Response Time**: < 500ms average
- **QA Extraction**: 150-300ms
- **Classification**: 50-150ms
- **Caching Hit Rate**: > 80%
- **Memory Usage**: 2-4GB (with models loaded)

### Optimization Features
- **Multi-level Caching**: Classification and QA results
- **GPU Acceleration**: CUDA support for inference
- **Async Processing**: Non-blocking operations
- **Connection Pooling**: Optimized database connections
- **Context Preprocessing**: Optimized text processing

## 🔧 Configuration

### Environment Variables
```bash
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
QA_CONFIDENCE_THRESHOLD=0.3
QA_CACHE_ENABLED=true
LOG_LEVEL=INFO
```

### Model Configuration
```python
# QA Model Settings
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
QA_MAX_ANSWER_LENGTH = 200
QA_MAX_CONTEXT_LENGTH = 512

# Classification Model Settings  
CLASSIFICATION_MODEL = "distilbert-base-uncased"
CONFIDENCE_THRESHOLD = 0.5
```

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### System Statistics
```bash
curl http://localhost:8000/hybrid-stats
```

**Response includes:**
- QA model performance
- Knowledge base statistics
- Cache hit rates
- Processing times
- Error rates

## 🛠️ Development

### Adding New Intents
1. **Update Knowledge Base**: Add context to `knowledge_base.json`
2. **Update Config**: Add intent to `KNOWLEDGE_INTENTS` or `CONVERSATIONAL_INTENTS`
3. **Add Keywords**: Update `INTENT_KEYWORDS` in config
4. **Add Templates**: For conversational intents, update `RESPONSE_TEMPLATES`
5. **Test**: Add test cases to `SAMPLE_TEST_DATA`

### Knowledge Base Format
```json
{
  "intent_name": {
    "category": "Product|Pricing|Guides|Policies|Account|Company|Conversational",
    "context": "Detailed 250-500 word context with specific information, examples, numbers, and comprehensive coverage of the topic..."
  }
}
```

## 🐛 Debugging

### Debug Mode
```json
{
  "message": "Your question here",
  "include_debug": true
}
```

**Debug Response includes:**
- Classification details
- QA extraction process
- Context selection
- Processing steps
- Performance metrics

### Common Issues
1. **Low QA Confidence**: Improve context quality in knowledge base
2. **Slow Responses**: Check GPU availability, optimize contexts
3. **Wrong Intent**: Review keywords and training data
4. **Memory Issues**: Reduce model cache size, optimize batch processing

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## 📞 Support

- **Documentation**: `/docs` endpoint
- **Health Check**: `/health` endpoint  
- **System Stats**: `/hybrid-stats` endpoint
- **Issues**: GitHub repository issues
- **Email**: support@company.com
