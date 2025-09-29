# 🤖 DistilBERT Campus Intent Classification Service

FastAPI service untuk klasifikasi intent chatbot kampus menggunakan hybrid approach dengan DistilBERT neural networks dan keyword matching untuk akurasi optimal.

## ✨ Features

- 🧠 **Hybrid Classification**: Kombinasi DistilBERT neural network + keyword matching
- 🎯 **Campus-Specific Intents**: 6 kategori intent khusus kampus
- 🚀 **High Performance**: Batch processing dan caching untuk performa optimal
- 📊 **Detailed Analytics**: Confidence scoring dan debugging information
- 🔄 **Smart Fallback**: Graceful degradation jika model gagal load
- 💾 **Intelligent Caching**: Cache untuk response yang sering diakses
- 🐳 **Docker Ready**: Containerized deployment
- 📖 **Auto Documentation**: Interactive API docs dengan FastAPI

## 🎯 Supported Intents

| Intent | Description | Example |
|--------|-------------|---------|
| `jadwal_kuliah` | Schedule and class information | "Kapan jadwal kuliah Informatika besok?" |
| `pembayaran` | Payment and tuition inquiries | "Bagaimana cara bayar UKT?" |
| `reset_password` | Account and authentication help | "Saya lupa password akun saya" |
| `faq_informasi` | General campus information | "Info beasiswa tersedia dimana?" |
| `smalltalk` | Casual conversation and greetings | "Halo, selamat pagi!" |
| `unknown` | Unrecognized intents | Fallback category |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM (untuk DistilBERT model)
- Internet connection (untuk download model pertama kali)

### Installation

#### 1. Manual Installation

```bash
# Clone repository
git clone <your-repo-url>
cd services/distilbert-service

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env sesuai kebutuhan

# Run service
python main.py
```

#### 2. Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Check status
docker-compose ps
docker-compose logs -f
```

#### 3. VPS Automated Setup

```bash
# Upload service files to /tmp/distilbert-service on VPS
scp -r . user@your-vps:/tmp/distilbert-service

# Run automated setup script on VPS
ssh user@your-vps
sudo /tmp/distilbert-service/setup-vps.sh
```

## 📡 API Endpoints

### Core Classification

#### Single Classification
```bash
POST /classify
{
  "text": "Kapan jadwal kuliah besok?",
  "include_debug": false
}
```

#### Batch Classification
```bash
POST /batch-classify
{
  "texts": [
    "Kapan jadwal kuliah besok?",
    "Bagaimana cara bayar UKT?",
    "Saya lupa password"
  ],
  "include_debug": false
}
```

### Monitoring & Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and service status |
| `/model-info` | GET | Detailed model information |
| `/intents` | GET | Available intent categories |
| `/system-info` | GET | System resource information |
| `/cache-stats` | GET | Cache performance statistics |

### Testing & Administration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/test-classification` | POST | Test with predefined samples |
| `/clear-cache` | POST | Clear classification cache |
| `/docs` | GET | Interactive API documentation |

## 🔧 Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
LOG_LEVEL=INFO

# Model Configuration
MODEL_NAME=distilbert-base-uncased
MODEL_CACHE_DIR=./models/cache
CONFIDENCE_THRESHOLD=0.7
MAX_SEQUENCE_LENGTH=128

# Performance
ENABLE_CACHING=true
CACHE_TTL=300
MAX_WORKERS=4

# Security (Optional)
API_KEY=your-secret-key
CORS_ORIGINS=["http://localhost:3002", "http://localhost:3003"]
```

### Advanced Configuration

Edit `config.py` untuk customization lebih lanjut:

```python
# Tambah intent baru
intent_labels = [
    "jadwal_kuliah",
    "pembayaran", 
    "reset_password",
    "faq_informasi",
    "smalltalk",
    "your_new_intent",  # Tambahkan intent baru
    "unknown"
]

# Tambah keywords untuk intent baru
INTENT_KEYWORDS = {
    "your_new_intent": [
        "keyword1", "keyword2", "keyword3"
    ],
    # ... existing intents
}
```

## 🔗 Integration dengan NestJS

### Update DistilBERT Classifier di NestJS

```typescript
// packages/models/src/distilbert-classifier.ts
export class DistilBERTClassifier {
  private serviceUrl: string;

  constructor() {
    this.serviceUrl = process.env.DISTILBERT_SERVICE_URL || 'http://localhost:8000';
  }

  async classifyIntent(text: string): Promise<ClassificationResult> {
    try {
      const response = await fetch(`${this.serviceUrl}/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.DISTILBERT_API_KEY}`
        },
        body: JSON.stringify({ text, include_debug: false })
      });

      const data = await response.json();
      
      return {
        intent: data.intent,
        confidence: data.confidence,
        source: 'distilbert_service'
      };
    } catch (error) {
      // Fallback to keyword matching
      return this.keywordFallback(text);
    }
  }
}
```

### Environment di NestJS

```bash
# apps/api/.env
DISTILBERT_SERVICE_URL=http://localhost:8000
DISTILBERT_API_KEY=distilbert-service-2024
```

## 📊 Performance Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "available_intents": ["jadwal_kuliah", "pembayaran", ...],
  "uptime_seconds": 3600.5,
  "model_info": { ... }
}
```

### Cache Statistics

```bash
curl http://localhost:8000/cache-stats
```

### System Information

```bash
curl http://localhost:8000/system-info
```

## 🧪 Testing

### Test Single Classification

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Kapan jadwal kuliah Informatika besok?", "include_debug": true}'
```

### Test Batch Classification

```bash
curl -X POST http://localhost:8000/batch-classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Kapan jadwal kuliah besok?",
      "Bagaimana cara bayar UKT?", 
      "Saya lupa password"
    ],
    "include_debug": false
  }'
```

### Test Predefined Samples

```bash
curl -X POST http://localhost:8000/test-classification \
  -H "Content-Type: application/json" \
  -d '{
    "sample_type": "random",
    "count": 5
  }'
```

## 📝 Logging

### View Real-time Logs

```bash
# Docker
docker-compose logs -f distilbert-service

# Systemd
sudo journalctl -u distilbert-service -f

# Log files
tail -f /var/log/distilbert-service/distilbert-service.log
```

### Log Levels

- `DEBUG`: Detailed execution information
- `INFO`: General service information
- `WARNING`: Warning messages
- `ERROR`: Error messages

## 🔒 Security

### API Key Authentication (Optional)

```bash
# Set API key in .env
API_KEY=your-secret-key

# Use in requests
curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "test message"}'
```

### CORS Configuration

```python
# config.py
cors_origins = [
    "http://localhost:3002",  # NestJS API
    "http://localhost:3003",  # Next.js Frontend
    "https://yourdomain.com"   # Production domain
]
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Loading Error
```bash
# Check available memory
free -h

# Check disk space for model cache
df -h ./models/cache

# Clear model cache and retry
rm -rf ./models/cache/*
```

#### 2. Service Won't Start
```bash
# Check service logs
sudo journalctl -u distilbert-service -n 50

# Check port availability
sudo netstat -tlnp | grep 8000

# Restart service
sudo systemctl restart distilbert-service
```

#### 3. Poor Classification Accuracy
```bash
# Test with known samples
curl -X POST http://localhost:8000/test-classification \
  -d '{"sample_type": "all", "count": 20}'

# Check confidence threshold
curl http://localhost:8000/model-info

# Adjust threshold in .env
CONFIDENCE_THRESHOLD=0.6  # Lower for more neural network usage
```

#### 4. Memory Issues
```bash
# Monitor memory usage
htop

# Reduce batch size in .env
BATCH_SIZE=8  # Default: 16

# Enable swap if needed
sudo swapon --show
```

## 📈 Performance Optimization

### Memory Optimization
- Use quantized model untuk mengurangi memory usage
- Enable swap file untuk VPS dengan RAM terbatas
- Adjust batch size sesuai available memory

### Speed Optimization
- Enable caching untuk frequent requests
- Use multiple workers (dengan caveats untuk model loading)
- Implement connection pooling untuk database jika diperlukan

### Accuracy Optimization
- Fine-tune confidence threshold berdasarkan testing
- Add more keywords untuk specific intents
- Implement feedback loop untuk improving model

## 🚀 Deployment

### Production Checklist

- [ ] Set proper LOG_LEVEL (INFO atau WARNING)
- [ ] Configure proper CORS origins
- [ ] Set strong API_KEY jika authentication enabled
- [ ] Setup proper firewall rules
- [ ] Configure Nginx reverse proxy
- [ ] Setup SSL certificate
- [ ] Configure monitoring dan alerting
- [ ] Setup backup untuk model cache
- [ ] Test failover scenarios

### Scaling

Untuk high-traffic deployment:

1. **Load Balancer**: Setup multiple instances behind load balancer
2. **Caching Layer**: Redis untuk shared caching
3. **Monitoring**: Prometheus + Grafana untuk metrics
4. **Alerting**: Setup alerts untuk model failures

## 📞 Support

Jika mengalami issues:

1. Check logs terlebih dahulu
2. Test dengan `/health` endpoint
3. Verify configuration di `.env`
4. Check system resources
5. Contact: support@kampus.ac.id

## 📄 License

MIT License - See LICENSE file for details.

---

**Developed for Campus Chatbot System**  
Version: 1.0.0  
Last Updated: September 2025
