# HifazatAI Security System Prototype

A comprehensive AI-powered security monitoring system that integrates threat intelligence analysis, video surveillance analytics, and border anomaly detection into a unified dashboard.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- 8GB RAM minimum (16GB recommended)
- 4+ CPU cores

### 1. Clone and Setup
```bash
git clone <repository-url>
cd hifazat-ai-prototype

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### 2. Start the System
```bash
# Terminal 1: Start backend API
cd backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev

# Terminal 3: Run demo (optional)
cd backend/demo
python demo_pipeline.py --interactive
```

### 3. Access the Dashboard
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Threat Intel  │    │  Video Surveillance  │    │ Border Anomaly  │
│    Pipeline     │    │     Pipeline         │    │    Pipeline     │
└─────────┬───────┘    └─────────┬────────────┘    └─────────┬───────┘
          │                      │                           │
          └──────────────────────┼───────────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │      Alert Broker API      │
                    │    (FastAPI + SQLite)      │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │     React Dashboard        │
                    │  (Real-time Monitoring)    │
                    └────────────────────────────┘
```

### AI Pipelines

1. **Threat Intelligence Pipeline**
   - IOC extraction from feeds (IP, domains, hashes)
   - DistilBERT-based risk classification
   - Real-time threat scoring

2. **Video Surveillance Pipeline**
   - YOLOv8 object detection
   - ByteTrack multi-object tracking
   - Behavior analysis (loitering, zone violations)

3. **Border Anomaly Detection Pipeline**
   - Trajectory extraction and analysis
   - Isolation Forest anomaly detection
   - Motion-based fallback system

## 📊 Features

### Dashboard Capabilities
- ✅ Real-time alert monitoring with SSE/WebSocket support
- ✅ Advanced filtering and search functionality
- ✅ Alert management (review, dismiss, status updates)
- ✅ Video analysis with bounding box visualization
- ✅ System monitoring and performance metrics
- ✅ Comprehensive evaluation framework

### AI Performance Targets
- **Object Detection**: mAP@0.5 ≥ 85%
- **Multi-Object Tracking**: MOTA ≥ 70%
- **Anomaly Detection**: F1-Score ≥ 70%
- **Threat Intelligence**: Precision ≥ 80%

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
VITE_API_URL=http://localhost:8000

# CPU Optimization (for i5 6th gen)
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

### Pipeline Configuration
Configuration files are located in `backend/configs/`:
- `detection_config.json` - Object detection settings
- `tracking_config.json` - Multi-object tracking parameters
- `anomaly_config.json` - Anomaly detection thresholds

## 🐳 Docker Deployment

### Quick Deploy
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

### Production Deployment
```bash
# Build production image
docker build -t hifazat-ai:latest .

# Run with resource limits (for i5 6th gen)
docker run -d \
  --name hifazat-ai \
  --memory=6g \
  --cpus=4 \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  hifazat-ai:latest
```

## 🧪 Testing and Evaluation

### Run Evaluation Suite
```bash
cd backend/evaluation

# Setup evaluation environment
python setup_evaluation.py

# Run comprehensive evaluation
cd scripts
python run_evaluation.py

# Run individual evaluations
python evaluate_detection.py --config ../configs/detection_eval.json --model yolov8n.pt --dataset ../datasets/visdrone --output ../results/detection
python evaluate_tracking.py --config ../configs/tracking_eval.json --sequence ../datasets/mot/MOT17/train/MOT17-02-FRCNN --output ../results/tracking
python evaluate_anomaly.py --config ../configs/anomaly_eval.json --data ../datasets/anomaly_test.json --output ../results/anomaly
```

### Generate Test Data
```bash
cd backend/evaluation

# Generate synthetic datasets
python test_data_manager.py --action generate --type threat --samples 200
python test_data_manager.py --action generate --type video --samples 100
python test_data_manager.py --action generate --type anomaly --samples 150

# Validate datasets
python test_data_manager.py --action validate --type threat --path datasets/threat_test.json
```

## 🎬 Demo Scenarios

### Interactive Demo
```bash
cd backend/demo
python demo_pipeline.py --interactive
```

### Automated Scenarios
```bash
# Mixed threats (10 minutes)
python demo_pipeline.py --scenario mixed_threats

# Security breach simulation
python demo_pipeline.py --scenario security_breach

# Border incident
python demo_pipeline.py --scenario border_incident

# Continuous monitoring
python demo_pipeline.py --scenario continuous_monitoring
```

## 📁 Project Structure

```
hifazat-ai-prototype/
├── backend/                    # Python backend
│   ├── api/                   # FastAPI application
│   ├── pipelines/             # AI processing pipelines
│   ├── models/                # Pydantic data models
│   ├── utils/                 # Utility functions
│   ├── evaluation/            # Evaluation framework
│   └── demo/                  # Demo scenarios
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API and WebSocket services
│   │   └── types/            # TypeScript type definitions
│   └── dist/                 # Built frontend assets
├── docker-compose.yml         # Docker deployment
├── Dockerfile                 # Container configuration
└── README.md                  # This file
```

## 🔍 API Documentation

### Core Endpoints

#### Alerts
- `POST /alerts/threat` - Create threat intelligence alert
- `POST /alerts/video` - Create video surveillance alert
- `POST /alerts/anomaly` - Create border anomaly alert
- `GET /alerts` - Retrieve alerts with filtering
- `GET /alerts/{id}` - Get specific alert
- `PUT /alerts/{id}/status` - Update alert status

#### System
- `GET /health` - Health check
- `GET /metrics/{pipeline}` - Pipeline metrics
- `POST /metrics` - Submit system metrics

#### Real-time
- `GET /events` - Server-Sent Events stream
- WebSocket support at `/ws` (optional)

### Example API Usage
```python
import requests

# Create a threat alert
alert_data = {
    "ioc_type": "ip",
    "ioc_value": "192.168.1.100",
    "risk_level": "High",
    "evidence_text": "Suspicious network activity",
    "source_feed": "ThreatFeed_1"
}

response = requests.post("http://localhost:8000/alerts/threat", json=alert_data)
print(response.json())
```

## 🚨 Troubleshooting

### Common Issues

#### Frontend Shows Blank Screen
```bash
# Check Vite configuration
cd frontend
npm run build
# Ensure base: './' is set in vite.config.ts
```

#### API Connection Failed
```bash
# Check API is running
curl http://localhost:8000/health

# Check CORS configuration
# Verify frontend URL is in CORS origins
```

#### High CPU Usage
```bash
# Apply CPU optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Use CPU-optimized models
# Check resource monitoring in dashboard
```

#### Demo Not Generating Alerts
```bash
# Verify API connectivity
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"

# Check demo pipeline logs
python demo_pipeline.py --scenario mixed_threats --verbose
```

## 📈 Performance Optimization

### For i5 6th Gen CPU
```bash
# Set thread limits
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Use optimized batch sizes
# Configure frame skipping in video pipeline
# Enable model quantization
```

### Memory Management
```bash
# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Configure garbage collection
export MALLOC_TRIM_THRESHOLD_=100000
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
cd backend && pip install -r requirements-dev.txt
cd frontend && npm install --include=dev

# Run tests
cd backend && python -m pytest
cd frontend && npm test

# Code formatting
cd backend && black . && isort .
cd frontend && npm run lint:fix
```

### Adding New Features
1. Create feature branch
2. Implement backend API changes
3. Add frontend components
4. Update tests and documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For technical support:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check system logs in `/logs` directory
4. Run health checks and diagnostics

## 🔮 Future Enhancements

- [ ] Advanced ML model optimization
- [ ] Multi-camera video fusion
- [ ] Enhanced threat intelligence feeds
- [ ] Mobile application support
- [ ] Cloud-native deployment options
- [ ] Advanced analytics and reporting

---

**HifazatAI** - Comprehensive AI Security Monitoring System