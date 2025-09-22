# HifazatAI Prototype - Complete Setup and Deployment Instructions

## Overview
HifazatAI is a comprehensive security AI system prototype with three core detection pipelines: threat intelligence analysis, video surveillance analytics, and border anomaly detection. This guide provides complete instructions for setting up, running, and deploying the system using various cloud services and platforms.

## System Requirements

### Minimum Hardware Requirements (i5 6th Gen Optimized)
- **CPU**: Intel i5 6th Gen or equivalent (4 cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space for models and data
- **GPU**: Optional (NVIDIA GTX 1050 or better for accelerated inference)
- **Network**: Stable internet connection for model downloads

### Software Requirements
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+
- **Python**: 3.10 or 3.11
- **Docker**: Latest version (optional but recommended)
- **Git**: Latest version

## Quick Start (Local Development)

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd hifazat-ai-prototype

# Create Python virtual environment
python3 -m venv backend/venv
source backend/venv/bin/activate  # Linux/Mac
# or
backend\venv\Scripts\activate  # Windows

# Install Python dependencies
cd backend
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install spaCy model for NLP
python -m spacy download en_core_web_sm

# Install Node.js dependencies (if using React frontend)
cd ../frontend
npm install
```

### 2. Model Setup (i5 6th Gen Optimized)

```bash
# Create models directory
mkdir -p backend/models

# Download lightweight models optimized for CPU
cd backend

# YOLOv8 Nano (fastest, CPU-optimized)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Download pre-trained threat intelligence model (if available)
# wget https://huggingface.co/models/secureBERT/resolve/main/model.bin -O models/threat_model.bin
```

### 3. Configuration

```bash
# Copy example configurations
cp config/config.example.json config/config.json

# Edit configuration for your system
nano config/config.json
```

**Optimized config.json for i5 6th Gen:**
```json
{
  "video_surveillance": {
    "model_size": "nano",
    "batch_size": 1,
    "inference_device": "cpu",
    "max_concurrent_streams": 2,
    "frame_skip": 2,
    "resolution": [640, 480]
  },
  "threat_intelligence": {
    "model_type": "distilbert",
    "batch_size": 8,
    "max_sequence_length": 256
  },
  "border_anomaly": {
    "detection_method": "isolation_forest",
    "min_trajectory_length": 5,
    "contamination": 0.1
  }
}
```

### 4. Run the System

```bash
# Terminal 1: Start Alert Broker API
cd backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Start Threat Intelligence Pipeline
python pipelines/threat_intelligence/pipeline.py

# Terminal 3: Start Video Surveillance Pipeline
python pipelines/video_surveillance/pipeline.py

# Terminal 4: Start Border Anomaly Pipeline
python pipelines/border_anomaly/pipeline.py

# Terminal 5: Start Frontend (if using React)
cd frontend
npm run dev
```

## Docker Deployment

### 1. Build Docker Images

```bash
# Build all services
docker-compose build

# Or build individually
docker build -t hifazat-backend ./backend
docker build -t hifazat-frontend ./frontend
```

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**docker-compose.yml (CPU Optimized):**
```yaml
version: '3.8'
services:
  alert-broker:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - INFERENCE_DEVICE=cpu
      - MODEL_SIZE=nano
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G

  threat-pipeline:
    build: ./backend
    command: python pipelines/threat_intelligence/pipeline.py
    depends_on:
      - alert-broker
    environment:
      - ALERT_BROKER_URL=http://alert-broker:8000
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G

  video-pipeline:
    build: ./backend
    command: python pipelines/video_surveillance/pipeline.py
    depends_on:
      - alert-broker
    volumes:
      - ./test_videos:/app/test_videos
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 3G

  anomaly-pipeline:
    build: ./backend
    command: python pipelines/border_anomaly/pipeline.py
    depends_on:
      - alert-broker
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - alert-broker
```

## Cloud Deployment Options

### 1. Google Cloud Platform (GCP)

#### Google Colab for Development/Testing
```python
# Install in Colab notebook
!git clone <your-repo-url>
%cd hifazat-ai-prototype/backend
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm

# Run lightweight version
!python pipelines/threat_intelligence/pipeline.py --mode=demo
```

#### Google Cloud Run Deployment
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/hifazat-backend
gcloud run deploy hifazat-backend \
  --image gcr.io/PROJECT_ID/hifazat-backend \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10
```

#### Google Kubernetes Engine (GKE)
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hifazat-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hifazat-backend
  template:
    metadata:
      labels:
        app: hifazat-backend
    spec:
      containers:
      - name: backend
        image: gcr.io/PROJECT_ID/hifazat-backend
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: INFERENCE_DEVICE
          value: "cpu"
```

### 2. Amazon Web Services (AWS)

#### AWS EC2 Deployment
```bash
# Launch EC2 instance (t3.large recommended for i5 6th gen equivalent)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx

# SSH and setup
ssh -i your-key.pem ubuntu@ec2-instance-ip
sudo apt update && sudo apt install -y docker.io docker-compose
git clone <your-repo-url>
cd hifazat-ai-prototype
docker-compose up -d
```

#### AWS ECS Fargate
```json
{
  "family": "hifazat-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/hifazat-backend",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "INFERENCE_DEVICE",
          "value": "cpu"
        }
      ]
    }
  ]
}
```

#### AWS Lambda (Serverless)
```python
# lambda_handler.py
import json
from pipelines.threat_intelligence.pipeline import ThreatIntelligencePipeline

def lambda_handler(event, context):
    pipeline = ThreatIntelligencePipeline()
    
    # Process threat intelligence data
    feed_data = json.loads(event['body'])
    results = pipeline.process_feed(feed_data)
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

### 3. Microsoft Azure

#### Azure Container Instances
```bash
# Create resource group
az group create --name hifazat-rg --location eastus

# Deploy container
az container create \
  --resource-group hifazat-rg \
  --name hifazat-backend \
  --image your-registry/hifazat-backend \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables INFERENCE_DEVICE=cpu
```

#### Azure Kubernetes Service (AKS)
```bash
# Create AKS cluster
az aks create \
  --resource-group hifazat-rg \
  --name hifazat-cluster \
  --node-count 2 \
  --node-vm-size Standard_D2s_v3 \
  --enable-addons monitoring

# Deploy application
kubectl apply -f k8s-deployment.yaml
```

### 4. Heroku Deployment

```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create hifazat-ai-prototype

# Set environment variables
heroku config:set INFERENCE_DEVICE=cpu
heroku config:set MODEL_SIZE=nano

# Deploy
git push heroku main
```

**Procfile:**
```
web: cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT
worker: cd backend && python pipelines/threat_intelligence/pipeline.py
```

## Performance Optimization for i5 6th Gen

### 1. CPU Optimization
```python
# backend/config/cpu_optimization.py
import os
import torch

# Set CPU threads for optimal performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

# Use CPU-optimized models
OPTIMIZED_CONFIG = {
    'yolo_model': 'yolov8n.pt',  # Nano version
    'batch_size': 1,
    'half_precision': False,  # Keep full precision for CPU
    'max_det': 100,  # Reduce max detections
    'conf_threshold': 0.5,
    'iou_threshold': 0.45
}
```

### 2. Memory Management
```python
# backend/utils/memory_manager.py
import gc
import psutil

class MemoryManager:
    def __init__(self, max_memory_percent=80):
        self.max_memory_percent = max_memory_percent
    
    def check_memory(self):
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.max_memory_percent:
            gc.collect()
            return True
        return False
    
    def optimize_for_cpu(self):
        # Disable GPU memory allocation
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Set conservative memory limits
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, -1))  # 4GB limit
```

### 3. Model Quantization
```python
# backend/utils/model_optimization.py
import torch
from ultralytics import YOLO

def quantize_model(model_path, output_path):
    """Quantize model for CPU inference"""
    model = YOLO(model_path)
    
    # Export to ONNX with optimization
    model.export(
        format='onnx',
        optimize=True,
        half=False,  # Keep FP32 for CPU
        simplify=True,
        opset=11
    )
    
    return f"{model_path.replace('.pt', '.onnx')}"
```

## Testing and Evaluation

### 1. Unit Tests
```bash
# Run all tests
cd backend
python -m pytest tests/ -v

# Run specific pipeline tests
python -m pytest tests/test_threat_intelligence.py -v
python -m pytest tests/test_video_surveillance.py -v
python -m pytest tests/test_border_anomaly.py -v
```

### 2. Performance Benchmarking
```bash
# Run performance tests
python scripts/benchmark.py --config config/cpu_optimized.json

# Memory profiling
python -m memory_profiler scripts/profile_memory.py

# CPU profiling
python -m cProfile -o profile.stats scripts/profile_cpu.py
```

### 3. Model Evaluation
```bash
# Evaluate detection accuracy
python scripts/evaluate_detection.py --dataset test_data/detection/
python scripts/evaluate_tracking.py --dataset test_data/tracking/
python scripts/evaluate_anomaly.py --dataset test_data/anomaly/

# Generate evaluation report
python scripts/generate_report.py --output evaluation_report.html
```

## Monitoring and Logging

### 1. Application Monitoring
```python
# backend/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
PROCESSED_FRAMES = Counter('processed_frames_total', 'Total processed frames')
PROCESSING_TIME = Histogram('processing_time_seconds', 'Frame processing time')
ACTIVE_STREAMS = Gauge('active_streams', 'Number of active video streams')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')

def monitor_performance():
    while True:
        memory_usage = psutil.virtual_memory().used
        MEMORY_USAGE.set(memory_usage)
        time.sleep(10)
```

### 2. Logging Configuration
```python
# backend/utils/logging.py
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/hifazat.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Monitor memory usage
htop
# or
watch -n 1 'free -h'

# Reduce batch size in config
# Set YOLO model to nano version
# Enable frame skipping
```

#### 2. CPU Performance
```bash
# Check CPU usage
top -p $(pgrep -f python)

# Optimize thread count
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### 3. Model Loading Issues
```bash
# Clear model cache
rm -rf ~/.cache/torch/hub/
rm -rf ~/.ultralytics/

# Re-download models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### 4. Docker Issues
```bash
# Check container resources
docker stats

# Increase memory limits
docker run --memory=4g --cpus=2 your-image

# Check logs
docker-compose logs -f service-name
```

## Security Considerations

### 1. API Security
```python
# backend/security/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. Data Privacy
```python
# backend/privacy/anonymization.py
import cv2
import numpy as np

def blur_faces(image, face_locations):
    """Blur detected faces for privacy"""
    for (top, right, bottom, left) in face_locations:
        face_region = image[top:bottom, left:right]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        image[top:bottom, left:right] = blurred_face
    return image
```

## Production Deployment Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database backups configured
- [ ] Monitoring and alerting setup
- [ ] Log rotation configured
- [ ] Resource limits set
- [ ] Security headers configured
- [ ] API rate limiting enabled
- [ ] Health checks implemented
- [ ] Graceful shutdown handling
- [ ] Error tracking setup (Sentry, etc.)
- [ ] Performance monitoring (New Relic, DataDog)

## Support and Maintenance

### Regular Maintenance Tasks
```bash
# Weekly tasks
./scripts/cleanup_old_logs.sh
./scripts/update_threat_feeds.sh
./scripts/backup_models.sh

# Monthly tasks
./scripts/retrain_models.sh
./scripts/performance_report.sh
./scripts/security_audit.sh
```

### Model Updates
```bash
# Update YOLO models
python scripts/update_yolo.py --version latest

# Update threat intelligence models
python scripts/update_threat_models.py --source huggingface

# Retrain anomaly detection
python scripts/retrain_anomaly.py --data new_training_data/
```

This comprehensive guide should help you deploy and run HifazatAI on various platforms while optimizing for your i5 6th gen processor constraints. The system is designed to be modular and scalable, allowing you to start with a basic setup and expand as needed.