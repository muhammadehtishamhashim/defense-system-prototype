# HifazatAI System Architecture

## Overview

HifazatAI is a comprehensive AI-powered security monitoring system designed with a microservices-inspired architecture. The system integrates multiple specialized AI pipelines into a unified platform for real-time threat detection, video surveillance analytics, and border anomaly detection.

## Architecture Principles

### Design Philosophy
- **Modularity**: Each AI pipeline operates independently with clear interfaces
- **Scalability**: Components can be scaled horizontally based on demand
- **Reliability**: Fault-tolerant design with graceful degradation
- **Performance**: Optimized for resource-constrained environments (i5 6th gen target)
- **Extensibility**: Easy to add new pipelines and detection capabilities

### Key Architectural Decisions

#### CPU-First Design
- Optimized for CPU-only inference (no GPU dependency)
- Lightweight model variants (YOLOv8n, DistilBERT)
- Thread-level parallelization with OpenMP optimization
- Memory-efficient processing with garbage collection tuning

#### Real-time Processing
- Event-driven architecture with asynchronous processing
- Server-Sent Events (SSE) for real-time frontend updates
- Streaming data processing with configurable batch sizes
- Low-latency alert generation and propagation

#### Data-Centric Approach
- Centralized alert broker for all pipeline outputs
- Standardized alert schema across different detection types
- Persistent storage with SQLite for development/small deployments
- Configurable data retention and archival policies

## System Components

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  React Dashboard (TypeScript)                                   │
│  ├── Real-time Alert Feed                                       │
│  ├── Video Analysis Interface                                   │
│  ├── System Monitoring Dashboard                                │
│  └── Configuration Management                                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP/WebSocket/SSE
┌─────────────────▼───────────────────────────────────────────────┐
│                     API Gateway Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Alert Broker                                          │
│  ├── RESTful API Endpoints                                     │
│  ├── Real-time Event Streaming                                 │
│  ├── Request Validation & Error Handling                       │
│  └── CORS & Security Middleware                                │
└─────────────────┬───────────────────────────────────────────────┘
                  │ Internal API Calls
┌─────────────────▼───────────────────────────────────────────────┐
│                   Processing Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Threat Intel    │ │ Video Surveillance│ │ Border Anomaly  │   │
│  │ Pipeline        │ │ Pipeline         │ │ Pipeline        │   │
│  │                 │ │                  │ │                 │   │
│  │ ├─IOC Extract   │ │ ├─Object Detect  │ │ ├─Trajectory    │   │
│  │ ├─Risk Classify │ │ ├─Multi-Track    │ │ ├─Feature Ext   │   │
│  │ └─Alert Gen     │ │ └─Behavior Anal  │ │ └─Anomaly Det   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────┬───────────────────────────────────────────────┘
                  │ Database Operations
┌─────────────────▼───────────────────────────────────────────────┐
│                     Data Layer                                  │
├─────────────────────────────────────────────────────────────────┤
│  ├── SQLite Database (Alerts, Metrics, Configuration)          │
│  ├── File Storage (Media, Models, Logs)                        │
│  └── Configuration Management (JSON/YAML)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### Frontend Layer (React/TypeScript)

**Technology Stack:**
- React 19+ with TypeScript for type safety
- Vite for fast development and optimized builds
- Tailwind CSS for responsive styling
- Axios for HTTP client with retry logic
- React Router for client-side routing

**Key Components:**
- **Dashboard Layout**: Main application shell with navigation
- **Alert Management**: Real-time alert feed with filtering and search
- **Video Analysis**: Video player with overlay visualization
- **System Monitor**: Pipeline status and performance metrics
- **Settings Interface**: Configuration management UI

**Real-time Communication:**
```typescript
// SSE Connection for Real-time Updates
const eventSource = new EventSource('/api/events');
eventSource.onmessage = (event) => {
  const alertData = JSON.parse(event.data);
  updateAlertFeed(alertData);
};
```

#### API Gateway Layer (FastAPI)

**Core Responsibilities:**
- RESTful API endpoint management
- Request validation using Pydantic models
- Real-time event streaming via Server-Sent Events
- CORS handling for cross-origin requests
- Error handling and logging

**API Structure:**
```python
# Alert Management Endpoints
POST /alerts/{type}     # Create new alert
GET  /alerts           # Retrieve alerts with filtering
GET  /alerts/{id}      # Get specific alert
PUT  /alerts/{id}/status # Update alert status

# System Monitoring
GET  /health           # Health check
GET  /metrics/{pipeline} # Pipeline metrics
POST /metrics          # Submit metrics

# Real-time Events
GET  /events           # SSE stream
```

**Middleware Stack:**
1. CORS middleware for frontend access
2. Request logging and timing
3. Error handling and standardization
4. Request validation and sanitization

#### Processing Layer (AI Pipelines)

##### Threat Intelligence Pipeline

**Architecture:**
```python
class ThreatIntelligencePipeline:
    def __init__(self):
        self.ioc_extractor = IOCExtractor()
        self.risk_classifier = RiskClassifier()
        self.alert_generator = AlertGenerator()
    
    async def process_feed(self, feed_data):
        # Extract IOCs using regex and NER
        iocs = self.ioc_extractor.extract(feed_data)
        
        # Classify risk using DistilBERT
        risk_scores = self.risk_classifier.classify(iocs)
        
        # Generate alerts for high-risk IOCs
        alerts = self.alert_generator.create_alerts(iocs, risk_scores)
        
        return alerts
```

**Components:**
- **IOC Extractor**: Regex patterns + spaCy NER for indicator extraction
- **Risk Classifier**: DistilBERT-based text classification
- **Feed Processor**: Multi-format ingestion (JSON, CSV, XML)
- **Alert Generator**: Structured alert creation with metadata

##### Video Surveillance Pipeline

**Architecture:**
```python
class VideoSurveillancePipeline:
    def __init__(self):
        self.detector = YOLOv8Detector()
        self.tracker = ByteTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
    
    async def process_frame(self, frame):
        # Object detection with YOLOv8n
        detections = self.detector.detect(frame)
        
        # Multi-object tracking
        tracks = self.tracker.update(detections)
        
        # Behavior analysis
        behaviors = self.behavior_analyzer.analyze(tracks)
        
        return behaviors
```

**Components:**
- **Object Detector**: YOLOv8 Nano for CPU-optimized detection
- **Multi-Object Tracker**: ByteTrack for lightweight tracking
- **Behavior Analyzer**: Rule-based behavior detection
- **Frame Processor**: Optimized frame handling with skipping

##### Border Anomaly Detection Pipeline

**Architecture:**
```python
class BorderAnomalyPipeline:
    def __init__(self):
        self.trajectory_extractor = TrajectoryExtractor()
        self.feature_computer = FeatureComputer()
        self.anomaly_detector = IsolationForestDetector()
    
    async def process_trajectory(self, trajectory):
        # Extract trajectory features
        features = self.feature_computer.compute(trajectory)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.predict(features)
        
        return anomaly_score
```

**Components:**
- **Trajectory Extractor**: Path extraction from tracking data
- **Feature Computer**: Speed, curvature, direction analysis
- **Anomaly Detector**: Isolation Forest for unsupervised detection
- **Motion Analyzer**: Fallback motion-based detection

#### Data Layer

##### Database Schema (SQLite)

```sql
-- Alerts table
CREATE TABLE alerts (
    id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    type TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    data JSON NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_name TEXT NOT NULL,
    processing_rate REAL,
    accuracy_score REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT,
    error_count INTEGER DEFAULT 0
);

-- Configuration table
CREATE TABLE configuration (
    key TEXT PRIMARY KEY,
    value JSON NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

##### File Storage Structure

```
data/
├── alerts.db              # SQLite database
├── media/
│   ├── snapshots/         # Video frame captures
│   ├── videos/           # Video segments
│   └── trajectories/     # Trajectory visualizations
├── models/
│   ├── yolov8n.pt        # Object detection model
│   ├── threat_classifier/ # Threat classification model
│   └── anomaly_detector/ # Anomaly detection model
├── logs/
│   ├── api.log           # API server logs
│   ├── pipelines.log     # Pipeline processing logs
│   └── errors.log        # Error logs
└── config/
    ├── default_config.json
    ├── pipeline_configs/
    └── user_preferences.json
```

## Data Flow Architecture

### Alert Processing Flow

```
1. Data Ingestion
   ├── Threat Feeds → IOC Extraction
   ├── Video Streams → Frame Processing
   └── Tracking Data → Trajectory Analysis

2. AI Processing
   ├── Feature Extraction
   ├── Model Inference
   └── Confidence Scoring

3. Alert Generation
   ├── Threshold Filtering
   ├── Alert Structuring
   └── Metadata Enrichment

4. Alert Broker
   ├── Database Storage
   ├── Real-time Broadcasting
   └── API Exposure

5. Frontend Display
   ├── Real-time Updates
   ├── User Interaction
   └── Status Management
```

### Real-time Event Flow

```
Pipeline → Alert Broker → SSE Stream → Frontend
    ↓           ↓            ↓           ↓
Database    Validation   Browser    UI Update
Storage     & Logging    Cache      & Display
```

## Scalability Considerations

### Horizontal Scaling

**Pipeline Scaling:**
- Each pipeline can run as independent processes
- Load balancing across multiple pipeline instances
- Queue-based work distribution for high-volume scenarios

**Database Scaling:**
- SQLite for development and small deployments
- PostgreSQL/MySQL for production scaling
- Read replicas for query performance
- Sharding strategies for large datasets

**Frontend Scaling:**
- CDN deployment for static assets
- Multiple frontend instances behind load balancer
- Client-side caching and optimization

### Vertical Scaling

**CPU Optimization:**
- Thread pool management for parallel processing
- ONNX model optimization for faster inference
- Memory mapping for large model files
- Batch processing for improved throughput

**Memory Management:**
- Garbage collection tuning for Python processes
- Memory pooling for frequent allocations
- Lazy loading of models and resources
- Configurable cache sizes

## Security Architecture

### API Security

**Input Validation:**
- Pydantic models for request validation
- SQL injection prevention with parameterized queries
- File upload restrictions and validation
- Rate limiting for API endpoints

**Access Control:**
- JWT-based authentication (production)
- Role-based access control (RBAC)
- API key management for pipeline access
- Session management and timeout

### Data Security

**Data Protection:**
- Encryption at rest for sensitive data
- Secure communication with HTTPS/TLS
- PII detection and anonymization
- Audit logging for data access

**Privacy Features:**
- Face redaction in video processing
- Configurable data retention policies
- GDPR compliance features
- Data export and deletion capabilities

## Performance Architecture

### CPU Optimization

**Model Optimization:**
```python
# ONNX Runtime optimization
import onnxruntime as ort

# CPU-optimized session
session = ort.InferenceSession(
    model_path,
    providers=['CPUExecutionProvider'],
    sess_options=ort.SessionOptions()
)

# Thread configuration
session.set_providers(['CPUExecutionProvider'], [
    {'intra_op_num_threads': 4}
])
```

**Threading Configuration:**
```bash
# Environment variables for CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### Memory Optimization

**Garbage Collection Tuning:**
```python
import gc
import os

# Optimize garbage collection
gc.set_threshold(700, 10, 10)

# Memory management
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
```

**Resource Monitoring:**
```python
import psutil

def monitor_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    
    return {
        'cpu_usage': cpu_percent,
        'memory_usage': memory_info.percent,
        'available_memory': memory_info.available
    }
```

## Deployment Architecture

### Development Environment

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./data:/app/data
    environment:
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=sqlite:///data/alerts.db
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    environment:
      - VITE_API_URL=http://localhost:8000
```

### Production Environment

**Container Orchestration:**
- Docker containers with resource limits
- Health checks and auto-restart policies
- Volume mounts for persistent data
- Environment-based configuration

**Resource Allocation:**
```dockerfile
# Production Dockerfile with resource optimization
FROM python:3.10-slim

# CPU optimization
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Memory limits
RUN echo 'vm.swappiness=10' >> /etc/sysctl.conf

# Application setup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring and Observability

### Application Monitoring

**Health Checks:**
- API endpoint health monitoring
- Database connection validation
- Model loading verification
- External service connectivity

**Performance Metrics:**
- Request latency and throughput
- Error rates and types
- Resource utilization trends
- Alert processing statistics

### Logging Architecture

**Structured Logging:**
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "Alert processed",
    alert_id=alert.id,
    pipeline=alert.source_pipeline,
    confidence=alert.confidence,
    processing_time=elapsed_time
)
```

**Log Aggregation:**
- Centralized logging with structured format
- Log rotation and retention policies
- Error alerting and notification
- Performance trend analysis

## Future Architecture Considerations

### Microservices Evolution

**Service Decomposition:**
- Separate services for each AI pipeline
- Dedicated alert broker service
- Independent configuration service
- Centralized logging and monitoring service

**Communication Patterns:**
- Message queues for asynchronous processing
- Service mesh for inter-service communication
- Event sourcing for audit trails
- CQRS for read/write separation

### Cloud-Native Architecture

**Kubernetes Deployment:**
- Pod-based pipeline scaling
- ConfigMaps for configuration management
- Persistent volumes for data storage
- Ingress controllers for traffic management

**Serverless Components:**
- Function-as-a-Service for event processing
- Managed databases for scalability
- Object storage for media files
- CDN for global content delivery

This architecture document provides a comprehensive overview of the HifazatAI system design, enabling developers and operators to understand, maintain, and extend the system effectively.