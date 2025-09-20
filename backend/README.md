# HifazatAI Alert Broker API

The Alert Broker API is the central component of the HifazatAI system that manages security alerts from multiple AI pipelines.

## Features

- **Multi-Pipeline Support**: Handles alerts from threat intelligence, video surveillance, and border anomaly detection pipelines
- **RESTful API**: Full CRUD operations for alert management
- **Database Persistence**: SQLite database with SQLAlchemy ORM
- **Real-time Processing**: Fast alert ingestion and retrieval
- **Filtering & Pagination**: Advanced query capabilities
- **System Metrics**: Pipeline performance monitoring
- **OpenAPI Documentation**: Auto-generated API docs

## Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
python start_api.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check

### Alert Management

- `POST /alerts/threat` - Create threat intelligence alert
- `POST /alerts/video` - Create video surveillance alert  
- `POST /alerts/anomaly` - Create border anomaly alert
- `GET /alerts` - Get alerts with filtering and pagination
- `GET /alerts/{alert_id}` - Get specific alert
- `PUT /alerts/{alert_id}/status` - Update alert status

### System Metrics

- `POST /metrics` - Create system metrics
- `GET /metrics/{pipeline_name}` - Get pipeline metrics

## Alert Types

### Threat Intelligence Alert
```json
{
  "id": "threat-001",
  "timestamp": "2024-01-01T12:00:00Z",
  "confidence": 0.85,
  "source_pipeline": "threat_intelligence",
  "status": "active",
  "ioc_type": "ip",
  "ioc_value": "192.168.1.100",
  "risk_level": "High",
  "evidence_text": "Suspicious IP detected",
  "source_feed": "threat_feed_1",
  "risk_score": 0.9
}
```

### Video Surveillance Alert
```json
{
  "id": "video-001",
  "timestamp": "2024-01-01T12:00:00Z",
  "confidence": 0.75,
  "source_pipeline": "video_surveillance",
  "status": "active",
  "event_type": "loitering",
  "bounding_box": [100, 200, 50, 80],
  "track_id": 123,
  "snapshot_path": "/media/snapshots/frame.jpg",
  "video_timestamp": 45.5,
  "metadata": {"camera_id": "cam_001"}
}
```

### Border Anomaly Alert
```json
{
  "id": "anomaly-001",
  "timestamp": "2024-01-01T12:00:00Z",
  "confidence": 0.65,
  "source_pipeline": "border_anomaly",
  "status": "active",
  "anomaly_type": "unusual_movement",
  "severity_score": 0.8,
  "trajectory_points": [[10, 20], [15, 25], [20, 30]],
  "feature_vector": [0.1, 0.2, 0.3, 0.4],
  "supporting_frames": ["/media/frames/frame1.jpg"]
}
```

## Query Parameters

### Get Alerts (`GET /alerts`)

- `alert_type`: Filter by alert type (threat, video, anomaly)
- `status`: Filter by status (active, reviewed, dismissed, resolved)
- `start_time`: Filter by start timestamp
- `end_time`: Filter by end timestamp
- `limit`: Number of results (1-1000, default: 100)
- `offset`: Number of results to skip (default: 0)

Example:
```
GET /alerts?alert_type=threat&status=active&limit=50&offset=0
```

## Database Schema

The API uses SQLite with the following tables:

- **alerts**: Main alert storage
- **alert_media**: Associated media files
- **system_metrics**: Pipeline performance metrics

## Error Handling

The API returns structured error responses:

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

Common status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Configuration

Environment variables:
- `DATABASE_URL`: Database connection string (default: sqlite:///./hifazat.db)
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_FILE`: Log file path (optional)
- `CORS_ORIGINS`: Allowed CORS origins (default: http://localhost:3000)

## Development

### Project Structure
```
backend/
├── api/
│   └── main.py          # FastAPI application
├── models/
│   ├── alerts.py        # Pydantic models
│   └── database.py      # SQLAlchemy models
├── utils/
│   └── logging.py       # Logging utilities
├── tests/
│   └── test_api.py      # API tests
├── requirements.txt     # Dependencies
└── start_api.py        # Startup script
```

### Adding New Alert Types

1. Define Pydantic model in `models/alerts.py`
2. Add endpoint in `api/main.py`
3. Update database schema if needed
4. Add tests in `tests/test_api.py`

## License

This project is part of the HifazatAI security system prototype.