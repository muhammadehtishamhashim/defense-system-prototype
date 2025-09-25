# HifazatAI API Documentation

## Overview

The HifazatAI Alert Broker API is a FastAPI-based service that manages security alerts from multiple AI pipelines. It provides RESTful endpoints for alert creation, retrieval, and management, along with real-time updates via Server-Sent Events.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API operates without authentication for prototype purposes. In production, JWT-based authentication would be implemented.

## Content Type

All requests and responses use `application/json` content type unless otherwise specified.

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

Error responses follow this format:
```json
{
  "detail": "Error description"
}
```

## Core Endpoints

### Health Check

#### GET /health
Returns the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "database": "connected"
}
```

### Root Information

#### GET /
Returns basic API information.

**Response:**
```json
{
  "name": "HifazatAI Alert Broker API",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Alert Management

### Create Threat Intelligence Alert

#### POST /alerts/threat

Creates a new threat intelligence alert.

**Request Body:**
```json
{
  "id": "threat_001",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "confidence": 0.85,
  "status": "active",
  "ioc_type": "ip",
  "ioc_value": "192.168.1.100",
  "risk_level": "High",
  "evidence_text": "Suspicious network activity detected",
  "source_feed": "ThreatFeed_1",
  "extracted_iocs": ["192.168.1.100"],
  "threat_categories": ["malware", "botnet"]
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "Threat alert created successfully",
  "data": {
    "id": "threat_001",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "confidence": 0.85,
    "status": "active",
    "source_pipeline": "threat_intelligence",
    "ioc_type": "ip",
    "ioc_value": "192.168.1.100",
    "risk_level": "High",
    "evidence_text": "Suspicious network activity detected",
    "source_feed": "ThreatFeed_1",
    "extracted_iocs": ["192.168.1.100"],
    "threat_categories": ["malware", "botnet"]
  }
}
```

### Create Video Surveillance Alert

#### POST /alerts/video

Creates a new video surveillance alert.

**Request Body:**
```json
{
  "id": "video_001",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "confidence": 0.92,
  "status": "active",
  "camera_id": "cam_001",
  "behavior_type": "loitering",
  "detected_objects": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 300],
      "track_id": "track_001"
    }
  ],
  "frame_path": "/media/snapshots/frame_001.jpg",
  "video_segment": "/media/videos/segment_001.mp4",
  "zone_info": {
    "zone_id": "restricted_area_1",
    "zone_name": "Restricted Area 1"
  }
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "Video alert created successfully",
  "data": {
    "id": "video_001",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "confidence": 0.92,
    "status": "active",
    "source_pipeline": "video_surveillance",
    "camera_id": "cam_001",
    "behavior_type": "loitering",
    "detected_objects": [...],
    "frame_path": "/media/snapshots/frame_001.jpg",
    "video_segment": "/media/videos/segment_001.mp4",
    "zone_info": {...}
  }
}
```

### Create Border Anomaly Alert

#### POST /alerts/anomaly

Creates a new border anomaly detection alert.

**Request Body:**
```json
{
  "id": "anomaly_001",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "confidence": 0.78,
  "status": "active",
  "anomaly_type": "unusual_trajectory",
  "anomaly_score": 0.85,
  "trajectory_data": {
    "points": [[100, 200], [110, 210], [120, 220]],
    "duration": 45.5,
    "speed_profile": [2.1, 2.3, 1.8],
    "curvature": 0.15
  },
  "location": {
    "sector": "border_sector_3",
    "coordinates": [31.7683, 35.2137]
  },
  "supporting_frames": [
    "/media/anomaly/frame_001.jpg",
    "/media/anomaly/frame_002.jpg"
  ]
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "Anomaly alert created successfully",
  "data": {
    "id": "anomaly_001",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "confidence": 0.78,
    "status": "active",
    "source_pipeline": "border_anomaly",
    "anomaly_type": "unusual_trajectory",
    "anomaly_score": 0.85,
    "trajectory_data": {...},
    "location": {...},
    "supporting_frames": [...]
  }
}
```

### Retrieve Alerts

#### GET /alerts

Retrieves alerts with optional filtering and pagination.

**Query Parameters:**
- `alert_type` (optional): Filter by alert type (`threat`, `video`, `anomaly`)
- `status` (optional): Filter by status (`active`, `reviewed`, `dismissed`)
- `start_time` (optional): Filter by start time (ISO 8601 format)
- `end_time` (optional): Filter by end time (ISO 8601 format)
- `limit` (optional): Number of alerts to return (1-1000, default: 100)
- `offset` (optional): Number of alerts to skip (default: 0)

**Example Request:**
```
GET /alerts?alert_type=threat&status=active&limit=50&offset=0
```

**Response (200):**
```json
{
  "alerts": [
    {
      "id": "threat_001",
      "timestamp": "2024-01-15T10:30:00.000Z",
      "confidence": 0.85,
      "status": "active",
      "source_pipeline": "threat_intelligence",
      "ioc_type": "ip",
      "ioc_value": "192.168.1.100",
      "risk_level": "High"
    }
  ],
  "total": 150,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
```

### Get Specific Alert

#### GET /alerts/{alert_id}

Retrieves a specific alert by ID.

**Path Parameters:**
- `alert_id`: Unique identifier of the alert

**Response (200):**
```json
{
  "success": true,
  "message": "Alert retrieved successfully",
  "data": {
    "id": "threat_001",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "confidence": 0.85,
    "status": "active",
    "source_pipeline": "threat_intelligence",
    "ioc_type": "ip",
    "ioc_value": "192.168.1.100",
    "risk_level": "High",
    "evidence_text": "Suspicious network activity detected"
  }
}
```

### Update Alert Status

#### PUT /alerts/{alert_id}/status

Updates the status of an alert.

**Path Parameters:**
- `alert_id`: Unique identifier of the alert

**Request Body:**
```json
{
  "status": "reviewed",
  "notes": "Investigated and confirmed as false positive"
}
```

**Response (200):**
```json
{
  "success": true,
  "message": "Alert status updated to reviewed",
  "data": null
}
```

## System Metrics

### Create System Metrics

#### POST /metrics

Creates a system metrics record for monitoring pipeline performance.

**Request Body:**
```json
{
  "pipeline_name": "threat_intelligence",
  "processing_rate": 150.5,
  "accuracy_score": 0.87,
  "last_update": "2024-01-15T10:30:00.000Z",
  "status": "healthy",
  "error_count": 2
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "System metrics created successfully"
}
```

### Get Pipeline Metrics

#### GET /metrics/{pipeline_name}

Retrieves the latest metrics for a specific pipeline.

**Path Parameters:**
- `pipeline_name`: Name of the pipeline (`threat_intelligence`, `video_surveillance`, `border_anomaly`)

**Response (200):**
```json
{
  "pipeline_name": "threat_intelligence",
  "processing_rate": 150.5,
  "accuracy_score": 0.87,
  "last_update": "2024-01-15T10:30:00.000Z",
  "status": "healthy",
  "error_count": 2
}
```

## Real-time Updates

### Server-Sent Events

#### GET /events

Establishes a Server-Sent Events connection for real-time alert updates.

**Response Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Event Format:**
```
event: new_alert
data: {"id": "alert_001", "type": "threat", "timestamp": "2024-01-15T10:30:00.000Z"}

event: alert_update
data: {"id": "alert_001", "status": "reviewed", "timestamp": "2024-01-15T10:35:00.000Z"}

event: system_status
data: {"pipeline": "video_surveillance", "status": "healthy", "timestamp": "2024-01-15T10:30:00.000Z"}
```

## Data Models

### Alert Status Enum
- `active`: Alert is active and requires attention
- `reviewed`: Alert has been reviewed by an analyst
- `dismissed`: Alert has been dismissed as false positive

### Alert Type Enum
- `threat`: Threat intelligence alert
- `video`: Video surveillance alert
- `anomaly`: Border anomaly alert

### Risk Level Enum (Threat Alerts)
- `High`: High-risk threat requiring immediate attention
- `Medium`: Medium-risk threat requiring investigation
- `Low`: Low-risk threat for monitoring

### Behavior Type Enum (Video Alerts)
- `loitering`: Person staying in area longer than threshold
- `zone_violation`: Unauthorized entry into restricted zone
- `abandoned_object`: Object left unattended for extended period

### Anomaly Type Enum (Anomaly Alerts)
- `unusual_trajectory`: Trajectory deviates from normal patterns
- `speed_anomaly`: Unusual speed profile detected
- `direction_anomaly`: Unexpected direction changes

## Rate Limits

Currently, no rate limits are enforced for the prototype. In production, consider implementing:
- 1000 requests per minute per IP
- 100 alert creations per minute per pipeline

## SDK Examples

### Python SDK Example

```python
import requests
from datetime import datetime

class HifazatAIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def create_threat_alert(self, ioc_type, ioc_value, risk_level, evidence_text):
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.8,
            "status": "active",
            "ioc_type": ioc_type,
            "ioc_value": ioc_value,
            "risk_level": risk_level,
            "evidence_text": evidence_text,
            "source_feed": "API_Client"
        }
        
        response = requests.post(f"{self.base_url}/alerts/threat", json=alert_data)
        return response.json()
    
    def get_alerts(self, alert_type=None, status=None, limit=100):
        params = {"limit": limit}
        if alert_type:
            params["alert_type"] = alert_type
        if status:
            params["status"] = status
        
        response = requests.get(f"{self.base_url}/alerts", params=params)
        return response.json()

# Usage
client = HifazatAIClient()
result = client.create_threat_alert("ip", "192.168.1.100", "High", "Malicious activity detected")
print(result)
```

### JavaScript SDK Example

```javascript
class HifazatAIClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async createThreatAlert(iocType, iocValue, riskLevel, evidenceText) {
        const alertData = {
            timestamp: new Date().toISOString(),
            confidence: 0.8,
            status: 'active',
            ioc_type: iocType,
            ioc_value: iocValue,
            risk_level: riskLevel,
            evidence_text: evidenceText,
            source_feed: 'API_Client'
        };
        
        const response = await fetch(`${this.baseUrl}/alerts/threat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(alertData)
        });
        
        return await response.json();
    }
    
    async getAlerts(alertType = null, status = null, limit = 100) {
        const params = new URLSearchParams({ limit: limit.toString() });
        if (alertType) params.append('alert_type', alertType);
        if (status) params.append('status', status);
        
        const response = await fetch(`${this.baseUrl}/alerts?${params}`);
        return await response.json();
    }
}

// Usage
const client = new HifazatAIClient();
client.createThreatAlert('ip', '192.168.1.100', 'High', 'Malicious activity detected')
    .then(result => console.log(result));
```

## Testing the API

### Using curl

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Create threat alert
curl -X POST "http://localhost:8000/alerts/threat" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T10:30:00.000Z",
    "confidence": 0.85,
    "status": "active",
    "ioc_type": "ip",
    "ioc_value": "192.168.1.100",
    "risk_level": "High",
    "evidence_text": "Suspicious network activity",
    "source_feed": "Test_Feed"
  }'

# Get alerts
curl -X GET "http://localhost:8000/alerts?limit=10"

# Update alert status
curl -X PUT "http://localhost:8000/alerts/threat_001/status" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "reviewed",
    "notes": "Investigated and resolved"
  }'
```

### Using Postman

1. Import the OpenAPI specification from `http://localhost:8000/openapi.json`
2. Set base URL to `http://localhost:8000`
3. Use the pre-configured requests for testing

## Interactive API Documentation

The API provides interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

These interfaces allow you to:
- Explore all available endpoints
- Test API calls directly from the browser
- View request/response schemas
- Download API specifications