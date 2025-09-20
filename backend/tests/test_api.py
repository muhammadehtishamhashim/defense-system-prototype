"""
Unit tests for HifazatAI Alert Broker API endpoints.
Tests all CRUD operations and error handling.
"""

import pytest
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from models.database import Base, get_db
from models.alerts import AlertStatus, AlertType

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestAPIEndpoints:
    """Test class for API endpoints"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear database before each test
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "HifazatAI Alert Broker API"
        assert data["status"] == "running"
        assert "timestamp" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
    
    def test_create_threat_alert(self):
        """Test creating a threat intelligence alert"""
        alert_data = {
            "id": "test-threat-001",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85,
            "source_pipeline": "threat_intelligence",
            "status": "active",
            "ioc_type": "ip",
            "ioc_value": "192.168.1.100",
            "risk_level": "High",
            "evidence_text": "Suspicious IP detected in threat feed",
            "source_feed": "test_feed",
            "risk_score": 0.9
        }
        
        response = client.post("/alerts/threat", json=alert_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Threat alert created successfully"
        assert data["data"]["id"] == "test-threat-001"
    
    def test_create_video_alert(self):
        """Test creating a video surveillance alert"""
        alert_data = {
            "id": "test-video-001",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.75,
            "source_pipeline": "video_surveillance",
            "status": "active",
            "event_type": "loitering",
            "bounding_box": [100, 200, 50, 80],
            "track_id": 123,
            "snapshot_path": "/media/snapshots/test.jpg",
            "video_timestamp": 45.5,
            "metadata": {"camera_id": "cam_001"}
        }
        
        response = client.post("/alerts/video", json=alert_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Video alert created successfully"
        assert data["data"]["id"] == "test-video-001"
    
    def test_create_anomaly_alert(self):
        """Test creating a border anomaly alert"""
        alert_data = {
            "id": "test-anomaly-001",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.65,
            "source_pipeline": "border_anomaly",
            "status": "active",
            "anomaly_type": "unusual_movement",
            "severity_score": 0.8,
            "trajectory_points": [[10, 20], [15, 25], [20, 30]],
            "feature_vector": [0.1, 0.2, 0.3, 0.4],
            "supporting_frames": ["/media/frames/frame1.jpg", "/media/frames/frame2.jpg"]
        }
        
        response = client.post("/alerts/anomaly", json=alert_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Anomaly alert created successfully"
        assert data["data"]["id"] == "test-anomaly-001"
    
    def test_get_alerts_empty(self):
        """Test getting alerts when database is empty"""
        response = client.get("/alerts")
        assert response.status_code == 200
        
        data = response.json()
        assert data["alerts"] == []
        assert data["total"] == 0
        assert data["has_more"] is False
    
    def test_get_alerts_with_data(self):
        """Test getting alerts with data in database"""
        # Create test alerts
        self.test_create_threat_alert()
        self.test_create_video_alert()
        
        response = client.get("/alerts")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["alerts"]) == 2
        assert data["total"] == 2
        assert data["has_more"] is False
    
    def test_get_alerts_with_filters(self):
        """Test getting alerts with type filter"""
        # Create test alerts
        self.test_create_threat_alert()
        self.test_create_video_alert()
        
        # Filter by threat type
        response = client.get("/alerts?alert_type=threat")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["alerts"]) == 1
        assert data["total"] == 1
        assert data["alerts"][0]["ioc_type"] == "ip"
    
    def test_get_alerts_pagination(self):
        """Test alert pagination"""
        # Create multiple alerts
        for i in range(5):
            alert_data = {
                "id": f"test-threat-{i:03d}",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.8,
                "source_pipeline": "threat_intelligence",
                "status": "active",
                "ioc_type": "ip",
                "ioc_value": f"192.168.1.{i}",
                "risk_level": "Medium",
                "evidence_text": f"Test threat {i}",
                "source_feed": "test_feed",
                "risk_score": 0.7
            }
            client.post("/alerts/threat", json=alert_data)
        
        # Test pagination
        response = client.get("/alerts?limit=2&offset=0")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["alerts"]) == 2
        assert data["total"] == 5
        assert data["has_more"] is True
        
        # Test second page
        response = client.get("/alerts?limit=2&offset=2")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["alerts"]) == 2
        assert data["has_more"] is True
    
    def test_get_alert_by_id(self):
        """Test getting a specific alert by ID"""
        # Create test alert
        self.test_create_threat_alert()
        
        response = client.get("/alerts/test-threat-001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "test-threat-001"
        assert data["data"]["ioc_value"] == "192.168.1.100"
    
    def test_get_alert_not_found(self):
        """Test getting non-existent alert"""
        response = client.get("/alerts/non-existent-id")
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"]
    
    def test_update_alert_status(self):
        """Test updating alert status"""
        # Create test alert
        self.test_create_threat_alert()
        
        update_data = {
            "status": "reviewed",
            "notes": "Reviewed by analyst"
        }
        
        response = client.put("/alerts/test-threat-001/status", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "reviewed" in data["message"]
    
    def test_update_alert_status_not_found(self):
        """Test updating status of non-existent alert"""
        update_data = {
            "status": "reviewed"
        }
        
        response = client.put("/alerts/non-existent-id/status", json=update_data)
        assert response.status_code == 404
    
    def test_create_system_metrics(self):
        """Test creating system metrics"""
        metrics_data = {
            "pipeline_name": "threat_intelligence",
            "processing_rate": 10.5,
            "accuracy_score": 0.85,
            "last_update": datetime.now().isoformat(),
            "status": "running",
            "error_count": 0
        }
        
        response = client.post("/metrics", json=metrics_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "System metrics created successfully"
    
    def test_get_pipeline_metrics(self):
        """Test getting pipeline metrics"""
        # Create metrics first
        self.test_create_system_metrics()
        
        response = client.get("/metrics/threat_intelligence")
        assert response.status_code == 200
        
        data = response.json()
        assert data["pipeline_name"] == "threat_intelligence"
        assert data["processing_rate"] == 10.5
        assert data["accuracy_score"] == 0.85
    
    def test_get_pipeline_metrics_not_found(self):
        """Test getting metrics for non-existent pipeline"""
        response = client.get("/metrics/non_existent_pipeline")
        assert response.status_code == 404
    
    def test_invalid_alert_data(self):
        """Test creating alert with invalid data"""
        invalid_data = {
            "id": "test-invalid",
            "timestamp": "invalid-timestamp",
            "confidence": 1.5,  # Invalid confidence > 1.0
            "source_pipeline": "test"
        }
        
        response = client.post("/alerts/threat", json=invalid_data)
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])