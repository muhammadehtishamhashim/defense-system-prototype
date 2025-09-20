"""
Pydantic models for HifazatAI alert system.
Defines data structures for different types of alerts and system metrics.
"""

from datetime import datetime
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class AlertStatus(str, Enum):
    """Alert status enumeration"""
    ACTIVE = "active"
    REVIEWED = "reviewed"
    DISMISSED = "dismissed"
    RESOLVED = "resolved"


class AlertType(str, Enum):
    """Alert type enumeration"""
    THREAT = "threat"
    VIDEO = "video"
    ANOMALY = "anomaly"


class BaseAlert(BaseModel):
    """Base alert model with common fields"""
    id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(..., description="Alert generation timestamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Alert confidence score")
    source_pipeline: str = Field(..., description="Source pipeline name")
    status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="Alert status")


class ThreatAlert(BaseAlert):
    """Threat intelligence alert model"""
    ioc_type: str = Field(..., description="IOC type (ip, domain, hash, cve)")
    ioc_value: str = Field(..., description="IOC value")
    risk_level: str = Field(..., description="Risk level (High, Medium, Low)")
    evidence_text: str = Field(..., description="Supporting evidence text")
    source_feed: str = Field(..., description="Source threat intelligence feed")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score")


class VideoAlert(BaseAlert):
    """Video surveillance alert model"""
    event_type: str = Field(..., description="Event type (loitering, zone_violation, abandoned_object)")
    bounding_box: List[int] = Field(..., description="Bounding box coordinates [x, y, w, h]")
    track_id: int = Field(..., description="Object tracking ID")
    snapshot_path: str = Field(..., description="Path to snapshot image")
    video_timestamp: float = Field(..., description="Timestamp in video")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class AnomalyAlert(BaseAlert):
    """Border anomaly detection alert model"""
    anomaly_type: str = Field(..., description="Type of anomaly detected")
    severity_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly severity score")
    trajectory_points: List[List[int]] = Field(..., description="Trajectory coordinates")
    feature_vector: List[float] = Field(..., description="Computed feature vector")
    supporting_frames: List[str] = Field(..., description="Paths to supporting frame images")


class AlertCreate(BaseModel):
    """Model for creating new alerts"""
    alert_type: AlertType
    data: Dict = Field(..., description="Alert-specific data")


class AlertUpdate(BaseModel):
    """Model for updating alert status"""
    status: AlertStatus
    notes: Optional[str] = None


class AlertQuery(BaseModel):
    """Model for alert query parameters"""
    alert_type: Optional[AlertType] = None
    status: Optional[AlertStatus] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class SystemMetrics(BaseModel):
    """System metrics model"""
    pipeline_name: str = Field(..., description="Pipeline name")
    processing_rate: float = Field(..., description="Processing rate (items/second)")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy score")
    last_update: datetime = Field(..., description="Last update timestamp")
    status: str = Field(..., description="Pipeline status")
    error_count: int = Field(default=0, description="Error count in last hour")


class AlertResponse(BaseModel):
    """Response model for alert operations"""
    success: bool
    message: str
    data: Optional[Union[BaseAlert, List[BaseAlert]]] = None


class PaginatedAlertResponse(BaseModel):
    """Paginated response for alert queries"""
    alerts: List[BaseAlert]
    total: int
    limit: int
    offset: int
    has_more: bool