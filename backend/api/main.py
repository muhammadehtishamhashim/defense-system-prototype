"""
HifazatAI Alert Broker API
FastAPI application for managing security alerts from multiple AI pipelines.
"""

import uuid
import json
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json
from typing import AsyncGenerator

from models.alerts import (
    ThreatAlert, VideoAlert, AnomalyAlert, BaseAlert,
    AlertCreate, AlertUpdate, AlertQuery, AlertResponse,
    PaginatedAlertResponse, SystemMetrics, AlertType, AlertStatus
)
from models.database import get_db, db_manager, AlertDB
from utils.logging import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="HifazatAI Alert Broker API",
    description="Centralized API for managing security alerts from AI pipelines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add middleware to handle OPTIONS requests
@app.middleware("http")
async def handle_options(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    response = await call_next(request)
    return response

# Initialize logger
logger = get_logger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting HifazatAI Alert Broker API")
    
    # Start SSE broadcast service
    await sse_manager.start_broadcast_service()
    
    # Start monitoring service
    await video_monitor.start_monitoring()
    
    # Set up mock alert generator callback
    mock_alert_generator.set_alert_callback(sse_manager.broadcast_alert)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down HifazatAI Alert Broker API")
    
    # Stop SSE broadcast service
    await sse_manager.stop_broadcast_service()
    
    # Stop monitoring service
    await video_monitor.stop_monitoring()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "HifazatAI Alert Broker API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }


# Alert Management Endpoints

@app.post("/alerts/threat", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_threat_alert(
    alert: ThreatAlert,
    db: Session = Depends(get_db)
):
    """Create a new threat intelligence alert"""
    try:
        # Generate unique ID if not provided
        if not alert.id:
            alert.id = str(uuid.uuid4())
        
        # Prepare alert data for database
        alert_data = {
            "id": alert.id,
            "pipeline": "threat_intelligence",
            "type": "threat",
            "timestamp": alert.timestamp,
            "confidence": alert.confidence,
            "status": alert.status.value,
            "data": alert.model_dump(mode='json')  # Use model_dump with json mode for proper serialization
        }
        
        # Create alert in database
        db_alert = db_manager.create_alert(db, alert_data)
        logger.info(f"Created threat alert: {alert.id}")
        
        return AlertResponse(
            success=True,
            message="Threat alert created successfully",
            data=alert
        )
    
    except Exception as e:
        logger.error(f"Error creating threat alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create threat alert: {str(e)}"
        )


@app.post("/alerts/video", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_video_alert(
    alert: VideoAlert,
    db: Session = Depends(get_db)
):
    """Create a new video surveillance alert"""
    try:
        # Generate unique ID if not provided
        if not alert.id:
            alert.id = str(uuid.uuid4())
        
        # Prepare alert data for database
        alert_data = {
            "id": alert.id,
            "pipeline": "video_surveillance",
            "type": "video",
            "timestamp": alert.timestamp,
            "confidence": alert.confidence,
            "status": alert.status.value,
            "data": alert.model_dump(mode='json')  # Use model_dump with json mode for proper serialization
        }
        
        # Create alert in database
        db_alert = db_manager.create_alert(db, alert_data)
        logger.info(f"Created video alert: {alert.id}")
        
        return AlertResponse(
            success=True,
            message="Video alert created successfully",
            data=alert
        )
    
    except Exception as e:
        logger.error(f"Error creating video alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create video alert: {str(e)}"
        )


@app.post("/alerts/anomaly", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_anomaly_alert(
    alert: AnomalyAlert,
    db: Session = Depends(get_db)
):
    """Create a new border anomaly alert"""
    try:
        # Generate unique ID if not provided
        if not alert.id:
            alert.id = str(uuid.uuid4())
        
        # Prepare alert data for database
        alert_data = {
            "id": alert.id,
            "pipeline": "border_anomaly",
            "type": "anomaly",
            "timestamp": alert.timestamp,
            "confidence": alert.confidence,
            "status": alert.status.value,
            "data": alert.model_dump(mode='json')  # Use model_dump with json mode for proper serialization
        }
        
        # Create alert in database
        db_alert = db_manager.create_alert(db, alert_data)
        logger.info(f"Created anomaly alert: {alert.id}")
        
        return AlertResponse(
            success=True,
            message="Anomaly alert created successfully",
            data=alert
        )
    
    except Exception as e:
        logger.error(f"Error creating anomaly alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create anomaly alert: {str(e)}"
        )


@app.get("/alerts", response_model=PaginatedAlertResponse)
async def get_alerts(
    alert_type: Optional[AlertType] = Query(None, description="Filter by alert type"),
    status: Optional[AlertStatus] = Query(None, description="Filter by alert status"),
    start_time: Optional[datetime] = Query(None, description="Filter by start time"),
    end_time: Optional[datetime] = Query(None, description="Filter by end time"),
    limit: int = Query(100, ge=1, le=1000, description="Number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip"),
    db: Session = Depends(get_db)
):
    """Get alerts with filtering and pagination"""
    try:
        # Convert enum values to strings
        type_filter = alert_type.value if alert_type else None
        status_filter = status.value if status else None
        
        # Get alerts from database
        alerts = db_manager.get_alerts(
            db=db,
            alert_type=type_filter,
            status=status_filter,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
        
        # Get total count
        total = db_manager.count_alerts(
            db=db,
            alert_type=type_filter,
            status=status_filter,
            start_time=start_time,
            end_time=end_time
        )
        
        # Convert database objects to Pydantic models
        alert_list = []
        for db_alert in alerts:
            alert_data = db_alert.data
            if isinstance(alert_data, str):
                alert_data = json.loads(alert_data)
            
            # Create appropriate alert model based on type
            if db_alert.type == "threat":
                alert_obj = ThreatAlert(**alert_data)
            elif db_alert.type == "video":
                alert_obj = VideoAlert(**alert_data)
            elif db_alert.type == "anomaly":
                alert_obj = AnomalyAlert(**alert_data)
            else:
                # Fallback to base alert
                alert_obj = BaseAlert(**{
                    "id": db_alert.id,
                    "timestamp": db_alert.timestamp,
                    "confidence": db_alert.confidence,
                    "source_pipeline": db_alert.pipeline,
                    "status": db_alert.status
                })
            
            alert_list.append(alert_obj)
        
        return PaginatedAlertResponse(
            alerts=alert_list,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total
        )
    
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@app.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific alert by ID"""
    try:
        db_alert = db_manager.get_alert(db, alert_id)
        if not db_alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found"
            )
        
        # Convert to appropriate Pydantic model
        alert_data = db_alert.data
        if isinstance(alert_data, str):
            alert_data = json.loads(alert_data)
        
        if db_alert.type == "threat":
            alert_obj = ThreatAlert(**alert_data)
        elif db_alert.type == "video":
            alert_obj = VideoAlert(**alert_data)
        elif db_alert.type == "anomaly":
            alert_obj = AnomalyAlert(**alert_data)
        else:
            alert_obj = BaseAlert(**{
                "id": db_alert.id,
                "timestamp": db_alert.timestamp,
                "confidence": db_alert.confidence,
                "source_pipeline": db_alert.pipeline,
                "status": db_alert.status
            })
        
        return AlertResponse(
            success=True,
            message="Alert retrieved successfully",
            data=alert_obj
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alert: {str(e)}"
        )


@app.put("/alerts/{alert_id}/status", response_model=AlertResponse)
async def update_alert_status(
    alert_id: str,
    update: AlertUpdate,
    db: Session = Depends(get_db)
):
    """Update alert status"""
    try:
        db_alert = db_manager.update_alert_status(
            db=db,
            alert_id=alert_id,
            status=update.status.value,
            notes=update.notes
        )
        
        if not db_alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert {alert_id} not found"
            )
        
        logger.info(f"Updated alert {alert_id} status to {update.status.value}")
        
        return AlertResponse(
            success=True,
            message=f"Alert status updated to {update.status.value}",
            data=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update alert: {str(e)}"
        )


# System Metrics Endpoints

@app.post("/metrics", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_system_metrics(
    metrics: SystemMetrics,
    db: Session = Depends(get_db)
):
    """Create system metrics record"""
    try:
        metrics_data = {
            "pipeline_name": metrics.pipeline_name,
            "processing_rate": metrics.processing_rate,
            "accuracy_score": metrics.accuracy_score,
            "status": metrics.status,
            "error_count": metrics.error_count
        }
        
        db_metrics = db_manager.create_system_metrics(db, metrics_data)
        logger.info(f"Created metrics for pipeline: {metrics.pipeline_name}")
        
        return {
            "success": True,
            "message": "System metrics created successfully"
        }
    
    except Exception as e:
        logger.error(f"Error creating system metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create system metrics: {str(e)}"
        )


@app.get("/metrics/{pipeline_name}", response_model=SystemMetrics)
async def get_pipeline_metrics(
    pipeline_name: str,
    db: Session = Depends(get_db)
):
    """Get latest metrics for a pipeline"""
    try:
        db_metrics = db_manager.get_latest_metrics(db, pipeline_name)
        if not db_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for pipeline: {pipeline_name}"
            )
        
        return SystemMetrics(
            pipeline_name=db_metrics.pipeline_name,
            processing_rate=db_metrics.processing_rate,
            accuracy_score=db_metrics.accuracy_score,
            last_update=db_metrics.timestamp,
            status=db_metrics.status,
            error_count=db_metrics.error_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics for {pipeline_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


# Server-Sent Events endpoint
from services.sse_manager import sse_manager
from services.mock_alert_generator import mock_alert_generator

@app.get("/events")
async def stream_events():
    """Server-Sent Events endpoint for real-time updates"""
    return await sse_manager.create_sse_response()


# System metrics endpoint
@app.get("/system/metrics", response_model=dict)
async def get_system_metrics(db: Session = Depends(get_db)):
    """Get current system metrics"""
    try:
        # Get latest metrics for each pipeline
        pipelines = ["threat_intelligence", "video_surveillance", "border_anomaly"]
        metrics_data = {}
        
        for pipeline in pipelines:
            db_metrics = db_manager.get_latest_metrics(db, pipeline)
            if db_metrics:
                metrics_data[pipeline] = {
                    "processing_rate": db_metrics.processing_rate,
                    "accuracy_score": db_metrics.accuracy_score,
                    "status": db_metrics.status,
                    "error_count": db_metrics.error_count,
                    "last_update": db_metrics.timestamp.isoformat()
                }
            else:
                # Provide default values if no metrics exist
                metrics_data[pipeline] = {
                    "processing_rate": 0.0,
                    "accuracy_score": 0.0,
                    "status": "offline",
                    "error_count": 0,
                    "last_update": datetime.now().isoformat()
                }
        
        # Calculate system-wide metrics
        total_alerts = db_manager.count_alerts(db)
        recent_alerts = db_manager.count_alerts(
            db, 
            start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
        
        return {
            "system_status": "healthy",
            "total_alerts": total_alerts,
            "alerts_today": recent_alerts,
            "pipelines": metrics_data,
            "uptime": "99.8%",  # Mock uptime
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving system metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )


# Video Streaming Endpoints

from services.video_streaming import video_streaming_service, VideoInfo
from services.video_analysis import video_analysis_coordinator, AnalysisSession, AnalysisStatus
from services.mock_alert_generator import mock_alert_generator
from services.monitoring import video_monitor, PerformanceMetrics
from pydantic import BaseModel

class StartAnalysisRequest(BaseModel):
    """Request model for starting video analysis"""
    video_filename: str
    mock_alerts: bool = True
    alert_interval: int = 30

class StartAnalysisResponse(BaseModel):
    """Response model for starting video analysis"""
    session_id: str
    message: str
    success: bool

@app.get("/api/videos", response_model=List[VideoInfo])
async def list_videos():
    """Get list of available video files"""
    try:
        videos = video_streaming_service.list_available_videos()
        logger.info(f"Listed {len(videos)} available videos")
        return videos
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list videos: {str(e)}"
        )


@app.get("/api/videos/{filename}")
async def stream_video(filename: str, request: Request):
    """Stream video file with range request support"""
    try:
        return await video_streaming_service.stream_video(filename, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream video: {str(e)}"
        )


@app.get("/api/videos/{filename}/info", response_model=VideoInfo)
async def get_video_info(filename: str):
    """Get metadata for a specific video file"""
    try:
        video_info = video_streaming_service.get_video_info(filename)
        logger.info(f"Retrieved info for video: {filename}")
        return video_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video info for {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video info: {str(e)}"
        )


# Video Analysis Endpoints

@app.post("/api/analysis/start", response_model=StartAnalysisResponse)
async def start_video_analysis(request: StartAnalysisRequest):
    """Start video analysis session"""
    try:
        # Start video analysis
        session_id = await video_analysis_coordinator.start_analysis(request.video_filename)
        
        # Start mock alert generation if requested
        if request.mock_alerts:
            await mock_alert_generator.start_mock_alerts(session_id, request.alert_interval)
        
        logger.info(f"Started video analysis session {session_id} for {request.video_filename}")
        
        return StartAnalysisResponse(
            session_id=session_id,
            message=f"Video analysis started for {request.video_filename}",
            success=True
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error starting video analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start video analysis: {str(e)}"
        )


@app.post("/api/analysis/stop")
async def stop_video_analysis(session_id: str):
    """Stop video analysis session"""
    try:
        # Stop video analysis
        analysis_stopped = await video_analysis_coordinator.stop_analysis(session_id)
        
        # Stop mock alert generation
        alerts_stopped = await mock_alert_generator.stop_mock_alerts(session_id)
        
        if not analysis_stopped:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis session {session_id} not found"
            )
        
        logger.info(f"Stopped video analysis session {session_id}")
        
        return {
            "success": True,
            "message": f"Video analysis session {session_id} stopped",
            "analysis_stopped": analysis_stopped,
            "alerts_stopped": alerts_stopped
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping video analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop video analysis: {str(e)}"
        )


@app.get("/api/analysis/status/{session_id}", response_model=AnalysisSession)
async def get_analysis_status(session_id: str):
    """Get status of video analysis session"""
    try:
        session_status = video_analysis_coordinator.get_analysis_status(session_id)
        
        if not session_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis session {session_id} not found"
            )
        
        return session_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status for {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analysis status: {str(e)}"
        )


@app.get("/api/analysis/sessions", response_model=List[AnalysisSession])
async def list_analysis_sessions():
    """Get list of all active analysis sessions"""
    try:
        sessions = video_analysis_coordinator.list_active_sessions()
        return sessions
    except Exception as e:
        logger.error(f"Error listing analysis sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list analysis sessions: {str(e)}"
        )


# Monitoring Endpoints

@app.get("/api/monitoring/metrics/{session_id}")
async def get_session_metrics(session_id: str):
    """Get performance metrics for a specific analysis session"""
    try:
        metrics = video_monitor.get_session_metrics(session_id)
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for session {session_id}"
            )
        
        return {
            "session_id": metrics.session_id,
            "frames_per_second": metrics.frames_per_second,
            "processing_latency": metrics.processing_latency,
            "memory_usage_mb": metrics.memory_usage_mb,
            "cpu_usage_percent": metrics.cpu_usage_percent,
            "error_count": metrics.error_count,
            "last_error": metrics.last_error,
            "uptime_seconds": metrics.uptime_seconds,
            "status": metrics.status.value,
            "timestamp": metrics.timestamp.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session metrics: {str(e)}"
        )


@app.get("/api/monitoring/metrics")
async def get_all_metrics():
    """Get performance metrics for all analysis sessions"""
    try:
        all_metrics = video_monitor.get_all_metrics()
        global_metrics = video_monitor.get_global_metrics()
        
        session_metrics = {}
        for session_id, metrics in all_metrics.items():
            session_metrics[session_id] = {
                "session_id": metrics.session_id,
                "frames_per_second": metrics.frames_per_second,
                "processing_latency": metrics.processing_latency,
                "error_count": metrics.error_count,
                "status": metrics.status.value,
                "uptime_seconds": metrics.uptime_seconds,
                "timestamp": metrics.timestamp.isoformat()
            }
        
        return {
            "global_metrics": {
                "frames_per_second": global_metrics.frames_per_second,
                "processing_latency": global_metrics.processing_latency,
                "error_count": global_metrics.error_count,
                "status": global_metrics.status.value,
                "timestamp": global_metrics.timestamp.isoformat()
            },
            "session_metrics": session_metrics,
            "active_sessions": len(session_metrics)
        }
    except Exception as e:
        logger.error(f"Error getting all metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


# Test Endpoints for Debugging

@app.post("/api/test/alert")
async def test_alert():
    """Generate a test alert to verify SSE functionality"""
    try:
        # Create a test alert
        test_alert = VideoAlert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            confidence=0.95,
            source_pipeline="video_surveillance",
            status=AlertStatus.ACTIVE,
            event_type="test_alert",
            bounding_box=[100, 100, 200, 200],
            track_id=999,
            snapshot_path="test_snapshot.jpg",
            video_timestamp=datetime.now().isoformat(),
            metadata={
                'test': True,
                'description': 'This is a test alert to verify SSE functionality'
            }
        )
        
        # Send through SSE
        await sse_manager.broadcast_alert(test_alert)
        
        logger.info(f"Generated test alert: {test_alert.id}")
        
        return {
            "success": True,
            "message": "Test alert generated and sent via SSE",
            "alert_id": test_alert.id,
            "connections": len(sse_manager.connections)
        }
        
    except Exception as e:
        logger.error(f"Error generating test alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate test alert: {str(e)}"
        )


@app.get("/api/test/sse-status")
async def get_sse_status():
    """Get SSE connection status"""
    return {
        "connections": len(sse_manager.connections),
        "queue_size": sse_manager.alert_queue.qsize(),
        "broadcast_task_running": sse_manager.broadcast_task is not None and not sse_manager.broadcast_task.done()
    }


# Snapshot serving endpoint
from fastapi.staticfiles import StaticFiles
import os

# Create snapshots directory if it doesn't exist
os.makedirs("media/snapshots", exist_ok=True)

# Mount static files for snapshots
app.mount("/snapshots", StaticFiles(directory="media/snapshots"), name="snapshots")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)