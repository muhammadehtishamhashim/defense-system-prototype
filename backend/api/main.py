"""
HifazatAI Alert Broker API
FastAPI application for managing security alerts from multiple AI pipelines.
"""

import uuid
import json
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, status
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting HifazatAI Alert Broker API")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down HifazatAI Alert Broker API")


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
@app.get("/events")
async def stream_events():
    """Server-Sent Events endpoint for real-time updates"""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events"""
        try:
            # Send initial connection confirmation
            initial_data = {
                "type": "connection",
                "data": {"status": "connected", "timestamp": datetime.now().isoformat()},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # Send heartbeat every 60 seconds (reduced frequency)
            while True:
                try:
                    heartbeat_data = {
                        "type": "heartbeat",
                        "data": {"timestamp": datetime.now().isoformat()},
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(heartbeat_data)}\n\n"
                    
                    # Wait before next heartbeat
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    logger.info("SSE connection cancelled by client")
                    break
                except Exception as e:
                    logger.error(f"SSE error in heartbeat: {str(e)}")
                    # Send error message before closing
                    error_data = {
                        "type": "error",
                        "data": {"message": "Connection error", "timestamp": datetime.now().isoformat()},
                        "timestamp": datetime.now().isoformat()
                    }
                    try:
                        yield f"data: {json.dumps(error_data)}\n\n"
                    except:
                        pass
                    break
                    
        except Exception as e:
            logger.error(f"SSE generator error: {str(e)}")
            return
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)