#!/usr/bin/env python3
"""
Initialize system metrics for HifazatAI.
Creates initial metrics data for all pipelines.
"""

from datetime import datetime
from models.database import SessionLocal, db_manager
from utils.logging import get_logger

def init_system_metrics():
    """Initialize system metrics for all pipelines"""
    logger = get_logger(__name__)
    
    try:
        db = SessionLocal()
        
        # Define pipelines and their initial metrics
        pipelines = [
            {
                "pipeline_name": "threat_intelligence",
                "processing_rate": 12.5,
                "accuracy_score": 85.2,
                "status": "online",
                "error_count": 0
            },
            {
                "pipeline_name": "video_surveillance", 
                "processing_rate": 8.3,
                "accuracy_score": 91.7,
                "status": "online",
                "error_count": 1
            },
            {
                "pipeline_name": "border_anomaly",
                "processing_rate": 15.8,
                "accuracy_score": 78.9,
                "status": "online", 
                "error_count": 0
            }
        ]
        
        # Create metrics for each pipeline
        for pipeline_data in pipelines:
            db_manager.create_system_metrics(db, pipeline_data)
            logger.info(f"Created metrics for pipeline: {pipeline_data['pipeline_name']}")
        
        db.close()
        logger.info("System metrics initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system metrics: {str(e)}")
        raise

if __name__ == "__main__":
    init_system_metrics()