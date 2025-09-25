#!/usr/bin/env python3
"""
Startup script for HifazatAI Alert Broker API.
Initializes database and starts the FastAPI server.
"""

import uvicorn
from models.database import create_tables
from utils.logging import get_logger

def main():
    """Main startup function"""
    logger = get_logger(__name__)
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        create_tables()
        logger.info("Database initialized successfully")
        
        # Initialize system metrics if needed
        try:
            from init_system_metrics import init_system_metrics
            init_system_metrics()
        except Exception as e:
            logger.warning(f"Could not initialize system metrics: {str(e)}")
            # Continue anyway as this is not critical
        
        # Start API server
        logger.info("Starting HifazatAI Alert Broker API...")
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        raise

if __name__ == "__main__":
    main()