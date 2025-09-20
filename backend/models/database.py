"""
Database models and connection management for HifazatAI.
Uses SQLAlchemy with SQLite for data persistence.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hifazat.db")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AlertDB(Base):
    """SQLAlchemy model for alerts table"""
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)
    pipeline = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    data = Column(JSON, nullable=False)
    status = Column(String, default="active", index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class AlertMediaDB(Base):
    """SQLAlchemy model for alert media table"""
    __tablename__ = "alert_media"

    id = Column(String, primary_key=True, index=True)
    alert_id = Column(String, nullable=False, index=True)
    media_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())


class SystemMetricsDB(Base):
    """SQLAlchemy model for system metrics table"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_name = Column(String, nullable=False, index=True)
    processing_rate = Column(Float, nullable=False)
    accuracy_score = Column(Float, nullable=True)
    status = Column(String, nullable=False)
    error_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=func.now(), index=True)


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        create_tables()
    
    def create_alert(self, db: Session, alert_data: Dict[str, Any]) -> AlertDB:
        """Create a new alert in the database"""
        db_alert = AlertDB(**alert_data)
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        return db_alert
    
    def get_alert(self, db: Session, alert_id: str) -> Optional[AlertDB]:
        """Get alert by ID"""
        return db.query(AlertDB).filter(AlertDB.id == alert_id).first()
    
    def get_alerts(
        self, 
        db: Session, 
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AlertDB]:
        """Get alerts with filtering and pagination"""
        query = db.query(AlertDB)
        
        if alert_type:
            query = query.filter(AlertDB.type == alert_type)
        if status:
            query = query.filter(AlertDB.status == status)
        if start_time:
            query = query.filter(AlertDB.timestamp >= start_time)
        if end_time:
            query = query.filter(AlertDB.timestamp <= end_time)
        
        return query.order_by(AlertDB.timestamp.desc()).offset(offset).limit(limit).all()
    
    def count_alerts(
        self,
        db: Session,
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """Count alerts with filtering"""
        query = db.query(AlertDB)
        
        if alert_type:
            query = query.filter(AlertDB.type == alert_type)
        if status:
            query = query.filter(AlertDB.status == status)
        if start_time:
            query = query.filter(AlertDB.timestamp >= start_time)
        if end_time:
            query = query.filter(AlertDB.timestamp <= end_time)
        
        return query.count()
    
    def update_alert_status(self, db: Session, alert_id: str, status: str, notes: Optional[str] = None) -> Optional[AlertDB]:
        """Update alert status"""
        db_alert = self.get_alert(db, alert_id)
        if db_alert:
            db_alert.status = status
            if notes:
                # Add notes to the data field
                if isinstance(db_alert.data, dict):
                    db_alert.data["notes"] = notes
                else:
                    data = json.loads(db_alert.data) if isinstance(db_alert.data, str) else {}
                    data["notes"] = notes
                    db_alert.data = data
            db.commit()
            db.refresh(db_alert)
        return db_alert
    
    def create_alert_media(self, db: Session, media_data: Dict[str, Any]) -> AlertMediaDB:
        """Create alert media record"""
        db_media = AlertMediaDB(**media_data)
        db.add(db_media)
        db.commit()
        db.refresh(db_media)
        return db_media
    
    def get_alert_media(self, db: Session, alert_id: str) -> List[AlertMediaDB]:
        """Get media files for an alert"""
        return db.query(AlertMediaDB).filter(AlertMediaDB.alert_id == alert_id).all()
    
    def create_system_metrics(self, db: Session, metrics_data: Dict[str, Any]) -> SystemMetricsDB:
        """Create system metrics record"""
        db_metrics = SystemMetricsDB(**metrics_data)
        db.add(db_metrics)
        db.commit()
        db.refresh(db_metrics)
        return db_metrics
    
    def get_latest_metrics(self, db: Session, pipeline_name: str) -> Optional[SystemMetricsDB]:
        """Get latest metrics for a pipeline"""
        return db.query(SystemMetricsDB).filter(
            SystemMetricsDB.pipeline_name == pipeline_name
        ).order_by(SystemMetricsDB.timestamp.desc()).first()


# Global database manager instance
db_manager = DatabaseManager()