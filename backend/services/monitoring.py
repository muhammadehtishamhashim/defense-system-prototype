"""
Video Processing Monitoring Service
Monitors performance metrics and handles errors for video streaming and analysis.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """System health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for video processing"""
    session_id: str
    frames_per_second: float = 0.0
    processing_latency: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: datetime = field(default_factory=datetime.now)


class VideoProcessingMonitor:
    """Monitors video processing performance and handles errors"""
    
    def __init__(self):
        """Initialize monitoring service"""
        self.session_metrics: Dict[str, PerformanceMetrics] = {}
        self.global_metrics = PerformanceMetrics(session_id="global")
        self.error_threshold = 10  # Max errors before marking as unhealthy
        self.latency_threshold = 5.0  # Max processing latency in seconds
        self.monitoring_task = None
        
        logger.info("Video processing monitor initialized")
    
    async def start_monitoring(self):
        """Start the monitoring service"""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Video processing monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Video processing monitoring stopped")
    
    def register_session(self, session_id: str):
        """
        Register a new video processing session for monitoring
        
        Args:
            session_id: ID of the session to monitor
        """
        self.session_metrics[session_id] = PerformanceMetrics(
            session_id=session_id,
            timestamp=datetime.now()
        )
        logger.info(f"Registered monitoring for session {session_id}")
    
    def unregister_session(self, session_id: str):
        """
        Unregister a video processing session
        
        Args:
            session_id: ID of the session to unregister
        """
        if session_id in self.session_metrics:
            del self.session_metrics[session_id]
            logger.info(f"Unregistered monitoring for session {session_id}")
    
    def record_frame_processed(self, session_id: str, processing_time: float):
        """
        Record a processed frame with timing information
        
        Args:
            session_id: ID of the processing session
            processing_time: Time taken to process the frame in seconds
        """
        if session_id not in self.session_metrics:
            self.register_session(session_id)
        
        metrics = self.session_metrics[session_id]
        
        # Update processing latency (moving average)
        if metrics.processing_latency == 0:
            metrics.processing_latency = processing_time
        else:
            metrics.processing_latency = (metrics.processing_latency * 0.9) + (processing_time * 0.1)
        
        # Update FPS calculation
        current_time = time.time()
        if not hasattr(metrics, '_last_frame_time'):
            metrics._last_frame_time = current_time
            metrics._frame_count = 1
        else:
            time_diff = current_time - metrics._last_frame_time
            if time_diff >= 1.0:  # Update FPS every second
                metrics.frames_per_second = metrics._frame_count / time_diff
                metrics._last_frame_time = current_time
                metrics._frame_count = 0
            else:
                metrics._frame_count += 1
        
        # Update uptime
        metrics.uptime_seconds = (datetime.now() - metrics.timestamp).total_seconds()
        
        # Check health status
        self._update_health_status(metrics)
    
    def record_error(self, session_id: str, error_message: str):
        """
        Record an error for a processing session
        
        Args:
            session_id: ID of the processing session
            error_message: Error message to record
        """
        if session_id not in self.session_metrics:
            self.register_session(session_id)
        
        metrics = self.session_metrics[session_id]
        metrics.error_count += 1
        metrics.last_error = error_message
        
        # Update health status
        self._update_health_status(metrics)
        
        logger.warning(f"Error recorded for session {session_id}: {error_message}")
    
    def get_session_metrics(self, session_id: str) -> Optional[PerformanceMetrics]:
        """
        Get performance metrics for a specific session
        
        Args:
            session_id: ID of the session
            
        Returns:
            PerformanceMetrics object or None if session not found
        """
        return self.session_metrics.get(session_id)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """
        Get performance metrics for all sessions
        
        Returns:
            Dictionary of session metrics
        """
        return self.session_metrics.copy()
    
    def get_global_metrics(self) -> PerformanceMetrics:
        """
        Get aggregated global metrics
        
        Returns:
            Global PerformanceMetrics object
        """
        if not self.session_metrics:
            return self.global_metrics
        
        # Aggregate metrics from all sessions
        total_fps = sum(m.frames_per_second for m in self.session_metrics.values())
        avg_latency = sum(m.processing_latency for m in self.session_metrics.values()) / len(self.session_metrics)
        total_errors = sum(m.error_count for m in self.session_metrics.values())
        
        # Determine overall health status
        statuses = [m.status for m in self.session_metrics.values()]
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.ERROR in statuses:
            overall_status = HealthStatus.ERROR
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        self.global_metrics.frames_per_second = total_fps
        self.global_metrics.processing_latency = avg_latency
        self.global_metrics.error_count = total_errors
        self.global_metrics.status = overall_status
        self.global_metrics.timestamp = datetime.now()
        
        return self.global_metrics
    
    def _update_health_status(self, metrics: PerformanceMetrics):
        """
        Update health status based on current metrics
        
        Args:
            metrics: PerformanceMetrics object to update
        """
        if metrics.error_count >= self.error_threshold:
            metrics.status = HealthStatus.CRITICAL
        elif metrics.processing_latency > self.latency_threshold:
            metrics.status = HealthStatus.ERROR
        elif metrics.error_count > 0 or metrics.processing_latency > self.latency_threshold * 0.7:
            metrics.status = HealthStatus.WARNING
        else:
            metrics.status = HealthStatus.HEALTHY
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while True:
                try:
                    # Update global metrics
                    self.get_global_metrics()
                    
                    # Clean up old sessions (inactive for more than 1 hour)
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    inactive_sessions = [
                        session_id for session_id, metrics in self.session_metrics.items()
                        if metrics.timestamp < cutoff_time
                    ]
                    
                    for session_id in inactive_sessions:
                        self.unregister_session(session_id)
                    
                    # Log system health periodically
                    global_metrics = self.get_global_metrics()
                    if global_metrics.status != HealthStatus.HEALTHY:
                        logger.warning(f"System health: {global_metrics.status.value} - "
                                     f"FPS: {global_metrics.frames_per_second:.1f}, "
                                     f"Latency: {global_metrics.processing_latency:.2f}s, "
                                     f"Errors: {global_metrics.error_count}")
                    
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(5)  # Short delay before retry
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")


# Global monitor instance
video_monitor = VideoProcessingMonitor()


class ErrorHandler:
    """Handles errors and implements retry logic for video processing"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize error handler
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts: Dict[str, int] = {}
        
    async def handle_with_retry(self, operation, *args, **kwargs):
        """
        Execute operation with retry logic
        
        Args:
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        operation_name = operation.__name__ if hasattr(operation, '__name__') else str(operation)
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs) if asyncio.iscoroutinefunction(operation) else operation(*args, **kwargs)
                
                # Reset error count on success
                if operation_name in self.error_counts:
                    del self.error_counts[operation_name]
                
                return result
                
            except Exception as e:
                self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
                
                if attempt < self.max_retries:
                    logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Operation {operation_name} failed after {self.max_retries + 1} attempts: {str(e)}")
                    raise
    
    def get_error_stats(self) -> Dict[str, int]:
        """
        Get error statistics
        
        Returns:
            Dictionary of operation names and their error counts
        """
        return self.error_counts.copy()


# Global error handler instance
error_handler = ErrorHandler()