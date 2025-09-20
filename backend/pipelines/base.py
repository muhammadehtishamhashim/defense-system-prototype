"""
Base pipeline class for HifazatAI AI pipelines.
Provides common functionality for threat intelligence, video surveillance, and border anomaly detection.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

import requests
from utils.logging import get_pipeline_logger, error_tracker
from utils.config import ConfigManager
from models.alerts import BaseAlert, SystemMetrics


class PipelineStatus(str, Enum):
    """Pipeline status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    alerts_generated: int = 0
    processing_time_total: float = 0.0
    last_processing_time: float = 0.0
    start_time: datetime = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.processed_items == 0:
            return 0.0
        return self.successful_items / self.processed_items
    
    @property
    def processing_rate(self) -> float:
        """Calculate processing rate (items per second)"""
        if self.processing_time_total == 0:
            return 0.0
        return self.processed_items / self.processing_time_total
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per item"""
        if self.processed_items == 0:
            return 0.0
        return self.processing_time_total / self.processed_items


class BasePipeline(ABC):
    """
    Abstract base class for all AI pipelines.
    Provides common functionality for configuration, logging, metrics, and alert management.
    """
    
    def __init__(self, pipeline_name: str, config_path: Optional[str] = None):
        """
        Initialize base pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            config_path: Optional path to configuration file
        """
        self.pipeline_name = pipeline_name
        self.logger = get_pipeline_logger(pipeline_name)
        self.config_manager = ConfigManager(config_path)
        self.metrics = PipelineMetrics()
        self.status = PipelineStatus.STOPPED
        self._stop_event = asyncio.Event()
        self._error_callbacks: List[Callable] = []
        
        # API configuration
        self.api_base_url = self.config_manager.get("api_base_url", "http://localhost:8000")
        self.confidence_threshold = self.config_manager.get("confidence_threshold", 0.5)
        self.processing_interval = self.config_manager.get("processing_interval", 1.0)
        
        self.logger.info(f"Initialized {pipeline_name} pipeline")
    
    @abstractmethod
    async def process_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data and return results.
        Must be implemented by subclasses.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing results dictionary
        """
        pass
    
    @abstractmethod
    def create_alert(self, processing_result: Dict[str, Any]) -> Optional[BaseAlert]:
        """
        Create alert from processing results.
        Must be implemented by subclasses.
        
        Args:
            processing_result: Results from process_input
            
        Returns:
            Alert object or None if no alert should be generated
        """
        pass
    
    async def start(self):
        """Start the pipeline"""
        try:
            self.logger.info(f"Starting {self.pipeline_name} pipeline")
            self.status = PipelineStatus.STARTING
            self.metrics.start_time = datetime.now()
            self._stop_event.clear()
            
            # Initialize pipeline-specific components
            await self.initialize()
            
            self.status = PipelineStatus.RUNNING
            self.logger.info(f"{self.pipeline_name} pipeline started successfully")
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            self.status = PipelineStatus.ERROR
            self.logger.error(f"Error starting {self.pipeline_name} pipeline: {str(e)}")
            error_tracker.track_error(self.pipeline_name, e)
            await self._handle_error(e)
            raise
    
    async def stop(self):
        """Stop the pipeline"""
        self.logger.info(f"Stopping {self.pipeline_name} pipeline")
        self.status = PipelineStatus.STOPPED
        self._stop_event.set()
        
        # Cleanup pipeline-specific resources
        await self.cleanup()
        
        self.logger.info(f"{self.pipeline_name} pipeline stopped")
    
    async def pause(self):
        """Pause the pipeline"""
        self.logger.info(f"Pausing {self.pipeline_name} pipeline")
        self.status = PipelineStatus.PAUSED
    
    async def resume(self):
        """Resume the pipeline"""
        self.logger.info(f"Resuming {self.pipeline_name} pipeline")
        self.status = PipelineStatus.RUNNING
    
    async def initialize(self):
        """
        Initialize pipeline-specific components.
        Override in subclasses if needed.
        """
        pass
    
    async def cleanup(self):
        """
        Cleanup pipeline-specific resources.
        Override in subclasses if needed.
        """
        pass
    
    async def _main_loop(self):
        """Main processing loop"""
        while not self._stop_event.is_set():
            try:
                if self.status == PipelineStatus.RUNNING:
                    # Get input data
                    input_data = await self.get_input_data()
                    
                    if input_data is not None:
                        await self._process_and_handle_result(input_data)
                    
                    # Send metrics periodically
                    await self._send_metrics()
                
                # Wait for next processing cycle
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                self.metrics.failed_items += 1
                self.logger.error(f"Error in main loop: {str(e)}")
                error_tracker.track_error(self.pipeline_name, e)
                await self._handle_error(e)
    
    async def _process_and_handle_result(self, input_data: Any):
        """Process input data and handle the result"""
        start_time = time.time()
        
        try:
            # Process the input
            result = await self.process_input(input_data)
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics.processed_items += 1
            self.metrics.successful_items += 1
            self.metrics.processing_time_total += processing_time
            self.metrics.last_processing_time = processing_time
            
            # Create alert if needed
            alert = self.create_alert(result)
            if alert and alert.confidence >= self.confidence_threshold:
                await self.send_alert(alert)
                self.metrics.alerts_generated += 1
                self.logger.info(f"Generated alert: {alert.id}")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.processed_items += 1
            self.metrics.failed_items += 1
            self.metrics.processing_time_total += processing_time
            
            self.logger.error(f"Error processing input: {str(e)}")
            error_tracker.track_error(self.pipeline_name, e)
            raise
    
    async def get_input_data(self) -> Any:
        """
        Get input data for processing.
        Override in subclasses to implement specific input sources.
        
        Returns:
            Input data or None if no data available
        """
        return None
    
    async def send_alert(self, alert: BaseAlert):
        """Send alert to the Alert Broker API"""
        try:
            # Determine alert endpoint based on pipeline type
            endpoint_map = {
                "threat_intelligence": "threat",
                "video_surveillance": "video",
                "border_anomaly": "anomaly"
            }
            
            endpoint = endpoint_map.get(self.pipeline_name, "threat")
            url = f"{self.api_base_url}/alerts/{endpoint}"
            
            # Convert alert to dictionary
            alert_data = alert.dict()
            
            # Send POST request
            response = requests.post(url, json=alert_data, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Successfully sent alert {alert.id} to API")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert {alert.id}: {str(e)}")
            error_tracker.track_error(f"{self.pipeline_name}_api", e)
            raise
    
    async def _send_metrics(self):
        """Send pipeline metrics to the API"""
        try:
            # Only send metrics every 60 seconds
            if self.metrics.processed_items % 60 == 0 and self.metrics.processed_items > 0:
                metrics = SystemMetrics(
                    pipeline_name=self.pipeline_name,
                    processing_rate=self.metrics.processing_rate,
                    accuracy_score=self.metrics.success_rate,
                    last_update=datetime.now(),
                    status=self.status.value,
                    error_count=self.metrics.failed_items
                )
                
                url = f"{self.api_base_url}/metrics"
                response = requests.post(url, json=metrics.dict(), timeout=10)
                response.raise_for_status()
                
                self.logger.debug(f"Sent metrics to API: {metrics.dict()}")
                
        except Exception as e:
            self.logger.warning(f"Failed to send metrics: {str(e)}")
            # Don't raise exception for metrics failures
    
    async def _handle_error(self, error: Exception):
        """Handle pipeline errors"""
        # Call registered error callbacks
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {str(e)}")
        
        # Implement exponential backoff for retries
        if self.status == PipelineStatus.ERROR:
            await asyncio.sleep(min(2 ** self.metrics.failed_items, 60))
    
    def add_error_callback(self, callback: Callable):
        """Add error callback function"""
        self._error_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics"""
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "metrics": {
                "processed_items": self.metrics.processed_items,
                "successful_items": self.metrics.successful_items,
                "failed_items": self.metrics.failed_items,
                "alerts_generated": self.metrics.alerts_generated,
                "success_rate": self.metrics.success_rate,
                "processing_rate": self.metrics.processing_rate,
                "average_processing_time": self.metrics.average_processing_time,
                "uptime": (datetime.now() - self.metrics.start_time).total_seconds() if self.metrics.start_time else 0
            },
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "processing_interval": self.processing_interval,
                "api_base_url": self.api_base_url
            }
        }
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update pipeline configuration"""
        for key, value in config_updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated {key} to {value}")
        
        # Save updated configuration
        self.config_manager.update(config_updates)


class PipelineManager:
    """Manager for multiple pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, BasePipeline] = {}
        self.logger = get_pipeline_logger("manager")
    
    def register_pipeline(self, pipeline: BasePipeline):
        """Register a pipeline with the manager"""
        self.pipelines[pipeline.pipeline_name] = pipeline
        self.logger.info(f"Registered pipeline: {pipeline.pipeline_name}")
    
    async def start_all(self):
        """Start all registered pipelines"""
        tasks = []
        for pipeline in self.pipelines.values():
            task = asyncio.create_task(pipeline.start())
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self):
        """Stop all registered pipelines"""
        tasks = []
        for pipeline in self.pipelines.values():
            task = asyncio.create_task(pipeline.stop())
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_pipeline_status(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific pipeline"""
        pipeline = self.pipelines.get(pipeline_name)
        return pipeline.get_status() if pipeline else None
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pipelines"""
        return {
            name: pipeline.get_status()
            for name, pipeline in self.pipelines.items()
        }