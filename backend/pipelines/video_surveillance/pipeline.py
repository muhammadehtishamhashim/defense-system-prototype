"""
Video Surveillance Pipeline implementation.
Processes video streams, performs object detection/tracking, analyzes behavior, and generates alerts.
"""

import asyncio
import uuid
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from pipelines.base import BasePipeline
from models.alerts import VideoAlert
from utils.logging import get_pipeline_logger
from .detection import CPUOptimizedVideoSource
from .analysis import VideoAnalysisPipeline, Zone

logger = get_pipeline_logger("video_surveillance")


class VideoSurveillancePipeline(BasePipeline):
    """
    Video Surveillance Pipeline that processes video streams and generates alerts.
    Extends BasePipeline with video-specific functionality.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Video Surveillance Pipeline

        Args:
            config_path: Path to pipeline configuration file
        """
        super().__init__("video_surveillance", config_path)

        # Pipeline-specific configuration
        self.video_sources = self.config_manager.get("video_sources", [])
        self.detection_zones = self.config_manager.get("detection_zones", [])
        self.snapshot_enabled = self.config_manager.get("snapshot_enabled", True)
        self.realtime_processing = self.config_manager.get("realtime_processing", True)

        # Initialize video analysis pipeline
        detector_config = self.config_manager.get("detector_config", {})
        analyzer_config = self.config_manager.get("analyzer_config", {})

        self.analysis_pipeline = VideoAnalysisPipeline(detector_config, analyzer_config)

        # Setup detection zones
        self._setup_zones()

        # Processing state
        self.active_streams = {}
        self.snapshot_dir = self.config_manager.get("snapshot_dir", "media/snapshots")
        self.media_storage = MediaStorage(self.snapshot_dir)

        logger.info("Video Surveillance Pipeline initialized")

    def _setup_zones(self):
        """Setup detection zones from configuration"""
        for zone_config in self.detection_zones:
            try:
                zone = Zone(
                    id=zone_config['id'],
                    name=zone_config['name'],
                    points=zone_config['points'],
                    color=tuple(zone_config.get('color', [0, 255, 0])),
                    restricted=zone_config.get('restricted', True),
                    min_dwell_time=zone_config.get('min_dwell_time', 0)
                )
                self.analysis_pipeline.add_zone(zone)
                logger.info(f"Configured zone: {zone.name}")
            except Exception as e:
                logger.error(f"Error setting up zone {zone_config.get('name', 'unknown')}: {str(e)}")

    async def initialize(self):
        """Initialize pipeline-specific components"""
        logger.info("Initializing Video Surveillance Pipeline")

        # Create snapshot directory
        import os
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize video sources
        await self._initialize_video_sources()

    async def _initialize_video_sources(self):
        """Initialize video input sources"""
        if not self.video_sources:
            logger.warning("No video sources configured")
            return

        for source_config in self.video_sources:
            source_name = source_config.get('name', 'unknown')
            source_url = source_config.get('url', '')

            if source_url:
                try:
                    video_source = CPUOptimizedVideoSource(
                        source_url,
                        target_fps=source_config.get('target_fps', 15),
                        max_resolution=source_config.get('max_resolution', (640, 480))
                    )
                    self.active_streams[source_name] = {
                        'source': video_source,
                        'config': source_config,
                        'status': 'initialized'
                    }
                    logger.info(f"Initialized video source: {source_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize video source {source_name}: {str(e)}")
            else:
                logger.warning(f"No URL configured for video source: {source_name}")

    async def get_input_data(self) -> Optional[Dict[str, Any]]:
        """
        Get video frame data for processing

        Returns:
            Video frame data or None if no data available
        """
        # Process all active video streams
        for stream_name, stream_info in self.active_streams.items():
            if stream_info['status'] == 'initialized':
                frame = stream_info['source'].read_frame()
                if frame is not None:
                    return {
                        'stream_name': stream_name,
                        'frame': frame,
                        'timestamp': datetime.now(),
                        'source_config': stream_info['config']
                    }

        return None

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video frame data

        Args:
            input_data: Dictionary containing video frame and metadata

        Returns:
            Processing results
        """
        stream_name = input_data.get('stream_name', 'unknown')
        frame = input_data.get('frame')
        timestamp = input_data.get('timestamp', datetime.now())

        if frame is None:
            return {'error': 'No frame data provided'}

        logger.debug(f"Processing frame from stream: {stream_name}")

        try:
            # Process frame through analysis pipeline
            results = self.analysis_pipeline.process_frame(frame, timestamp)

            # Add stream information
            results.update({
                'stream_name': stream_name,
                'video_timestamp': timestamp.isoformat(),
                'frame_shape': frame.shape if frame is not None else None
            })

            logger.info(f"Processed frame from {stream_name}: {results['detections']} detections, {results['tracks']} tracks, {len(results['alerts'])} alerts")

            return results

        except Exception as e:
            logger.error(f"Error processing video frame: {str(e)}")
            return {
                'error': str(e),
                'stream_name': stream_name,
                'timestamp': timestamp.isoformat()
            }

    def create_alert(self, processing_result: Dict[str, Any]) -> Optional[VideoAlert]:
        """
        Create video alert from processing results

        Args:
            processing_result: Results from process_input

        Returns:
            VideoAlert object or None if no alert should be generated
        """
        if 'error' in processing_result:
            return None

        try:
            alerts = processing_result.get('alerts', [])
            if not alerts:
                return None

            # Take the first (most critical) alert
            alert_data = alerts[0]
            alert_type = alert_data.get('type')

            # Map alert types to event types
            event_type_map = {
                'loitering': 'loitering',
                'zone_violation': 'zone_violation',
                'abandoned_object': 'abandoned_object'
            }

            event_type = event_type_map.get(alert_type, 'unknown')

            # Get track information for the alert
            track_id = alert_data.get('track_id')
            track_info = None

            # Find track information from processing results
            # This would need to be passed from the analysis pipeline
            # For now, create a basic alert
            confidence = alert_data.get('confidence', 0.5)
            if confidence < self.confidence_threshold:
                return None

            # Create snapshot if enabled
            snapshot_path = None
            if self.snapshot_enabled:
                try:
                    # This would need access to the current frame
                    # For now, use placeholder
                    snapshot_path = f"snapshots/{uuid.uuid4()}.jpg"
                except Exception as e:
                    logger.warning(f"Could not create snapshot: {str(e)}")

            # Create video alert
            alert = VideoAlert(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                confidence=confidence,
                source_pipeline="video_surveillance",
                event_type=event_type,
                bounding_box=[0, 0, 100, 100],  # Placeholder - would come from track data
                track_id=track_id or 0,
                snapshot_path=snapshot_path or "",
                video_timestamp=processing_result.get('timestamp', datetime.now().isoformat()),
                metadata={
                    'stream_name': processing_result.get('stream_name', 'unknown'),
                    'alert_data': alert_data,
                    'frame_number': processing_result.get('frame_number', 0)
                }
            )

            return alert

        except Exception as e:
            logger.error(f"Error creating video alert: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("Cleaning up Video Surveillance Pipeline")

        # Close all video streams
        for stream_name, stream_info in self.active_streams.items():
            try:
                stream_info['source'].release()
                logger.info(f"Released video source: {stream_name}")
            except Exception as e:
                logger.error(f"Error releasing video source {stream_name}: {str(e)}")

        self.active_streams.clear()

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status"""
        base_status = super().get_pipeline_status()

        # Add video-specific metrics
        video_metrics = {
            'active_streams': len(self.active_streams),
            'configured_zones': len(self.detection_zones),
            'snapshot_enabled': self.snapshot_enabled,
            'realtime_processing': self.realtime_processing,
            'analysis_stats': self.analysis_pipeline.get_pipeline_status() if self.analysis_pipeline else {}
        }

        base_status['video_specific'] = video_metrics
        return base_status

    def add_video_source(self, name: str, url: str, config: Dict[str, Any] = None):
        """Add a video source to the pipeline"""
        if config is None:
            config = {}

        try:
            video_source = CPUOptimizedVideoSource(
                url,
                target_fps=config.get('target_fps', 15),
                max_resolution=config.get('max_resolution', (640, 480))
            )
            self.active_streams[name] = {
                'source': video_source,
                'config': config,
                'status': 'active'
            }
            logger.info(f"Added video source: {name}")
        except Exception as e:
            logger.error(f"Failed to add video source {name}: {str(e)}")

    def remove_video_source(self, name: str):
        """Remove a video source from the pipeline"""
        if name in self.active_streams:
            try:
                self.active_streams[name]['source'].release()
                del self.active_streams[name]
                logger.info(f"Removed video source: {name}")
            except Exception as e:
                logger.error(f"Error removing video source {name}: {str(e)}")

    def get_stream_status(self) -> Dict[str, Any]:
        """Get status of all video streams"""
        stream_status = {}
        for name, stream_info in self.active_streams.items():
            try:
                source = stream_info['source']
                frame_size = source.get_frame_size()
                stream_status[name] = {
                    'status': stream_info['status'],
                    'frame_size': frame_size,
                    'config': stream_info['config']
                }
            except Exception as e:
                stream_status[name] = {
                    'status': 'error',
                    'error': str(e)
                }

        return stream_status


class MediaStorage:
    """Handles storage of video snapshots and media files"""

    def __init__(self, base_dir: str = "media"):
        """
        Initialize media storage

        Args:
            base_dir: Base directory for storing media files
        """
        self.base_dir = base_dir
        self.snapshots_dir = f"{base_dir}/snapshots"
        self.videos_dir = f"{base_dir}/videos"

        # Create directories
        import os
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        logger.info(f"Media storage initialized: {base_dir}")

    def save_snapshot(self, frame: np.ndarray, filename: str = None) -> str:
        """
        Save video frame as snapshot image

        Args:
            frame: Video frame to save
            filename: Optional filename (generated if not provided)

        Returns:
            Path to saved snapshot file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{timestamp}.jpg"

        filepath = f"{self.snapshots_dir}/{filename}"

        try:
            # Save image
            success = cv2.imwrite(filepath, frame)
            if success:
                logger.debug(f"Snapshot saved: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save snapshot: {filepath}")
                return ""
        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")
            return ""

    def get_snapshot_as_base64(self, filepath: str) -> Optional[str]:
        """
        Get snapshot image as base64 encoded string

        Args:
            filepath: Path to snapshot file

        Returns:
            Base64 encoded image string or None if error
        """
        try:
            with open(filepath, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return base64_data
        except Exception as e:
            logger.error(f"Error reading snapshot {filepath}: {str(e)}")
            return None

    def cleanup_old_snapshots(self, max_age_hours: int = 24):
        """
        Clean up old snapshot files

        Args:
            max_age_hours: Maximum age of snapshots to keep (hours)
        """
        try:
            import os
            import time

            cutoff_time = time.time() - (max_age_hours * 3600)

            for filename in os.listdir(self.snapshots_dir):
                filepath = f"{self.snapshots_dir}/{filename}"
                if os.path.isfile(filepath):
                    file_modified = os.path.getmtime(filepath)
                    if file_modified < cutoff_time:
                        os.remove(filepath)
                        logger.debug(f"Removed old snapshot: {filename}")

            logger.info(f"Cleaned up old snapshots (max age: {max_age_hours}h)")

        except Exception as e:
            logger.error(f"Error cleaning up snapshots: {str(e)}")


class VideoStreamProcessor:
    """Processes video streams in real-time"""

    def __init__(self, pipeline: VideoSurveillancePipeline):
        """
        Initialize video stream processor

        Args:
            pipeline: Video surveillance pipeline instance
        """
        self.pipeline = pipeline
        self.is_running = False
        self.streams = {}

    async def start_stream(self, stream_name: str, source_url: str):
        """Start processing a video stream"""
        if stream_name in self.streams:
            logger.warning(f"Stream {stream_name} already exists")
            return

        try:
            # Add stream to pipeline
            self.pipeline.add_video_source(stream_name, source_url)

            # Create processing task
            self.streams[stream_name] = {
                'task': asyncio.create_task(self._process_stream(stream_name)),
                'status': 'starting'
            }

            logger.info(f"Started video stream: {stream_name}")

        except Exception as e:
            logger.error(f"Failed to start stream {stream_name}: {str(e)}")

    async def stop_stream(self, stream_name: str):
        """Stop processing a video stream"""
        if stream_name not in self.streams:
            logger.warning(f"Stream {stream_name} not found")
            return

        try:
            # Cancel processing task
            self.streams[stream_name]['task'].cancel()
            await self.streams[stream_name]['task']

            # Remove from pipeline
            self.pipeline.remove_video_source(stream_name)

            del self.streams[stream_name]
            logger.info(f"Stopped video stream: {stream_name}")

        except Exception as e:
            logger.error(f"Error stopping stream {stream_name}: {str(e)}")

    async def _process_stream(self, stream_name: str):
        """Process a single video stream"""
        try:
            self.streams[stream_name]['status'] = 'running'

            while self.is_running:
                try:
                    # Get frame from pipeline
                    input_data = await self.pipeline.get_input_data()

                    if input_data and input_data.get('stream_name') == stream_name:
                        # Process the frame
                        results = await self.pipeline.process_input(input_data)

                        # Generate alerts if needed
                        alert = self.pipeline.create_alert(results)
                        if alert:
                            await self.pipeline.send_alert(alert)
                            logger.info(f"Generated alert for stream {stream_name}: {alert.event_type}")

                        # Clean up old snapshots periodically
                        if self.pipeline.frame_count % 1000 == 0:
                            self.pipeline.media_storage.cleanup_old_snapshots()

                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing stream {stream_name}: {str(e)}")
                    await asyncio.sleep(1)  # Wait before retrying

        except Exception as e:
            logger.error(f"Stream processing error for {stream_name}: {str(e)}")
        finally:
            self.streams[stream_name]['status'] = 'stopped'

    async def start_all(self):
        """Start the stream processor"""
        self.is_running = True
        logger.info("Video stream processor started")

    async def stop_all(self):
        """Stop the stream processor"""
        self.is_running = False

        # Stop all streams
        stop_tasks = []
        for stream_name in self.streams:
            stop_tasks.append(self.stop_stream(stream_name))

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        logger.info("Video stream processor stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get stream processor status"""
        return {
            'is_running': self.is_running,
            'active_streams': list(self.streams.keys()),
            'stream_status': {
                name: info['status'] for name, info in self.streams.items()
            }
        }
