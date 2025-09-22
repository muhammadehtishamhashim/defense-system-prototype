"""
Behavior analysis engine for video surveillance.
Implements loitering detection, zone violation detection, and abandoned object detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import math
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("behavior_analysis")


@dataclass
class Zone:
    """Configurable zone for zone violation detection"""
    id: str
    name: str
    points: List[Tuple[int, int]]  # List of (x, y) coordinates
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR color
    restricted: bool = True  # Whether this zone is restricted
    min_dwell_time: int = 0  # Minimum time (seconds) for loitering detection


@dataclass
class TrackHistory:
    """Track history for behavior analysis"""
    track_id: int
    positions: List[Tuple[float, float]]  # List of (x, y) center positions
    timestamps: List[datetime]
    first_seen: datetime
    last_seen: datetime
    velocities: List[Tuple[float, float]]  # List of (vx, vy) velocities
    bounding_boxes: List[List[float]]  # List of [x1, y1, x2, y2]


class BehaviorAnalyzer:
    """Analyzes object behavior for suspicious activities"""

    def __init__(self, loitering_threshold: int = 30, abandoned_object_threshold: int = 60):
        """
        Initialize behavior analyzer

        Args:
            loitering_threshold: Time in seconds to trigger loitering alert
            abandoned_object_threshold: Time in seconds to trigger abandoned object alert
        """
        self.loitering_threshold = loitering_threshold
        self.abandoned_object_threshold = abandoned_object_threshold

        # Track histories for all objects
        self.track_histories: Dict[int, TrackHistory] = {}

        # Zones for zone violation detection
        self.zones: List[Zone] = []

        # Abandoned objects tracking
        self.abandoned_objects: Dict[str, Dict[str, Any]] = {}

        logger.info("Behavior analyzer initialized")

    def add_zone(self, zone: Zone):
        """Add a zone for zone violation detection"""
        self.zones.append(zone)
        logger.info(f"Added zone: {zone.name} ({zone.id})")

    def update_track(self, tracked_objects: List[Dict[str, Any]], current_time: datetime):
        """
        Update track histories with new frame data

        Args:
            tracked_objects: List of tracked objects from current frame
            current_time: Timestamp of current frame
        """
        # Update existing tracks
        for obj in tracked_objects:
            track_id = obj['track_id']
            center = obj['center']
            bbox = obj['bbox']

            if track_id not in self.track_histories:
                # New track
                self.track_histories[track_id] = TrackHistory(
                    track_id=track_id,
                    positions=[center],
                    timestamps=[current_time],
                    first_seen=current_time,
                    last_seen=current_time,
                    velocities=[],
                    bounding_boxes=[bbox]
                )
            else:
                # Update existing track
                history = self.track_histories[track_id]
                history.positions.append(center)
                history.timestamps.append(current_time)
                history.last_seen = current_time
                history.bounding_boxes.append(bbox)

                # Calculate velocity if we have enough history
                if len(history.positions) >= 2:
                    prev_pos = history.positions[-2]
                    dt = (current_time - history.timestamps[-2]).total_seconds()
                    if dt > 0:
                        vx = (center[0] - prev_pos[0]) / dt
                        vy = (center[1] - prev_pos[1]) / dt
                        history.velocities.append((vx, vy))

        # Clean up old tracks (not seen for more than 10 seconds)
        current_time = datetime.now()
        tracks_to_remove = []
        for track_id, history in self.track_histories.items():
            if (current_time - history.last_seen).total_seconds() > 10:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_histories[track_id]

    def detect_loitering(self, track_id: int, current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Detect loitering behavior for a specific track

        Args:
            track_id: Object track ID
            current_time: Current timestamp

        Returns:
            Loitering alert if detected, None otherwise
        """
        if track_id not in self.track_histories:
            return None

        history = self.track_histories[track_id]

        # Check if object has been in roughly the same area for long enough
        if len(history.positions) < 10:  # Need some history
            return None

        # Calculate movement over time
        recent_positions = history.positions[-10:]  # Last 10 positions
        if len(recent_positions) < 2:
            return None

        # Calculate average displacement
        total_displacement = 0
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            total_displacement += math.sqrt(dx*dx + dy*dy)

        avg_displacement = total_displacement / (len(recent_positions) - 1)

        # Check if movement is below threshold (loitering)
        if avg_displacement < 5.0:  # Less than 5 pixels per frame on average
            dwell_time = (current_time - history.first_seen).total_seconds()

            if dwell_time >= self.loitering_threshold:
                return {
                    'type': 'loitering',
                    'track_id': track_id,
                    'dwell_time': dwell_time,
                    'position': recent_positions[-1],
                    'confidence': min(1.0, dwell_time / (self.loitering_threshold * 2)),
                    'timestamp': current_time
                }

        return None

    def detect_zone_violation(self, track_id: int, current_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Detect zone violations for a specific track

        Args:
            track_id: Object track ID
            current_time: Current timestamp

        Returns:
            Zone violation alert if detected, None otherwise
        """
        if track_id not in self.track_histories:
            return None

        history = self.track_histories[track_id]
        current_position = history.positions[-1] if history.positions else None

        if not current_position:
            return None

        # Check each zone
        for zone in self.zones:
            if self._is_point_in_zone(current_position, zone.points):
                return {
                    'type': 'zone_violation',
                    'track_id': track_id,
                    'zone_id': zone.id,
                    'zone_name': zone.name,
                    'position': current_position,
                    'timestamp': current_time,
                    'confidence': 0.9
                }

        return None

    def detect_abandoned_objects(self, current_time: datetime) -> List[Dict[str, Any]]:
        """
        Detect abandoned objects based on track disappearance

        Args:
            current_time: Current timestamp

        Returns:
            List of abandoned object alerts
        """
        alerts = []

        # Check for tracks that disappeared recently
        for track_id, history in self.track_histories.items():
            time_since_last_seen = (current_time - history.last_seen).total_seconds()

            # Object disappeared within last 5 seconds
            if 1 <= time_since_last_seen <= 5:
                # Get the last known position and bounding box
                last_position = history.positions[-1]
                last_bbox = history.bounding_boxes[-1]

                # Check if this looks like an object that was left behind
                # (stationary for a while before disappearing)
                if len(history.positions) >= 10:
                    recent_positions = history.positions[-10:]
                    total_movement = 0
                    for i in range(1, len(recent_positions)):
                        dx = recent_positions[i][0] - recent_positions[i-1][0]
                        dy = recent_positions[i][1] - recent_positions[i-1][1]
                        total_movement += math.sqrt(dx*dx + dy*dy)

                    avg_movement = total_movement / (len(recent_positions) - 1)

                    # If object was relatively stationary before disappearing
                    if avg_movement < 3.0:
                        # Check if we should alert (avoid duplicate alerts)
                        alert_id = f"abandoned_{track_id}_{current_time.strftime('%Y%m%d_%H%M%S')}"

                        if alert_id not in self.abandoned_objects:
                            self.abandoned_objects[alert_id] = {
                                'track_id': track_id,
                                'first_detected': history.first_seen,
                                'last_seen': history.last_seen,
                                'position': last_position,
                                'bbox': last_bbox
                            }

                            alerts.append({
                                'type': 'abandoned_object',
                                'track_id': track_id,
                                'alert_id': alert_id,
                                'position': last_position,
                                'bbox': last_bbox,
                                'dwell_time': (history.last_seen - history.first_seen).total_seconds(),
                                'timestamp': current_time,
                                'confidence': 0.8
                            })

        # Clean up old abandoned object records
        cutoff_time = current_time - timedelta(hours=1)
        alerts_to_remove = []
        for alert_id, alert_data in self.abandoned_objects.items():
            if alert_data['last_seen'] < cutoff_time:
                alerts_to_remove.append(alert_id)

        for alert_id in alerts_to_remove:
            del self.abandoned_objects[alert_id]

        return alerts

    def _is_point_in_zone(self, point: Tuple[float, float], zone_points: List[Tuple[int, int]]) -> bool:
        """
        Check if a point is inside a zone using ray casting algorithm

        Args:
            point: (x, y) coordinates
            zone_points: List of zone vertices

        Returns:
            True if point is inside zone
        """
        x, y = point
        n = len(zone_points)
        inside = False

        p1x, p1y = zone_points[0]
        for i in range(1, n + 1):
            p2x, p2y = zone_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def analyze_frame(self, tracked_objects: List[Dict[str, Any]], current_time: datetime) -> List[Dict[str, Any]]:
        """
        Analyze a frame for suspicious behaviors

        Args:
            tracked_objects: List of tracked objects from current frame
            current_time: Timestamp of current frame

        Returns:
            List of behavior alerts
        """
        alerts = []

        # Update track histories
        self.update_track(tracked_objects, current_time)

        # Check each track for suspicious behavior
        for obj in tracked_objects:
            track_id = obj['track_id']

            # Loitering detection
            loitering_alert = self.detect_loitering(track_id, current_time)
            if loitering_alert:
                alerts.append(loitering_alert)

            # Zone violation detection
            zone_alert = self.detect_zone_violation(track_id, current_time)
            if zone_alert:
                alerts.append(zone_alert)

        # Abandoned object detection
        abandoned_alerts = self.detect_abandoned_objects(current_time)
        alerts.extend(abandoned_alerts)

        logger.debug(f"Generated {len(alerts)} behavior alerts")
        return alerts

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get behavior analysis statistics"""
        total_tracks = len(self.track_histories)
        zones_count = len(self.zones)
        abandoned_count = len(self.abandoned_objects)

        return {
            'total_active_tracks': total_tracks,
            'configured_zones': zones_count,
            'abandoned_objects_tracked': abandoned_count,
            'loitering_threshold_seconds': self.loitering_threshold,
            'abandoned_threshold_seconds': self.abandoned_object_threshold
        }


class VideoAnalysisPipeline:
    """Complete video analysis pipeline with detection, tracking, and behavior analysis"""

    def __init__(self, detector_config: Dict[str, Any] = None, analyzer_config: Dict[str, Any] = None):
        """
        Initialize video analysis pipeline

        Args:
            detector_config: Configuration for object detector
            analyzer_config: Configuration for behavior analyzer
        """
        # Initialize components
        detector_config = detector_config or {}
        analyzer_config = analyzer_config or {}

        from .detection import CPUOptimizedDetector, ByteTrackTracker, CPUOptimizedVideoProcessor
        from utils.logging import get_pipeline_logger

        logger = get_pipeline_logger("video_pipeline")

        self.detector = CPUOptimizedDetector(
            model_path=detector_config.get('model_path', 'yolov8n.pt'),
            confidence_threshold=detector_config.get('confidence_threshold', 0.5),
            use_onnx=detector_config.get('use_onnx', True),
            input_size=detector_config.get('input_size', 416)
        )

        self.tracker = ByteTrackTracker(
            frame_rate=detector_config.get('frame_rate', 15),
            track_thresh=detector_config.get('track_thresh', 0.5),
            track_buffer=detector_config.get('track_buffer', 30),
            match_thresh=detector_config.get('match_thresh', 0.8)
        )

        max_resolution = detector_config.get('max_resolution', (640, 480))
        self.processor = CPUOptimizedVideoProcessor(self.detector, self.tracker, max_resolution)
        self.analyzer = BehaviorAnalyzer(
            loitering_threshold=analyzer_config.get('loitering_threshold', 30),
            abandoned_object_threshold=analyzer_config.get('abandoned_threshold', 60)
        )

        self.frame_count = 0
        self.start_time = datetime.now()

        logger.info("Video analysis pipeline initialized")

    def add_zone(self, zone: Zone):
        """Add a zone for zone violation detection"""
        self.analyzer.add_zone(zone)

    def process_frame(self, frame: np.ndarray, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process a single frame through the complete pipeline

        Args:
            frame: Input video frame
            timestamp: Frame timestamp (optional)

        Returns:
            Analysis results with detections, tracks, and alerts
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.frame_count += 1

        # Step 1: Process frame for detection and tracking
        processed_frame, tracked_objects = self.processor.process_frame(frame)

        # Step 2: Analyze behavior
        behavior_alerts = self.analyzer.analyze_frame(tracked_objects, timestamp)

        # Step 3: Prepare results
        results = {
            'frame_number': self.frame_count,
            'timestamp': timestamp.isoformat(),
            'detections': len([obj for obj in tracked_objects if 'confidence' in obj]),
            'tracks': len(tracked_objects),
            'alerts': behavior_alerts,
            'processing_stats': self.processor.get_processing_stats(),
            'analysis_stats': self.analyzer.get_analysis_stats()
        }

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            'frame_count': self.frame_count,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'detector_status': 'active' if self.detector.model else 'inactive',
            'tracker_status': 'active' if self.tracker.tracker else 'fallback',
            'analyzer_status': 'active',
            'processing_stats': self.processor.get_processing_stats(),
            'analysis_stats': self.analyzer.get_analysis_stats()
        }
