"""
Border Anomaly Detection Pipeline for HifazatAI.
Integrates trajectory extraction, analysis, and anomaly detection.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
import os

from .trajectory import TrajectoryExtractor, TrajectoryAnalyzer, Trajectory
from .anomaly_detector import (
    AnomalyDetector, IsolationForestDetector, AutoencoderDetector, 
    MotionBasedDetector, EnsembleDetector, AnomalyResult
)
from utils.logging import get_pipeline_logger
from utils.base_pipeline import BasePipeline

logger = get_pipeline_logger("border_anomaly")


class BorderAnomalyPipeline(BasePipeline):
    """Main pipeline for border anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize border anomaly detection pipeline
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        
        # Configuration
        self.frame_width = config.get('frame_width', 1920)
        self.frame_height = config.get('frame_height', 1080)
        self.fps = config.get('fps', 30.0)
        self.min_trajectory_length = config.get('min_trajectory_length', 5)
        self.max_gap_frames = config.get('max_gap_frames', 10)
        
        # Detection configuration
        self.detection_method = config.get('detection_method', 'ensemble')
        self.contamination = config.get('contamination', 0.1)
        self.speed_threshold = config.get('speed_threshold', 100.0)
        self.direction_change_threshold = config.get('direction_change_threshold', 5)
        
        # Model paths
        self.model_dir = config.get('model_dir', 'models/border_anomaly')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize components
        self.trajectory_extractor = TrajectoryExtractor(
            min_trajectory_length=self.min_trajectory_length,
            max_gap_frames=self.max_gap_frames
        )
        
        self.trajectory_analyzer = TrajectoryAnalyzer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=self.fps
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = self._create_anomaly_detector()
        
        # Training data storage
        self.training_trajectories: List[Trajectory] = []
        self.is_trained = False
        
        # Alert configuration
        self.alert_broker_url = config.get('alert_broker_url', 'http://localhost:8000')
        self.enable_alerts = config.get('enable_alerts', True)
        
        logger.info(f"BorderAnomalyPipeline initialized ({self.detection_method} detector)")
    
    def _create_anomaly_detector(self) -> AnomalyDetector:
        """Create anomaly detector based on configuration"""
        if self.detection_method == 'isolation_forest':
            return IsolationForestDetector(
                contamination=self.contamination,
                random_state=42
            )
        elif self.detection_method == 'autoencoder':
            return AutoencoderDetector(
                hidden_dim=8,
                learning_rate=0.001,
                epochs=100
            )
        elif self.detection_method == 'motion_based':
            return MotionBasedDetector(
                speed_threshold=self.speed_threshold,
                direction_change_threshold=self.direction_change_threshold
            )
        elif self.detection_method == 'ensemble':
            detectors = [
                IsolationForestDetector(contamination=self.contamination),
                MotionBasedDetector(
                    speed_threshold=self.speed_threshold,
                    direction_change_threshold=self.direction_change_threshold
                )
            ]
            # Add autoencoder if PyTorch is available
            try:
                detectors.append(AutoencoderDetector())
                weights = [0.4, 0.3, 0.3]
            except ImportError:
                weights = [0.6, 0.4]
            
            return EnsembleDetector(detectors, weights)
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")
    
    async def process_frame(self, frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single frame for anomaly detection
        
        Args:
            frame_data: Frame data with tracking information
            
        Returns:
            List of anomaly alerts
        """
        frame_number = frame_data.get('frame_number', 0)
        timestamp = datetime.fromisoformat(frame_data.get('timestamp', datetime.now().isoformat()))
        tracked_objects = frame_data.get('tracked_objects', [])
        
        # Update trajectories
        completed_trajectories = self.trajectory_extractor.update(
            tracked_objects, frame_number, timestamp
        )
        
        # Process completed trajectories for anomalies
        alerts = []
        for trajectory in completed_trajectories:
            if self.is_trained:
                anomaly_result = self.anomaly_detector.predict(trajectory)
                
                if anomaly_result.is_anomaly:
                    alert = await self._create_anomaly_alert(trajectory, anomaly_result, frame_data)
                    alerts.append(alert)
                    
                    if self.enable_alerts:
                        await self._send_alert(alert)
            else:
                # Store trajectory for training
                self.training_trajectories.append(trajectory)
        
        return alerts
    
    async def train_model(self, additional_trajectories: Optional[List[Trajectory]] = None) -> Dict[str, Any]:
        """
        Train the anomaly detection model
        
        Args:
            additional_trajectories: Optional additional training data
            
        Returns:
            Training results
        """
        # Combine all available trajectories
        all_trajectories = self.training_trajectories.copy()
        if additional_trajectories:
            all_trajectories.extend(additional_trajectories)
        
        if len(all_trajectories) < 10:
            raise ValueError(f"Insufficient training data: {len(all_trajectories)} trajectories (minimum 10 required)")
        
        logger.info(f"Training anomaly detector with {len(all_trajectories)} trajectories")
        
        # Train the model
        start_time = datetime.now()
        self.anomaly_detector.fit(all_trajectories)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save the trained model
        model_path = os.path.join(self.model_dir, f"{self.detection_method}_model.pkl")
        self.anomaly_detector.save_model(model_path)
        
        self.is_trained = True
        
        # Clear training trajectories to save memory
        self.training_trajectories.clear()
        
        training_results = {
            'training_trajectories': len(all_trajectories),
            'training_time_seconds': training_time,
            'model_path': model_path,
            'detection_method': self.detection_method,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model training completed in {training_time:.2f}s")
        return training_results
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a pre-trained model
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            True if model loaded successfully
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.detection_method}_model.pkl")
        
        try:
            self.anomaly_detector.load_model(model_path)
            self.is_trained = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    async def _create_anomaly_alert(self, trajectory: Trajectory, anomaly_result: AnomalyResult, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create anomaly alert from detection result"""
        # Analyze trajectory features for alert details
        features = self.trajectory_analyzer.analyze_trajectory(trajectory)
        
        alert = {
            'alert_type': 'AnomalyAlert',
            'timestamp': anomaly_result.timestamp.isoformat(),
            'severity': self._calculate_severity(anomaly_result),
            'source': 'border_anomaly_pipeline',
            'title': f'Anomalous Movement Detected - Track {trajectory.track_id}',
            'description': self._generate_alert_description(trajectory, anomaly_result, features),
            'metadata': {
                'trajectory_id': trajectory.track_id,
                'detection_method': anomaly_result.detection_method,
                'anomaly_score': anomaly_result.anomaly_score,
                'confidence': anomaly_result.confidence,
                'trajectory_features': asdict(features),
                'detection_details': anomaly_result.details,
                'frame_number': frame_data.get('frame_number'),
                'video_source': frame_data.get('source', 'unknown')
            }
        }
        
        return alert
    
    def _calculate_severity(self, anomaly_result: AnomalyResult) -> str:
        """Calculate alert severity based on anomaly score and confidence"""
        if anomaly_result.confidence > 0.8:
            return 'HIGH'
        elif anomaly_result.confidence > 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_alert_description(self, trajectory: Trajectory, anomaly_result: AnomalyResult, features) -> str:
        """Generate human-readable alert description"""
        description_parts = [
            f"Anomalous movement pattern detected for object track {trajectory.track_id}."
        ]
        
        # Add specific details based on detection method
        if anomaly_result.detection_method == 'motion_based' and anomaly_result.details:
            reasons = anomaly_result.details.get('anomaly_reasons', [])
            if 'excessive_speed' in reasons:
                description_parts.append(f"Object moving at excessive speed ({features.max_speed:.1f} px/s).")
            if 'erratic_movement' in reasons:
                description_parts.append(f"Erratic movement with {features.direction_changes} direction changes.")
            if 'zigzag_pattern' in reasons:
                description_parts.append(f"Zigzag movement pattern (straightness: {features.straightness_ratio:.2f}).")
        
        # Add trajectory summary
        description_parts.append(
            f"Trajectory duration: {features.duration:.1f}s, "
            f"distance: {features.total_distance:.1f}px, "
            f"avg speed: {features.average_speed:.1f}px/s."
        )
        
        return " ".join(description_parts)
    
    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to Alert Broker"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.alert_broker_url}/alerts",
                    json=alert,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 201:
                        logger.info(f"Alert sent successfully: {alert['title']}")
                    else:
                        logger.warning(f"Failed to send alert: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        active_trajectories = self.trajectory_extractor.get_active_trajectories()
        completed_trajectories = self.trajectory_extractor.get_completed_trajectories()
        
        stats = {
            'active_trajectories': len(active_trajectories),
            'completed_trajectories': len(completed_trajectories),
            'training_trajectories': len(self.training_trajectories),
            'is_trained': self.is_trained,
            'detection_method': self.detection_method,
            'configuration': {
                'frame_size': f"{self.frame_width}x{self.frame_height}",
                'fps': self.fps,
                'min_trajectory_length': self.min_trajectory_length,
                'max_gap_frames': self.max_gap_frames,
                'contamination': self.contamination,
                'speed_threshold': self.speed_threshold,
                'direction_change_threshold': self.direction_change_threshold
            }
        }
        
        return stats
    
    async def force_complete_trajectories(self) -> List[Trajectory]:
        """Force completion of all active trajectories"""
        return self.trajectory_extractor.force_complete_all()
    
    async def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        # Force complete any remaining trajectories
        remaining = await self.force_complete_trajectories()
        if remaining:
            logger.info(f"Completed {len(remaining)} remaining trajectories during cleanup")
        
        # Clear training data
        self.training_trajectories.clear()
        
        logger.info("BorderAnomalyPipeline cleanup completed")


# Configuration templates
DEFAULT_CONFIG = {
    'frame_width': 1920,
    'frame_height': 1080,
    'fps': 30.0,
    'min_trajectory_length': 5,
    'max_gap_frames': 10,
    'detection_method': 'ensemble',
    'contamination': 0.1,
    'speed_threshold': 100.0,
    'direction_change_threshold': 5,
    'model_dir': 'models/border_anomaly',
    'alert_broker_url': 'http://localhost:8000',
    'enable_alerts': True
}

LIGHTWEIGHT_CONFIG = {
    **DEFAULT_CONFIG,
    'detection_method': 'motion_based',
    'min_trajectory_length': 3,
    'speed_threshold': 150.0,
    'direction_change_threshold': 8
}

HIGH_ACCURACY_CONFIG = {
    **DEFAULT_CONFIG,
    'detection_method': 'ensemble',
    'contamination': 0.05,
    'min_trajectory_length': 8,
    'max_gap_frames': 5,
    'speed_threshold': 80.0,
    'direction_change_threshold': 3
}