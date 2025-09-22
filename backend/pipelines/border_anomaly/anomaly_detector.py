"""
Anomaly detection models for border surveillance.
Implements multiple approaches: Isolation Forest, Autoencoder, and motion-based detection.
"""

import numpy as np
import joblib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.combination import aom, moa, average, maximization
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

from .trajectory import Trajectory, TrajectoryFeatures, TrajectoryAnalyzer
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("border_anomaly")


def create_synthetic_anomaly_data(num_normal: int = 50, num_anomalies: int = 10) -> Tuple[List['Trajectory'], List[bool]]:
    """
    Create synthetic trajectory data with known anomalies for testing
    
    Args:
        num_normal: Number of normal trajectories to generate
        num_anomalies: Number of anomalous trajectories to generate
    
    Returns:
        Tuple of (trajectories, labels) where labels indicate anomalies
    """
    from .trajectory import TrajectoryPoint, Trajectory
    from datetime import timedelta
    trajectories = []
    labels = []
    
    # Generate normal trajectories
    for i in range(num_normal):
        # Normal trajectory: smooth, moderate speed, straight-ish path
        start_time = datetime.now()
        points = []
        
        # Random starting position
        start_x = np.random.uniform(100, 800)
        start_y = np.random.uniform(100, 500)
        
        # Normal movement parameters
        direction = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(20, 60)  # Normal speed
        
        for j in range(np.random.randint(10, 25)):  # Normal trajectory length
            # Add some noise but keep movement smooth
            noise_x = np.random.normal(0, 5)
            noise_y = np.random.normal(0, 5)
            
            x = start_x + j * speed * np.cos(direction) + noise_x
            y = start_y + j * speed * np.sin(direction) + noise_y
            
            # Slight direction changes (normal behavior)
            direction += np.random.normal(0, 0.1)
            
            point = TrajectoryPoint(
                x=max(0, min(1920, x)),
                y=max(0, min(1080, y)),
                timestamp=start_time + timedelta(seconds=j * 0.5),
                frame_number=j,
                confidence=np.random.uniform(0.8, 1.0)
            )
            points.append(point)
        
        trajectory = Trajectory(
            track_id=i,
            points=points,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=len(points) * 0.5)
        )
        trajectories.append(trajectory)
        labels.append(False)  # Normal
    
    # Generate anomalous trajectories
    for i in range(num_anomalies):
        start_time = datetime.now()
        points = []
        
        # Random starting position
        start_x = np.random.uniform(100, 800)
        start_y = np.random.uniform(100, 500)
        
        # Choose anomaly type
        anomaly_type = np.random.choice(['high_speed', 'erratic_movement', 'zigzag', 'loitering'])
        
        if anomaly_type == 'high_speed':
            # Very high speed trajectory
            direction = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(150, 300)  # Anomalously high speed
            
            for j in range(np.random.randint(5, 15)):
                x = start_x + j * speed * np.cos(direction)
                y = start_y + j * speed * np.sin(direction)
                
                point = TrajectoryPoint(
                    x=max(0, min(1920, x)),
                    y=max(0, min(1080, y)),
                    timestamp=start_time + timedelta(seconds=j * 0.2),  # Fast movement
                    frame_number=j,
                    confidence=np.random.uniform(0.7, 0.9)
                )
                points.append(point)
        
        elif anomaly_type == 'erratic_movement':
            # Frequent direction changes
            direction = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(30, 80)
            
            for j in range(np.random.randint(15, 30)):
                # Frequent large direction changes
                direction += np.random.uniform(-np.pi/2, np.pi/2)
                
                x = start_x + j * speed * np.cos(direction)
                y = start_y + j * speed * np.sin(direction)
                
                point = TrajectoryPoint(
                    x=max(0, min(1920, x)),
                    y=max(0, min(1080, y)),
                    timestamp=start_time + timedelta(seconds=j * 0.5),
                    frame_number=j,
                    confidence=np.random.uniform(0.6, 0.9)
                )
                points.append(point)
        
        elif anomaly_type == 'zigzag':
            # Zigzag pattern
            for j in range(np.random.randint(20, 40)):
                # Alternating direction
                direction = np.pi/4 if j % 4 < 2 else -np.pi/4
                speed = np.random.uniform(40, 80)
                
                x = start_x + j * speed * np.cos(direction) * 0.5
                y = start_y + j * speed * np.sin(direction) * 0.5
                
                point = TrajectoryPoint(
                    x=max(0, min(1920, x)),
                    y=max(0, min(1080, y)),
                    timestamp=start_time + timedelta(seconds=j * 0.3),
                    frame_number=j,
                    confidence=np.random.uniform(0.7, 0.9)
                )
                points.append(point)
        
        elif anomaly_type == 'loitering':
            # Staying in small area for long time
            for j in range(np.random.randint(30, 60)):  # Long trajectory
                # Small random movements around starting position
                x = start_x + np.random.normal(0, 20)
                y = start_y + np.random.normal(0, 20)
                
                point = TrajectoryPoint(
                    x=max(0, min(1920, x)),
                    y=max(0, min(1080, y)),
                    timestamp=start_time + timedelta(seconds=j * 0.5),
                    frame_number=j,
                    confidence=np.random.uniform(0.8, 1.0)
                )
                points.append(point)
        
        trajectory = Trajectory(
            track_id=num_normal + i,
            points=points,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=len(points) * 0.5)
        )
        trajectories.append(trajectory)
        labels.append(True)  # Anomaly
    
    return trajectories, labels


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    trajectory_id: int
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    detection_method: str
    timestamp: datetime
    details: Dict[str, Any] = None


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    @abstractmethod
    def fit(self, trajectories: List[Trajectory]) -> None:
        """Train the anomaly detector"""
        pass
    
    @abstractmethod
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Detect anomalies in a trajectory"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save trained model"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load trained model"""
        pass


class IsolationForestDetector(AnomalyDetector):
    """CPU-optimized Isolation Forest-based anomaly detector"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42, 
                 n_estimators: int = 50, max_samples: str = 'auto', n_jobs: int = -1):
        """
        Initialize CPU-optimized Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            n_estimators: Number of trees (reduced for CPU optimization)
            max_samples: Number of samples to draw for each tree
            n_jobs: Number of CPU cores to use (-1 for all available)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        
        # CPU-optimized parameters
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=n_jobs,
            bootstrap=False,  # Faster for CPU
            warm_start=False
        )
        self.scaler = StandardScaler()
        self.analyzer = TrajectoryAnalyzer()
        self.is_fitted = False
        self.threshold = None
        
        logger.info(f"CPU-optimized IsolationForestDetector initialized "
                   f"(contamination={contamination}, n_estimators={n_estimators})")
    
    def fit(self, trajectories: List[Trajectory]) -> None:
        """Train the Isolation Forest model"""
        if not trajectories:
            raise ValueError("No trajectories provided for training")
        
        # Extract features from trajectories
        features = []
        for trajectory in trajectories:
            if trajectory.is_valid:
                trajectory_features = self.analyzer.analyze_trajectory(trajectory)
                feature_vector = self._extract_feature_vector(trajectory_features)
                features.append(feature_vector)
        
        if not features:
            raise ValueError("No valid trajectories found for training")
        
        features_array = np.array(features)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Train model
        self.model.fit(features_normalized)
        self.is_fitted = True
        
        logger.info(f"IsolationForest trained on {len(features)} trajectories")
    
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Detect anomalies using Isolation Forest"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not trajectory.is_valid:
            return AnomalyResult(
                trajectory_id=trajectory.track_id,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                detection_method="isolation_forest",
                timestamp=datetime.now(),
                details={"error": "Invalid trajectory"}
            )
        
        # Extract features
        trajectory_features = self.analyzer.analyze_trajectory(trajectory)
        feature_vector = self._extract_feature_vector(trajectory_features)
        
        # Normalize features
        feature_normalized = self.scaler.transform([feature_vector])
        
        # Predict
        prediction = self.model.predict(feature_normalized)[0]
        anomaly_score = self.model.decision_function(feature_normalized)[0]
        
        # Convert to probability-like score (0-1 range)
        confidence = max(0.0, min(1.0, (anomaly_score + 0.5) / 1.0))
        
        is_anomaly = prediction == -1
        
        return AnomalyResult(
            trajectory_id=trajectory.track_id,
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            confidence=confidence,
            detection_method="isolation_forest",
            timestamp=datetime.now(),
            details={
                "feature_vector": feature_vector.tolist(),
                "normalized_score": float(anomaly_score)
            }
        )
    
    def _extract_feature_vector(self, features: TrajectoryFeatures) -> np.ndarray:
        """Extract feature vector from TrajectoryFeatures"""
        return np.array([
            features.total_distance,
            features.displacement,
            features.duration,
            features.average_speed,
            features.max_speed,
            features.path_curvature,
            features.direction_changes,
            features.straightness_ratio,
            features.entry_angle,
            features.exit_angle,
            features.bounding_box_area,
            features.path_complexity,
            features.acceleration_variance,
            features.stop_duration,
            features.movement_consistency
        ])
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, path)
        logger.info(f"IsolationForest model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"IsolationForest model loaded from {path}")


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder-based anomaly detector using PyTorch"""
    
    def __init__(self, hidden_dim: int = 8, learning_rate: float = 0.001, epochs: int = 100):
        """
        Initialize Autoencoder detector
        
        Args:
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install torch to use AutoencoderDetector.")
        
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.analyzer = TrajectoryAnalyzer()
        self.is_fitted = False
        self.threshold = None
        
        logger.info(f"AutoencoderDetector initialized (hidden_dim={hidden_dim})")
    
    def fit(self, trajectories: List[Trajectory]) -> None:
        """Train the Autoencoder model"""
        if not trajectories:
            raise ValueError("No trajectories provided for training")
        
        # Extract features
        features = []
        for trajectory in trajectories:
            if trajectory.is_valid:
                trajectory_features = self.analyzer.analyze_trajectory(trajectory)
                feature_vector = self._extract_feature_vector(trajectory_features)
                features.append(feature_vector)
        
        if not features:
            raise ValueError("No valid trajectories found for training")
        
        features_array = np.array(features)
        input_dim = features_array.shape[1]
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Create model
        self.model = self._create_autoencoder(input_dim)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(features_normalized)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_X)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Calculate reconstruction threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.threshold = float(torch.quantile(reconstruction_errors, 0.95))
        
        self.is_fitted = True
        logger.info(f"Autoencoder trained on {len(features)} trajectories, threshold: {self.threshold:.6f}")
    
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Detect anomalies using Autoencoder"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not trajectory.is_valid:
            return AnomalyResult(
                trajectory_id=trajectory.track_id,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                detection_method="autoencoder",
                timestamp=datetime.now(),
                details={"error": "Invalid trajectory"}
            )
        
        # Extract and normalize features
        trajectory_features = self.analyzer.analyze_trajectory(trajectory)
        feature_vector = self._extract_feature_vector(trajectory_features)
        feature_normalized = self.scaler.transform([feature_vector])
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(feature_normalized)
            reconstructed = self.model(X_tensor)
            reconstruction_error = float(torch.mean((X_tensor - reconstructed) ** 2))
        
        is_anomaly = reconstruction_error > self.threshold
        confidence = min(1.0, reconstruction_error / (self.threshold * 2))
        
        return AnomalyResult(
            trajectory_id=trajectory.track_id,
            is_anomaly=is_anomaly,
            anomaly_score=reconstruction_error,
            confidence=confidence,
            detection_method="autoencoder",
            timestamp=datetime.now(),
            details={
                "reconstruction_error": reconstruction_error,
                "threshold": self.threshold
            }
        )
    
    def _create_autoencoder(self, input_dim: int) -> nn.Module:
        """Create autoencoder neural network"""
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return Autoencoder(input_dim, self.hidden_dim)
    
    def _extract_feature_vector(self, features: TrajectoryFeatures) -> np.ndarray:
        """Extract feature vector from TrajectoryFeatures"""
        return np.array([
            features.total_distance,
            features.displacement,
            features.duration,
            features.average_speed,
            features.max_speed,
            features.path_curvature,
            features.direction_changes,
            features.straightness_ratio,
            features.entry_angle,
            features.exit_angle,
            features.bounding_box_area,
            features.path_complexity,
            features.acceleration_variance,
            features.stop_duration,
            features.movement_consistency
        ])
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs
        }
        
        torch.save(model_data, path)
        logger.info(f"Autoencoder model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = torch.load(path)
        
        # Recreate model architecture
        input_dim = len(self._extract_feature_vector(
            # Create dummy features to get input dimension
            type('DummyFeatures', (), {
                'total_distance': 0, 'displacement': 0, 'duration': 0,
                'average_speed': 0, 'max_speed': 0, 'path_curvature': 0,
                'direction_changes': 0, 'straightness_ratio': 0,
                'entry_angle': 0, 'exit_angle': 0, 'bounding_box_area': 0,
                'path_complexity': 0, 'acceleration_variance': 0,
                'stop_duration': 0, 'movement_consistency': 0
            })()
        ))
        
        self.model = self._create_autoencoder(input_dim)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.hidden_dim = model_data['hidden_dim']
        self.learning_rate = model_data['learning_rate']
        self.epochs = model_data['epochs']
        self.is_fitted = True
        
        logger.info(f"Autoencoder model loaded from {path}")


class MotionBasedDetector(AnomalyDetector):
    """Enhanced motion-based anomaly detector with adaptive thresholds"""
    
    def __init__(self, speed_threshold: float = 100.0, direction_change_threshold: int = 5,
                 curvature_threshold: float = 0.5, stop_duration_threshold: float = 5.0,
                 adaptive_thresholds: bool = True):
        """
        Initialize enhanced motion-based detector
        
        Args:
            speed_threshold: Maximum normal speed (pixels/second)
            direction_change_threshold: Maximum normal direction changes
            curvature_threshold: Maximum normal path curvature
            stop_duration_threshold: Maximum normal stop duration (seconds)
            adaptive_thresholds: Whether to adapt thresholds based on training data
        """
        self.speed_threshold = speed_threshold
        self.direction_change_threshold = direction_change_threshold
        self.curvature_threshold = curvature_threshold
        self.stop_duration_threshold = stop_duration_threshold
        self.adaptive_thresholds = adaptive_thresholds
        
        # Adaptive threshold parameters
        self.training_features = []
        self.computed_thresholds = {}
        
        self.analyzer = TrajectoryAnalyzer()
        self.is_fitted = not adaptive_thresholds  # If adaptive, needs training
        
        logger.info(f"Enhanced MotionBasedDetector initialized "
                   f"(adaptive_thresholds={adaptive_thresholds})")
    
    def fit(self, trajectories: List[Trajectory]) -> None:
        """Train adaptive thresholds if enabled"""
        if not self.adaptive_thresholds:
            logger.info("MotionBasedDetector requires no training (adaptive_thresholds=False)")
            return
        
        if not trajectories:
            logger.warning("No trajectories provided for adaptive threshold training")
            self.is_fitted = True
            return
        
        # Extract features from training trajectories
        self.training_features = []
        for trajectory in trajectories:
            if trajectory.is_valid:
                features = self.analyzer.analyze_trajectory(trajectory)
                self.training_features.append(features)
        
        if not self.training_features:
            logger.warning("No valid trajectories found for training")
            self.is_fitted = True
            return
        
        # Compute adaptive thresholds based on training data
        speeds = [f.max_speed for f in self.training_features]
        direction_changes = [f.direction_changes for f in self.training_features]
        curvatures = [f.path_curvature for f in self.training_features]
        stop_durations = [f.stop_duration for f in self.training_features]
        
        # Use 95th percentile as threshold for anomaly detection
        self.computed_thresholds = {
            'speed': np.percentile(speeds, 95) if speeds else self.speed_threshold,
            'direction_changes': np.percentile(direction_changes, 95) if direction_changes else self.direction_change_threshold,
            'curvature': np.percentile(curvatures, 95) if curvatures else self.curvature_threshold,
            'stop_duration': np.percentile(stop_durations, 95) if stop_durations else self.stop_duration_threshold
        }
        
        self.is_fitted = True
        logger.info(f"Adaptive thresholds computed: {self.computed_thresholds}")
    
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Detect anomalies based on enhanced motion rules with adaptive thresholds"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first for adaptive thresholds.")
        
        if not trajectory.is_valid:
            return AnomalyResult(
                trajectory_id=trajectory.track_id,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                detection_method="motion_based",
                timestamp=datetime.now(),
                details={"error": "Invalid trajectory"}
            )
        
        # Analyze trajectory
        features = self.analyzer.analyze_trajectory(trajectory)
        
        # Use adaptive thresholds if available, otherwise use default thresholds
        thresholds = self.computed_thresholds if self.adaptive_thresholds and self.computed_thresholds else {
            'speed': self.speed_threshold,
            'direction_changes': self.direction_change_threshold,
            'curvature': self.curvature_threshold,
            'stop_duration': self.stop_duration_threshold
        }
        
        # Check for anomalies with weighted scoring
        anomaly_reasons = []
        anomaly_scores = {}
        
        # High speed anomaly
        if features.max_speed > thresholds['speed']:
            anomaly_reasons.append("excessive_speed")
            anomaly_scores['speed'] = (features.max_speed - thresholds['speed']) / thresholds['speed']
        
        # Excessive direction changes
        if features.direction_changes > thresholds['direction_changes']:
            anomaly_reasons.append("erratic_movement")
            anomaly_scores['direction_changes'] = (features.direction_changes - thresholds['direction_changes']) / thresholds['direction_changes']
        
        # High path curvature (complex path)
        if features.path_curvature > thresholds['curvature']:
            anomaly_reasons.append("complex_path")
            anomaly_scores['curvature'] = (features.path_curvature - thresholds['curvature']) / thresholds['curvature']
        
        # Excessive stop duration
        if features.stop_duration > thresholds['stop_duration']:
            anomaly_reasons.append("excessive_stopping")
            anomaly_scores['stop_duration'] = (features.stop_duration - thresholds['stop_duration']) / thresholds['stop_duration']
        
        # Very low straightness (zigzag pattern)
        if features.straightness_ratio < 0.3:
            anomaly_reasons.append("zigzag_pattern")
            anomaly_scores['straightness'] = (0.3 - features.straightness_ratio) / 0.3
        
        # Unusual acceleration patterns
        if features.acceleration_variance > 1000:  # High variance in acceleration
            anomaly_reasons.append("erratic_acceleration")
            anomaly_scores['acceleration'] = min(1.0, features.acceleration_variance / 2000)
        
        # Calculate weighted anomaly score
        total_anomaly_score = sum(anomaly_scores.values())
        is_anomaly = len(anomaly_reasons) > 0
        confidence = min(1.0, total_anomaly_score)
        
        return AnomalyResult(
            trajectory_id=trajectory.track_id,
            is_anomaly=is_anomaly,
            anomaly_score=total_anomaly_score,
            confidence=confidence,
            detection_method="motion_based",
            timestamp=datetime.now(),
            details={
                "anomaly_reasons": anomaly_reasons,
                "anomaly_scores": anomaly_scores,
                "thresholds_used": thresholds,
                "features": {
                    "max_speed": features.max_speed,
                    "direction_changes": features.direction_changes,
                    "path_curvature": features.path_curvature,
                    "stop_duration": features.stop_duration,
                    "straightness_ratio": features.straightness_ratio,
                    "acceleration_variance": features.acceleration_variance
                }
            }
        )
    
    def save_model(self, path: str) -> None:
        """Save detector configuration and adaptive thresholds"""
        config = {
            'speed_threshold': self.speed_threshold,
            'direction_change_threshold': self.direction_change_threshold,
            'curvature_threshold': self.curvature_threshold,
            'stop_duration_threshold': self.stop_duration_threshold,
            'adaptive_thresholds': self.adaptive_thresholds,
            'computed_thresholds': self.computed_thresholds,
            'training_features_count': len(self.training_features) if self.training_features else 0
        }
        joblib.dump(config, path)
        logger.info(f"Enhanced MotionBasedDetector config saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load detector configuration and adaptive thresholds"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        config = joblib.load(path)
        self.speed_threshold = config['speed_threshold']
        self.direction_change_threshold = config['direction_change_threshold']
        self.curvature_threshold = config.get('curvature_threshold', 0.5)
        self.stop_duration_threshold = config.get('stop_duration_threshold', 5.0)
        self.adaptive_thresholds = config.get('adaptive_thresholds', True)
        self.computed_thresholds = config.get('computed_thresholds', {})
        self.is_fitted = True
        
        logger.info(f"Enhanced MotionBasedDetector config loaded from {path}")


class PyODEnsembleDetector(AnomalyDetector):
    """PyOD-based ensemble detector for baseline comparison"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42, 
                 combination_method: str = 'average'):
        """
        Initialize PyOD ensemble detector
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            combination_method: Method to combine detector scores ('average', 'max', 'aom', 'moa')
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD not available. Install pyod to use PyODEnsembleDetector.")
        
        self.contamination = contamination
        self.random_state = random_state
        self.combination_method = combination_method
        
        # Initialize individual detectors
        self.detectors = {
            'iforest': IForest(contamination=contamination, random_state=random_state, n_estimators=50),
            'lof': LOF(contamination=contamination, n_neighbors=10),
            'ocsvm': OCSVM(contamination=contamination, kernel='rbf', gamma='scale')
        }
        
        self.scaler = StandardScaler()
        self.analyzer = TrajectoryAnalyzer()
        self.is_fitted = False
        self.decision_scores_ = None
        
        logger.info(f"PyODEnsembleDetector initialized with {len(self.detectors)} detectors")
    
    def fit(self, trajectories: List[Trajectory]) -> None:
        """Train all PyOD detectors in the ensemble"""
        if not trajectories:
            raise ValueError("No trajectories provided for training")
        
        # Extract features from trajectories
        features = []
        for trajectory in trajectories:
            if trajectory.is_valid:
                trajectory_features = self.analyzer.analyze_trajectory(trajectory)
                feature_vector = self._extract_feature_vector(trajectory_features)
                features.append(feature_vector)
        
        if not features:
            raise ValueError("No valid trajectories found for training")
        
        features_array = np.array(features)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Train all detectors
        detector_scores = {}
        for name, detector in self.detectors.items():
            logger.info(f"Training {name} detector...")
            detector.fit(features_normalized)
            detector_scores[name] = detector.decision_scores_
        
        # Store decision scores for combination
        self.decision_scores_ = detector_scores
        self.is_fitted = True
        
        logger.info(f"PyODEnsemble trained on {len(features)} trajectories")
    
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Detect anomalies using PyOD ensemble"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not trajectory.is_valid:
            return AnomalyResult(
                trajectory_id=trajectory.track_id,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                detection_method="pyod_ensemble",
                timestamp=datetime.now(),
                details={"error": "Invalid trajectory"}
            )
        
        # Extract and normalize features
        trajectory_features = self.analyzer.analyze_trajectory(trajectory)
        feature_vector = self._extract_feature_vector(trajectory_features)
        feature_normalized = self.scaler.transform([feature_vector])
        
        # Get predictions from all detectors
        individual_scores = {}
        individual_predictions = {}
        
        for name, detector in self.detectors.items():
            score = detector.decision_function(feature_normalized)[0]
            prediction = detector.predict(feature_normalized)[0]
            individual_scores[name] = float(score)
            individual_predictions[name] = bool(prediction == 1)
        
        # Combine scores using specified method
        scores_array = np.array(list(individual_scores.values())).reshape(1, -1)
        
        if self.combination_method == 'average':
            combined_score = float(np.mean(scores_array))
        elif self.combination_method == 'max':
            combined_score = float(np.max(scores_array))
        elif self.combination_method == 'aom':
            combined_score = float(aom(scores_array, n_buckets=3)[0])
        elif self.combination_method == 'moa':
            combined_score = float(moa(scores_array, n_buckets=3)[0])
        else:
            combined_score = float(np.mean(scores_array))
        
        # Determine if anomaly based on combined score and contamination threshold
        # Use the average threshold from training data
        avg_threshold = np.mean([
            np.percentile(scores, (1 - self.contamination) * 100)
            for scores in self.decision_scores_.values()
        ])
        
        is_anomaly = combined_score > avg_threshold
        confidence = min(1.0, max(0.0, combined_score / (avg_threshold * 2)))
        
        return AnomalyResult(
            trajectory_id=trajectory.track_id,
            is_anomaly=is_anomaly,
            anomaly_score=combined_score,
            confidence=confidence,
            detection_method="pyod_ensemble",
            timestamp=datetime.now(),
            details={
                "individual_scores": individual_scores,
                "individual_predictions": individual_predictions,
                "combination_method": self.combination_method,
                "threshold": float(avg_threshold)
            }
        )
    
    def _extract_feature_vector(self, features: TrajectoryFeatures) -> np.ndarray:
        """Extract feature vector from TrajectoryFeatures"""
        return np.array([
            features.total_distance,
            features.displacement,
            features.duration,
            features.average_speed,
            features.max_speed,
            features.path_curvature,
            features.direction_changes,
            features.straightness_ratio,
            features.entry_angle,
            features.exit_angle,
            features.bounding_box_area,
            features.path_complexity,
            features.acceleration_variance,
            features.stop_duration,
            features.movement_consistency
        ])
    
    def save_model(self, path: str) -> None:
        """Save the trained ensemble model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        model_data = {
            'detectors': self.detectors,
            'scaler': self.scaler,
            'decision_scores': self.decision_scores_,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'combination_method': self.combination_method
        }
        
        joblib.dump(model_data, path)
        logger.info(f"PyODEnsemble model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained ensemble model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        self.detectors = model_data['detectors']
        self.scaler = model_data['scaler']
        self.decision_scores_ = model_data['decision_scores']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.combination_method = model_data['combination_method']
        self.is_fitted = True
        
        logger.info(f"PyODEnsemble model loaded from {path}")


class EnsembleDetector(AnomalyDetector):
    """Ensemble detector combining multiple detection methods"""
    
    def __init__(self, detectors: List[AnomalyDetector], weights: Optional[List[float]] = None):
        """
        Initialize ensemble detector
        
        Args:
            detectors: List of anomaly detectors
            weights: Optional weights for each detector
        """
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        
        if len(self.weights) != len(self.detectors):
            raise ValueError("Number of weights must match number of detectors")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"EnsembleDetector initialized with {len(detectors)} detectors")
    
    def fit(self, trajectories: List[Trajectory]) -> None:
        """Train all detectors in the ensemble"""
        for i, detector in enumerate(self.detectors):
            logger.info(f"Training detector {i+1}/{len(self.detectors)}: {type(detector).__name__}")
            detector.fit(trajectories)
    
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Predict using ensemble of detectors"""
        results = []
        for detector in self.detectors:
            result = detector.predict(trajectory)
            results.append(result)
        
        # Combine results using weighted voting
        weighted_score = sum(r.anomaly_score * w for r, w in zip(results, self.weights))
        weighted_confidence = sum(r.confidence * w for r, w in zip(results, self.weights))
        
        # Majority voting for binary decision
        anomaly_votes = sum(1 for r in results if r.is_anomaly)
        is_anomaly = anomaly_votes > len(results) / 2
        
        return AnomalyResult(
            trajectory_id=trajectory.track_id,
            is_anomaly=is_anomaly,
            anomaly_score=weighted_score,
            confidence=weighted_confidence,
            detection_method="ensemble",
            timestamp=datetime.now(),
            details={
                "individual_results": [
                    {
                        "method": r.detection_method,
                        "is_anomaly": r.is_anomaly,
                        "score": r.anomaly_score,
                        "confidence": r.confidence
                    }
                    for r in results
                ],
                "weights": self.weights,
                "anomaly_votes": anomaly_votes
            }
        )
    
    def save_model(self, path: str) -> None:
        """Save all models in the ensemble"""
        os.makedirs(path, exist_ok=True)
        
        for i, detector in enumerate(self.detectors):
            detector_path = os.path.join(path, f"detector_{i}_{type(detector).__name__}.pkl")
            detector.save_model(detector_path)
        
        # Save ensemble configuration
        config = {
            'detector_types': [type(d).__name__ for d in self.detectors],
            'weights': self.weights
        }
        config_path = os.path.join(path, "ensemble_config.pkl")
        joblib.dump(config, config_path)
        
        logger.info(f"EnsembleDetector saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load all models in the ensemble"""
        config_path = os.path.join(path, "ensemble_config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Ensemble config not found: {config_path}")
        
        config = joblib.load(config_path)
        self.weights = config['weights']
        
        # Load individual detectors
        for i, detector in enumerate(self.detectors):
            detector_path = os.path.join(path, f"detector_{i}_{type(detector).__name__}.pkl")
            detector.load_model(detector_path)
        
        logger.info(f"EnsembleDetector loaded from {path}")