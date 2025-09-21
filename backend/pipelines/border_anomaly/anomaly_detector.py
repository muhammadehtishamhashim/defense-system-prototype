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

from .trajectory import Trajectory, TrajectoryFeatures, TrajectoryAnalyzer
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("border_anomaly")


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
    """Isolation Forest-based anomaly detector"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.analyzer = TrajectoryAnalyzer()
        self.is_fitted = False
        
        logger.info(f"IsolationForestDetector initialized (contamination={contamination})")
    
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
    """Simple motion-based anomaly detector as fallback"""
    
    def __init__(self, speed_threshold: float = 100.0, direction_change_threshold: int = 5):
        """
        Initialize motion-based detector
        
        Args:
            speed_threshold: Maximum normal speed (pixels/second)
            direction_change_threshold: Maximum normal direction changes
        """
        self.speed_threshold = speed_threshold
        self.direction_change_threshold = direction_change_threshold
        self.analyzer = TrajectoryAnalyzer()
        self.is_fitted = True  # No training required
        
        logger.info(f"MotionBasedDetector initialized (speed_threshold={speed_threshold})")
    
    def fit(self, trajectories: List[Trajectory]) -> None:
        """No training required for motion-based detector"""
        logger.info("MotionBasedDetector requires no training")
    
    def predict(self, trajectory: Trajectory) -> AnomalyResult:
        """Detect anomalies based on simple motion rules"""
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
        
        # Check for anomalies
        anomaly_reasons = []
        anomaly_score = 0.0
        
        # High speed anomaly
        if features.max_speed > self.speed_threshold:
            anomaly_reasons.append("excessive_speed")
            anomaly_score += (features.max_speed - self.speed_threshold) / self.speed_threshold
        
        # Excessive direction changes
        if features.direction_changes > self.direction_change_threshold:
            anomaly_reasons.append("erratic_movement")
            anomaly_score += (features.direction_changes - self.direction_change_threshold) / self.direction_change_threshold
        
        # Very low straightness (zigzag pattern)
        if features.straightness_ratio < 0.3:
            anomaly_reasons.append("zigzag_pattern")
            anomaly_score += (0.3 - features.straightness_ratio) / 0.3
        
        is_anomaly = len(anomaly_reasons) > 0
        confidence = min(1.0, anomaly_score)
        
        return AnomalyResult(
            trajectory_id=trajectory.track_id,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            detection_method="motion_based",
            timestamp=datetime.now(),
            details={
                "anomaly_reasons": anomaly_reasons,
                "max_speed": features.max_speed,
                "direction_changes": features.direction_changes,
                "straightness_ratio": features.straightness_ratio
            }
        )
    
    def save_model(self, path: str) -> None:
        """Save detector configuration"""
        config = {
            'speed_threshold': self.speed_threshold,
            'direction_change_threshold': self.direction_change_threshold
        }
        joblib.dump(config, path)
        logger.info(f"MotionBasedDetector config saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load detector configuration"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        config = joblib.load(path)
        self.speed_threshold = config['speed_threshold']
        self.direction_change_threshold = config['direction_change_threshold']
        
        logger.info(f"MotionBasedDetector config loaded from {path}")


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