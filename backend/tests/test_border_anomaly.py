"""
Unit tests for border anomaly detection pipeline.
Tests trajectory extraction, analysis, and anomaly detection.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from pipelines.border_anomaly.trajectory import (
    TrajectoryPoint, Trajectory, TrajectoryExtractor, TrajectoryAnalyzer,
    TrajectoryVisualizer, create_sample_trajectory
)
from pipelines.border_anomaly.anomaly_detector import (
    IsolationForestDetector, MotionBasedDetector, AnomalyResult,
    PyODEnsembleDetector, create_synthetic_anomaly_data
)
from pipelines.border_anomaly.pipeline import BorderAnomalyPipeline, DEFAULT_CONFIG
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("test_border_anomaly")


class TestTrajectoryPoint(unittest.TestCase):
    """Test TrajectoryPoint dataclass"""
    
    def test_trajectory_point_creation(self):
        """Test creating a trajectory point"""
        timestamp = datetime.now()
        point = TrajectoryPoint(
            x=100.0,
            y=200.0,
            timestamp=timestamp,
            frame_number=1,
            confidence=0.9
        )
        
        self.assertEqual(point.x, 100.0)
        self.assertEqual(point.y, 200.0)
        self.assertEqual(point.timestamp, timestamp)
        self.assertEqual(point.frame_number, 1)
        self.assertEqual(point.confidence, 0.9)


class TestTrajectory(unittest.TestCase):
    """Test Trajectory dataclass"""
    
    def setUp(self):
        """Set up test trajectory"""
        self.start_time = datetime.now()
        self.points = [
            TrajectoryPoint(0, 0, self.start_time, 0),
            TrajectoryPoint(10, 10, self.start_time + timedelta(seconds=1), 1),
            TrajectoryPoint(20, 20, self.start_time + timedelta(seconds=2), 2),
        ]
        self.trajectory = Trajectory(
            track_id=1,
            points=self.points,
            start_time=self.start_time,
            end_time=self.start_time + timedelta(seconds=2)
        )
    
    def test_trajectory_properties(self):
        """Test trajectory properties"""
        self.assertEqual(self.trajectory.duration, 2.0)
        self.assertEqual(self.trajectory.length, 3)
        self.assertTrue(self.trajectory.is_valid)
    
    def test_invalid_trajectory(self):
        """Test trajectory with insufficient points"""
        short_trajectory = Trajectory(
            track_id=2,
            points=self.points[:2],  # Only 2 points
            start_time=self.start_time,
            end_time=self.start_time + timedelta(seconds=1)
        )
        self.assertFalse(short_trajectory.is_valid)


class TestTrajectoryExtractor(unittest.TestCase):
    """Test TrajectoryExtractor class"""
    
    def setUp(self):
        """Set up trajectory extractor"""
        self.extractor = TrajectoryExtractor(min_trajectory_length=3, max_gap_frames=5)
        self.timestamp = datetime.now()
    
    def test_trajectory_extraction(self):
        """Test basic trajectory extraction"""
        # Simulate tracked objects over multiple frames
        tracked_objects_frame1 = [
            {'track_id': 1, 'bbox': [100, 100, 150, 150], 'confidence': 0.9}
        ]
        tracked_objects_frame2 = [
            {'track_id': 1, 'bbox': [110, 110, 160, 160], 'confidence': 0.8}
        ]
        tracked_objects_frame3 = [
            {'track_id': 1, 'bbox': [120, 120, 170, 170], 'confidence': 0.85}
        ]
        
        # Update with frames
        completed1 = self.extractor.update(tracked_objects_frame1, 1, self.timestamp)
        completed2 = self.extractor.update(tracked_objects_frame2, 2, self.timestamp + timedelta(seconds=1))
        completed3 = self.extractor.update(tracked_objects_frame3, 3, self.timestamp + timedelta(seconds=2))
        
        # No trajectories should be completed yet
        self.assertEqual(len(completed1), 0)
        self.assertEqual(len(completed2), 0)
        self.assertEqual(len(completed3), 0)
        
        # Check active trajectories
        active = self.extractor.get_active_trajectories()
        self.assertEqual(len(active), 1)
        self.assertEqual(len(active[1]), 3)  # 3 points for track_id 1
    
    def test_trajectory_completion(self):
        """Test trajectory completion when track disappears"""
        # Add points for a trajectory
        for i in range(5):
            tracked_objects = [
                {'track_id': 1, 'bbox': [100+i*10, 100+i*10, 150+i*10, 150+i*10]}
            ]
            self.extractor.update(tracked_objects, i, self.timestamp + timedelta(seconds=i))
        
        # Track disappears - simulate gap larger than max_gap_frames
        completed = self.extractor.update([], 10, self.timestamp + timedelta(seconds=10))
        
        # Should complete the trajectory
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].track_id, 1)
        self.assertEqual(len(completed[0].points), 5)
    
    def test_force_complete_all(self):
        """Test forcing completion of all active trajectories"""
        # Add some active trajectories
        for i in range(3):
            tracked_objects = [
                {'track_id': 1, 'bbox': [100+i*10, 100, 150+i*10, 150]},
                {'track_id': 2, 'bbox': [200, 100+i*10, 250, 150+i*10]}
            ]
            self.extractor.update(tracked_objects, i, self.timestamp + timedelta(seconds=i))
        
        # Force complete all
        completed = self.extractor.force_complete_all()
        
        # Should complete both trajectories
        self.assertEqual(len(completed), 2)
        track_ids = {t.track_id for t in completed}
        self.assertEqual(track_ids, {1, 2})


class TestTrajectoryAnalyzer(unittest.TestCase):
    """Test TrajectoryAnalyzer class"""
    
    def setUp(self):
        """Set up trajectory analyzer"""
        self.analyzer = TrajectoryAnalyzer(frame_width=1920, frame_height=1080, fps=30.0)
        self.trajectory = create_sample_trajectory(track_id=1, num_points=10)
    
    def test_analyze_trajectory(self):
        """Test trajectory analysis"""
        features = self.analyzer.analyze_trajectory(self.trajectory)
        
        # Check that all features are computed
        self.assertIsInstance(features.total_distance, float)
        self.assertIsInstance(features.displacement, float)
        self.assertIsInstance(features.duration, float)
        self.assertIsInstance(features.average_speed, float)
        self.assertIsInstance(features.max_speed, float)
        self.assertIsInstance(features.path_curvature, float)
        self.assertIsInstance(features.direction_changes, int)
        self.assertIsInstance(features.straightness_ratio, float)
        
        # Check reasonable values
        self.assertGreaterEqual(features.total_distance, 0)
        self.assertGreaterEqual(features.displacement, 0)
        self.assertGreater(features.duration, 0)
        self.assertGreaterEqual(features.average_speed, 0)
        self.assertGreaterEqual(features.straightness_ratio, 0)
        self.assertLessEqual(features.straightness_ratio, 1)
    
    def test_straight_line_trajectory(self):
        """Test analysis of straight line trajectory"""
        # Create straight line trajectory
        start_time = datetime.now()
        points = []
        for i in range(5):
            point = TrajectoryPoint(
                x=100 + i * 50,  # Straight line in x
                y=200,           # Constant y
                timestamp=start_time + timedelta(seconds=i),
                frame_number=i
            )
            points.append(point)
        
        trajectory = Trajectory(
            track_id=1,
            points=points,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=4)
        )
        
        features = self.analyzer.analyze_trajectory(trajectory)
        
        # Straight line should have high straightness ratio
        self.assertGreater(features.straightness_ratio, 0.9)
        # Should have minimal direction changes
        self.assertLessEqual(features.direction_changes, 1)
    
    def test_invalid_trajectory_analysis(self):
        """Test analysis of invalid trajectory"""
        # Create trajectory with insufficient points
        invalid_trajectory = Trajectory(
            track_id=1,
            points=[TrajectoryPoint(0, 0, datetime.now(), 0)],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_trajectory(invalid_trajectory)


class TestMotionBasedDetector(unittest.TestCase):
    """Test MotionBasedDetector class"""
    
    def setUp(self):
        """Set up motion-based detector"""
        self.detector = MotionBasedDetector(
            speed_threshold=50.0,
            direction_change_threshold=3
        )
    
    def test_normal_trajectory(self):
        """Test detection on normal trajectory"""
        trajectory = create_sample_trajectory(track_id=1, num_points=5)
        result = self.detector.predict(trajectory)
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.trajectory_id, 1)
        self.assertEqual(result.detection_method, "motion_based")
        # Normal trajectory should not be anomalous
        self.assertFalse(result.is_anomaly)
    
    def test_high_speed_anomaly(self):
        """Test detection of high-speed anomaly"""
        # Create high-speed trajectory
        start_time = datetime.now()
        points = []
        for i in range(5):
            point = TrajectoryPoint(
                x=i * 200,  # Large jumps = high speed
                y=100,
                timestamp=start_time + timedelta(milliseconds=i * 100),  # Short time intervals
                frame_number=i
            )
            points.append(point)
        
        trajectory = Trajectory(
            track_id=1,
            points=points,
            start_time=start_time,
            end_time=start_time + timedelta(milliseconds=400)
        )
        
        result = self.detector.predict(trajectory)
        
        # Should detect as anomaly due to high speed
        self.assertTrue(result.is_anomaly)
        self.assertIn("excessive_speed", result.details.get("anomaly_reasons", []))
    
    def test_save_load_model(self):
        """Test saving and loading detector configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "motion_detector.pkl")
            
            # Save model
            self.detector.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new detector and load
            new_detector = MotionBasedDetector()
            new_detector.load_model(model_path)
            
            # Check configuration was loaded
            self.assertEqual(new_detector.speed_threshold, 50.0)
            self.assertEqual(new_detector.direction_change_threshold, 3)


class TestIsolationForestDetector(unittest.TestCase):
    """Test IsolationForestDetector class"""
    
    def setUp(self):
        """Set up isolation forest detector"""
        self.detector = IsolationForestDetector(contamination=0.1)
    
    def test_training_and_prediction(self):
        """Test training and prediction"""
        # Create training trajectories
        training_trajectories = []
        for i in range(20):
            trajectory = create_sample_trajectory(track_id=i, num_points=8)
            training_trajectories.append(trajectory)
        
        # Train detector
        self.detector.fit(training_trajectories)
        self.assertTrue(self.detector.is_fitted)
        
        # Test prediction
        test_trajectory = create_sample_trajectory(track_id=100, num_points=8)
        result = self.detector.predict(test_trajectory)
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.detection_method, "isolation_forest")
        self.assertIsInstance(result.is_anomaly, bool)
        self.assertIsInstance(result.anomaly_score, float)
    
    def test_insufficient_training_data(self):
        """Test error handling with insufficient training data"""
        with self.assertRaises(ValueError):
            self.detector.fit([])
    
    def test_prediction_without_training(self):
        """Test error handling when predicting without training"""
        trajectory = create_sample_trajectory(track_id=1, num_points=5)
        
        with self.assertRaises(ValueError):
            self.detector.predict(trajectory)


class TestPyODEnsembleDetector(unittest.TestCase):
    """Test PyODEnsembleDetector class"""
    
    def setUp(self):
        """Set up PyOD ensemble detector"""
        try:
            self.detector = PyODEnsembleDetector(contamination=0.2)
            self.pyod_available = True
        except ImportError:
            self.pyod_available = False
            self.skipTest("PyOD not available")
    
    def test_training_and_prediction(self):
        """Test training and prediction with synthetic data"""
        if not self.pyod_available:
            self.skipTest("PyOD not available")
        
        # Create synthetic training data
        trajectories, labels = create_synthetic_anomaly_data(num_normal=30, num_anomalies=5)
        normal_trajectories = [t for t, l in zip(trajectories, labels) if not l]
        
        # Train detector on normal data only
        self.detector.fit(normal_trajectories)
        self.assertTrue(self.detector.is_fitted)
        
        # Test prediction on all data
        results = []
        for trajectory in trajectories:
            result = self.detector.predict(trajectory)
            results.append(result)
            self.assertIsInstance(result, AnomalyResult)
            self.assertEqual(result.detection_method, "pyod_ensemble")
        
        # Check that some anomalies are detected
        detected_anomalies = sum(1 for r in results if r.is_anomaly)
        self.assertGreater(detected_anomalies, 0)
    
    def test_combination_methods(self):
        """Test different combination methods"""
        if not self.pyod_available:
            self.skipTest("PyOD not available")
        
        methods = ['average', 'max', 'aom', 'moa']
        trajectories, _ = create_synthetic_anomaly_data(num_normal=20, num_anomalies=3)
        normal_trajectories = [t for t, l in zip(trajectories, _) if not l]
        
        for method in methods:
            detector = PyODEnsembleDetector(combination_method=method)
            detector.fit(normal_trajectories)
            
            # Test prediction
            test_trajectory = trajectories[0]
            result = detector.predict(test_trajectory)
            self.assertIsInstance(result, AnomalyResult)
            self.assertIn('combination_method', result.details)
            self.assertEqual(result.details['combination_method'], method)


class TestEnhancedMotionBasedDetector(unittest.TestCase):
    """Test enhanced MotionBasedDetector with adaptive thresholds"""
    
    def setUp(self):
        """Set up enhanced motion-based detector"""
        self.detector = MotionBasedDetector(adaptive_thresholds=True)
    
    def test_adaptive_threshold_training(self):
        """Test adaptive threshold computation"""
        # Create training data
        trajectories, _ = create_synthetic_anomaly_data(num_normal=20, num_anomalies=0)
        
        # Train detector
        self.detector.fit(trajectories)
        self.assertTrue(self.detector.is_fitted)
        self.assertIsInstance(self.detector.computed_thresholds, dict)
        
        # Check that thresholds were computed
        expected_keys = ['speed', 'direction_changes', 'curvature', 'stop_duration']
        for key in expected_keys:
            self.assertIn(key, self.detector.computed_thresholds)
            self.assertIsInstance(self.detector.computed_thresholds[key], (int, float))
    
    def test_enhanced_anomaly_detection(self):
        """Test enhanced anomaly detection with multiple criteria"""
        # Create synthetic data with known anomalies
        trajectories, labels = create_synthetic_anomaly_data(num_normal=15, num_anomalies=5)
        normal_trajectories = [t for t, l in zip(trajectories, labels) if not l]
        
        # Train on normal data
        self.detector.fit(normal_trajectories)
        
        # Test on all data
        results = []
        for trajectory, is_anomaly in zip(trajectories, labels):
            result = self.detector.predict(trajectory)
            results.append((result, is_anomaly))
            
            # Check result structure
            self.assertIsInstance(result, AnomalyResult)
            self.assertIn('anomaly_reasons', result.details)
            self.assertIn('thresholds_used', result.details)
            self.assertIn('features', result.details)
        
        # Calculate detection performance
        true_positives = sum(1 for r, l in results if r.is_anomaly and l)
        false_positives = sum(1 for r, l in results if r.is_anomaly and not l)
        true_negatives = sum(1 for r, l in results if not r.is_anomaly and not l)
        false_negatives = sum(1 for r, l in results if not r.is_anomaly and l)
        
        # Should detect at least some anomalies
        self.assertGreater(true_positives, 0)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        logger.info(f"Enhanced MotionBasedDetector - Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    def test_non_adaptive_mode(self):
        """Test detector in non-adaptive mode"""
        detector = MotionBasedDetector(adaptive_thresholds=False)
        self.assertTrue(detector.is_fitted)  # Should be fitted immediately
        
        # Should work without training
        trajectory = create_sample_trajectory(track_id=1, num_points=10)
        result = detector.predict(trajectory)
        self.assertIsInstance(result, AnomalyResult)


class TestCPUOptimizedIsolationForest(unittest.TestCase):
    """Test CPU-optimized IsolationForestDetector"""
    
    def setUp(self):
        """Set up CPU-optimized detector"""
        self.detector = IsolationForestDetector(
            n_estimators=25,  # Reduced for faster testing
            n_jobs=1  # Single thread for testing
        )
    
    def test_cpu_optimization_parameters(self):
        """Test that CPU optimization parameters are set correctly"""
        self.assertEqual(self.detector.n_estimators, 25)
        self.assertEqual(self.detector.n_jobs, 1)
        self.assertFalse(self.detector.model.bootstrap)
        self.assertFalse(self.detector.model.warm_start)
    
    def test_performance_with_synthetic_data(self):
        """Test performance on synthetic anomaly data"""
        # Create larger dataset for better evaluation
        trajectories, labels = create_synthetic_anomaly_data(num_normal=40, num_anomalies=10)
        normal_trajectories = [t for t, l in zip(trajectories, labels) if not l]
        
        # Train detector
        start_time = datetime.now()
        self.detector.fit(normal_trajectories)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Test prediction speed
        start_time = datetime.now()
        results = []
        for trajectory in trajectories:
            result = self.detector.predict(trajectory)
            results.append(result)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate performance metrics
        true_labels = labels
        predicted_labels = [r.is_anomaly for r in results]
        
        true_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t and p)
        false_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and p)
        true_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and not p)
        false_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if t and not p)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        logger.info(f"CPU-optimized IsolationForest - Training time: {training_time:.3f}s, "
                   f"Prediction time: {prediction_time:.3f}s, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        # Should achieve reasonable performance
        self.assertGreater(recall, 0.3)  # At least 30% recall


class TestSyntheticAnomalyData(unittest.TestCase):
    """Test synthetic anomaly data generation"""
    
    def test_data_generation(self):
        """Test synthetic data generation function"""
        trajectories, labels = create_synthetic_anomaly_data(num_normal=10, num_anomalies=5)
        
        # Check correct number of trajectories
        self.assertEqual(len(trajectories), 15)
        self.assertEqual(len(labels), 15)
        
        # Check label distribution
        normal_count = sum(1 for l in labels if not l)
        anomaly_count = sum(1 for l in labels if l)
        self.assertEqual(normal_count, 10)
        self.assertEqual(anomaly_count, 5)
        
        # Check trajectory validity
        for trajectory in trajectories:
            self.assertIsInstance(trajectory, Trajectory)
            self.assertGreater(len(trajectory.points), 0)
            self.assertTrue(trajectory.track_id >= 0)
    
    def test_anomaly_types(self):
        """Test that different anomaly types are generated"""
        # Generate more anomalies to increase chance of different types
        trajectories, labels = create_synthetic_anomaly_data(num_normal=5, num_anomalies=20)
        anomaly_trajectories = [t for t, l in zip(trajectories, labels) if l]
        
        # Analyze trajectories to check for different anomaly characteristics
        analyzer = TrajectoryAnalyzer()
        features = []
        for trajectory in anomaly_trajectories:
            if trajectory.is_valid:
                feature = analyzer.analyze_trajectory(trajectory)
                features.append(feature)
        
        # Check for variety in features (indicating different anomaly types)
        speeds = [f.max_speed for f in features]
        direction_changes = [f.direction_changes for f in features]
        durations = [f.duration for f in features]
        
        # Should have variety in characteristics
        self.assertGreater(max(speeds) - min(speeds), 50)  # Speed variation
        self.assertGreater(max(direction_changes) - min(direction_changes), 5)  # Direction change variation


class TestBorderAnomalyPipeline(unittest.TestCase):
    """Test BorderAnomalyPipeline class"""
    
    def setUp(self):
        """Set up pipeline"""
        config = DEFAULT_CONFIG.copy()
        config['detection_method'] = 'motion_based'  # Use simple detector for testing
        config['enable_alerts'] = False  # Disable alerts for testing
        self.pipeline = BorderAnomalyPipeline(config)
    
    @patch('aiohttp.ClientSession.post')
    async def test_process_frame(self, mock_post):
        """Test frame processing"""
        frame_data = {
            'frame_number': 1,
            'timestamp': datetime.now().isoformat(),
            'tracked_objects': [
                {'track_id': 1, 'bbox': [100, 100, 150, 150], 'confidence': 0.9}
            ]
        }
        
        alerts = await self.pipeline.process_frame(frame_data)
        
        # Should not generate alerts without training or completed trajectories
        self.assertEqual(len(alerts), 0)
    
    async def test_get_statistics(self):
        """Test getting pipeline statistics"""
        stats = await self.pipeline.get_statistics()
        
        self.assertIn('active_trajectories', stats)
        self.assertIn('completed_trajectories', stats)
        self.assertIn('is_trained', stats)
        self.assertIn('detection_method', stats)
        self.assertEqual(stats['detection_method'], 'motion_based')
    
    async def test_cleanup(self):
        """Test pipeline cleanup"""
        # Add some test data
        frame_data = {
            'frame_number': 1,
            'timestamp': datetime.now().isoformat(),
            'tracked_objects': [
                {'track_id': 1, 'bbox': [100, 100, 150, 150]}
            ]
        }
        await self.pipeline.process_frame(frame_data)
        
        # Cleanup
        await self.pipeline.cleanup()
        
        # Check that training trajectories are cleared
        self.assertEqual(len(self.pipeline.training_trajectories), 0)


class TestTrajectoryVisualizer(unittest.TestCase):
    """Test TrajectoryVisualizer class"""
    
    def setUp(self):
        """Set up visualizer"""
        self.visualizer = TrajectoryVisualizer(frame_width=1920, frame_height=1080)
        self.trajectory = create_sample_trajectory(track_id=1, num_points=10)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_trajectory(self, mock_close, mock_show):
        """Test plotting single trajectory"""
        self.visualizer.plot_trajectory(self.trajectory)
        
        # Check that matplotlib functions were called
        mock_show.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_multiple_trajectories(self, mock_close, mock_show):
        """Test plotting multiple trajectories"""
        trajectories = [
            create_sample_trajectory(track_id=1, num_points=8),
            create_sample_trajectory(track_id=2, num_points=6),
            create_sample_trajectory(track_id=3, num_points=10)
        ]
        
        self.visualizer.plot_multiple_trajectories(trajectories)
        
        # Check that matplotlib functions were called
        mock_show.assert_called_once()
        mock_close.assert_called_once()
    
    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectories"""
        empty_trajectory = Trajectory(
            track_id=1,
            points=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Should not raise exception
        with patch('matplotlib.pyplot.show'):
            self.visualizer.plot_trajectory(empty_trajectory)


class TestSampleTrajectoryGeneration(unittest.TestCase):
    """Test sample trajectory generation"""
    
    def test_create_sample_trajectory(self):
        """Test creating sample trajectory"""
        trajectory = create_sample_trajectory(track_id=5, num_points=15)
        
        self.assertEqual(trajectory.track_id, 5)
        self.assertEqual(len(trajectory.points), 15)
        self.assertTrue(trajectory.is_valid)
        
        # Check that points have reasonable values
        for point in trajectory.points:
            self.assertGreaterEqual(point.x, 0)
            self.assertLessEqual(point.x, 1920)
            self.assertGreaterEqual(point.y, 0)
            self.assertLessEqual(point.y, 1080)
            self.assertGreaterEqual(point.confidence, 0.8)
            self.assertLessEqual(point.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()