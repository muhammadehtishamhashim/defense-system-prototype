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
    IsolationForestDetector, MotionBasedDetector, AnomalyResult
)
from pipelines.border_anomaly.pipeline import BorderAnomalyPipeline, DEFAULT_CONFIG


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