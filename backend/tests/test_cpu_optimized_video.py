"""
Tests for CPU-optimized video surveillance pipeline.
Tests YOLOv8 Nano detection, ByteTrack tracking, and ONNX optimization.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from pipelines.video_surveillance.detection import (
    CPUOptimizedDetector,
    ByteTrackTracker,
    CPUOptimizedVideoProcessor,
    CPUOptimizedVideoSource
)
from pipelines.video_surveillance.analysis import VideoAnalysisPipeline


class TestCPUOptimizedDetector:
    """Test CPU-optimized YOLOv8 detector"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create a mock detector to avoid downloading models in tests
        self.detector = CPUOptimizedDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5,
            use_onnx=False,  # Disable ONNX for tests
            input_size=320  # Smaller size for faster tests
        )

    def test_detector_initialization(self):
        """Test detector initialization"""
        assert self.detector.confidence_threshold == 0.5
        assert self.detector.input_size == 320
        assert self.detector.device == 'cpu'
        assert self.detector.frame_skip == 2

    def test_frame_skipping(self):
        """Test frame skipping functionality"""
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the model to avoid actual inference
        with patch.object(self.detector, 'model') as mock_model:
            mock_model.return_value = [Mock(boxes=Mock(
                xyxy=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([]))))),
                conf=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([]))))),
                cls=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([])))))
            ))]
            
            # First frame should be processed
            detections1 = self.detector.detect(test_frame)
            
            # Second frame should return cached results (frame skipping)
            detections2 = self.detector.detect(test_frame)
            
            # Results should be the same due to frame skipping
            assert detections1 == detections2

    @patch('torch.no_grad')
    def test_pytorch_inference(self, mock_no_grad):
        """Test PyTorch inference fallback"""
        test_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        
        # Mock PyTorch model
        mock_result = Mock()
        mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
        mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        
        with patch.object(self.detector, 'model') as mock_model:
            mock_model.return_value = [mock_result]
            self.detector.class_names = {0: 'person'}
            
            detections = self.detector._detect_pytorch(test_frame, 640, 480)
            
            assert len(detections) == 1
            assert detections[0]['class_name'] == 'person'
            assert detections[0]['confidence'] == 0.8

    def test_frame_preprocessing(self):
        """Test frame preprocessing for ONNX"""
        test_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        preprocessed = self.detector._preprocess_frame(test_frame)
        
        # Check output shape: [1, 3, 320, 320]
        assert preprocessed.shape == (1, 3, 320, 320)
        # Check normalization: values should be in [0, 1]
        assert preprocessed.min() >= 0.0
        assert preprocessed.max() <= 1.0


class TestByteTrackTracker:
    """Test ByteTrack tracker"""

    def setup_method(self):
        """Setup test fixtures"""
        self.tracker = ByteTrackTracker(
            frame_rate=15,
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )

    def test_tracker_initialization(self):
        """Test tracker initialization"""
        assert self.tracker.frame_rate == 15
        assert self.tracker.track_thresh == 0.5
        assert self.tracker.track_buffer == 30
        assert self.tracker.match_thresh == 0.8

    def test_simple_tracking_fallback(self):
        """Test simple tracking fallback when ByteTrack is not available"""
        # Force fallback mode
        self.tracker.tracker = None
        
        # Create test detections
        detections = [
            {
                'bbox': [10, 10, 50, 50],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'person',
                'center': [30, 30]
            }
        ]
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First update should create new track
        tracked_objects = self.tracker.update(detections, test_frame)
        
        assert len(tracked_objects) == 1
        assert 'track_id' in tracked_objects[0]
        assert tracked_objects[0]['track_id'] > 0

    def test_iou_calculation(self):
        """Test IoU calculation"""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([20, 20, 60, 60])
        
        iou = self.tracker._calculate_iou(bbox1, bbox2)
        
        # Should be positive overlap
        assert 0 < iou < 1

    def test_track_cleanup(self):
        """Test old track cleanup in simple tracking"""
        self.tracker.tracker = None  # Force simple tracking
        
        # Add a track manually
        self.tracker.simple_tracks[1] = {'bbox': [10, 10, 50, 50], 'age': 35}
        
        # Update with empty detections
        tracked_objects = self.tracker._simple_tracking_update([])
        
        # Old track should be removed (age > 30)
        assert 1 not in self.tracker.simple_tracks


class TestCPUOptimizedVideoProcessor:
    """Test CPU-optimized video processor"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create mock detector and tracker
        self.mock_detector = Mock(spec=CPUOptimizedDetector)
        self.mock_tracker = Mock(spec=ByteTrackTracker)
        
        self.processor = CPUOptimizedVideoProcessor(
            detector=self.mock_detector,
            tracker=self.mock_tracker,
            max_resolution=(640, 480)
        )

    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.max_resolution == (640, 480)
        assert self.processor.frame_count == 0
        assert len(self.processor.processing_times) == 0

    def test_frame_resizing(self):
        """Test frame resizing for CPU optimization"""
        # Create large frame
        large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        resized_frame = self.processor._resize_for_cpu(large_frame)
        
        # Should be resized to fit within max resolution
        height, width = resized_frame.shape[:2]
        assert width <= 640
        assert height <= 480

    def test_frame_processing(self):
        """Test complete frame processing pipeline"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock detector output
        mock_detections = [
            {
                'bbox': [10, 10, 50, 50],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'person',
                'center': [30, 30]
            }
        ]
        self.mock_detector.detect.return_value = mock_detections
        
        # Mock tracker output
        mock_tracked = [
            {
                'track_id': 1,
                'bbox': [10, 10, 50, 50],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'person',
                'center': [30, 30]
            }
        ]
        self.mock_tracker.update.return_value = mock_tracked
        
        # Process frame
        processed_frame, tracked_objects = self.processor.process_frame(test_frame)
        
        assert processed_frame is not None
        assert len(tracked_objects) == 1
        assert tracked_objects[0]['track_id'] == 1
        assert self.processor.frame_count == 1

    def test_performance_monitoring(self):
        """Test performance monitoring and FPS calculation"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.mock_detector.detect.return_value = []
        self.mock_tracker.update.return_value = []
        
        # Process multiple frames
        for _ in range(5):
            self.processor.process_frame(test_frame)
        
        # Check performance metrics
        stats = self.processor.get_processing_stats()
        
        assert stats['total_frames'] == 5
        assert 'current_fps' in stats
        assert 'avg_processing_time_ms' in stats
        assert len(self.processor.processing_times) == 5


class TestCPUOptimizedVideoSource:
    """Test CPU-optimized video source"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create a temporary test video file
        self.temp_video = self._create_test_video()

    def teardown_method(self):
        """Cleanup test fixtures"""
        if os.path.exists(self.temp_video):
            os.remove(self.temp_video)

    def _create_test_video(self):
        """Create a temporary test video file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, 15.0, (320, 240))
        
        # Write 30 frames
        for i in range(30):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        return temp_file.name

    def test_video_source_initialization(self):
        """Test video source initialization"""
        source = CPUOptimizedVideoSource(
            source=self.temp_video,
            target_fps=15,
            max_resolution=(640, 480)
        )
        
        assert source.target_fps == 15
        assert source.max_resolution == (640, 480)
        
        source.release()

    def test_frame_reading(self):
        """Test frame reading with FPS control"""
        source = CPUOptimizedVideoSource(
            source=self.temp_video,
            target_fps=30,  # High FPS for testing
            max_resolution=(640, 480)
        )
        
        if source.cap is not None:
            frame = source.read_frame()
            assert frame is not None
            assert len(frame.shape) == 3  # Height, Width, Channels
        
        source.release()

    def test_frame_resizing(self):
        """Test automatic frame resizing"""
        # Create source with small max resolution
        source = CPUOptimizedVideoSource(
            source=self.temp_video,
            target_fps=15,
            max_resolution=(160, 120)  # Very small resolution
        )
        
        if source.cap is not None:
            frame = source.read_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                assert width <= 160
                assert height <= 120
        
        source.release()

    def test_source_info(self):
        """Test source information retrieval"""
        source = CPUOptimizedVideoSource(
            source=self.temp_video,
            target_fps=15,
            max_resolution=(640, 480)
        )
        
        info = source.get_source_info()
        
        if source.cap is not None:
            assert info['status'] == 'connected'
            assert 'frame_size' in info
            assert 'target_fps' in info
        
        source.release()


class TestVideoAnalysisPipeline:
    """Test complete video analysis pipeline with CPU optimizations"""

    def setup_method(self):
        """Setup test fixtures"""
        detector_config = {
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'use_onnx': False,  # Disable for tests
            'input_size': 320,
            'frame_rate': 15,
            'max_resolution': (640, 480)
        }
        
        analyzer_config = {
            'loitering_threshold': 30,
            'abandoned_threshold': 60
        }
        
        # Mock the detector and tracker to avoid model loading
        with patch('pipelines.video_surveillance.analysis.CPUOptimizedDetector'), \
             patch('pipelines.video_surveillance.analysis.ByteTrackTracker'), \
             patch('pipelines.video_surveillance.analysis.CPUOptimizedVideoProcessor'):
            
            self.pipeline = VideoAnalysisPipeline(detector_config, analyzer_config)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, 'detector')
        assert hasattr(self.pipeline, 'tracker')
        assert hasattr(self.pipeline, 'processor')
        assert hasattr(self.pipeline, 'analyzer')

    def test_frame_processing(self):
        """Test frame processing through complete pipeline"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the processor to return test results
        mock_results = (test_frame, [])
        self.pipeline.processor.process_frame.return_value = mock_results
        self.pipeline.analyzer.analyze_frame.return_value = []
        
        results = self.pipeline.process_frame(test_frame)
        
        assert 'frame_number' in results
        assert 'timestamp' in results
        assert 'detections' in results
        assert 'tracks' in results
        assert 'alerts' in results

    def test_pipeline_status(self):
        """Test pipeline status reporting"""
        status = self.pipeline.get_pipeline_status()
        
        assert 'frame_count' in status
        assert 'uptime' in status
        assert 'detector_status' in status
        assert 'tracker_status' in status
        assert 'analyzer_status' in status


@pytest.mark.integration
class TestCPUOptimizationIntegration:
    """Integration tests for CPU optimization features"""

    def test_cpu_thread_configuration(self):
        """Test CPU thread configuration for i5 6th gen"""
        import torch
        import os
        
        # Initialize detector (should set thread limits)
        detector = CPUOptimizedDetector(use_onnx=False)
        
        # Check thread configuration
        assert os.environ.get('OMP_NUM_THREADS') == '4'
        assert os.environ.get('MKL_NUM_THREADS') == '4'
        # Note: torch.get_num_threads() might not reflect the set value immediately

    @patch('onnxruntime.InferenceSession')
    @patch('os.path.exists')
    def test_onnx_export_and_loading(self, mock_exists, mock_session):
        """Test ONNX export and loading process"""
        mock_exists.return_value = False  # Force export
        
        # Mock YOLO model export
        mock_model = Mock()
        mock_model.export = Mock()
        
        with patch('ultralytics.YOLO', return_value=mock_model):
            detector = CPUOptimizedDetector(use_onnx=True)
            
            # Should attempt to export to ONNX
            mock_model.export.assert_called_once()

    def test_memory_efficiency(self):
        """Test memory efficiency of CPU optimizations"""
        # This test would measure memory usage in a real scenario
        # For now, just verify that components can be created without issues
        
        detector = CPUOptimizedDetector(
            use_onnx=False,
            input_size=320  # Small input size
        )
        
        tracker = ByteTrackTracker(frame_rate=15)
        
        processor = CPUOptimizedVideoProcessor(
            detector=detector,
            tracker=tracker,
            max_resolution=(640, 480)
        )
        
        # All components should be created successfully
        assert detector is not None
        assert tracker is not None
        assert processor is not None


if __name__ == '__main__':
    pytest.main([__file__])