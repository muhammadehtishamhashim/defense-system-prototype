"""
CPU-optimized object detection and tracking module for video surveillance.
Uses YOLOv8 Nano for detection and ByteTrack for lightweight multi-object tracking.
Optimized for i5 6th gen CPU performance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import torch
from ultralytics import YOLO
import logging
from datetime import datetime
import os
import threading
import time
from collections import deque

# ByteTrack imports - use simple implementation
try:
    from .simple_bytetrack import SimpleBYTETracker, TrackState
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    logging.warning("Simple ByteTrack not available")

# ONNX Runtime imports for CPU optimization
try:
    import onnxruntime as ort
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install onnxruntime onnx")

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("video_surveillance")


class CPUOptimizedDetector:
    """CPU-optimized YOLOv8 Nano detector with ONNX support for i5 6th gen"""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, 
                 use_onnx: bool = True, input_size: int = 416):
        """
        Initialize CPU-optimized YOLOv8 detector

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Detection confidence threshold
            use_onnx: Whether to use ONNX runtime for faster CPU inference
            input_size: Input image size (smaller = faster on CPU)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.device = 'cpu'
        
        # CPU optimization settings for i5 6th gen (4 cores, 4 threads)
        torch.set_num_threads(4)  # Match CPU cores
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        
        # Frame skipping for real-time performance
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_counter = 0
        self.last_detections = []
        
        # Initialize model
        self.model = None
        self.onnx_session = None
        self.class_names = {}
        
        self._initialize_model()

    def _initialize_model(self):
        """Initialize YOLOv8 model with CPU optimizations"""
        try:
            # Load YOLOv8 nano model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Set to evaluation mode for better performance
            if hasattr(self.model.model, 'eval'):
                self.model.model.eval()
                
            self.class_names = self.model.names
            
            # Export to ONNX for faster CPU inference if requested
            if self.use_onnx:
                self._export_to_onnx()
                
            logger.info(f"YOLOv8 nano model loaded (ONNX: {self.use_onnx}, input_size: {self.input_size})")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {str(e)}")
            self.model = None

    def _export_to_onnx(self):
        """Export YOLOv8 model to ONNX format for CPU optimization"""
        try:
            onnx_path = self.model_path.replace('.pt', '_cpu_optimized.onnx')
            
            if not os.path.exists(onnx_path):
                logger.info("Exporting YOLOv8 to ONNX format for CPU optimization...")
                # Export with CPU-optimized settings
                self.model.export(
                    format='onnx',
                    imgsz=self.input_size,
                    optimize=True,
                    simplify=True,
                    opset=11,  # Compatible with older ONNX runtime versions
                    dynamic=False  # Static shapes for better CPU performance
                )
                
            # Initialize ONNX Runtime session with CPU optimizations
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 4  # Match CPU cores
            session_options.inter_op_num_threads = 1  # Single thread for inter-op
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.onnx_session = ort.InferenceSession(
                onnx_path, 
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"ONNX model loaded successfully: {onnx_path}")
            
        except Exception as e:
            logger.warning(f"ONNX export failed, falling back to PyTorch: {str(e)}")
            self.onnx_session = None
            self.use_onnx = False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        CPU-optimized object detection with frame skipping

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List of detection results with bounding boxes and confidence scores
        """
        if self.model is None:
            logger.warning("YOLOv8 model not available, returning empty detections")
            return []

        # Frame skipping for real-time performance on CPU
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return self.last_detections  # Return cached detections

        try:
            # Resize frame for faster processing on CPU
            original_height, original_width = frame.shape[:2]
            resized_frame = cv2.resize(frame, (self.input_size, self.input_size))
            
            detections = []
            
            if self.use_onnx and self.onnx_session:
                # Use ONNX Runtime for faster CPU inference
                detections = self._detect_onnx(resized_frame, original_width, original_height)
            else:
                # Fallback to PyTorch
                detections = self._detect_pytorch(resized_frame, original_width, original_height)

            # Cache detections for frame skipping
            self.last_detections = detections
            
            logger.debug(f"Detected {len(detections)} objects (frame {self.frame_counter})")
            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return self.last_detections

    def _detect_onnx(self, frame: np.ndarray, orig_width: int, orig_height: int) -> List[Dict[str, Any]]:
        """ONNX Runtime inference for CPU optimization"""
        try:
            # Preprocess frame for ONNX
            input_tensor = self._preprocess_frame(frame)
            
            # Run ONNX inference
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_tensor})
            
            # Post-process ONNX outputs
            return self._postprocess_onnx_outputs(outputs[0], orig_width, orig_height)
            
        except Exception as e:
            logger.error(f"ONNX inference error: {str(e)}")
            return []

    def _detect_pytorch(self, frame: np.ndarray, orig_width: int, orig_height: int) -> List[Dict[str, Any]]:
        """PyTorch inference fallback"""
        try:
            # Run YOLOv8 inference with CPU optimizations
            with torch.no_grad():  # Disable gradient computation for inference
                results = self.model(frame, conf=self.confidence_threshold, verbose=False, 
                                   imgsz=self.input_size, device='cpu')

            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    if conf >= self.confidence_threshold:
                        # Scale bounding box back to original frame size
                        x1, y1, x2, y2 = box
                        x1 = x1 * orig_width / self.input_size
                        y1 = y1 * orig_height / self.input_size
                        x2 = x2 * orig_width / self.input_size
                        y2 = y2 * orig_height / self.input_size
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names.get(class_id, f'class_{class_id}'),
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        }
                        detections.append(detection)

            return detections
            
        except Exception as e:
            logger.error(f"PyTorch inference error: {str(e)}")
            return []

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX inference"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(frame_transposed, axis=0)
        
        return input_tensor

    def _postprocess_onnx_outputs(self, outputs: np.ndarray, orig_width: int, orig_height: int) -> List[Dict[str, Any]]:
        """Post-process ONNX model outputs"""
        detections = []
        
        # ONNX output format: [batch, num_detections, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
        for detection in outputs[0]:  # Remove batch dimension
            # Extract confidence and class scores
            confidence = detection[4]
            class_scores = detection[5:]
            
            if confidence >= self.confidence_threshold:
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                if class_confidence >= self.confidence_threshold:
                    # Extract and scale bounding box
                    x_center, y_center, width, height = detection[:4]
                    
                    # Convert from center format to corner format
                    x1 = (x_center - width / 2) * orig_width / self.input_size
                    y1 = (y_center - height / 2) * orig_height / self.input_size
                    x2 = (x_center + width / 2) * orig_width / self.input_size
                    y2 = (y_center + height / 2) * orig_height / self.input_size
                    
                    detection_dict = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence * class_confidence),
                        'class_id': int(class_id),
                        'class_name': self.class_names.get(class_id, f'class_{class_id}'),
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    }
                    detections.append(detection_dict)
        
        return detections

    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes"""
        return list(self.class_names.values())


class ByteTrackTracker:
    """ByteTrack-based lightweight multi-object tracker optimized for CPU"""

    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5, 
                 track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initialize ByteTrack tracker

        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to buffer tracks
            match_thresh: Matching threshold for track association
        """
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        if not BYTETRACK_AVAILABLE:
            logger.warning("ByteTrack not available, using simple tracking fallback")
            self.tracker = None
        else:
            try:
                # Simple ByteTrack implementation optimized for CPU performance
                self.tracker = SimpleBYTETracker(
                    frame_rate=frame_rate,
                    track_thresh=track_thresh,
                    track_buffer=track_buffer,
                    match_thresh=match_thresh
                )
                logger.info("Simple ByteTrack tracker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Simple ByteTrack: {str(e)}")
                self.tracker = None
                
        # Fallback simple tracker
        self.simple_tracks = {}
        self.next_track_id = 1

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections using ByteTrack

        Args:
            detections: List of object detections from detector
            frame: Current video frame

        Returns:
            List of tracked objects with tracking IDs
        """
        if self.tracker is None:
            # Fallback to simple tracking
            return self._simple_tracking_update(detections)

        try:
            # For now, use simple tracking fallback until ByteTrack is fully implemented
            return self._simple_tracking_update(detections)

        except Exception as e:
            logger.error(f"Error during tracking: {str(e)}")
            return self._simple_tracking_update(detections)

    def _simple_tracking_update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple fallback tracking using IoU matching"""
        tracked_objects = []
        
        for detection in detections:
            bbox = detection['bbox']
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold
            
            # Find best matching existing track
            for track_id, track_info in self.simple_tracks.items():
                iou = self._calculate_iou(np.array(bbox), np.array(track_info['bbox']))
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track_id = best_match_id
                self.simple_tracks[track_id]['bbox'] = bbox
                self.simple_tracks[track_id]['age'] += 1
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                self.simple_tracks[track_id] = {
                    'bbox': bbox,
                    'age': 1
                }
            
            tracked_object = detection.copy()
            tracked_object['track_id'] = track_id
            tracked_object['age'] = self.simple_tracks[track_id]['age']
            tracked_objects.append(tracked_object)
        
        # Clean up old tracks
        tracks_to_remove = []
        for track_id, track_info in self.simple_tracks.items():
            if track_info['age'] > 30:  # Remove tracks older than 30 frames
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.simple_tracks[track_id]
        
        return tracked_objects

    def _find_matching_detection(self, track_bbox: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the detection that best matches a track"""
        best_match = {}
        best_iou = 0.3
        
        for detection in detections:
            det_bbox = np.array(detection['bbox'])
            iou = self._calculate_iou(track_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = detection
        
        return best_match

    def _calculate_track_velocity(self, track) -> Optional[List[float]]:
        """Calculate velocity from ByteTrack track"""
        try:
            if hasattr(track, 'kalman_filter') and hasattr(track.kalman_filter, 'mean'):
                # Extract velocity from Kalman filter state
                state = track.kalman_filter.mean
                if len(state) >= 6:  # [x, y, a, h, vx, vy, va, vh]
                    return [float(state[4]), float(state[5])]  # vx, vy
        except Exception:
            pass
        return None

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # bbox format: [x1, y1, x2, y2]
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = inter_area / (bbox1_area + bbox2_area - inter_area)
        return iou

    def _calculate_velocity(self, track) -> Optional[List[float]]:
        """Calculate object velocity from track history"""
        try:
            if hasattr(track, 'track_history') and len(track.track_history) >= 2:
                # Get last two positions
                recent_positions = track.track_history[-2:]
                velocity_x = recent_positions[1][0] - recent_positions[0][0]
                velocity_y = recent_positions[1][1] - recent_positions[0][1]
                return [velocity_x, velocity_y]
        except Exception:
            pass
        return None


class CPUOptimizedVideoProcessor:
    """CPU-optimized video frame processing pipeline for i5 6th gen"""

    def __init__(self, detector: CPUOptimizedDetector, tracker: ByteTrackTracker, 
                 max_resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize CPU-optimized video processor

        Args:
            detector: CPU-optimized YOLOv8 detector
            tracker: ByteTrack tracker
            max_resolution: Maximum processing resolution for CPU optimization
        """
        self.detector = detector
        self.tracker = tracker
        self.max_resolution = max_resolution
        self.frame_count = 0
        self.start_time = datetime.now()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)  # Keep last 100 frame times
        self.last_fps_update = time.time()
        self.current_fps = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single video frame with CPU optimizations

        Args:
            frame: Input video frame

        Returns:
            Tuple of (processed_frame, tracked_objects)
        """
        start_time = time.time()
        self.frame_count += 1

        # Step 1: Resize frame if too large for CPU processing
        processed_frame = self._resize_for_cpu(frame)

        # Step 2: Object detection with CPU optimizations
        detections = self.detector.detect(processed_frame)

        # Step 3: Object tracking with ByteTrack
        tracked_objects = self.tracker.update(detections, processed_frame)

        # Step 4: Add visualization (optional, lightweight)
        display_frame = self._draw_tracking_info(processed_frame, tracked_objects)

        # Step 5: Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self._update_fps()

        return display_frame, tracked_objects

    def _resize_for_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to optimal size for CPU processing"""
        height, width = frame.shape[:2]
        max_width, max_height = self.max_resolution
        
        # Only resize if frame is larger than max resolution
        if width > max_width or height > max_height:
            # Calculate scaling factor to maintain aspect ratio
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            return resized_frame
        
        return frame

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                self.current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
            self.last_fps_update = current_time

    def _draw_tracking_info(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and tracking info on frame"""
        display_frame = frame.copy()

        for obj in tracked_objects:
            # Draw bounding box
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Color based on object class
            class_id = obj['class_id']
            color = self._get_class_color(class_id)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw tracking ID
            track_id = obj['track_id']
            cv2.putText(display_frame, f"ID:{track_id}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw class name and confidence
            class_name = obj['class_name']
            confidence = obj['confidence']
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(display_frame, label,
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return display_frame

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for object class"""
        colors = [
            (255, 0, 0),    # Red - person
            (0, 255, 0),    # Green - car
            (0, 0, 255),    # Blue - truck
            (255, 255, 0),  # Yellow - bus
            (255, 0, 255),  # Magenta - motorcycle
            (0, 255, 255),  # Cyan - bicycle
            (128, 128, 128), # Gray - others
        ]
        return colors[class_id % len(colors)]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        overall_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        avg_processing_time = 0.0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)

        return {
            'total_frames': self.frame_count,
            'overall_fps': overall_fps,
            'current_fps': self.current_fps,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'elapsed_time': elapsed_time,
            'detector_classes': len(self.detector.get_supported_classes()),
            'using_onnx': self.detector.use_onnx,
            'frame_skip_rate': self.detector.frame_skip,
            'max_resolution': self.max_resolution
        }


class CPUOptimizedVideoSource:
    """CPU-optimized video input source with reduced resolution support"""

    def __init__(self, source: str, target_fps: int = 15, max_resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize CPU-optimized video input source

        Args:
            source: Video file path or camera index/URL
            target_fps: Target processing FPS (reduced for CPU)
            max_resolution: Maximum resolution for CPU processing
        """
        self.source = source
        self.target_fps = target_fps
        self.max_resolution = max_resolution
        self.cap = None
        self.frame_time = 1.0 / target_fps
        self.last_frame_time = 0
        
        # Frame buffering for smooth playback
        self.frame_buffer = deque(maxlen=5)
        self.buffer_thread = None
        self.is_buffering = False

        self._initialize_capture()

    def _initialize_capture(self):
        """Initialize video capture with CPU optimizations"""
        try:
            # Try to open video source
            if isinstance(self.source, str):
                if self.source.isdigit():
                    self.cap = cv2.VideoCapture(int(self.source))  # Camera index
                else:
                    self.cap = cv2.VideoCapture(self.source)  # File or URL
            else:
                self.cap = cv2.VideoCapture(self.source)  # Numeric camera index

            if not self.cap or not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                self.cap = None
                return

            # Set capture properties for CPU optimization
            if self.cap.isOpened():
                # Set resolution to reduce CPU load
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_resolution[1])
                
                # Set FPS if it's a camera
                if isinstance(self.source, (int, str)) and (isinstance(self.source, int) or self.source.isdigit()):
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                # Disable auto-exposure and auto-focus for consistent performance (cameras only)
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
                except:
                    pass  # Not all cameras support these properties

            logger.info(f"Video source opened: {self.source} at {self.max_resolution}")

        except Exception as e:
            logger.error(f"Error initializing video capture: {str(e)}")
            self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame with FPS control"""
        if self.cap is None:
            return None

        current_time = time.time()
        
        # FPS control - skip frames if processing too fast
        if current_time - self.last_frame_time < self.frame_time:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.last_frame_time = current_time
            
            # Ensure frame is within max resolution
            frame = self._resize_frame(frame)
            return frame
        else:
            logger.warning("Failed to read frame from video source")
            return None

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to max resolution if needed"""
        height, width = frame.shape[:2]
        max_width, max_height = self.max_resolution
        
        if width > max_width or height > max_height:
            # Calculate scaling factor to maintain aspect ratio
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return frame

    def get_frame_size(self) -> Tuple[int, int]:
        """Get video frame dimensions"""
        if self.cap is None:
            return (0, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure dimensions don't exceed max resolution
        max_width, max_height = self.max_resolution
        width = min(width, max_width)
        height = min(height, max_height)
        
        return (width, height)

    def get_source_info(self) -> Dict[str, Any]:
        """Get detailed source information"""
        if self.cap is None:
            return {'status': 'disconnected'}
        
        try:
            return {
                'status': 'connected',
                'source': self.source,
                'frame_size': self.get_frame_size(),
                'target_fps': self.target_fps,
                'max_resolution': self.max_resolution,
                'backend': self.cap.getBackendName() if hasattr(self.cap, 'getBackendName') else 'unknown'
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            logger.info(f"Video source released: {self.source}")


# Aliases for backward compatibility
VideoInputSource = CPUOptimizedVideoSource
ObjectDetector = CPUOptimizedDetector
ObjectTracker = ByteTrackTracker
VideoProcessor = CPUOptimizedVideoProcessor
