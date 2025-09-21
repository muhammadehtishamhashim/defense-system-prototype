"""
Object detection and tracking module for video surveillance.
Uses YOLOv8 for detection and DeepSORT for multi-object tracking.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import torch
from ultralytics import YOLO
import logging
from datetime import datetime

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSORT not available. Install with: pip install deep-sort-realtime")

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("video_surveillance")


class ObjectDetector:
    """YOLOv8-based object detection for video surveillance"""

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLOv8 object detector

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.class_names = self.model.names
            logger.info(f"YOLOv8 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {str(e)}")
            self.model = None
            self.class_names = {}

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a video frame

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List of detection results with bounding boxes and confidence scores
        """
        if self.model is None:
            logger.warning("YOLOv8 model not available, returning empty detections")
            return []

        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    if conf >= self.confidence_threshold:
                        detection = {
                            'bbox': box.tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names.get(class_id, f'class_{class_id}'),
                            'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]  # Center point
                        }
                        detections.append(detection)

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return []

    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes"""
        return list(self.class_names.values())


class ObjectTracker:
    """DeepSORT-based multi-object tracker"""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Initialize DeepSORT tracker

        Args:
            max_age: Maximum number of frames to keep alive a track without associated detections
            n_init: Number of consecutive detections before the track is confirmed
        """
        self.max_age = max_age
        self.n_init = n_init

        if not DEEPSORT_AVAILABLE:
            logger.warning("DeepSORT not available, using simple tracking fallback")
            self.tracker = None
        else:
            try:
                self.tracker = DeepSort(
                    max_age=max_age,
                    n_init=n_init,
                    max_iou_distance=0.7,
                    max_cosine_distance=0.2,
                    nn_budget=100
                )
                logger.info("DeepSORT tracker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DeepSORT: {str(e)}")
                self.tracker = None

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections

        Args:
            detections: List of object detections from detector
            frame: Current video frame

        Returns:
            List of tracked objects with tracking IDs
        """
        if self.tracker is None:
            # Fallback: just add dummy tracking IDs
            for i, detection in enumerate(detections):
                detection['track_id'] = i
            return detections

        try:
            # Prepare detections for DeepSORT format
            # DeepSORT expects: [[x1, y1, x2, y2, confidence], ...]
            bboxes = []
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])

            bboxes = np.array(bboxes) if bboxes else np.empty((0, 5))

            # Update tracker
            if len(bboxes) > 0:
                tracks = self.tracker.update_tracks(bboxes, frame=frame)
            else:
                # Get tracks without new detections
                tracks = self.tracker.tracker.tracks

            # Process tracks
            tracked_objects = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                # Get the latest detection for this track
                bbox = track.to_tlbr()  # [x1, y1, x2, y2]

                # Find corresponding detection (if any)
                detection = None
                for det in detections:
                    det_bbox = np.array(det['bbox'])
                    iou = self._calculate_iou(bbox, det_bbox)
                    if iou > 0.5:  # Match threshold
                        detection = det
                        break

                if detection:
                    tracked_object = {
                        'track_id': int(track_id),
                        'bbox': bbox.tolist(),
                        'confidence': detection['confidence'],
                        'class_id': detection['class_id'],
                        'class_name': detection['class_name'],
                        'center': detection['center'],
                        'velocity': self._calculate_velocity(track),
                        'age': track.age
                    }
                    tracked_objects.append(tracked_object)

            logger.debug(f"Tracked {len(tracked_objects)} objects")
            return tracked_objects

        except Exception as e:
            logger.error(f"Error during tracking: {str(e)}")
            return detections

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


class VideoProcessor:
    """Video frame processing pipeline"""

    def __init__(self, detector: ObjectDetector, tracker: ObjectTracker):
        """
        Initialize video processor

        Args:
            detector: YOLOv8 object detector
            tracker: DeepSORT object tracker
        """
        self.detector = detector
        self.tracker = tracker
        self.frame_count = 0
        self.start_time = datetime.now()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single video frame

        Args:
            frame: Input video frame

        Returns:
            Tuple of (processed_frame, tracked_objects)
        """
        self.frame_count += 1

        # Step 1: Object detection
        detections = self.detector.detect(frame)

        # Step 2: Object tracking
        tracked_objects = self.tracker.update(detections, frame)

        # Step 3: Add visualization (optional)
        processed_frame = self._draw_tracking_info(frame, tracked_objects)

        return processed_frame, tracked_objects

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
            label = f"{class_name} {confidence".2f"}"
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
        """Get processing statistics"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        return {
            'total_frames': self.frame_count,
            'processing_fps': fps,
            'elapsed_time': elapsed_time,
            'detector_classes': len(self.detector.get_supported_classes())
        }


class VideoInputSource:
    """Video input source abstraction"""

    def __init__(self, source: str, fps: int = 30):
        """
        Initialize video input source

        Args:
            source: Video file path or camera index/URL
            fps: Target processing FPS
        """
        self.source = source
        self.target_fps = fps
        self.cap = None
        self.frame_time = 1.0 / fps

        # Try to open video source
        if isinstance(source, str):
            if source.isdigit():
                self.cap = cv2.VideoCapture(int(source))  # Camera index
            else:
                self.cap = cv2.VideoCapture(source)  # File or URL
        else:
            self.cap = cv2.VideoCapture(source)  # Numeric camera index

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            self.cap = None
        else:
            logger.info(f"Video source opened: {source}")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video source"""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            logger.warning("Failed to read frame from video source")
            return None

    def get_frame_size(self) -> Tuple[int, int]:
        """Get video frame dimensions"""
        if self.cap is None:
            return (0, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            logger.info("Video source released")
