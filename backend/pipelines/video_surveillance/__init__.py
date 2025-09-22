"""
Video Surveillance Pipeline package.
Provides object detection, tracking, behavior analysis, and alert generation for video streams.
"""

from .pipeline import VideoSurveillancePipeline
from .detection import (
    CPUOptimizedVideoSource, 
    CPUOptimizedDetector, 
    ByteTrackTracker, 
    CPUOptimizedVideoProcessor,
    # Backward compatibility aliases
    VideoInputSource, 
    ObjectDetector
)
from .analysis import VideoAnalysisPipeline, Zone

__all__ = [
    'VideoSurveillancePipeline', 
    'VideoAnalysisPipeline', 
    'Zone',
    'CPUOptimizedVideoSource', 
    'CPUOptimizedDetector', 
    'ByteTrackTracker', 
    'CPUOptimizedVideoProcessor',
    # Backward compatibility
    'VideoInputSource', 
    'ObjectDetector'
]