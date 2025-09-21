"""
Video Surveillance Pipeline package.
Provides object detection, tracking, behavior analysis, and alert generation for video streams.
"""

from .pipeline import VideoSurveillancePipeline
from .detection import VideoInputSource, ObjectDetector
from .analysis import VideoAnalysisPipeline, Zone

__all__ = ['VideoSurveillancePipeline', 'VideoInputSource', 'ObjectDetector', 'VideoAnalysisPipeline', 'Zone']