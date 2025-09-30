"""
Video Analysis Coordinator
Coordinates video file processing with the existing video surveillance pipeline.
"""

import asyncio
import uuid
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

from pipelines.video_surveillance.pipeline import VideoSurveillancePipeline
from utils.logging import get_logger
from .monitoring import video_monitor, error_handler

logger = get_logger(__name__)


class AnalysisStatus(str, Enum):
    """Analysis session status"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AnalysisSession(BaseModel):
    """Analysis session information"""
    session_id: str
    video_filename: str
    status: AnalysisStatus
    started_at: datetime
    frames_processed: int
    alerts_generated: int
    current_frame: int
    total_frames: int
    fps: float
    progress_percent: float


class VideoAnalysisCoordinator:
    """Coordinates video file processing with video surveillance pipeline"""
    
    def __init__(self, media_directory: str = "backend/media/videos"):
        """
        Initialize video analysis coordinator
        
        Args:
            media_directory: Path to directory containing video files
        """
        self.media_directory = Path(media_directory)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.pipeline = None
        
        logger.info("Video analysis coordinator initialized")
    
    async def initialize_pipeline(self):
        """Initialize the video surveillance pipeline"""
        try:
            if not self.pipeline:
                self.pipeline = VideoSurveillancePipeline()
                await self.pipeline.initialize()
                logger.info("Video surveillance pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    async def start_analysis(self, video_filename: str) -> str:
        """
        Start video analysis session
        
        Args:
            video_filename: Name of the video file to analyze
            
        Returns:
            Session ID for the analysis session
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If analysis cannot be started
        """
        video_path = self.media_directory / video_filename
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file '{video_filename}' not found")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        try:
            # Initialize pipeline if needed
            await self.initialize_pipeline()
            
            # Extract video metadata
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_filename}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Create session data
            session_data = {
                'session_id': session_id,
                'video_filename': video_filename,
                'video_path': video_path,
                'status': AnalysisStatus.STARTING,
                'started_at': datetime.now(),
                'frames_processed': 0,
                'alerts_generated': 0,
                'current_frame': 0,
                'total_frames': total_frames,
                'fps': fps,
                'task': None,
                'stop_event': asyncio.Event()
            }
            
            self.active_sessions[session_id] = session_data
            
            # Register session with monitoring
            video_monitor.register_session(session_id)
            
            # Start analysis task
            session_data['task'] = asyncio.create_task(
                self._process_video_analysis(session_id)
            )
            
            logger.info(f"Started video analysis session {session_id} for {video_filename}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting analysis for {video_filename}: {str(e)}")
            # Clean up session if it was created
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            raise RuntimeError(f"Failed to start analysis: {str(e)}")
    
    async def stop_analysis(self, session_id: str) -> bool:
        """
        Stop video analysis session
        
        Args:
            session_id: ID of the session to stop
            
        Returns:
            True if session was stopped, False if session not found
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Analysis session {session_id} not found")
            return False
        
        try:
            session_data = self.active_sessions[session_id]
            
            # Signal stop
            session_data['stop_event'].set()
            session_data['status'] = AnalysisStatus.STOPPED
            
            # Cancel task if running
            if session_data['task'] and not session_data['task'].done():
                session_data['task'].cancel()
                try:
                    await session_data['task']
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"Stopped video analysis session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping analysis session {session_id}: {str(e)}")
            return False
    
    def get_analysis_status(self, session_id: str) -> Optional[AnalysisSession]:
        """
        Get status of analysis session
        
        Args:
            session_id: ID of the session
            
        Returns:
            AnalysisSession object or None if session not found
        """
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        
        # Calculate progress
        progress_percent = 0.0
        if session_data['total_frames'] > 0:
            progress_percent = (session_data['current_frame'] / session_data['total_frames']) * 100
        
        return AnalysisSession(
            session_id=session_data['session_id'],
            video_filename=session_data['video_filename'],
            status=session_data['status'],
            started_at=session_data['started_at'],
            frames_processed=session_data['frames_processed'],
            alerts_generated=session_data['alerts_generated'],
            current_frame=session_data['current_frame'],
            total_frames=session_data['total_frames'],
            fps=session_data['fps'],
            progress_percent=progress_percent
        )
    
    def list_active_sessions(self) -> List[AnalysisSession]:
        """
        Get list of all active analysis sessions
        
        Returns:
            List of AnalysisSession objects
        """
        sessions = []
        for session_id in self.active_sessions:
            session = self.get_analysis_status(session_id)
            if session:
                sessions.append(session)
        return sessions
    
    async def cleanup_session(self, session_id: str):
        """
        Clean up analysis session resources
        
        Args:
            session_id: ID of the session to clean up
        """
        if session_id in self.active_sessions:
            try:
                await self.stop_analysis(session_id)
                del self.active_sessions[session_id]
                
                # Unregister from monitoring
                video_monitor.unregister_session(session_id)
                
                logger.info(f"Cleaned up analysis session {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {str(e)}")
    
    async def _process_frame_async(self, input_data: Dict[str, Any], session_data: Dict[str, Any]):
        """
        Process a single frame asynchronously without blocking main loop
        
        Args:
            input_data: Frame data for processing
            session_data: Session information
        """
        session_id = session_data['session_id']
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process the frame with error handling
            results = await error_handler.handle_with_retry(
                self.pipeline.process_input, input_data
            )
            
            # Record successful processing
            processing_time = asyncio.get_event_loop().time() - start_time
            video_monitor.record_frame_processed(session_id, processing_time)
            
            # Update session stats
            if results.get('alerts'):
                session_data['alerts_generated'] += len(results['alerts'])
                
        except Exception as e:
            # Record error in monitoring
            video_monitor.record_error(session_id, str(e))
            logger.warning(f"Error in async frame processing for session {session_id}: {str(e)}")
    
    async def _process_video_analysis(self, session_id: str):
        """
        Process video frames for analysis
        
        Args:
            session_id: ID of the analysis session
        """
        session_data = self.active_sessions[session_id]
        
        try:
            session_data['status'] = AnalysisStatus.RUNNING
            video_path = session_data['video_path']
            
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            
            frame_skip = 5  # Process every 5th frame for performance
            frame_count = 0
            
            logger.info(f"Starting video analysis for session {session_id}")
            
            while not session_data['stop_event'].is_set():
                ret, frame = cap.read()
                
                if not ret:
                    # End of video, restart from beginning for continuous analysis
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    session_data['current_frame'] = 0
                    continue
                
                frame_count += 1
                session_data['current_frame'] = frame_count
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    continue
                
                try:
                    # Process frame through pipeline (non-blocking)
                    if self.pipeline:
                        input_data = {
                            'stream_name': f'analysis_{session_id}',
                            'frame': frame,
                            'timestamp': datetime.now(),
                            'source_config': {'analysis_session': session_id}
                        }
                        
                        # Process the frame asynchronously
                        processing_task = asyncio.create_task(
                            self._process_frame_async(input_data, session_data)
                        )
                        
                        # Don't wait for processing to complete - allows concurrent processing
                        session_data['frames_processed'] += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing frame in session {session_id}: {str(e)}")
                    continue
                
                # Adaptive delay based on processing load
                await asyncio.sleep(0.05)  # Reduced delay for better performance
            
            cap.release()
            session_data['status'] = AnalysisStatus.STOPPED
            logger.info(f"Video analysis completed for session {session_id}")
            
        except asyncio.CancelledError:
            logger.info(f"Video analysis cancelled for session {session_id}")
            session_data['status'] = AnalysisStatus.STOPPED
        except Exception as e:
            logger.error(f"Error in video analysis for session {session_id}: {str(e)}")
            session_data['status'] = AnalysisStatus.ERROR
        finally:
            # Clean up resources
            try:
                if 'cap' in locals():
                    cap.release()
            except:
                pass


# Global coordinator instance
video_analysis_coordinator = VideoAnalysisCoordinator()