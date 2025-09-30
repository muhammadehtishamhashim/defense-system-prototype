"""
Video Streaming Service
Handles HTTP video streaming with range request support and metadata extraction.
"""

import os
import mimetypes
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
import cv2
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.logging import get_logger

logger = get_logger(__name__)


class VideoInfo(BaseModel):
    """Video file information model"""
    filename: str
    display_name: str
    duration: float
    resolution: Tuple[int, int]
    format: str
    file_size: int
    created_at: datetime
    fps: float


class VideoStreamingService:
    """Service for streaming video files with HTTP range request support"""
    
    def __init__(self, media_directory: str = "backend/media/videos"):
        """
        Initialize video streaming service
        
        Args:
            media_directory: Path to directory containing video files
        """
        # Make path absolute to avoid working directory issues
        if not os.path.isabs(media_directory):
            # Get the directory where this script is located
            current_dir = Path(__file__).parent.parent  # Go up from services to backend
            self.media_directory = current_dir / media_directory.replace("backend/", "")
        else:
            self.media_directory = Path(media_directory)
            
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        
        # Ensure media directory exists
        self.media_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Video streaming service initialized with directory: {self.media_directory.absolute()}")
    
    def list_available_videos(self) -> List[VideoInfo]:
        """
        Get list of available video files with metadata
        
        Returns:
            List of VideoInfo objects
        """
        videos = []
        
        try:
            logger.info(f"Scanning directory: {self.media_directory}")
            logger.info(f"Supported formats: {self.supported_formats}")
            
            for file_path in self.media_directory.iterdir():
                logger.info(f"Found file: {file_path.name}, suffix: {file_path.suffix.lower()}, is_file: {file_path.is_file()}")
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    try:
                        video_info = self._extract_video_info(file_path)
                        videos.append(video_info)
                        logger.info(f"Successfully extracted info for {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not extract info for {file_path.name}: {str(e)}")
                        # Create basic video info even if metadata extraction fails
                        try:
                            stat = file_path.stat()
                            basic_info = VideoInfo(
                                filename=file_path.name,
                                display_name=self._generate_display_name(file_path.name),
                                duration=0.0,  # Unknown duration
                                resolution=(0, 0),  # Unknown resolution
                                format=file_path.suffix.lower(),
                                file_size=stat.st_size,
                                created_at=datetime.fromtimestamp(stat.st_ctime),
                                fps=0.0  # Unknown FPS
                            )
                            videos.append(basic_info)
                            logger.info(f"Created basic info for {file_path.name}")
                        except Exception as e2:
                            logger.error(f"Failed to create basic info for {file_path.name}: {str(e2)}")
                            continue
            
            # Sort by creation time, newest first
            videos.sort(key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing videos: {str(e)}")
        
        return videos
    
    def get_video_info(self, filename: str) -> VideoInfo:
        """
        Get metadata for a specific video file
        
        Args:
            filename: Name of the video file
            
        Returns:
            VideoInfo object
            
        Raises:
            HTTPException: If file not found or invalid
        """
        file_path = self._get_safe_file_path(filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")
        
        return self._extract_video_info(file_path)
    
    async def stream_video(self, filename: str, request: Request) -> StreamingResponse:
        """
        Stream video file with HTTP range request support
        
        Args:
            filename: Name of the video file
            request: FastAPI request object for range headers
            
        Returns:
            StreamingResponse for video streaming
            
        Raises:
            HTTPException: If file not found or streaming error
        """
        file_path = self._get_safe_file_path(filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Parse range header
        range_header = request.headers.get('range')
        
        if range_header:
            # Handle range request
            return await self._stream_range_response(file_path, range_header, file_size)
        else:
            # Stream entire file
            return await self._stream_full_response(file_path, file_size)
    
    def _extract_video_info(self, file_path: Path) -> VideoInfo:
        """
        Extract metadata from video file using OpenCV
        
        Args:
            file_path: Path to video file
            
        Returns:
            VideoInfo object with extracted metadata
        """
        cap = None
        try:
            # Use OpenCV to extract video metadata
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {file_path.name}")
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate extracted values
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                logger.warning(f"Invalid video properties for {file_path.name}: fps={fps}, frames={frame_count}, size={width}x{height}")
                # Try to read at least one frame to verify it's a valid video
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Could not read any frames from video file")
                
                # Use default values if properties are invalid
                fps = fps if fps > 0 else 25.0  # Default FPS
                width = width if width > 0 else 640  # Default width
                height = height if height > 0 else 480  # Default height
                frame_count = frame_count if frame_count > 0 else 1000  # Default frame count
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file system info
            stat = file_path.stat()
            
            return VideoInfo(
                filename=file_path.name,
                display_name=self._generate_display_name(file_path.name),
                duration=duration,
                resolution=(width, height),
                format=file_path.suffix.lower(),
                file_size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                fps=fps
            )
            
        except Exception as e:
            logger.error(f"Error extracting video info from {file_path}: {str(e)}")
            raise  # Re-raise to be handled by the calling method
        finally:
            if cap is not None:
                cap.release()
    
    def _generate_display_name(self, filename: str) -> str:
        """
        Generate user-friendly display name from filename
        
        Args:
            filename: Original filename
            
        Returns:
            User-friendly display name
        """
        # Remove extension and replace underscores/hyphens with spaces
        name = Path(filename).stem
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        return ' '.join(word.capitalize() for word in name.split())
    
    def _get_safe_file_path(self, filename: str) -> Path:
        """
        Get safe file path preventing directory traversal
        
        Args:
            filename: Requested filename
            
        Returns:
            Safe file path
            
        Raises:
            HTTPException: If filename is invalid
        """
        # Prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = self.media_directory / filename
        
        # Ensure the resolved path is within media directory
        try:
            file_path.resolve().relative_to(self.media_directory.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        return file_path
    
    async def _stream_range_response(self, file_path: Path, range_header: str, file_size: int) -> StreamingResponse:
        """
        Handle HTTP range request for partial content
        
        Args:
            file_path: Path to video file
            range_header: HTTP Range header value
            file_size: Total file size
            
        Returns:
            StreamingResponse with partial content
        """
        try:
            # Parse range header (e.g., "bytes=0-1023")
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Validate range
            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(status_code=416, detail="Range not satisfiable")
            
            chunk_size = end - start + 1
            
            def generate_chunk():
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        read_size = min(8192, remaining)  # 8KB chunks
                        data = f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            
            # Get content type
            content_type = mimetypes.guess_type(str(file_path))[0] or 'video/mp4'
            
            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(chunk_size),
                'Content-Type': content_type,
            }
            
            return StreamingResponse(
                generate_chunk(),
                status_code=206,  # Partial Content
                headers=headers
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid range header: {str(e)}")
        except Exception as e:
            logger.error(f"Error streaming range response: {str(e)}")
            raise HTTPException(status_code=500, detail="Error streaming video")
    
    async def _stream_full_response(self, file_path: Path, file_size: int) -> StreamingResponse:
        """
        Stream entire video file with optimized chunking
        
        Args:
            file_path: Path to video file
            file_size: Total file size
            
        Returns:
            StreamingResponse with full content
        """
        async def generate_full():
            # Use larger chunks for better performance
            chunk_size = min(65536, file_size)  # 64KB chunks or file size if smaller
            
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data
                    # Allow other coroutines to run
                    await asyncio.sleep(0)
        
        # Get content type
        content_type = mimetypes.guess_type(str(file_path))[0] or 'video/mp4'
        
        headers = {
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size),
            'Content-Type': content_type,
            'Cache-Control': 'public, max-age=3600',  # Cache for 1 hour
        }
        
        return StreamingResponse(
            generate_full(),
            status_code=200,
            headers=headers
        )


# Global service instance
video_streaming_service = VideoStreamingService()