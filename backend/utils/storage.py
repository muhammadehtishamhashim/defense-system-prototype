"""
File storage utilities for HifazatAI.
Handles storage and retrieval of media files, snapshots, and other data.
"""

import os
import shutil
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
from dataclasses import dataclass
from enum import Enum
import json
import base64

from utils.logging import get_logger
from utils.config import config_manager


class MediaType(str, Enum):
    """Media file type enumeration"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    DATA = "data"


@dataclass
class StoredFile:
    """Information about a stored file"""
    file_id: str
    original_name: str
    stored_path: str
    media_type: MediaType
    file_size: int
    mime_type: str
    checksum: str
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_id": self.file_id,
            "original_name": self.original_name,
            "stored_path": self.stored_path,
            "media_type": self.media_type.value,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredFile':
        """Create from dictionary"""
        return cls(
            file_id=data["file_id"],
            original_name=data["original_name"],
            stored_path=data["stored_path"],
            media_type=MediaType(data["media_type"]),
            file_size=data["file_size"],
            mime_type=data["mime_type"],
            checksum=data["checksum"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


class FileStorageManager:
    """Manager for file storage operations"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize file storage manager.
        
        Args:
            base_path: Base directory for file storage
        """
        self.logger = get_logger(__name__)
        self.base_path = Path(base_path or config_manager.get("storage.media_path", "media"))
        
        # Create storage directories
        self.directories = {
            MediaType.IMAGE: self.base_path / "images",
            MediaType.VIDEO: self.base_path / "videos", 
            MediaType.AUDIO: self.base_path / "audio",
            MediaType.DOCUMENT: self.base_path / "documents",
            MediaType.DATA: self.base_path / "data"
        }
        
        # Create all directories
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.base_path / "file_metadata.json"
        self.metadata_cache: Dict[str, StoredFile] = {}
        self._load_metadata()
        
        self.logger.info(f"Initialized file storage at {self.base_path}")
    
    def store_file(
        self,
        file_data: bytes,
        original_name: str,
        media_type: MediaType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredFile:
        """
        Store a file and return storage information.
        
        Args:
            file_data: File content as bytes
            original_name: Original filename
            media_type: Type of media file
            metadata: Optional metadata dictionary
            
        Returns:
            StoredFile object with storage information
        """
        try:
            # Generate file ID and checksum
            file_id = self._generate_file_id(file_data, original_name)
            checksum = self._calculate_checksum(file_data)
            
            # Check if file already exists
            if file_id in self.metadata_cache:
                existing_file = self.metadata_cache[file_id]
                if existing_file.checksum == checksum:
                    self.logger.info(f"File already exists: {file_id}")
                    return existing_file
            
            # Determine storage path
            file_extension = Path(original_name).suffix
            storage_dir = self.directories[media_type]
            
            # Create subdirectory based on date
            date_subdir = storage_dir / datetime.now().strftime("%Y/%m/%d")
            date_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            stored_filename = f"{file_id}{file_extension}"
            stored_path = date_subdir / stored_filename
            
            # Write file
            with open(stored_path, 'wb') as f:
                f.write(file_data)
            
            # Get file info
            file_size = len(file_data)
            mime_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
            
            # Create stored file object
            stored_file = StoredFile(
                file_id=file_id,
                original_name=original_name,
                stored_path=str(stored_path.relative_to(self.base_path)),
                media_type=media_type,
                file_size=file_size,
                mime_type=mime_type,
                checksum=checksum,
                created_at=datetime.now(),
                metadata=metadata
            )
            
            # Update metadata cache and save
            self.metadata_cache[file_id] = stored_file
            self._save_metadata()
            
            self.logger.info(f"Stored file: {file_id} ({file_size} bytes)")
            return stored_file
            
        except Exception as e:
            self.logger.error(f"Error storing file {original_name}: {str(e)}")
            raise
    
    def store_file_from_path(
        self,
        source_path: str,
        media_type: MediaType,
        metadata: Optional[Dict[str, Any]] = None,
        copy_file: bool = True
    ) -> StoredFile:
        """
        Store a file from filesystem path.
        
        Args:
            source_path: Path to source file
            media_type: Type of media file
            metadata: Optional metadata dictionary
            copy_file: Whether to copy file (True) or move it (False)
            
        Returns:
            StoredFile object with storage information
        """
        source_file = Path(source_path)
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Read file data
        with open(source_file, 'rb') as f:
            file_data = f.read()
        
        # Store the file
        stored_file = self.store_file(
            file_data=file_data,
            original_name=source_file.name,
            media_type=media_type,
            metadata=metadata
        )
        
        # Remove source file if moving
        if not copy_file:
            source_file.unlink()
            self.logger.info(f"Moved file from {source_path}")
        
        return stored_file
    
    def retrieve_file(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve file content by ID.
        
        Args:
            file_id: File identifier
            
        Returns:
            File content as bytes or None if not found
        """
        try:
            stored_file = self.get_file_info(file_id)
            if not stored_file:
                return None
            
            file_path = self.base_path / stored_file.stored_path
            
            if not file_path.exists():
                self.logger.warning(f"File not found on disk: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_id}: {str(e)}")
            return None
    
    def get_file_info(self, file_id: str) -> Optional[StoredFile]:
        """
        Get file information by ID.
        
        Args:
            file_id: File identifier
            
        Returns:
            StoredFile object or None if not found
        """
        return self.metadata_cache.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """
        Get full file path by ID.
        
        Args:
            file_id: File identifier
            
        Returns:
            Path object or None if not found
        """
        stored_file = self.get_file_info(file_id)
        if stored_file:
            return self.base_path / stored_file.stored_path
        return None
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by ID.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            stored_file = self.get_file_info(file_id)
            if not stored_file:
                self.logger.warning(f"File not found in metadata: {file_id}")
                return False
            
            file_path = self.base_path / stored_file.stored_path
            
            # Remove file from disk
            if file_path.exists():
                file_path.unlink()
            
            # Remove from metadata
            del self.metadata_cache[file_id]
            self._save_metadata()
            
            self.logger.info(f"Deleted file: {file_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
    
    def list_files(
        self,
        media_type: Optional[MediaType] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[StoredFile]:
        """
        List stored files with optional filtering.
        
        Args:
            media_type: Optional media type filter
            limit: Maximum number of files to return
            offset: Number of files to skip
            
        Returns:
            List of StoredFile objects
        """
        files = list(self.metadata_cache.values())
        
        # Filter by media type
        if media_type:
            files = [f for f in files if f.media_type == media_type]
        
        # Sort by creation time (newest first)
        files.sort(key=lambda f: f.created_at, reverse=True)
        
        # Apply pagination
        if offset > 0:
            files = files[offset:]
        if limit:
            files = files[:limit]
        
        return files
    
    def cleanup_orphaned_files(self) -> int:
        """
        Remove files from disk that are not in metadata.
        
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        try:
            # Get all files in storage directories
            for media_type, directory in self.directories.items():
                if not directory.exists():
                    continue
                
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        # Check if file is in metadata
                        relative_path = str(file_path.relative_to(self.base_path))
                        
                        # Find if any stored file has this path
                        found = False
                        for stored_file in self.metadata_cache.values():
                            if stored_file.stored_path == relative_path:
                                found = True
                                break
                        
                        if not found:
                            file_path.unlink()
                            cleaned_count += 1
                            self.logger.info(f"Cleaned orphaned file: {relative_path}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "total_files": len(self.metadata_cache),
            "total_size": 0,
            "by_media_type": {},
            "storage_path": str(self.base_path)
        }
        
        # Calculate statistics
        for stored_file in self.metadata_cache.values():
            stats["total_size"] += stored_file.file_size
            
            media_type = stored_file.media_type.value
            if media_type not in stats["by_media_type"]:
                stats["by_media_type"][media_type] = {"count": 0, "size": 0}
            
            stats["by_media_type"][media_type]["count"] += 1
            stats["by_media_type"][media_type]["size"] += stored_file.file_size
        
        return stats
    
    def _generate_file_id(self, file_data: bytes, original_name: str) -> str:
        """Generate unique file ID"""
        content_hash = hashlib.sha256(file_data).hexdigest()[:16]
        name_hash = hashlib.md5(original_name.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{content_hash}_{name_hash}"
    
    def _calculate_checksum(self, file_data: bytes) -> str:
        """Calculate file checksum"""
        return hashlib.sha256(file_data).hexdigest()
    
    def _load_metadata(self):
        """Load file metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                # Convert to StoredFile objects
                for file_id, file_data in metadata_dict.items():
                    self.metadata_cache[file_id] = StoredFile.from_dict(file_data)
                
                self.logger.info(f"Loaded metadata for {len(self.metadata_cache)} files")
            
        except Exception as e:
            self.logger.error(f"Error loading metadata: {str(e)}")
            self.metadata_cache = {}
    
    def _save_metadata(self):
        """Save file metadata to disk"""
        try:
            # Convert to serializable format
            metadata_dict = {
                file_id: stored_file.to_dict()
                for file_id, stored_file in self.metadata_cache.items()
            }
            
            # Write to temporary file first
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Atomic replace
            temp_file.replace(self.metadata_file)
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")


# Utility functions for common operations

def store_image_from_base64(
    base64_data: str,
    filename: str,
    storage_manager: Optional[FileStorageManager] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> StoredFile:
    """
    Store image from base64 encoded data.
    
    Args:
        base64_data: Base64 encoded image data
        filename: Original filename
        storage_manager: Optional storage manager instance
        metadata: Optional metadata dictionary
        
    Returns:
        StoredFile object
    """
    if storage_manager is None:
        storage_manager = FileStorageManager()
    
    # Decode base64 data
    image_data = base64.b64decode(base64_data)
    
    return storage_manager.store_file(
        file_data=image_data,
        original_name=filename,
        media_type=MediaType.IMAGE,
        metadata=metadata
    )


def create_thumbnail(
    image_path: str,
    thumbnail_size: tuple = (150, 150),
    storage_manager: Optional[FileStorageManager] = None
) -> Optional[StoredFile]:
    """
    Create thumbnail for an image.
    
    Args:
        image_path: Path to source image
        thumbnail_size: Thumbnail dimensions (width, height)
        storage_manager: Optional storage manager instance
        
    Returns:
        StoredFile object for thumbnail or None if failed
    """
    try:
        from PIL import Image
        
        if storage_manager is None:
            storage_manager = FileStorageManager()
        
        # Open and resize image
        with Image.open(image_path) as img:
            img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            import io
            thumbnail_bytes = io.BytesIO()
            img.save(thumbnail_bytes, format='JPEG', quality=85)
            thumbnail_data = thumbnail_bytes.getvalue()
        
        # Generate thumbnail filename
        original_name = Path(image_path).name
        thumbnail_name = f"thumb_{original_name}"
        
        return storage_manager.store_file(
            file_data=thumbnail_data,
            original_name=thumbnail_name,
            media_type=MediaType.IMAGE,
            metadata={"is_thumbnail": True, "original_image": original_name}
        )
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating thumbnail for {image_path}: {str(e)}")
        return None


# Global storage manager instance
storage_manager = FileStorageManager()