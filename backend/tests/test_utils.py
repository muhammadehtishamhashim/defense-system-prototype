"""
Unit tests for HifazatAI utility functions.
Tests configuration management, file storage, and logging utilities.
"""

import pytest
import tempfile
import json
import yaml
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.config import ConfigManager, ThreatIntelligenceConfig, VideoSurveillanceConfig, BorderAnomalyConfig
from utils.storage import FileStorageManager, MediaType, StoredFile, store_image_from_base64
from utils.logging import get_logger, setup_logging, ErrorTracker
from pipelines.base import BasePipeline, PipelineStatus, PipelineMetrics


class TestConfigManager:
    """Test configuration management functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
    
    def test_default_config_loading(self):
        """Test loading default configuration"""
        config_manager = ConfigManager()
        
        # Check that default pipelines are loaded
        assert config_manager.get("pipelines.threat_intelligence") is not None
        assert config_manager.get("pipelines.video_surveillance") is not None
        assert config_manager.get("pipelines.border_anomaly") is not None
    
    def test_json_config_loading(self):
        """Test loading JSON configuration file"""
        test_config = {
            "pipelines": {
                "threat_intelligence": {
                    "enabled": True,
                    "confidence_threshold": 0.8
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        config_manager = ConfigManager(str(self.config_path))
        assert config_manager.get("pipelines.threat_intelligence.confidence_threshold") == 0.8
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration file"""
        yaml_path = Path(self.temp_dir) / "test_config.yaml"
        test_config = {
            "pipelines": {
                "video_surveillance": {
                    "enabled": False,
                    "processing_interval": 2.0
                }
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(test_config, f)
        
        config_manager = ConfigManager(str(yaml_path))
        assert config_manager.get("pipelines.video_surveillance.processing_interval") == 2.0
    
    def test_get_set_operations(self):
        """Test get and set operations"""
        config_manager = ConfigManager()
        
        # Test setting and getting values
        config_manager.set("test.nested.value", 42)
        assert config_manager.get("test.nested.value") == 42
        
        # Test default values
        assert config_manager.get("nonexistent.key", "default") == "default"
    
    def test_pipeline_config_objects(self):
        """Test pipeline configuration objects"""
        config_manager = ConfigManager()
        
        # Test threat intelligence config
        threat_config = config_manager.get_pipeline_config("threat_intelligence")
        assert isinstance(threat_config, ThreatIntelligenceConfig)
        assert threat_config.enabled is True
        
        # Test video surveillance config
        video_config = config_manager.get_pipeline_config("video_surveillance")
        assert isinstance(video_config, VideoSurveillanceConfig)
        assert video_config.tracking_enabled is True
        
        # Test border anomaly config
        border_config = config_manager.get_pipeline_config("border_anomaly")
        assert isinstance(border_config, BorderAnomalyConfig)
        assert border_config.sensitivity == 0.7
    
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        
        # Set invalid configuration
        config_manager.set("pipelines.threat_intelligence.confidence_threshold", 1.5)
        config_manager.set("pipelines.video_surveillance.processing_interval", -1)
        
        errors = config_manager.validate_config()
        assert "threat_intelligence" in errors
        assert "video_surveillance" in errors
    
    def test_save_config(self):
        """Test saving configuration to file"""
        config_manager = ConfigManager()
        config_manager.set("test.value", "saved")
        
        save_path = Path(self.temp_dir) / "saved_config.json"
        success = config_manager.save_config(str(save_path))
        
        assert success is True
        assert save_path.exists()
        
        # Load and verify
        with open(save_path, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config["test"]["value"] == "saved"


class TestFileStorageManager:
    """Test file storage functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_manager = FileStorageManager(self.temp_dir)
    
    def test_store_and_retrieve_file(self):
        """Test storing and retrieving files"""
        test_data = b"This is test file content"
        filename = "test_file.txt"
        
        # Store file
        stored_file = self.storage_manager.store_file(
            file_data=test_data,
            original_name=filename,
            media_type=MediaType.DOCUMENT
        )
        
        assert stored_file.original_name == filename
        assert stored_file.file_size == len(test_data)
        assert stored_file.media_type == MediaType.DOCUMENT
        
        # Retrieve file
        retrieved_data = self.storage_manager.retrieve_file(stored_file.file_id)
        assert retrieved_data == test_data
    
    def test_store_duplicate_file(self):
        """Test storing duplicate files"""
        test_data = b"Duplicate content"
        filename = "duplicate.txt"
        
        # Store file twice
        stored_file1 = self.storage_manager.store_file(
            file_data=test_data,
            original_name=filename,
            media_type=MediaType.DOCUMENT
        )
        
        stored_file2 = self.storage_manager.store_file(
            file_data=test_data,
            original_name=filename,
            media_type=MediaType.DOCUMENT
        )
        
        # Should return the same file
        assert stored_file1.file_id == stored_file2.file_id
        assert stored_file1.checksum == stored_file2.checksum
    
    def test_store_file_from_path(self):
        """Test storing file from filesystem path"""
        # Create temporary source file
        source_path = Path(self.temp_dir) / "source.txt"
        test_content = "Source file content"
        
        with open(source_path, 'w') as f:
            f.write(test_content)
        
        # Store file (copy mode)
        stored_file = self.storage_manager.store_file_from_path(
            source_path=str(source_path),
            media_type=MediaType.DOCUMENT,
            copy_file=True
        )
        
        assert stored_file.original_name == "source.txt"
        assert source_path.exists()  # Original should still exist
        
        # Verify content
        retrieved_data = self.storage_manager.retrieve_file(stored_file.file_id)
        assert retrieved_data.decode() == test_content
    
    def test_delete_file(self):
        """Test deleting files"""
        test_data = b"File to delete"
        
        stored_file = self.storage_manager.store_file(
            file_data=test_data,
            original_name="delete_me.txt",
            media_type=MediaType.DOCUMENT
        )
        
        file_id = stored_file.file_id
        
        # Verify file exists
        assert self.storage_manager.get_file_info(file_id) is not None
        
        # Delete file
        success = self.storage_manager.delete_file(file_id)
        assert success is True
        
        # Verify file is gone
        assert self.storage_manager.get_file_info(file_id) is None
        assert self.storage_manager.retrieve_file(file_id) is None
    
    def test_list_files(self):
        """Test listing files with filters"""
        # Store multiple files
        for i in range(5):
            self.storage_manager.store_file(
                file_data=f"Content {i}".encode(),
                original_name=f"file_{i}.txt",
                media_type=MediaType.DOCUMENT
            )
        
        # Store an image
        self.storage_manager.store_file(
            file_data=b"fake image data",
            original_name="image.jpg",
            media_type=MediaType.IMAGE
        )
        
        # Test listing all files
        all_files = self.storage_manager.list_files()
        assert len(all_files) == 6
        
        # Test filtering by media type
        documents = self.storage_manager.list_files(media_type=MediaType.DOCUMENT)
        assert len(documents) == 5
        
        images = self.storage_manager.list_files(media_type=MediaType.IMAGE)
        assert len(images) == 1
        
        # Test pagination
        limited_files = self.storage_manager.list_files(limit=3)
        assert len(limited_files) == 3
        
        offset_files = self.storage_manager.list_files(limit=3, offset=3)
        assert len(offset_files) == 3
    
    def test_storage_stats(self):
        """Test storage statistics"""
        # Store files of different types
        self.storage_manager.store_file(
            file_data=b"Document content",
            original_name="doc.txt",
            media_type=MediaType.DOCUMENT
        )
        
        self.storage_manager.store_file(
            file_data=b"Image data" * 100,  # Larger file
            original_name="image.jpg",
            media_type=MediaType.IMAGE
        )
        
        stats = self.storage_manager.get_storage_stats()
        
        assert stats["total_files"] == 2
        assert stats["total_size"] > 0
        assert "document" in stats["by_media_type"]
        assert "image" in stats["by_media_type"]
        assert stats["by_media_type"]["document"]["count"] == 1
        assert stats["by_media_type"]["image"]["count"] == 1
    
    def test_base64_image_storage(self):
        """Test storing image from base64 data"""
        # Create fake base64 image data
        import base64
        fake_image_data = b"fake jpeg data"
        base64_data = base64.b64encode(fake_image_data).decode()
        
        stored_file = store_image_from_base64(
            base64_data=base64_data,
            filename="test_image.jpg",
            storage_manager=self.storage_manager
        )
        
        assert stored_file.media_type == MediaType.IMAGE
        assert stored_file.original_name == "test_image.jpg"
        
        # Verify content
        retrieved_data = self.storage_manager.retrieve_file(stored_file.file_id)
        assert retrieved_data == fake_image_data


class TestBasePipeline:
    """Test base pipeline functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create a concrete implementation for testing
        class TestPipeline(BasePipeline):
            def __init__(self):
                super().__init__("test_pipeline")
                self.processed_data = []
            
            async def process_input(self, input_data):
                self.processed_data.append(input_data)
                return {"result": f"processed_{input_data}"}
            
            def create_alert(self, processing_result):
                # Don't create alerts for testing
                return None
            
            async def get_input_data(self):
                # Return test data for a few iterations
                if len(self.processed_data) < 3:
                    return f"test_input_{len(self.processed_data)}"
                return None
        
        self.pipeline = TestPipeline()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.pipeline_name == "test_pipeline"
        assert self.pipeline.status == PipelineStatus.STOPPED
        assert isinstance(self.pipeline.metrics, PipelineMetrics)
    
    def test_pipeline_status_management(self):
        """Test pipeline status changes"""
        assert self.pipeline.status == PipelineStatus.STOPPED
        
        # Test status changes
        import asyncio
        
        async def test_status():
            # Start pipeline (will run briefly)
            task = asyncio.create_task(self.pipeline.start())
            
            # Give it time to start
            await asyncio.sleep(0.1)
            
            # Check it's running
            assert self.pipeline.status == PipelineStatus.RUNNING
            
            # Stop pipeline
            await self.pipeline.stop()
            assert self.pipeline.status == PipelineStatus.STOPPED
            
            # Wait for task to complete
            try:
                await task
            except:
                pass  # Expected to be cancelled
        
        asyncio.run(test_status())
    
    def test_pipeline_metrics(self):
        """Test pipeline metrics calculation"""
        metrics = PipelineMetrics()
        
        # Test initial state
        assert metrics.processed_items == 0
        assert metrics.success_rate == 0.0
        assert metrics.processing_rate == 0.0
        
        # Simulate processing
        metrics.processed_items = 10
        metrics.successful_items = 8
        metrics.processing_time_total = 5.0
        
        assert metrics.success_rate == 0.8
        assert metrics.processing_rate == 2.0
        assert metrics.average_processing_time == 0.5
    
    @patch('requests.post')
    def test_send_alert(self, mock_post):
        """Test sending alerts to API"""
        from models.alerts import BaseAlert
        from datetime import datetime
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create test alert
        alert = BaseAlert(
            id="test_alert",
            timestamp=datetime.now(),
            confidence=0.8,
            source_pipeline="test_pipeline"
        )
        
        # Test sending alert
        import asyncio
        asyncio.run(self.pipeline.send_alert(alert))
        
        # Verify API call was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "alerts/threat" in call_args[1]["url"]  # Default endpoint
        assert call_args[1]["json"]["id"] == "test_alert"


class TestLoggingUtilities:
    """Test logging utilities"""
    
    def test_logger_creation(self):
        """Test logger creation and configuration"""
        logger = get_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output"""
        temp_dir = tempfile.mkdtemp()
        log_file = Path(temp_dir) / "test.log"
        
        logger = setup_logging(
            name="file_test",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        # Test logging
        logger.info("Test message")
        
        # Check file was created and contains message
        assert log_file.exists()
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_error_tracker(self):
        """Test error tracking functionality"""
        tracker = ErrorTracker()
        
        # Track some errors
        error1 = ValueError("Test error 1")
        error2 = RuntimeError("Test error 2")
        
        tracker.track_error("component1", error1)
        tracker.track_error("component1", error1)  # Same error again
        tracker.track_error("component2", error2)
        
        # Get summary
        summary = tracker.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert "component1:ValueError" in summary["error_breakdown"]
        assert "component2:RuntimeError" in summary["error_breakdown"]
        assert summary["error_breakdown"]["component1:ValueError"] == 2


if __name__ == "__main__":
    pytest.main([__file__])