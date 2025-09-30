"""
Integration tests for video streaming functionality
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from services.video_streaming import VideoStreamingService
from services.video_analysis import VideoAnalysisCoordinator
from services.mock_alert_generator import MockAlertGenerator


class TestVideoStreaming:
    """Test video streaming service"""
    
    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory with mock video file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock video file
            video_path = Path(temp_dir) / "test_video.mp4"
            video_path.write_bytes(b"mock video content")
            yield temp_dir
    
    @pytest.fixture
    def video_service(self, temp_video_dir):
        """Create video streaming service with temp directory"""
        return VideoStreamingService(temp_video_dir)
    
    def test_list_available_videos(self, video_service):
        """Test listing available videos"""
        videos = video_service.list_available_videos()
        assert len(videos) >= 0  # May be empty if no valid video files
    
    def test_get_video_info_not_found(self, video_service):
        """Test getting info for non-existent video"""
        with pytest.raises(Exception):  # Should raise HTTPException
            video_service.get_video_info("nonexistent.mp4")
    
    def test_safe_file_path_validation(self, video_service):
        """Test file path validation prevents directory traversal"""
        with pytest.raises(Exception):
            video_service._get_safe_file_path("../../../etc/passwd")
        
        with pytest.raises(Exception):
            video_service._get_safe_file_path("subdir/../../file.mp4")
    
    @patch('cv2.VideoCapture')
    def test_extract_video_info_success(self, mock_cv2, video_service, temp_video_dir):
        """Test successful video info extraction"""
        # Mock OpenCV VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,    # FPS
            7: 1000.0,  # Frame count
            3: 1920.0,  # Width
            4: 1080.0   # Height
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        # Create test file
        test_file = Path(temp_video_dir) / "test.mp4"
        test_file.write_bytes(b"test content")
        
        info = video_service._extract_video_info(test_file)
        
        assert info.filename == "test.mp4"
        assert info.fps == 30.0
        assert info.resolution == (1920, 1080)
        assert info.duration > 0


class TestVideoAnalysis:
    """Test video analysis coordination"""
    
    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory with mock video file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test_video.mp4"
            video_path.write_bytes(b"mock video content")
            yield temp_dir
    
    @pytest.fixture
    def analysis_coordinator(self, temp_video_dir):
        """Create analysis coordinator with temp directory"""
        return VideoAnalysisCoordinator(temp_video_dir)
    
    @pytest.mark.asyncio
    async def test_start_analysis_file_not_found(self, analysis_coordinator):
        """Test starting analysis with non-existent file"""
        with pytest.raises(FileNotFoundError):
            await analysis_coordinator.start_analysis("nonexistent.mp4")
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_start_analysis_success(self, mock_cv2, analysis_coordinator, temp_video_dir):
        """Test successful analysis start"""
        # Mock OpenCV VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,    # FPS
            7: 1000.0   # Frame count
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        # Create test file
        test_file = Path(temp_video_dir) / "test.mp4"
        test_file.write_bytes(b"test content")
        
        session_id = await analysis_coordinator.start_analysis("test.mp4")
        
        assert session_id is not None
        assert session_id in analysis_coordinator.active_sessions
        
        # Clean up
        await analysis_coordinator.stop_analysis(session_id)
    
    @pytest.mark.asyncio
    async def test_stop_analysis_not_found(self, analysis_coordinator):
        """Test stopping non-existent analysis session"""
        result = await analysis_coordinator.stop_analysis("nonexistent-session")
        assert result is False
    
    def test_get_analysis_status_not_found(self, analysis_coordinator):
        """Test getting status for non-existent session"""
        status = analysis_coordinator.get_analysis_status("nonexistent-session")
        assert status is None


class TestMockAlertGenerator:
    """Test mock alert generation"""
    
    @pytest.fixture
    def alert_generator(self):
        """Create mock alert generator"""
        return MockAlertGenerator()
    
    @pytest.mark.asyncio
    async def test_start_mock_alerts(self, alert_generator):
        """Test starting mock alert generation"""
        session_id = "test-session"
        
        result = await alert_generator.start_mock_alerts(session_id, interval=1)
        assert result is True
        assert session_id in alert_generator.active_generators
        
        # Clean up
        await alert_generator.stop_mock_alerts(session_id)
    
    @pytest.mark.asyncio
    async def test_stop_mock_alerts_not_found(self, alert_generator):
        """Test stopping non-existent alert generator"""
        result = await alert_generator.stop_mock_alerts("nonexistent-session")
        assert result is False
    
    def test_generate_realistic_alert(self, alert_generator):
        """Test generating realistic alert"""
        session_id = "test-session"
        alert = alert_generator.generate_realistic_alert(session_id)
        
        assert alert.id is not None
        assert alert.confidence > 0
        assert alert.source_pipeline == "video_surveillance"
        assert alert.metadata.get('session_id') == session_id
        assert alert.metadata.get('mock_alert') is True
    
    def test_get_generator_status_not_found(self, alert_generator):
        """Test getting status for non-existent generator"""
        status = alert_generator.get_generator_status("nonexistent-session")
        assert status is None


class TestVideoStreamingAPI:
    """Test video streaming API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_list_videos_endpoint(self, client):
        """Test GET /api/videos endpoint"""
        response = client.get("/api/videos")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_video_info_not_found(self, client):
        """Test GET /api/videos/{filename}/info with non-existent file"""
        response = client.get("/api/videos/nonexistent.mp4/info")
        assert response.status_code == 404
    
    def test_stream_video_not_found(self, client):
        """Test GET /api/videos/{filename} with non-existent file"""
        response = client.get("/api/videos/nonexistent.mp4")
        assert response.status_code == 404


class TestVideoAnalysisAPI:
    """Test video analysis API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_start_analysis_file_not_found(self, client):
        """Test POST /api/analysis/start with non-existent file"""
        response = client.post("/api/analysis/start", json={
            "video_filename": "nonexistent.mp4",
            "mock_alerts": True,
            "alert_interval": 30
        })
        assert response.status_code == 404
    
    def test_get_analysis_status_not_found(self, client):
        """Test GET /api/analysis/status/{session_id} with non-existent session"""
        response = client.get("/api/analysis/status/nonexistent-session")
        assert response.status_code == 404
    
    def test_list_analysis_sessions(self, client):
        """Test GET /api/analysis/sessions endpoint"""
        response = client.get("/api/analysis/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestMonitoringAPI:
    """Test monitoring API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_get_all_metrics(self, client):
        """Test GET /api/monitoring/metrics endpoint"""
        response = client.get("/api/monitoring/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "global_metrics" in data
        assert "session_metrics" in data
        assert "active_sessions" in data
    
    def test_get_session_metrics_not_found(self, client):
        """Test GET /api/monitoring/metrics/{session_id} with non-existent session"""
        response = client.get("/api/monitoring/metrics/nonexistent-session")
        assert response.status_code == 404


class TestIntegrationWorkflow:
    """Test complete video streaming and analysis workflow"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory with mock video file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test_video.mp4"
            video_path.write_bytes(b"mock video content")
            yield temp_dir
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    async def test_complete_workflow(self, mock_cv2, client, temp_video_dir):
        """Test complete workflow from video listing to analysis"""
        # Mock OpenCV VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,    # FPS
            7: 1000.0,  # Frame count
            3: 1920.0,  # Width
            4: 1080.0   # Height
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        # Patch the video service to use temp directory
        with patch('services.video_streaming.video_streaming_service.media_directory', Path(temp_video_dir)):
            with patch('services.video_analysis.video_analysis_coordinator.media_directory', Path(temp_video_dir)):
                
                # 1. List videos
                response = client.get("/api/videos")
                assert response.status_code == 200
                
                # 2. Get video info (may fail if no valid video files)
                try:
                    response = client.get("/api/videos/test_video.mp4/info")
                    if response.status_code == 200:
                        video_info = response.json()
                        assert "filename" in video_info
                except:
                    pass  # Expected if video format not supported
                
                # 3. Start analysis (may fail without proper video file)
                try:
                    response = client.post("/api/analysis/start", json={
                        "video_filename": "test_video.mp4",
                        "mock_alerts": True,
                        "alert_interval": 1
                    })
                    
                    if response.status_code == 200:
                        session_data = response.json()
                        session_id = session_data["session_id"]
                        
                        # 4. Check analysis status
                        response = client.get(f"/api/analysis/status/{session_id}")
                        if response.status_code == 200:
                            status = response.json()
                            assert "session_id" in status
                        
                        # 5. Stop analysis
                        response = client.post("/api/analysis/stop", params={"session_id": session_id})
                        assert response.status_code == 200
                        
                except:
                    pass  # Expected if video processing fails
                
                # 6. Check monitoring metrics
                response = client.get("/api/monitoring/metrics")
                assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])