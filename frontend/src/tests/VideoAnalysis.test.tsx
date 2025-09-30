/**
 * Integration tests for video analysis functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import VideoAnalysis from '../pages/VideoAnalysis';
import { videoAnalysisService } from '../services/videoAnalysisService';
import { sseService } from '../services/sseService';

// Mock the services
vi.mock('../services/videoAnalysisService');
vi.mock('../services/sseService');
vi.mock('../services/alertService');

// Mock video player component
vi.mock('../components/video/VideoPlayer', () => ({
  default: ({ src, onAnalysisStart, onAnalysisStop, analysisStatus }: any) => (
    <div data-testid="video-player">
      <div>Video Source: {src}</div>
      <div>Analysis Status: {analysisStatus}</div>
      {onAnalysisStart && (
        <button onClick={onAnalysisStart} data-testid="start-analysis">
          Start Analysis
        </button>
      )}
      {onAnalysisStop && (
        <button onClick={onAnalysisStop} data-testid="stop-analysis">
          Stop Analysis
        </button>
      )}
    </div>
  )
}));

// Mock other components
vi.mock('../components/video/SnapshotGallery', () => ({
  default: () => <div data-testid="snapshot-gallery">Snapshot Gallery</div>
}));

vi.mock('../components/video/VideoTimeline', () => ({
  default: () => <div data-testid="video-timeline">Video Timeline</div>
}));

const mockVideoAnalysisService = videoAnalysisService as any;
const mockSSEService = sseService as any;

describe('VideoAnalysis Integration Tests', () => {
  const mockVideos = [
    {
      filename: 'test-video.mp4',
      display_name: 'Test Video',
      duration: 120,
      resolution: [1920, 1080] as [number, number],
      format: '.mp4',
      file_size: 1024000,
      created_at: '2023-01-01T00:00:00Z',
      fps: 30
    },
    {
      filename: 'demo-video.mp4',
      display_name: 'Demo Video',
      duration: 60,
      resolution: [1280, 720] as [number, number],
      format: '.mp4',
      file_size: 512000,
      created_at: '2023-01-02T00:00:00Z',
      fps: 25
    }
  ];

  const mockAnalysisSession = {
    session_id: 'test-session-123',
    video_filename: 'test-video.mp4',
    status: 'running' as const,
    started_at: '2023-01-01T10:00:00Z',
    frames_processed: 150,
    alerts_generated: 5,
    current_frame: 150,
    total_frames: 3600,
    fps: 30,
    progress_percent: 4.17
  };

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    
    // Setup default mock implementations
    mockVideoAnalysisService.getAvailableVideos.mockResolvedValue(mockVideos);
    mockVideoAnalysisService.getVideoStreamUrl.mockResolvedValue('http://localhost:8000/api/videos/test-video.mp4');
    mockVideoAnalysisService.startAnalysis.mockResolvedValue({
      session_id: 'test-session-123',
      message: 'Analysis started',
      success: true
    });
    mockVideoAnalysisService.stopAnalysis.mockResolvedValue({
      success: true,
      message: 'Analysis stopped'
    });
    mockVideoAnalysisService.getAnalysisStatus.mockResolvedValue(mockAnalysisSession);
    
    mockSSEService.isConnected.mockReturnValue(false);
    mockSSEService.connect.mockResolvedValue(undefined);
    mockSSEService.onNewAlert.mockReturnValue(() => {});
    
    // Mock window.addEventListener
    global.addEventListener = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders video analysis page with loading state', async () => {
    mockVideoAnalysisService.getAvailableVideos.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve(mockVideos), 100))
    );

    render(<VideoAnalysis />);
    
    expect(screen.getByText('Loading videos...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Available Videos')).toBeInTheDocument();
    });
  });

  it('displays available videos correctly', async () => {
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Video')).toBeInTheDocument();
      expect(screen.getByText('Demo Video')).toBeInTheDocument();
      expect(screen.getByText('Duration: 120s')).toBeInTheDocument();
      expect(screen.getByText('Resolution: 1920x1080')).toBeInTheDocument();
    });
  });

  it('handles video selection and displays video player', async () => {
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Video')).toBeInTheDocument();
    });
    
    // Click on first video
    fireEvent.click(screen.getByText('Test Video'));
    
    await waitFor(() => {
      expect(mockVideoAnalysisService.getVideoStreamUrl).toHaveBeenCalledWith('test-video.mp4');
      expect(screen.getByTestId('video-player')).toBeInTheDocument();
      expect(screen.getByText('Video Source: http://localhost:8000/api/videos/test-video.mp4')).toBeInTheDocument();
    });
  });

  it('starts video analysis when start button is clicked', async () => {
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Video')).toBeInTheDocument();
    });
    
    // Select video
    fireEvent.click(screen.getByText('Test Video'));
    
    await waitFor(() => {
      expect(screen.getByText('Start Analysis')).toBeInTheDocument();
    });
    
    // Start analysis
    fireEvent.click(screen.getByText('Start Analysis'));
    
    await waitFor(() => {
      expect(mockVideoAnalysisService.startAnalysis).toHaveBeenCalledWith({
        video_filename: 'test-video.mp4',
        mock_alerts: true,
        alert_interval: 30
      });
    });
  });

  it('displays analysis session metrics', async () => {
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Video')).toBeInTheDocument();
    });
    
    // Select video and start analysis
    fireEvent.click(screen.getByText('Test Video'));
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Start Analysis'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument(); // Frames processed
      expect(screen.getByText('5')).toBeInTheDocument(); // Alerts generated
      expect(screen.getByText('4%')).toBeInTheDocument(); // Progress
      expect(screen.getByText('30')).toBeInTheDocument(); // FPS
    });
  });

  it('stops video analysis when stop button is clicked', async () => {
    // Mock running analysis session
    mockVideoAnalysisService.getAnalysisStatus.mockResolvedValue({
      ...mockAnalysisSession,
      status: 'running'
    });
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Test Video'));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Start Analysis'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Stop Analysis')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('Stop Analysis'));
    
    await waitFor(() => {
      expect(mockVideoAnalysisService.stopAnalysis).toHaveBeenCalledWith('test-session-123');
    });
  });

  it('handles SSE connection and displays real-time alerts', async () => {
    let alertCallback: (alert: any) => void = () => {};
    
    mockSSEService.onNewAlert.mockImplementation((callback: any) => {
      alertCallback = callback;
      return () => {};
    });
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(mockSSEService.connect).toHaveBeenCalled();
    });
    
    // Simulate receiving an alert
    const mockAlert = {
      id: 'alert-123',
      timestamp: '2023-01-01T10:30:00Z',
      confidence: 0.85,
      source_pipeline: 'video_surveillance',
      status: 'active',
      event_type: 'loitering',
      metadata: {
        description: 'Person detected loitering near entrance'
      }
    };
    
    alertCallback(mockAlert);
    
    await waitFor(() => {
      expect(screen.getByText('1 alerts received')).toBeInTheDocument();
      expect(screen.getByText('loitering')).toBeInTheDocument();
      expect(screen.getByText('Confidence: 85%')).toBeInTheDocument();
    });
  });

  it('handles errors gracefully', async () => {
    mockVideoAnalysisService.getAvailableVideos.mockRejectedValue(new Error('Network error'));
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load videos')).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  it('refreshes video list when retry button is clicked', async () => {
    mockVideoAnalysisService.getAvailableVideos
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(mockVideos);
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('Retry'));
    
    await waitFor(() => {
      expect(mockVideoAnalysisService.getAvailableVideos).toHaveBeenCalledTimes(2);
      expect(screen.getByText('Test Video')).toBeInTheDocument();
    });
  });

  it('displays appropriate message when no videos are available', async () => {
    mockVideoAnalysisService.getAvailableVideos.mockResolvedValue([]);
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      expect(screen.getByText('No videos available. Please add video files to the backend/media/videos directory.')).toBeInTheDocument();
    });
  });

  it('shows waiting message when analysis is running but no alerts received', async () => {
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Test Video'));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Start Analysis'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Waiting for alerts... (Mock alerts generate every 30 seconds)')).toBeInTheDocument();
    });
  });
});

describe('VideoAnalysis Service Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('handles API errors correctly', async () => {
    const mockError = new Error('API Error');
    mockVideoAnalysisService.startAnalysis.mockRejectedValue(mockError);
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Test Video'));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Start Analysis'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to start analysis')).toBeInTheDocument();
    });
  });

  it('polls analysis status correctly', async () => {
    vi.useFakeTimers();
    
    render(<VideoAnalysis />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Test Video'));
    });
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Start Analysis'));
    });
    
    // Fast-forward time to trigger polling
    vi.advanceTimersByTime(2000);
    
    await waitFor(() => {
      expect(mockVideoAnalysisService.getAnalysisStatus).toHaveBeenCalledWith('test-session-123');
    });
    
    vi.useRealTimers();
  });
});