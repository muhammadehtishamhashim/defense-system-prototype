import { useState, useEffect, useCallback } from 'react';
import { videoAnalysisService } from '../services/videoAnalysisService';
import type { AnalysisSession, VideoInfo } from '../services/videoAnalysisService';

export interface UseVideoAnalysisReturn {
  // Video data
  videos: VideoInfo[];
  selectedVideo: VideoInfo | null;
  videoStreamUrl: string | null;
  
  // Analysis state
  analysisSession: AnalysisSession | null;
  analysisStatus: 'idle' | 'starting' | 'running' | 'paused' | 'stopped' | 'error';
  
  // Loading states
  loading: boolean;
  error: string | null;
  
  // Actions
  selectVideo: (video: VideoInfo) => void;
  startAnalysis: (mockAlerts?: boolean, alertInterval?: number) => Promise<void>;
  stopAnalysis: () => Promise<void>;
  refreshStatus: () => Promise<void>;
  loadVideos: () => Promise<void>;
}

export const useVideoAnalysis = (): UseVideoAnalysisReturn => {
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoInfo | null>(null);
  const [videoStreamUrl, setVideoStreamUrl] = useState<string | null>(null);
  const [analysisSession, setAnalysisSession] = useState<AnalysisSession | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'starting' | 'running' | 'paused' | 'stopped' | 'error'>('idle');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available videos
  const loadVideos = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const videoList = await videoAnalysisService.getAvailableVideos();
      setVideos(videoList);
    } catch (err) {
      setError('Failed to load videos');
      console.error('Error loading videos:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Select a video
  const selectVideo = useCallback(async (video: VideoInfo) => {
    try {
      setSelectedVideo(video);
      const streamUrl = await videoAnalysisService.getVideoStreamUrl(video.filename);
      setVideoStreamUrl(streamUrl);
      
      // Reset analysis state when selecting new video
      setAnalysisSession(null);
      setAnalysisStatus('idle');
    } catch (err) {
      setError('Failed to get video stream URL');
      console.error('Error getting video stream URL:', err);
    }
  }, []);

  // Start video analysis
  const startAnalysis = useCallback(async (mockAlerts = true, alertInterval = 30) => {
    if (!selectedVideo) {
      setError('No video selected');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setAnalysisStatus('starting');

      const response = await videoAnalysisService.startAnalysis({
        video_filename: selectedVideo.filename,
        mock_alerts: mockAlerts,
        alert_interval: alertInterval
      });

      if (response.success) {
        // Start polling for status updates
        const sessionId = response.session_id;
        pollAnalysisStatus(sessionId);
      } else {
        throw new Error(response.message);
      }
    } catch (err) {
      setError('Failed to start analysis');
      setAnalysisStatus('error');
      console.error('Error starting analysis:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedVideo]);

  // Stop video analysis
  const stopAnalysis = useCallback(async () => {
    if (!analysisSession) {
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await videoAnalysisService.stopAnalysis(analysisSession.session_id);
      
      if (response.success) {
        setAnalysisStatus('stopped');
        setAnalysisSession(null);
      } else {
        throw new Error(response.message);
      }
    } catch (err) {
      setError('Failed to stop analysis');
      console.error('Error stopping analysis:', err);
    } finally {
      setLoading(false);
    }
  }, [analysisSession]);

  // Refresh analysis status
  const refreshStatus = useCallback(async () => {
    if (!analysisSession) {
      return;
    }

    try {
      const status = await videoAnalysisService.getAnalysisStatus(analysisSession.session_id);
      setAnalysisSession(status);
      setAnalysisStatus(status.status);
    } catch (err) {
      console.error('Error refreshing analysis status:', err);
      // Don't set error for status refresh failures
    }
  }, [analysisSession]);

  // Poll analysis status
  const pollAnalysisStatus = useCallback(async (sessionId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await videoAnalysisService.getAnalysisStatus(sessionId);
        setAnalysisSession(status);
        setAnalysisStatus(status.status);

        // Stop polling if analysis is stopped or errored
        if (status.status === 'stopped' || status.status === 'error') {
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error('Error polling analysis status:', err);
        clearInterval(pollInterval);
        setAnalysisStatus('error');
      }
    }, 2000); // Poll every 2 seconds

    // Clean up interval on unmount
    return () => clearInterval(pollInterval);
  }, []);

  // Load videos on mount
  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  return {
    // Video data
    videos,
    selectedVideo,
    videoStreamUrl,
    
    // Analysis state
    analysisSession,
    analysisStatus,
    
    // Loading states
    loading,
    error,
    
    // Actions
    selectVideo,
    startAnalysis,
    stopAnalysis,
    refreshStatus,
    loadVideos
  };
};