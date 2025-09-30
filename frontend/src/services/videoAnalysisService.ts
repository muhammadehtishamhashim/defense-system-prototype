import api from './api';

export interface VideoInfo {
  filename: string;
  display_name: string;
  duration: number;
  resolution: [number, number];
  format: string;
  file_size: number;
  created_at: string;
  fps: number;
}

export interface AnalysisSession {
  session_id: string;
  video_filename: string;
  status: 'starting' | 'running' | 'paused' | 'stopped' | 'error';
  started_at: string;
  frames_processed: number;
  alerts_generated: number;
  current_frame: number;
  total_frames: number;
  fps: number;
  progress_percent: number;
}

export interface StartAnalysisRequest {
  video_filename: string;
  mock_alerts?: boolean;
  alert_interval?: number;
}

export interface StartAnalysisResponse {
  session_id: string;
  message: string;
  success: boolean;
}

class VideoAnalysisService {
  async getAvailableVideos(): Promise<VideoInfo[]> {
    const response = await api.get('/api/videos');
    return response.data;
  }

  async getVideoInfo(filename: string): Promise<VideoInfo> {
    const response = await api.get(`/api/videos/${filename}/info`);
    return response.data;
  }

  async getVideoStreamUrl(filename: string): Promise<string> {
    const baseUrl = api.defaults.baseURL || 'http://localhost:8000';
    return `${baseUrl}/api/videos/${filename}`;
  }

  async startAnalysis(request: StartAnalysisRequest): Promise<StartAnalysisResponse> {
    const response = await api.post('/api/analysis/start', request);
    return response.data;
  }

  async stopAnalysis(sessionId: string): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/api/analysis/stop', null, {
      params: { session_id: sessionId }
    });
    return response.data;
  }

  async getAnalysisStatus(sessionId: string): Promise<AnalysisSession> {
    const response = await api.get(`/api/analysis/status/${sessionId}`);
    return response.data;
  }

  async listAnalysisSessions(): Promise<AnalysisSession[]> {
    const response = await api.get('/api/analysis/sessions');
    return response.data;
  }
}

export const videoAnalysisService = new VideoAnalysisService();