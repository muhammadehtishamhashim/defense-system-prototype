import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { VideoPlayer, SnapshotGallery, VideoTimeline } from '../components/video';
import { alertService } from '../services/alertService';
import { useVideoAnalysis } from '../hooks/useVideoAnalysis';
import { sseService } from '../services/sseService';
import type { VideoAlert, BaseAlert } from '../types';
import Badge from '../components/ui/Badge';
import Button from '../components/ui/Button';
import { 
  VideoCameraIcon,
  PhotoIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  PlayIcon,
  StopIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

const VideoAnalysis = () => {
  const {
    videos,
    selectedVideo,
    videoStreamUrl,
    analysisSession,
    analysisStatus,
    loading,
    error,
    selectVideo,
    startAnalysis,
    stopAnalysis,
    loadVideos
  } = useVideoAnalysis();

  const [videoAlerts, setVideoAlerts] = useState<VideoAlert[]>([]);
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [realtimeAlerts, setRealtimeAlerts] = useState<BaseAlert[]>([]);
  const [sseConnected, setSseConnected] = useState(false);

  useEffect(() => {
    fetchVideoAlerts();
    setupSSEConnection();
  }, []);

  const setupSSEConnection = async () => {
    try {
      if (!sseService.isConnected()) {
        await sseService.connect();
      }
      setSseConnected(true);
      
      // Subscribe to new alerts
      sseService.onNewAlert((alert: BaseAlert) => {
        console.log('Received real-time alert via onNewAlert:', alert);
        
        // Add to real-time alerts for display
        setRealtimeAlerts(prev => [alert, ...prev].slice(0, 10));
        
        // Convert to VideoAlert and add to videoAlerts for statistics
        if (alert.source_pipeline === 'video_surveillance') {
          const videoAlert = {
            ...alert,
            event_type: (alert as any).event_type || 'unknown',
            bounding_box: (alert as any).bounding_box || [0, 0, 0, 0],
            track_id: (alert as any).track_id || 0,
            snapshot_path: (alert as any).snapshot_path || '',
            video_timestamp: (alert as any).video_timestamp || 0
          } as VideoAlert;
          
          setVideoAlerts(prev => [videoAlert, ...prev].slice(0, 50));
          
          // Generate snapshot from alert
          const snapshotPath = (alert as any).snapshot_path || '';
          const snapshotUrl = snapshotPath ? `http://localhost:8000/${snapshotPath}` : '/placeholder-snapshot.jpg';
          
          const snapshot = {
            id: alert.id,
            url: snapshotUrl,
            timestamp: new Date(alert.timestamp).getTime() / 1000,
            description: `${(alert as any).event_type || 'unknown'} detection - Track ${(alert as any).track_id || 0}`,
            boundingBoxes: (alert as any).bounding_box ? [{
              x: (alert as any).bounding_box[0],
              y: (alert as any).bounding_box[1],
              width: (alert as any).bounding_box[2] - (alert as any).bounding_box[0],
              height: (alert as any).bounding_box[3] - (alert as any).bounding_box[1],
              label: (alert as any).event_type || 'unknown',
              confidence: alert.confidence
            }] : []
          };
          
          setSnapshots(prev => [snapshot, ...prev].slice(0, 20));
        }
      });
      
      // Also subscribe to all events for debugging
      sseService.onAnyEvent((message: any) => {
        console.log('Received SSE message:', message);
        if (message.type === 'alert' && message.data.source_pipeline === 'video_surveillance') {
          console.log('Video alert received via onAnyEvent:', message.data);
          
          // Add to real-time alerts for display
          setRealtimeAlerts(prev => [message.data, ...prev].slice(0, 10));
          
          // Convert to VideoAlert and add to videoAlerts for statistics
          const videoAlert = {
            ...message.data,
            event_type: message.data.event_type || 'unknown',
            bounding_box: message.data.bounding_box || [0, 0, 0, 0],
            track_id: message.data.track_id || 0,
            snapshot_path: message.data.snapshot_path || '',
            video_timestamp: message.data.video_timestamp || 0
          } as VideoAlert;
          
          setVideoAlerts(prev => [videoAlert, ...prev].slice(0, 50));
          
          // Generate snapshot from alert
          const snapshotPath = message.data.snapshot_path || '';
          const snapshotUrl = snapshotPath ? `http://localhost:8000/${snapshotPath}` : '/placeholder-snapshot.jpg';
          
          const snapshot = {
            id: message.data.id,
            url: snapshotUrl,
            timestamp: new Date(message.data.timestamp).getTime() / 1000,
            description: `${message.data.event_type || 'unknown'} detection - Track ${message.data.track_id || 0}`,
            boundingBoxes: message.data.bounding_box ? [{
              x: message.data.bounding_box[0],
              y: message.data.bounding_box[1],
              width: message.data.bounding_box[2] - message.data.bounding_box[0],
              height: message.data.bounding_box[3] - message.data.bounding_box[1],
              label: message.data.event_type || 'unknown',
              confidence: message.data.confidence
            }] : []
          };
          
          setSnapshots(prev => [snapshot, ...prev].slice(0, 20));
        }
      });
      
      // Listen for connection changes
      window.addEventListener('sse:connection', (event: any) => {
        setSseConnected(event.detail.connected);
      });
      
    } catch (error) {
      console.error('Failed to setup SSE connection:', error);
      setSseConnected(false);
    }
  };

  const fetchVideoAlerts = async () => {
    try {
      const response = await alertService.getAlerts({ 
        pipeline: 'video_surveillance',
        limit: 50 
      });
      
      const videoAlerts = response.alerts
        .filter(alert => alert.pipeline === 'video_surveillance')
        .map(alert => ({
          ...alert,
          source_pipeline: alert.pipeline,
          event_type: alert.data?.event_type || 'unknown',
          bounding_box: alert.data?.bounding_box || [0, 0, 0, 0],
          track_id: alert.data?.track_id || 0,
          snapshot_path: alert.data?.snapshot_path || '',
          video_timestamp: alert.data?.video_timestamp || 0
        })) as VideoAlert[];
      
      setVideoAlerts(videoAlerts);
      
      // Generate mock snapshots from alerts
      const mockSnapshots = videoAlerts.map(alert => ({
        id: alert.id,
        url: alert.snapshot_path || '/placeholder-snapshot.jpg',
        timestamp: new Date(alert.timestamp).getTime() / 1000,
        description: `${alert.event_type} detection - Track ${alert.track_id}`,
        boundingBoxes: alert.bounding_box ? [{
          x: alert.bounding_box[0],
          y: alert.bounding_box[1],
          width: alert.bounding_box[2] - alert.bounding_box[0],
          height: alert.bounding_box[3] - alert.bounding_box[1],
          label: alert.event_type,
          confidence: alert.confidence
        }] : []
      }));
      
      setSnapshots(mockSnapshots);
    } catch (error) {
      console.error('Error fetching video alerts:', error);
    }
  };

  const handleVideoSelect = (video: any) => {
    selectVideo(video);
    setCurrentTime(0);
  };

  const handleStartAnalysis = async () => {
    await startAnalysis(true, 30); // Enable mock alerts with 30s interval
  };

  const handleStopAnalysis = async () => {
    await stopAnalysis();
  };

  const handleTimeUpdate = (time: number) => {
    setCurrentTime(time);
  };

  const handleSeek = (time: number) => {
    setCurrentTime(time);
    // In real implementation, seek the video player
  };

  const getBoundingBoxesForTime = (time: number) => {
    // Find alerts that match the current time (within 1 second)
    const relevantAlerts = videoAlerts.filter(alert => {
      const alertTime = new Date(alert.timestamp).getTime() / 1000;
      return Math.abs(alertTime - time) < 1;
    });

    return relevantAlerts.map((alert) => ({
      x: alert.bounding_box?.[0] || 0,
      y: alert.bounding_box?.[1] || 0,
      width: (alert.bounding_box?.[2] || 0) - (alert.bounding_box?.[0] || 0),
      height: (alert.bounding_box?.[3] || 0) - (alert.bounding_box?.[1] || 0),
      label: alert.event_type,
      confidence: alert.confidence,
      trackId: alert.track_id
    }));
  };

  const getTimelineEvents = () => {
    return videoAlerts.map(alert => ({
      timestamp: new Date(alert.timestamp).getTime() / 1000,
      type: 'alert' as const,
      label: `${alert.event_type} - Track ${alert.track_id}`,
      color: alert.event_type === 'loitering' ? '#ef4444' : 
             alert.event_type === 'zone_violation' ? '#f59e0b' : '#8b5cf6',
      data: alert
    }));
  };

  if (loading && videos.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-gray-900">Video Analysis</h2>
          <p className="text-gray-600">Real-time video surveillance and analysis</p>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading videos...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">Video Analysis</h2>
        <p className="text-gray-600">Real-time video surveillance and analysis</p>
      </div>

      {/* Error Display */}
      {error && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2 text-red-600">
              <ExclamationTriangleIcon className="h-5 w-5" />
              <span>{error}</span>
              <Button variant="outline" size="sm" onClick={loadVideos}>
                <ArrowPathIcon className="h-4 w-4 mr-1" />
                Retry
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Video Sources */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <VideoCameraIcon className="h-5 w-5" />
              <span>Available Videos</span>
            </div>
            <Button variant="outline" size="sm" onClick={loadVideos} disabled={loading}>
              <ArrowPathIcon className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {videos.map(video => (
              <div
                key={video.filename}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedVideo?.filename === video.filename 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleVideoSelect(video)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900">{video.display_name}</h3>
                  <Badge variant="success">Available</Badge>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Duration: {Math.round(video.duration)}s</div>
                  <div>Resolution: {video.resolution[0]}x{video.resolution[1]}</div>
                  <div>Format: {video.format}</div>
                </div>
              </div>
            ))}
          </div>
          {videos.length === 0 && !loading && (
            <div className="text-center py-8 text-gray-500">
              No videos available. Please add video files to the backend/media/videos directory.
            </div>
          )}
        </CardContent>
      </Card>

      {selectedVideo && videoStreamUrl && (
        <>
          {/* Analysis Controls */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Analysis Controls - {selectedVideo.display_name}</span>
                <div className="flex items-center space-x-2">
                  {analysisSession && (
                    <Badge variant={
                      analysisStatus === 'running' ? 'success' :
                      analysisStatus === 'starting' ? 'warning' :
                      analysisStatus === 'error' ? 'danger' : 'default'
                    }>
                      {analysisStatus}
                    </Badge>
                  )}
                  {analysisStatus === 'idle' || analysisStatus === 'stopped' ? (
                    <Button onClick={handleStartAnalysis} disabled={loading}>
                      <PlayIcon className="h-4 w-4 mr-1" />
                      Start Analysis
                    </Button>
                  ) : (
                    <Button variant="destructive" onClick={handleStopAnalysis} disabled={loading}>
                      <StopIcon className="h-4 w-4 mr-1" />
                      Stop Analysis
                    </Button>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {analysisSession && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="font-medium text-gray-700">Frames Processed</div>
                    <div className="text-lg font-semibold">{analysisSession.frames_processed}</div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-700">Alerts Generated</div>
                    <div className="text-lg font-semibold">{analysisSession.alerts_generated}</div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-700">Progress</div>
                    <div className="text-lg font-semibold">{Math.round(analysisSession.progress_percent)}%</div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-700">FPS</div>
                    <div className="text-lg font-semibold">{Math.round(analysisSession.fps)}</div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Video Player */}
          <Card>
            <CardHeader>
              <CardTitle>Video Stream - {selectedVideo.display_name}</CardTitle>
            </CardHeader>
            <CardContent>
              <VideoPlayer
                src={videoStreamUrl}
                boundingBoxes={getBoundingBoxesForTime(currentTime)}
                onTimeUpdate={handleTimeUpdate}
                showControls={true}
                autoPlay={true}
                loop={true}
                muted={true}
                analysisStatus={analysisStatus}
                onAnalysisStart={handleStartAnalysis}
                onAnalysisStop={handleStopAnalysis}
                className="w-full"
              />
            </CardContent>
          </Card>

          {/* Timeline */}
          <Card>
            <CardHeader>
              <CardTitle>Video Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-gray-600">
                Timeline showing {getTimelineEvents().length} alert events
              </div>
              <div className="mt-2 space-y-1">
                {getTimelineEvents().slice(0, 5).map((event, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">{event.label}</span>
                    <span className="text-xs text-gray-500">
                      {new Date(event.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Alert Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Video Alerts</CardTitle>
            <ExclamationTriangleIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{videoAlerts.length}</div>
            <p className="text-xs text-muted-foreground">
              From all video sources
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            <ClockIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {videoAlerts.filter(a => a.status === 'active').length}
            </div>
            <p className="text-xs text-muted-foreground">
              Requiring attention
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Snapshots</CardTitle>
            <PhotoIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{snapshots.length}</div>
            <p className="text-xs text-muted-foreground">
              Captured from alerts
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Debug Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Debug Information</span>
            <div className="flex space-x-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={async () => {
                  try {
                    const response = await fetch('http://localhost:8000/api/test/alert', { method: 'POST' });
                    const result = await response.json();
                    console.log('Test alert result:', result);
                  } catch (error) {
                    console.error('Test alert error:', error);
                  }
                }}
              >
                Test Alert
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={async () => {
                  try {
                    const response = await fetch('http://localhost:8000/api/test/sse-status');
                    const result = await response.json();
                    console.log('SSE Status:', result);
                  } catch (error) {
                    console.error('SSE Status error:', error);
                  }
                }}
              >
                SSE Status
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <strong>Analysis Status:</strong> {analysisStatus}
            </div>
            <div>
              <strong>Session ID:</strong> {analysisSession?.session_id || 'None'}
            </div>
            <div>
              <strong>SSE Connected:</strong> {sseConnected ? 'Yes' : 'No'}
            </div>
            <div>
              <strong>Alerts Generated:</strong> {analysisSession?.alerts_generated || 0}
            </div>
            <div>
              <strong>Real-time Alerts:</strong> {realtimeAlerts.length}
            </div>
            <div>
              <strong>Video Alerts:</strong> {videoAlerts.length}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Real-time Alerts Display */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="h-5 w-5" />
              <span>Real-time Alerts</span>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant={sseConnected ? 'success' : 'danger'}>
                {sseConnected ? 'Connected' : 'Disconnected'}
              </Badge>
              <span className="text-sm text-gray-500">
                {realtimeAlerts.length} alerts received
              </span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {realtimeAlerts.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              {analysisStatus === 'running' 
                ? 'Waiting for alerts... (Mock alerts generate every 30 seconds)'
                : 'Start video analysis to receive real-time alerts'
              }
              <div className="mt-2 text-xs">
                Check browser console for SSE messages
              </div>
            </div>
          ) : (
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {realtimeAlerts.map((alert, index) => (
                <div key={`${alert.id}-${index}`} className="p-3 border rounded-lg bg-gray-50">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant={
                      alert.confidence > 0.8 ? 'danger' :
                      alert.confidence > 0.6 ? 'warning' : 'default'
                    }>
                      {(alert as any).event_type || alert.source_pipeline}
                    </Badge>
                    <span className="text-sm text-gray-500">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-sm">
                    <div>Confidence: {Math.round(alert.confidence * 100)}%</div>
                    <div>Pipeline: {alert.source_pipeline}</div>
                    {(alert as any).metadata?.description && (
                      <div className="text-gray-600 mt-1">
                        {(alert as any).metadata.description}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Snapshot Gallery */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <PhotoIcon className="h-5 w-5" />
            <span>Alert Snapshots</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <SnapshotGallery
            snapshots={snapshots}
            onSnapshotSelect={(snapshot) => {
              // Jump to snapshot time in video
              setCurrentTime(snapshot.timestamp);
            }}
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default VideoAnalysis;