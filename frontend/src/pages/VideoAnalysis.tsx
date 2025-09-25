import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { VideoPlayer, SnapshotGallery, VideoTimeline } from '../components/video';
import { alertService } from '../services/alertService';
import type { VideoAlert } from '../types';
import Badge from '../components/ui/Badge';
import { 
  VideoCameraIcon,
  PhotoIcon,
  ClockIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

interface VideoSource {
  id: string;
  name: string;
  url: string;
  status: 'online' | 'offline' | 'error';
}

const VideoAnalysis = () => {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [videoAlerts, setVideoAlerts] = useState<VideoAlert[]>([]);
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loading, setLoading] = useState(false);

  // Mock video sources - in real implementation, these would come from API
  const videoSources: VideoSource[] = [
    { id: '1', name: 'Camera 1 - Main Entrance', url: '/demo-video-1.mp4', status: 'online' },
    { id: '2', name: 'Camera 2 - Parking Lot', url: '/demo-video-2.mp4', status: 'online' },
    { id: '3', name: 'Camera 3 - Perimeter', url: '/demo-video-3.mp4', status: 'offline' },
  ];

  useEffect(() => {
    fetchVideoAlerts();
  }, []);

  const fetchVideoAlerts = async () => {
    try {
      setLoading(true);
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
    } finally {
      setLoading(false);
    }
  };

  const handleVideoSelect = (videoId: string) => {
    setSelectedVideo(videoId);
    setCurrentTime(0);
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

  const selectedVideoSource = videoSources.find(v => v.id === selectedVideo);

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-gray-900">Video Analysis</h2>
          <p className="text-gray-600">Real-time video surveillance and behavior analysis</p>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading video analysis...</div>
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

      {/* Video Sources */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <VideoCameraIcon className="h-5 w-5" />
            <span>Video Sources</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {videoSources.map(source => (
              <div
                key={source.id}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedVideo === source.id 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleVideoSelect(source.id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900">{source.name}</h3>
                  <Badge 
                    variant={source.status === 'online' ? 'success' : 
                            source.status === 'offline' ? 'default' : 'danger'}
                  >
                    {source.status}
                  </Badge>
                </div>
                <div className="text-sm text-gray-600">
                  {source.status === 'online' ? 'Live feed available' : 
                   source.status === 'offline' ? 'Camera offline' : 'Connection error'}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {selectedVideoSource && (
        <>
          {/* Video Player */}
          <Card>
            <CardHeader>
              <CardTitle>Live Video Feed - {selectedVideoSource.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <VideoPlayer
                src={selectedVideoSource.url}
                boundingBoxes={getBoundingBoxesForTime(currentTime)}
                onTimeUpdate={handleTimeUpdate}
                showControls={true}
                className="w-full"
              />
            </CardContent>
          </Card>

          {/* Timeline */}
          <VideoTimeline
            duration={duration || 300} // Mock duration
            currentTime={currentTime}
            events={getTimelineEvents()}
            onSeek={handleSeek}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            isPlaying={isPlaying}
            showFrameControls={true}
          />
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