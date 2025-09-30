import React, { useState, useRef, useEffect } from 'react';
import Button from '../ui/Button';
import { 
  PlayIcon, 
  PauseIcon, 
  ForwardIcon, 
  BackwardIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ArrowsPointingOutIcon,
  MagnifyingGlassPlusIcon,
  MagnifyingGlassMinusIcon
} from '@heroicons/react/24/outline';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
  confidence?: number;
  trackId?: number;
}

interface VideoPlayerProps {
  src: string;
  boundingBoxes?: BoundingBox[];
  onTimeUpdate?: (currentTime: number) => void;
  onFrameChange?: (frameNumber: number) => void;
  showControls?: boolean;
  autoPlay?: boolean;
  loop?: boolean;
  muted?: boolean;
  className?: string;
  analysisStatus?: 'idle' | 'starting' | 'running' | 'paused' | 'stopped' | 'error';
  onAnalysisStart?: () => void;
  onAnalysisStop?: () => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  boundingBoxes = [],
  onTimeUpdate,
  onFrameChange,
  showControls = true,
  autoPlay = true,  // Default to true for streaming
  loop = true,      // Default to true for streaming
  muted = true,     // Default to true for streaming
  className = '',
  analysisStatus = 'idle',
  onAnalysisStart,
  onAnalysisStop
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(muted);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      const time = video.currentTime;
      setCurrentTime(time);
      onTimeUpdate?.(time);
      
      // Calculate frame number (assuming 30 fps)
      const frameNumber = Math.floor(time * 30);
      onFrameChange?.(frameNumber);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [onTimeUpdate, onFrameChange]);

  useEffect(() => {
    drawBoundingBoxes();
  }, [boundingBoxes, currentTime, zoom, pan]);

  const drawBoundingBoxes = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Apply zoom and pan transformations
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.translate(pan.x, pan.y);

    // Draw bounding boxes
    boundingBoxes.forEach((box, index) => {
      const { x, y, width, height, label, confidence, trackId } = box;
      
      // Set box style
      ctx.strokeStyle = `hsl(${(index * 137.5) % 360}, 70%, 50%)`;
      ctx.lineWidth = 2;
      ctx.fillStyle = `hsla(${(index * 137.5) % 360}, 70%, 50%, 0.1)`;

      // Draw bounding box
      ctx.fillRect(x, y, width, height);
      ctx.strokeRect(x, y, width, height);

      // Draw label
      if (label || trackId !== undefined) {
        const labelText = trackId !== undefined 
          ? `${label || 'Object'} #${trackId}${confidence ? ` (${Math.round(confidence * 100)}%)` : ''}`
          : `${label}${confidence ? ` (${Math.round(confidence * 100)}%)` : ''}`;
        
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = '14px Arial';
        const textWidth = ctx.measureText(labelText).width;
        
        // Background for text
        ctx.fillRect(x, y - 20, textWidth + 8, 20);
        
        // Text
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, x + 4, y - 6);
      }
    });

    ctx.restore();
  };

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const time = parseFloat(e.target.value);
    video.currentTime = time;
    setCurrentTime(time);
  };

  const skipTime = (seconds: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = Math.max(0, Math.min(duration, video.currentTime + seconds));
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const newVolume = parseFloat(e.target.value);
    video.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isMuted) {
      video.volume = volume;
      setIsMuted(false);
    } else {
      video.volume = 0;
      setIsMuted(true);
    }
  };

  const toggleFullscreen = () => {
    const container = containerRef.current;
    if (!container) return;

    if (!isFullscreen) {
      container.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleZoom = (delta: number) => {
    setZoom(prev => Math.max(0.5, Math.min(3, prev + delta)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoom > 1) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div 
      ref={containerRef}
      className={`relative bg-black rounded-lg overflow-hidden ${className}`}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <div className="relative">
        <video
          ref={videoRef}
          src={src}
          autoPlay={autoPlay}
          loop={loop}
          muted={muted}
          className="w-full h-auto"
          style={{
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
            cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
          }}
          onMouseDown={handleMouseDown}
        />
        
        {/* Analysis Status Indicator */}
        {analysisStatus !== 'idle' && (
          <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-1 rounded-lg flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              analysisStatus === 'running' ? 'bg-green-500 animate-pulse' :
              analysisStatus === 'starting' ? 'bg-yellow-500 animate-pulse' :
              analysisStatus === 'error' ? 'bg-red-500' :
              'bg-gray-500'
            }`} />
            <span className="text-sm capitalize">{analysisStatus}</span>
          </div>
        )}
        
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`
          }}
        />
      </div>

      {showControls && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
          {/* Timeline */}
          <div className="mb-4">
            <input
              type="range"
              min="0"
              max={duration}
              value={currentTime}
              onChange={handleSeek}
              className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-white text-sm mt-1">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => skipTime(-10)}
                className="text-white hover:bg-white/20"
              >
                <BackwardIcon className="h-5 w-5" />
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={togglePlayPause}
                className="text-white hover:bg-white/20"
              >
                {isPlaying ? (
                  <PauseIcon className="h-6 w-6" />
                ) : (
                  <PlayIcon className="h-6 w-6" />
                )}
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => skipTime(10)}
                className="text-white hover:bg-white/20"
              >
                <ForwardIcon className="h-5 w-5" />
              </Button>

              <div className="flex items-center space-x-2 ml-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleMute}
                  className="text-white hover:bg-white/20"
                >
                  {isMuted ? (
                    <SpeakerXMarkIcon className="h-5 w-5" />
                  ) : (
                    <SpeakerWaveIcon className="h-5 w-5" />
                  )}
                </Button>
                
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={isMuted ? 0 : volume}
                  onChange={handleVolumeChange}
                  className="w-20 h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Analysis Controls */}
              {(onAnalysisStart || onAnalysisStop) && (
                <>
                  {analysisStatus === 'idle' || analysisStatus === 'stopped' ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onAnalysisStart}
                      className="text-white hover:bg-white/20"
                      disabled={analysisStatus === 'starting'}
                    >
                      <PlayIcon className="h-4 w-4 mr-1" />
                      <span className="text-sm">Analyze</span>
                    </Button>
                  ) : analysisStatus === 'running' || analysisStatus === 'starting' || analysisStatus === 'paused' ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onAnalysisStop}
                      className="text-white hover:bg-white/20"
                      disabled={analysisStatus === 'starting'}
                    >
                      <PauseIcon className="h-4 w-4 mr-1" />
                      <span className="text-sm">Stop</span>
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onAnalysisStart}
                      className="text-white hover:bg-white/20"
                      disabled={true}
                    >
                      <PlayIcon className="h-4 w-4 mr-1" />
                      <span className="text-sm">Error</span>
                    </Button>
                  )}
                  <div className="w-px h-6 bg-white/30 mx-2" />
                </>
              )}
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleZoom(-0.25)}
                className="text-white hover:bg-white/20"
                disabled={zoom <= 0.5}
              >
                <MagnifyingGlassMinusIcon className="h-5 w-5" />
              </Button>
              
              <span className="text-white text-sm min-w-[3rem] text-center">
                {Math.round(zoom * 100)}%
              </span>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleZoom(0.25)}
                className="text-white hover:bg-white/20"
                disabled={zoom >= 3}
              >
                <MagnifyingGlassPlusIcon className="h-5 w-5" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={toggleFullscreen}
                className="text-white hover:bg-white/20"
              >
                <ArrowsPointingOutIcon className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;