import React, { useState, useRef, useEffect } from 'react';
import Button from '../ui/Button';
import { 
  PlayIcon, 
  PauseIcon,
  ForwardIcon,
  BackwardIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface TimelineEvent {
  timestamp: number;
  type: 'alert' | 'detection' | 'marker';
  label: string;
  color?: string;
  data?: any;
}

interface VideoTimelineProps {
  duration: number;
  currentTime: number;
  events?: TimelineEvent[];
  onSeek: (time: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
  isPlaying?: boolean;
  showFrameControls?: boolean;
  frameRate?: number;
  className?: string;
}

const VideoTimeline: React.FC<VideoTimelineProps> = ({
  duration,
  currentTime,
  events = [],
  onSeek,
  onPlay,
  onPause,
  isPlaying = false,
  showFrameControls = true,
  frameRate = 30,
  className = ''
}) => {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);

  const formatTime = (time: number) => {
    const hours = Math.floor(time / 3600);
    const minutes = Math.floor((time % 3600) / 60);
    const seconds = Math.floor(time % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const formatFrame = (time: number) => {
    return Math.floor(time * frameRate);
  };

  const getTimeFromPosition = (clientX: number) => {
    const timeline = timelineRef.current;
    if (!timeline) return 0;

    const rect = timeline.getBoundingClientRect();
    const position = (clientX - rect.left) / rect.width;
    return Math.max(0, Math.min(duration, position * duration));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    const time = getTimeFromPosition(e.clientX);
    onSeek(time);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const time = getTimeFromPosition(e.clientX);
    setHoverTime(time);

    if (isDragging) {
      onSeek(time);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    setHoverTime(null);
    setIsDragging(false);
  };

  const skipToFrame = (direction: 'prev' | 'next') => {
    const frameTime = 1 / frameRate;
    const newTime = direction === 'next' 
      ? Math.min(duration, currentTime + frameTime)
      : Math.max(0, currentTime - frameTime);
    onSeek(newTime);
  };

  const skipTime = (seconds: number) => {
    const newTime = Math.max(0, Math.min(duration, currentTime + seconds));
    onSeek(newTime);
  };

  const getEventColor = (event: TimelineEvent) => {
    if (event.color) return event.color;
    
    switch (event.type) {
      case 'alert':
        return '#ef4444'; // red
      case 'detection':
        return '#f59e0b'; // amber
      case 'marker':
        return '#3b82f6'; // blue
      default:
        return '#6b7280'; // gray
    }
  };

  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const time = getTimeFromPosition(e.clientX);
        onSeek(time);
      }
    };

    if (isDragging) {
      document.addEventListener('mouseup', handleGlobalMouseUp);
      document.addEventListener('mousemove', handleGlobalMouseMove);
    }

    return () => {
      document.removeEventListener('mouseup', handleGlobalMouseUp);
      document.removeEventListener('mousemove', handleGlobalMouseMove);
    };
  }, [isDragging, onSeek]);

  return (
    <div className={`bg-white rounded-lg shadow p-4 ${className}`}>
      {/* Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          {showFrameControls && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => skipToFrame('prev')}
                title="Previous Frame"
              >
                <BackwardIcon className="h-4 w-4" />
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={isPlaying ? onPause : onPlay}
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? (
                  <PauseIcon className="h-4 w-4" />
                ) : (
                  <PlayIcon className="h-4 w-4" />
                )}
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => skipToFrame('next')}
                title="Next Frame"
              >
                <ForwardIcon className="h-4 w-4" />
              </Button>
            </>
          )}

          <div className="flex items-center space-x-4 ml-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => skipTime(-10)}
              title="Skip -10s"
            >
              -10s
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => skipTime(-1)}
              title="Skip -1s"
            >
              -1s
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => skipTime(1)}
              title="Skip +1s"
            >
              +1s
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => skipTime(10)}
              title="Skip +10s"
            >
              +10s
            </Button>
          </div>
        </div>

        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <div className="flex items-center space-x-1">
            <ClockIcon className="h-4 w-4" />
            <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
          </div>
          {showFrameControls && (
            <div>
              Frame: {formatFrame(currentTime)}
            </div>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div className="relative">
        <div
          ref={timelineRef}
          className="relative h-12 bg-gray-200 rounded-lg cursor-pointer overflow-hidden"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          {/* Progress bar */}
          <div
            className="absolute top-0 left-0 h-full bg-blue-500 rounded-lg transition-all duration-100"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />

          {/* Events */}
          {events.map((event, index) => (
            <div
              key={index}
              className="absolute top-0 w-1 h-full cursor-pointer hover:w-2 transition-all"
              style={{
                left: `${(event.timestamp / duration) * 100}%`,
                backgroundColor: getEventColor(event)
              }}
              onClick={(e) => {
                e.stopPropagation();
                setSelectedEvent(event);
                onSeek(event.timestamp);
              }}
              title={`${event.label} at ${formatTime(event.timestamp)}`}
            />
          ))}

          {/* Hover indicator */}
          {hoverTime !== null && (
            <div
              className="absolute top-0 w-0.5 h-full bg-gray-600 pointer-events-none"
              style={{ left: `${(hoverTime / duration) * 100}%` }}
            />
          )}

          {/* Current time indicator */}
          <div
            className="absolute top-0 w-0.5 h-full bg-red-500 pointer-events-none"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        </div>

        {/* Time markers */}
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0:00</span>
          <span>{formatTime(duration / 2)}</span>
          <span>{formatTime(duration)}</span>
        </div>

        {/* Hover time display */}
        {hoverTime !== null && (
          <div
            className="absolute -top-8 bg-black text-white text-xs px-2 py-1 rounded pointer-events-none"
            style={{ 
              left: `${(hoverTime / duration) * 100}%`,
              transform: 'translateX(-50%)'
            }}
          >
            {formatTime(hoverTime)}
          </div>
        )}
      </div>

      {/* Event details */}
      {selectedEvent && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-gray-900">{selectedEvent.label}</div>
              <div className="text-sm text-gray-600">
                {formatTime(selectedEvent.timestamp)} • {selectedEvent.type}
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedEvent(null)}
            >
              ×
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoTimeline;