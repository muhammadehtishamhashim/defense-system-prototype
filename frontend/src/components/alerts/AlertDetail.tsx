import React from 'react';
import type { BaseAlert, ThreatAlert, VideoAlert, AnomalyAlert } from '../../types';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { 
  ExclamationTriangleIcon, 
  VideoCameraIcon, 
  ShieldExclamationIcon,
  ClockIcon,
  CheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

interface AlertDetailProps {
  alert: BaseAlert;
  onStatusUpdate?: (alertId: string, status: BaseAlert['status']) => void;
  onClose?: () => void;
}

const AlertDetail: React.FC<AlertDetailProps> = ({ alert, onStatusUpdate, onClose }) => {
  const getAlertIcon = () => {
    switch (alert.source_pipeline) {
      case 'threat_intelligence':
        return <ShieldExclamationIcon className="h-6 w-6 text-red-500" />;
      case 'video_surveillance':
        return <VideoCameraIcon className="h-6 w-6 text-blue-500" />;
      case 'border_anomaly':
        return <ExclamationTriangleIcon className="h-6 w-6 text-yellow-500" />;
      default:
        return <ExclamationTriangleIcon className="h-6 w-6 text-gray-500" />;
    }
  };

  const getStatusBadge = () => {
    switch (alert.status) {
      case 'active':
        return <Badge variant="danger">Active</Badge>;
      case 'reviewed':
        return <Badge variant="warning">Reviewed</Badge>;
      case 'dismissed':
        return <Badge variant="default">Dismissed</Badge>;
      default:
        return <Badge variant="default">{alert.status}</Badge>;
    }
  };

  const renderThreatDetails = (threatAlert: ThreatAlert) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="font-medium text-gray-900 mb-2">IOC Information</h4>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Type:</span> {threatAlert.ioc_type}</div>
            <div><span className="font-medium">Value:</span> {threatAlert.ioc_value}</div>
            <div><span className="font-medium">Risk Level:</span> 
              <Badge variant={threatAlert.risk_level === 'High' ? 'danger' : threatAlert.risk_level === 'Medium' ? 'warning' : 'success'} className="ml-2">
                {threatAlert.risk_level}
              </Badge>
            </div>
          </div>
        </div>
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Source Information</h4>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Feed:</span> {threatAlert.source_feed}</div>
            <div><span className="font-medium">Confidence:</span> {Math.round(threatAlert.confidence * 100)}%</div>
          </div>
        </div>
      </div>
      
      <div>
        <h4 className="font-medium text-gray-900 mb-2">Evidence</h4>
        <div className="bg-gray-50 p-3 rounded-md text-sm">
          {threatAlert.evidence_text}
        </div>
      </div>
    </div>
  );

  const renderVideoDetails = (videoAlert: VideoAlert) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Event Information</h4>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Event Type:</span> {videoAlert.event_type?.replace('_', ' ')}</div>
            <div><span className="font-medium">Track ID:</span> {videoAlert.track_id}</div>
            <div><span className="font-medium">Video Timestamp:</span> {videoAlert.video_timestamp}s</div>
          </div>
        </div>
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Detection Details</h4>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Confidence:</span> {Math.round(videoAlert.confidence * 100)}%</div>
            <div><span className="font-medium">Bounding Box:</span> [{videoAlert.bounding_box?.join(', ')}]</div>
          </div>
        </div>
      </div>
      
      {videoAlert.snapshot_path && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Snapshot</h4>
          <div className="bg-gray-100 p-4 rounded-md">
            <img 
              src={videoAlert.snapshot_path} 
              alt="Alert snapshot"
              className="max-w-full h-auto rounded"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
            <div className="mt-2 text-sm text-gray-600">
              Snapshot from video at {videoAlert.video_timestamp}s
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderAnomalyDetails = (anomalyAlert: AnomalyAlert) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Anomaly Information</h4>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Type:</span> {anomalyAlert.anomaly_type}</div>
            <div><span className="font-medium">Severity Score:</span> 
              <Badge variant={anomalyAlert.severity_score > 0.8 ? 'danger' : anomalyAlert.severity_score > 0.6 ? 'warning' : 'info'} className="ml-2">
                {Math.round((anomalyAlert.severity_score || 0) * 100)}%
              </Badge>
            </div>
            <div><span className="font-medium">Confidence:</span> {Math.round(anomalyAlert.confidence * 100)}%</div>
          </div>
        </div>
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Trajectory Data</h4>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Points:</span> {anomalyAlert.trajectory_points?.length || 0}</div>
            <div><span className="font-medium">Features:</span> {anomalyAlert.feature_vector?.length || 0}</div>
          </div>
        </div>
      </div>
      
      {anomalyAlert.supporting_frames && anomalyAlert.supporting_frames.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Supporting Frames</h4>
          <div className="grid grid-cols-3 gap-2">
            {anomalyAlert.supporting_frames.slice(0, 6).map((frame, index) => (
              <div key={index} className="bg-gray-100 p-2 rounded">
                <img 
                  src={frame} 
                  alt={`Supporting frame ${index + 1}`}
                  className="w-full h-auto rounded"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderAlertContent = () => {
    switch (alert.source_pipeline) {
      case 'threat_intelligence':
        return renderThreatDetails(alert as ThreatAlert);
      case 'video_surveillance':
        return renderVideoDetails(alert as VideoAlert);
      case 'border_anomaly':
        return renderAnomalyDetails(alert as AnomalyAlert);
      default:
        return (
          <div className="text-gray-600">
            No detailed information available for this alert type.
          </div>
        );
    }
  };

  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            {getAlertIcon()}
            <div>
              <CardTitle className="text-xl">Alert Details</CardTitle>
              <div className="flex items-center space-x-2 mt-1">
                {getStatusBadge()}
                <Badge variant="info">ID: {alert.id.substring(0, 8)}</Badge>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {alert.status === 'active' && onStatusUpdate && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onStatusUpdate(alert.id, 'reviewed')}
                  className="flex items-center space-x-1"
                >
                  <CheckIcon className="h-4 w-4" />
                  <span>Mark Reviewed</span>
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onStatusUpdate(alert.id, 'dismissed')}
                  className="flex items-center space-x-1"
                >
                  <XMarkIcon className="h-4 w-4" />
                  <span>Dismiss</span>
                </Button>
              </>
            )}
            {onClose && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
              >
                <XMarkIcon className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-4 text-sm text-gray-600 mt-2">
          <div className="flex items-center space-x-1">
            <ClockIcon className="h-4 w-4" />
            <span>{new Date(alert.timestamp).toLocaleString()}</span>
          </div>
          <span className="capitalize">{alert.source_pipeline.replace('_', ' ')}</span>
        </div>
      </CardHeader>
      
      <CardContent>
        {renderAlertContent()}
      </CardContent>
    </Card>
  );
};

export default AlertDetail;