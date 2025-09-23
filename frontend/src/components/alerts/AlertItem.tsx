import React from 'react';
import type { BaseAlert, ThreatAlert, VideoAlert, AnomalyAlert } from '../../types';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import { 
  ExclamationTriangleIcon, 
  VideoCameraIcon, 
  ShieldExclamationIcon,
  ClockIcon,
  CheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

interface AlertItemProps {
  alert: BaseAlert;
  onSelect?: (alert: BaseAlert) => void;
  onStatusUpdate?: (alertId: string, status: BaseAlert['status']) => void;
}

const AlertItem: React.FC<AlertItemProps> = ({ alert, onSelect, onStatusUpdate }) => {
  const getAlertIcon = () => {
    switch (alert.source_pipeline) {
      case 'threat_intelligence':
        return <ShieldExclamationIcon className="h-5 w-5 text-red-500" />;
      case 'video_surveillance':
        return <VideoCameraIcon className="h-5 w-5 text-blue-500" />;
      case 'border_anomaly':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      default:
        return <ExclamationTriangleIcon className="h-5 w-5 text-gray-500" />;
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

  const getConfidenceBadge = () => {
    const confidence = Math.round(alert.confidence * 100);
    if (confidence >= 80) {
      return <Badge variant="success">{confidence}%</Badge>;
    } else if (confidence >= 60) {
      return <Badge variant="warning">{confidence}%</Badge>;
    } else {
      return <Badge variant="danger">{confidence}%</Badge>;
    }
  };

  const getAlertTitle = () => {
    const threatAlert = alert as ThreatAlert;
    const videoAlert = alert as VideoAlert;
    const anomalyAlert = alert as AnomalyAlert;

    switch (alert.source_pipeline) {
      case 'threat_intelligence':
        return `${threatAlert.risk_level} Risk: ${threatAlert.ioc_type?.toUpperCase()} - ${threatAlert.ioc_value}`;
      case 'video_surveillance':
        return `${videoAlert.event_type?.replace('_', ' ').toUpperCase()} - Track ${videoAlert.track_id}`;
      case 'border_anomaly':
        return `${anomalyAlert.anomaly_type} - Severity: ${Math.round((anomalyAlert.severity_score || 0) * 100)}%`;
      default:
        return `Alert from ${alert.source_pipeline}`;
    }
  };

  const getAlertDescription = () => {
    const threatAlert = alert as ThreatAlert;
    const videoAlert = alert as VideoAlert;
    const anomalyAlert = alert as AnomalyAlert;

    switch (alert.source_pipeline) {
      case 'threat_intelligence':
        return `Source: ${threatAlert.source_feed} | Evidence: ${threatAlert.evidence_text?.substring(0, 100)}...`;
      case 'video_surveillance':
        return `Video timestamp: ${videoAlert.video_timestamp}s | Bounding box: [${videoAlert.bounding_box?.join(', ')}]`;
      case 'border_anomaly':
        return `Trajectory points: ${anomalyAlert.trajectory_points?.length || 0} | Features: ${anomalyAlert.feature_vector?.length || 0}`;
      default:
        return 'No additional details available';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="px-6 py-4 hover:bg-gray-50 cursor-pointer" onClick={() => onSelect?.(alert)}>
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3 flex-1">
          <div className="flex-shrink-0 mt-1">
            {getAlertIcon()}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h4 className="text-sm font-medium text-gray-900 truncate">
                {getAlertTitle()}
              </h4>
              {getStatusBadge()}
              {getConfidenceBadge()}
            </div>
            
            <p className="text-sm text-gray-600 mb-2">
              {getAlertDescription()}
            </p>
            
            <div className="flex items-center text-xs text-gray-500 space-x-4">
              <div className="flex items-center space-x-1">
                <ClockIcon className="h-3 w-3" />
                <span>{formatTimestamp(alert.timestamp)}</span>
              </div>
              <span className="capitalize">{alert.source_pipeline.replace('_', ' ')}</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2 ml-4">
          {alert.status === 'active' && onStatusUpdate && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onStatusUpdate(alert.id, 'reviewed');
                }}
                className="text-xs"
              >
                <CheckIcon className="h-3 w-3 mr-1" />
                Review
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onStatusUpdate(alert.id, 'dismissed');
                }}
                className="text-xs"
              >
                <XMarkIcon className="h-3 w-3 mr-1" />
                Dismiss
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AlertItem;