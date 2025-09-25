import React, { useState, useEffect } from 'react';
import { dataService } from '../../services/dataService';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import Badge from '../ui/Badge';
import LoadingSpinner from '../ui/LoadingSpinner';
import { 
  ExclamationTriangleIcon, 
  VideoCameraIcon, 
  ShieldExclamationIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

interface AlertStats {
  total: number;
  active: number;
  reviewed: number;
  dismissed: number;
  byPipeline: {
    threat_intelligence: number;
    video_surveillance: number;
    border_anomaly: number;
  };
  byType: {
    threat: number;
    video: number;
    anomaly: number;
  };
}

const AlertSummary: React.FC = () => {
  const [stats, setStats] = useState<AlertStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      // Only show loading if we don't have stats yet
      if (!stats) {
        setLoading(true);
      }
      setError(null);

      // Fetch all alerts to calculate statistics using cached data service
      const [allAlerts, activeAlerts, reviewedAlerts, dismissedAlerts] = await Promise.all([
        dataService.getAllAlerts(),
        dataService.getActiveAlerts(),
        dataService.getReviewedAlerts(),
        dataService.getDismissedAlerts()
      ]);

      // Calculate pipeline statistics
      const pipelineStats = allAlerts.alerts.reduce((acc, alert) => {
        acc[alert.pipeline as keyof typeof acc] = (acc[alert.pipeline as keyof typeof acc] || 0) + 1;
        return acc;
      }, {
        threat_intelligence: 0,
        video_surveillance: 0,
        border_anomaly: 0
      });

      // Calculate type statistics
      const typeStats = allAlerts.alerts.reduce((acc, alert) => {
        acc[alert.type as keyof typeof acc] = (acc[alert.type as keyof typeof acc] || 0) + 1;
        return acc;
      }, {
        threat: 0,
        video: 0,
        anomaly: 0
      });

      setStats({
        total: allAlerts.total,
        active: activeAlerts.total,
        reviewed: reviewedAlerts.total,
        dismissed: dismissedAlerts.total,
        byPipeline: pipelineStats,
        byType: typeStats
      });
    } catch (err) {
      setError('Failed to fetch alert statistics');
      console.error('Error fetching alert stats:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    
    // Refresh stats every 60 seconds
    const interval = setInterval(fetchStats, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-32">
        <LoadingSpinner />
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
        {error || 'Failed to load statistics'}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {/* Total Alerts */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Alerts</CardTitle>
          <ChartBarIcon className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.total}</div>
          <div className="flex items-center space-x-2 mt-2">
            <Badge variant="danger">{stats.active} Active</Badge>
            <Badge variant="warning">{stats.reviewed} Reviewed</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Threat Intelligence */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Threat Intelligence</CardTitle>
          <ShieldExclamationIcon className="h-4 w-4 text-red-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.byPipeline.threat_intelligence}</div>
          <p className="text-xs text-muted-foreground">
            IOC detections and risk assessments
          </p>
        </CardContent>
      </Card>

      {/* Video Surveillance */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Video Surveillance</CardTitle>
          <VideoCameraIcon className="h-4 w-4 text-blue-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.byPipeline.video_surveillance}</div>
          <p className="text-xs text-muted-foreground">
            Behavior and object detection alerts
          </p>
        </CardContent>
      </Card>

      {/* Border Anomaly */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Border Anomaly</CardTitle>
          <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.byPipeline.border_anomaly}</div>
          <p className="text-xs text-muted-foreground">
            Unusual movement pattern detections
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default AlertSummary;