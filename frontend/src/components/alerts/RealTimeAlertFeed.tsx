import React, { useState, useEffect, useRef } from 'react';
import type { BaseAlert } from '../../types';
import { alertService } from '../../services/alertService';
import AlertItem from './AlertItem';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import { 
  PlayIcon, 
  PauseIcon, 
  ArrowPathIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

interface RealTimeAlertFeedProps {
  onAlertSelect?: (alert: BaseAlert) => void;
  maxAlerts?: number;
  refreshInterval?: number;
}

const RealTimeAlertFeed: React.FC<RealTimeAlertFeedProps> = ({ 
  onAlertSelect, 
  maxAlerts = 50,
  refreshInterval = 5000 // 5 seconds
}) => {
  const [alerts, setAlerts] = useState<BaseAlert[]>([]);
  const [isLive, setIsLive] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newAlertCount, setNewAlertCount] = useState(0);
  const intervalRef = useRef<number | null>(null);
  const lastAlertIdRef = useRef<string | null>(null);

  const fetchLatestAlerts = async () => {
    try {
      setError(null);
      const response = await alertService.getAlerts({
        limit: maxAlerts,
        page: 1
      });
      
      // Transform the response to match our types
      const transformedAlerts = response.alerts.map(alert => ({
        id: alert.id,
        timestamp: alert.timestamp,
        confidence: alert.confidence,
        source_pipeline: alert.pipeline,
        status: alert.status,
        ...alert.data
      })) as BaseAlert[];
      
      // Check for new alerts
      if (transformedAlerts.length > 0 && lastAlertIdRef.current) {
        const newAlerts = transformedAlerts.filter(alert => 
          new Date(alert.timestamp) > new Date(lastAlertIdRef.current || 0)
        );
        if (newAlerts.length > 0) {
          setNewAlertCount(prev => prev + newAlerts.length);
        }
      }
      
      setAlerts(transformedAlerts);
      setLastUpdate(new Date());
      
      // Update the last alert ID reference
      if (transformedAlerts.length > 0) {
        lastAlertIdRef.current = transformedAlerts[0].id;
      }
    } catch (err) {
      setError('Failed to fetch alerts');
      console.error('Error fetching alerts:', err);
    }
  };

  const startPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    intervalRef.current = setInterval(fetchLatestAlerts, refreshInterval);
    setIsLive(true);
  };

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsLive(false);
  };

  const toggleLive = () => {
    if (isLive) {
      stopPolling();
    } else {
      startPolling();
      setNewAlertCount(0); // Reset new alert count when resuming
    }
  };

  const handleRefresh = () => {
    fetchLatestAlerts();
    setNewAlertCount(0);
  };

  const handleStatusUpdate = async (alertId: string, status: BaseAlert['status']) => {
    try {
      await alertService.updateAlertStatus(alertId, status);
      // Update the local state
      setAlerts(prev => 
        prev.map(alert => 
          alert.id === alertId ? { ...alert, status } : alert
        )
      );
    } catch (err) {
      console.error('Error updating alert status:', err);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchLatestAlerts();
    
    // Start polling if live mode is enabled
    if (isLive) {
      startPolling();
    }
    
    // Cleanup on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const getActiveAlertCount = () => {
    return alerts.filter(alert => alert.status === 'active').length;
  };

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-medium text-gray-900">
              Live Alert Feed
            </h3>
            <div className="flex items-center space-x-2">
              <Badge variant={isLive ? 'success' : 'default'}>
                {isLive ? 'Live' : 'Paused'}
              </Badge>
              {getActiveAlertCount() > 0 && (
                <Badge variant="danger">
                  {getActiveAlertCount()} Active
                </Badge>
              )}
              {newAlertCount > 0 && !isLive && (
                <Badge variant="warning">
                  {newAlertCount} New
                </Badge>
              )}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="flex items-center space-x-1"
            >
              <ArrowPathIcon className="h-4 w-4" />
              <span>Refresh</span>
            </Button>
            
            <Button
              variant={isLive ? 'destructive' : 'default'}
              size="sm"
              onClick={toggleLive}
              className="flex items-center space-x-1"
            >
              {isLive ? (
                <>
                  <PauseIcon className="h-4 w-4" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <PlayIcon className="h-4 w-4" />
                  <span>Resume</span>
                </>
              )}
            </Button>
          </div>
        </div>
        
        {lastUpdate && (
          <div className="text-sm text-gray-500 mt-2">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
        
        {error && (
          <div className="flex items-center space-x-2 mt-2 text-red-600">
            <ExclamationTriangleIcon className="h-4 w-4" />
            <span className="text-sm">{error}</span>
          </div>
        )}
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="px-6 py-8 text-center text-gray-500">
            No alerts available
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {alerts.map(alert => (
              <AlertItem
                key={alert.id}
                alert={alert}
                onSelect={onAlertSelect}
                onStatusUpdate={handleStatusUpdate}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default RealTimeAlertFeed;