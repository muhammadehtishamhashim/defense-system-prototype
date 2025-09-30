import React, { useState, useEffect, useRef } from 'react';
import type { BaseAlert } from '../../types';
import { dataService } from '../../services/dataService';
import { sseService } from '../../services/sseService';
import AlertItem from './AlertItem';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import { 
  PlayIcon, 
  PauseIcon, 
  ArrowPathIcon,
  ExclamationTriangleIcon,
  WifiIcon
} from '@heroicons/react/24/outline';

interface RealTimeAlertFeedProps {
  onAlertSelect?: (alert: BaseAlert) => void;
  maxAlerts?: number;
  refreshInterval?: number;
  useSSE?: boolean;
}

const RealTimeAlertFeed: React.FC<RealTimeAlertFeedProps> = ({ 
  onAlertSelect, 
  maxAlerts = 50,
  refreshInterval = 60000, // 60 seconds
  useSSE = true
}) => {
  const [alerts, setAlerts] = useState<BaseAlert[]>([]);
  const [isLive, setIsLive] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newAlertCount, setNewAlertCount] = useState(0);
  const [sseConnected, setSseConnected] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastAlertIdRef = useRef<string | null>(null);
  const sseUnsubscribeRef = useRef<(() => void) | null>(null);

  const fetchLatestAlerts = async () => {
    try {
      setError(null);
      const response = await dataService.getAllAlerts();
      
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
      const { alertService } = await import('../../services/alertService');
      await alertService.updateAlertStatus(alertId, status);
      // Clear cache to force refresh
      dataService.clearCache();
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
    // Disable initial fetch to reduce requests
    // fetchLatestAlerts();
    
    if (useSSE) {
      // Setup SSE connection
      setupSSE();
    } else if (isLive) {
      // Start polling if live mode is enabled and not using SSE
      // Temporarily disable auto-polling
      // startPolling();
    }
    
    // Cleanup on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (sseUnsubscribeRef.current) {
        sseUnsubscribeRef.current();
      }
    };
  }, [useSSE]);

  const setupSSE = async () => {
    try {
      if (!sseService.isConnected()) {
        await sseService.connect();
      }
      
      setSseConnected(true);
      setError(null);
      
      // Subscribe to new alerts
      sseUnsubscribeRef.current = sseService.onNewAlert((alert: BaseAlert) => {
        setAlerts(prev => {
          const newAlerts = [alert, ...prev].slice(0, maxAlerts);
          return newAlerts;
        });
        setNewAlertCount(prev => prev + 1);
        setLastUpdate(new Date());
      });
      
      // Listen for SSE connection changes
      const handleSSEConnection = (event: CustomEvent) => {
        setSseConnected(event.detail.connected);
        if (!event.detail.connected) {
          setError('Real-time connection lost, falling back to polling');
          if (isLive) {
            startPolling();
          }
        } else {
          setError(null);
          stopPolling();
        }
      };
      
      window.addEventListener('sse:connection', handleSSEConnection as EventListener);
      
      return () => {
        window.removeEventListener('sse:connection', handleSSEConnection as EventListener);
      };
      
    } catch (err) {
      console.error('SSE connection failed:', err);
      setSseConnected(false);
      setError('Real-time connection failed, falling back to polling');
      
      // Fallback to polling
      if (isLive) {
        startPolling();
      }
    }
  };

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
              {useSSE && (
                <Badge variant={sseConnected ? 'success' : 'danger'}>
                  <WifiIcon className="h-3 w-3 mr-1" />
                  {sseConnected ? 'Real-time' : 'Polling'}
                </Badge>
              )}
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