import React, { useState, useEffect } from 'react';
import { checkAPIHealth, getConnectionStatus } from '../../services/api';
import { sseService } from '../../services/sseService';
import Badge from './Badge';
import { 
  WifiIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';

interface ConnectionStatusProps {
  showDetails?: boolean;
  className?: string;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ 
  showDetails = false, 
  className = '' 
}) => {
  const [apiConnected, setApiConnected] = useState(true);
  const [sseConnected, setSseConnected] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  useEffect(() => {
    // Initial status check
    checkStatus();

    // Listen for connection events
    const handleAPIConnection = (event: CustomEvent) => {
      setApiConnected(event.detail.connected);
      setLastCheck(new Date());
    };

    const handleSSEConnection = (event: CustomEvent) => {
      setSseConnected(event.detail.connected);
    };

    window.addEventListener('api:connection', handleAPIConnection as EventListener);
    window.addEventListener('sse:connection', handleSSEConnection as EventListener);

    // Periodic status check
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds

    return () => {
      window.removeEventListener('api:connection', handleAPIConnection as EventListener);
      window.removeEventListener('sse:connection', handleSSEConnection as EventListener);
      clearInterval(interval);
    };
  }, []);

  const checkStatus = async () => {
    const apiStatus = await checkAPIHealth();
    setApiConnected(apiStatus);
    setSseConnected(sseService.isConnected());
    setLastCheck(new Date());
  };

  const getOverallStatus = () => {
    if (apiConnected && sseConnected) return 'connected';
    if (apiConnected && !sseConnected) return 'partial';
    return 'disconnected';
  };

  const getStatusIcon = () => {
    const status = getOverallStatus();
    switch (status) {
      case 'connected':
        return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
      case 'partial':
        return <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500" />;
      case 'disconnected':
        return <XCircleIcon className="h-4 w-4 text-red-500" />;
      default:
        return <WifiIcon className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = () => {
    const status = getOverallStatus();
    switch (status) {
      case 'connected':
        return <Badge variant="success">Connected</Badge>;
      case 'partial':
        return <Badge variant="warning">Partial</Badge>;
      case 'disconnected':
        return <Badge variant="danger">Disconnected</Badge>;
      default:
        return <Badge variant="default">Unknown</Badge>;
    }
  };

  const getStatusMessage = () => {
    const status = getOverallStatus();
    switch (status) {
      case 'connected':
        return 'All services connected';
      case 'partial':
        return 'API connected, real-time updates unavailable';
      case 'disconnected':
        return 'Connection lost - check network';
      default:
        return 'Checking connection...';
    }
  };

  if (!showDetails) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        {getStatusIcon()}
        {getStatusBadge()}
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-gray-900 flex items-center space-x-2">
          <WifiIcon className="h-5 w-5" />
          <span>Connection Status</span>
        </h3>
        {getStatusBadge()}
      </div>

      <div className="space-y-3">
        {/* API Connection */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {apiConnected ? (
              <CheckCircleIcon className="h-4 w-4 text-green-500" />
            ) : (
              <XCircleIcon className="h-4 w-4 text-red-500" />
            )}
            <span className="text-sm font-medium">API Server</span>
          </div>
          <Badge variant={apiConnected ? 'success' : 'danger'}>
            {apiConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>

        {/* Real-time Connection */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {sseConnected ? (
              <CheckCircleIcon className="h-4 w-4 text-green-500" />
            ) : (
              <XCircleIcon className="h-4 w-4 text-red-500" />
            )}
            <span className="text-sm font-medium">Real-time Updates</span>
          </div>
          <Badge variant={sseConnected ? 'success' : 'danger'}>
            {sseConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
      </div>

      {/* Status Message */}
      <div className="mt-3 pt-3 border-t border-gray-200">
        <p className="text-sm text-gray-600">{getStatusMessage()}</p>
        {lastCheck && (
          <p className="text-xs text-gray-500 mt-1">
            Last checked: {lastCheck.toLocaleTimeString()}
          </p>
        )}
      </div>

      {/* Reconnect Button */}
      {!apiConnected && (
        <div className="mt-3">
          <button
            onClick={checkStatus}
            className="w-full px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      )}
    </div>
  );
};

export default ConnectionStatus;