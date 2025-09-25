import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import LoadingSpinner from '../ui/LoadingSpinner';
import { 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ArrowPathIcon,
  CpuChipIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface PipelineMetrics {
  pipeline_name: string;
  status: 'healthy' | 'warning' | 'error' | 'offline';
  processing_rate: number;
  accuracy_score?: number;
  last_update: string;
  error_count: number;
  uptime: number;
  memory_usage?: number;
  cpu_usage?: number;
}

interface PipelineStatusProps {
  onRefresh?: () => void;
  refreshInterval?: number;
}

const PipelineStatus: React.FC<PipelineStatusProps> = ({
  onRefresh,
  refreshInterval = 60000 // 60 seconds
}) => {
  const [pipelines, setPipelines] = useState<PipelineMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  // Mock data - in real implementation, this would come from API
  const mockPipelines: PipelineMetrics[] = [
    {
      pipeline_name: 'threat_intelligence',
      status: 'healthy',
      processing_rate: 15.2,
      accuracy_score: 0.87,
      last_update: new Date(Date.now() - 30000).toISOString(),
      error_count: 0,
      uptime: 99.8,
      memory_usage: 45.2,
      cpu_usage: 23.1
    },
    {
      pipeline_name: 'video_surveillance',
      status: 'warning',
      processing_rate: 8.7,
      accuracy_score: 0.82,
      last_update: new Date(Date.now() - 120000).toISOString(),
      error_count: 2,
      uptime: 97.5,
      memory_usage: 78.9,
      cpu_usage: 67.3
    },
    {
      pipeline_name: 'border_anomaly',
      status: 'healthy',
      processing_rate: 12.4,
      accuracy_score: 0.75,
      last_update: new Date(Date.now() - 15000).toISOString(),
      error_count: 0,
      uptime: 99.9,
      memory_usage: 52.1,
      cpu_usage: 34.8
    }
  ];

  const fetchPipelineStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Add some randomness to simulate real-time updates
      const updatedPipelines = mockPipelines.map(pipeline => ({
        ...pipeline,
        processing_rate: pipeline.processing_rate + (Math.random() - 0.5) * 2,
        cpu_usage: Math.max(0, Math.min(100, (pipeline.cpu_usage || 0) + (Math.random() - 0.5) * 10)),
        memory_usage: Math.max(0, Math.min(100, (pipeline.memory_usage || 0) + (Math.random() - 0.5) * 5)),
        last_update: new Date().toISOString()
      }));
      
      setPipelines(updatedPipelines);
      setLastRefresh(new Date());
      onRefresh?.();
    } catch (err) {
      setError('Failed to fetch pipeline status');
      console.error('Error fetching pipeline status:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPipelineStatus();
    
    const interval = setInterval(fetchPipelineStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'offline':
        return <XCircleIcon className="h-5 w-5 text-gray-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'healthy':
        return <Badge variant="success">Healthy</Badge>;
      case 'warning':
        return <Badge variant="warning">Warning</Badge>;
      case 'error':
        return <Badge variant="danger">Error</Badge>;
      case 'offline':
        return <Badge variant="default">Offline</Badge>;
      default:
        return <Badge variant="default">Unknown</Badge>;
    }
  };

  const formatPipelineName = (name: string) => {
    return name.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const formatLastUpdate = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    
    if (diffSecs < 60) {
      return `${diffSecs}s ago`;
    } else if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else {
      return date.toLocaleTimeString();
    }
  };

  if (loading && pipelines.length === 0) {
    return (
      <Card>
        <CardContent className="flex justify-center items-center h-64">
          <LoadingSpinner />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <CpuChipIcon className="h-5 w-5" />
            <span>Pipeline Status</span>
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            {lastRefresh && (
              <span className="text-sm text-gray-500">
                Updated {formatLastUpdate(lastRefresh.toISOString())}
              </span>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={fetchPipelineStatus}
              disabled={loading}
              className="flex items-center space-x-1"
            >
              <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        
        <div className="space-y-4">
          {pipelines.map(pipeline => (
            <div
              key={pipeline.pipeline_name}
              className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(pipeline.status)}
                  <div>
                    <h3 className="font-medium text-gray-900">
                      {formatPipelineName(pipeline.pipeline_name)}
                    </h3>
                    <p className="text-sm text-gray-600">
                      Last update: {formatLastUpdate(pipeline.last_update)}
                    </p>
                  </div>
                </div>
                {getStatusBadge(pipeline.status)}
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-gray-600">Processing Rate</div>
                  <div className="font-medium">
                    {pipeline.processing_rate.toFixed(1)} items/min
                  </div>
                </div>
                
                {pipeline.accuracy_score && (
                  <div>
                    <div className="text-gray-600">Accuracy</div>
                    <div className="font-medium">
                      {Math.round(pipeline.accuracy_score * 100)}%
                    </div>
                  </div>
                )}
                
                <div>
                  <div className="text-gray-600">Uptime</div>
                  <div className="font-medium">{pipeline.uptime.toFixed(1)}%</div>
                </div>
                
                <div>
                  <div className="text-gray-600">Errors</div>
                  <div className={`font-medium ${pipeline.error_count > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {pipeline.error_count}
                  </div>
                </div>
              </div>
              
              {(pipeline.cpu_usage !== undefined || pipeline.memory_usage !== undefined) && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {pipeline.cpu_usage !== undefined && (
                      <div>
                        <div className="flex justify-between text-gray-600 mb-1">
                          <span>CPU Usage</span>
                          <span>{pipeline.cpu_usage.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              pipeline.cpu_usage > 80 ? 'bg-red-500' :
                              pipeline.cpu_usage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${pipeline.cpu_usage}%` }}
                          />
                        </div>
                      </div>
                    )}
                    
                    {pipeline.memory_usage !== undefined && (
                      <div>
                        <div className="flex justify-between text-gray-600 mb-1">
                          <span>Memory Usage</span>
                          <span>{pipeline.memory_usage.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              pipeline.memory_usage > 80 ? 'bg-red-500' :
                              pipeline.memory_usage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${pipeline.memory_usage}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default PipelineStatus;