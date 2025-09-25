import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import Button from '../ui/Button';
import Badge from '../ui/Badge';
import { 
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  ArrowPathIcon,
  FunnelIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'error' | 'warning' | 'info' | 'debug';
  source: string;
  message: string;
  details?: string;
  stackTrace?: string;
}

interface ErrorLogViewerProps {
  refreshInterval?: number;
}

const ErrorLogViewer: React.FC<ErrorLogViewerProps> = ({
  refreshInterval = 30000
}) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedLevel, setSelectedLevel] = useState<string>('all');
  const [selectedSource, setSelectedSource] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());

  // Mock log data
  const mockLogs: LogEntry[] = [
    {
      id: '1',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      level: 'error',
      source: 'video_surveillance',
      message: 'Failed to process video frame from camera 2',
      details: 'Connection timeout after 30 seconds',
      stackTrace: 'VideoProcessor.processFrame() line 145\nCameraManager.getFrame() line 67'
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 600000).toISOString(),
      level: 'warning',
      source: 'threat_intelligence',
      message: 'Low confidence threat classification',
      details: 'IOC confidence score: 0.45 (below threshold of 0.7)'
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 900000).toISOString(),
      level: 'error',
      source: 'border_anomaly',
      message: 'Anomaly detection model failed to load',
      details: 'Model file not found: /models/anomaly_detector.pkl',
      stackTrace: 'AnomalyDetector.__init__() line 23\nModelLoader.load() line 89'
    },
    {
      id: '4',
      timestamp: new Date(Date.now() - 1200000).toISOString(),
      level: 'info',
      source: 'system',
      message: 'System startup completed successfully',
      details: 'All pipelines initialized and running'
    },
    {
      id: '5',
      timestamp: new Date(Date.now() - 1500000).toISOString(),
      level: 'warning',
      source: 'video_surveillance',
      message: 'High CPU usage detected',
      details: 'CPU usage: 87% (above warning threshold of 80%)'
    },
    {
      id: '6',
      timestamp: new Date(Date.now() - 1800000).toISOString(),
      level: 'error',
      source: 'api',
      message: 'Database connection failed',
      details: 'Connection refused to localhost:5432',
      stackTrace: 'DatabaseManager.connect() line 34\nApp.initialize() line 12'
    }
  ];

  useEffect(() => {
    const fetchLogs = () => {
      setLoading(true);
      // Simulate API call
      setTimeout(() => {
        setLogs(mockLogs);
        setLoading(false);
      }, 500);
    };

    fetchLogs();
    const interval = setInterval(fetchLogs, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  useEffect(() => {
    let filtered = logs;

    // Filter by level
    if (selectedLevel !== 'all') {
      filtered = filtered.filter(log => log.level === selectedLevel);
    }

    // Filter by source
    if (selectedSource !== 'all') {
      filtered = filtered.filter(log => log.source === selectedSource);
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(log =>
        log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.details?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredLogs(filtered);
  }, [logs, selectedLevel, selectedSource, searchTerm]);

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'info':
        return <InformationCircleIcon className="h-5 w-5 text-blue-500" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getLevelBadge = (level: string) => {
    switch (level) {
      case 'error':
        return <Badge variant="danger">Error</Badge>;
      case 'warning':
        return <Badge variant="warning">Warning</Badge>;
      case 'info':
        return <Badge variant="info">Info</Badge>;
      default:
        return <Badge variant="default">Debug</Badge>;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatSource = (source: string) => {
    return source.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const toggleLogExpansion = (logId: string) => {
    setExpandedLogs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(logId)) {
        newSet.delete(logId);
      } else {
        newSet.add(logId);
      }
      return newSet;
    });
  };

  const exportLogs = () => {
    const logData = filteredLogs.map(log => ({
      timestamp: log.timestamp,
      level: log.level,
      source: log.source,
      message: log.message,
      details: log.details,
      stackTrace: log.stackTrace
    }));

    const blob = new Blob([JSON.stringify(logData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `hifazat-logs-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };



  const sources = Array.from(new Set(logs.map(log => log.source)));
  const levels = ['error', 'warning', 'info', 'debug'];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <ExclamationTriangleIcon className="h-5 w-5" />
            <span>Error Logs & Diagnostics</span>
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={exportLogs}
              disabled={filteredLogs.length === 0}
              className="flex items-center space-x-1"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
              <span>Export</span>
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setLoading(true)}
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
        {/* Filters */}
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-4 mb-4">
            <FunnelIcon className="h-5 w-5 text-gray-500" />
            <span className="font-medium text-gray-900">Filters</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Level
              </label>
              <select
                value={selectedLevel}
                onChange={(e) => setSelectedLevel(e.target.value)}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Levels</option>
                {levels.map(level => (
                  <option key={level} value={level}>
                    {level.charAt(0).toUpperCase() + level.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Source
              </label>
              <select
                value={selectedSource}
                onChange={(e) => setSelectedSource(e.target.value)}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Sources</option>
                {sources.map(source => (
                  <option key={source} value={source}>
                    {formatSource(source)}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Search
              </label>
              <input
                type="text"
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Log Entries */}
        <div className="space-y-2">
          {filteredLogs.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <div className="text-lg mb-2">No logs found</div>
              <div className="text-sm">
                {logs.length === 0 ? 'No logs available' : 'Try adjusting your filters'}
              </div>
            </div>
          ) : (
            filteredLogs.map(log => {
              const isExpanded = expandedLogs.has(log.id);
              
              return (
                <div
                  key={log.id}
                  className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1">
                      {getLevelIcon(log.level)}
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          {getLevelBadge(log.level)}
                          <Badge variant="default">{formatSource(log.source)}</Badge>
                          <span className="text-sm text-gray-500">
                            {formatTimestamp(log.timestamp)}
                          </span>
                        </div>
                        
                        <p className="text-gray-900 font-medium mb-1">
                          {log.message}
                        </p>
                        
                        {log.details && (
                          <p className="text-sm text-gray-600">
                            {log.details}
                          </p>
                        )}
                        
                        {isExpanded && log.stackTrace && (
                          <div className="mt-3 p-3 bg-gray-100 rounded text-sm font-mono text-gray-800">
                            <div className="font-medium mb-2">Stack Trace:</div>
                            <pre className="whitespace-pre-wrap">{log.stackTrace}</pre>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {log.stackTrace && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleLogExpansion(log.id)}
                        className="ml-2"
                      >
                        {isExpanded ? 'Less' : 'More'}
                      </Button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
        
        {/* Summary */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-red-600">
                {logs.filter(l => l.level === 'error').length}
              </div>
              <div className="text-sm text-gray-600">Errors</div>
            </div>
            
            <div>
              <div className="text-2xl font-bold text-yellow-600">
                {logs.filter(l => l.level === 'warning').length}
              </div>
              <div className="text-sm text-gray-600">Warnings</div>
            </div>
            
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {logs.filter(l => l.level === 'info').length}
              </div>
              <div className="text-sm text-gray-600">Info</div>
            </div>
            
            <div>
              <div className="text-2xl font-bold text-gray-600">
                {filteredLogs.length} / {logs.length}
              </div>
              <div className="text-sm text-gray-600">Showing</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ErrorLogViewer;