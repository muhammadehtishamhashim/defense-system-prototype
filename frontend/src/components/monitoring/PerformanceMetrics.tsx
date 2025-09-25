import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import Button from '../ui/Button';
import { 
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface MetricDataPoint {
  timestamp: string;
  value: number;
}

interface PerformanceMetric {
  name: string;
  current: number;
  previous: number;
  unit: string;
  data: MetricDataPoint[];
  threshold?: {
    warning: number;
    critical: number;
  };
}

interface PerformanceMetricsProps {
  refreshInterval?: number;
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  refreshInterval = 30000
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [loading, setLoading] = useState(false);

  // Generate mock data
  const generateMockData = (baseValue: number, points: number = 20) => {
    const data: MetricDataPoint[] = [];
    const now = new Date();
    
    for (let i = points - 1; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60000).toISOString(); // 1 minute intervals
      const variation = (Math.random() - 0.5) * 0.2 * baseValue;
      const value = Math.max(0, baseValue + variation);
      data.push({ timestamp, value });
    }
    
    return data;
  };

  const mockMetrics: PerformanceMetric[] = [
    {
      name: 'Alert Processing Rate',
      current: 15.2,
      previous: 14.8,
      unit: 'alerts/min',
      data: generateMockData(15),
      threshold: { warning: 10, critical: 5 }
    },
    {
      name: 'Detection Accuracy',
      current: 87.3,
      previous: 86.1,
      unit: '%',
      data: generateMockData(87),
      threshold: { warning: 80, critical: 70 }
    },
    {
      name: 'Response Time',
      current: 1.2,
      previous: 1.5,
      unit: 'seconds',
      data: generateMockData(1.2),
      threshold: { warning: 2, critical: 5 }
    },
    {
      name: 'System Load',
      current: 45.6,
      previous: 48.2,
      unit: '%',
      data: generateMockData(45),
      threshold: { warning: 70, critical: 90 }
    },
    {
      name: 'Memory Usage',
      current: 62.1,
      previous: 59.8,
      unit: '%',
      data: generateMockData(62),
      threshold: { warning: 80, critical: 95 }
    },
    {
      name: 'Error Rate',
      current: 0.3,
      previous: 0.5,
      unit: '%',
      data: generateMockData(0.3),
      threshold: { warning: 1, critical: 5 }
    }
  ];

  useEffect(() => {
    const fetchMetrics = () => {
      setLoading(true);
      // Simulate API call
      setTimeout(() => {
        setMetrics(mockMetrics);
        setLoading(false);
      }, 500);
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval, selectedTimeRange]);

  const getMetricStatus = (metric: PerformanceMetric) => {
    if (!metric.threshold) return 'normal';
    
    if (metric.current >= metric.threshold.critical) return 'critical';
    if (metric.current >= metric.threshold.warning) return 'warning';
    return 'normal';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default:
        return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) {
      return <ArrowTrendingUpIcon className="h-4 w-4 text-green-500" />;
    } else if (current < previous) {
      return <ArrowTrendingDownIcon className="h-4 w-4 text-red-500" />;
    }
    return null;
  };

  const formatValue = (value: number, unit: string) => {
    if (unit === '%') {
      return `${value.toFixed(1)}%`;
    } else if (unit === 'seconds') {
      return `${value.toFixed(2)}s`;
    } else {
      return `${value.toFixed(1)} ${unit}`;
    }
  };

  const renderMiniChart = (data: MetricDataPoint[]) => {
    if (data.length === 0) return null;

    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const range = maxValue - minValue || 1;

    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = 100 - ((point.value - minValue) / range) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="w-full h-12" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          points={points}
          className="text-blue-500"
        />
      </svg>
    );
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <ChartBarIcon className="h-5 w-5" />
            <span>Performance Metrics</span>
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              {(['1h', '6h', '24h', '7d'] as const).map(range => (
                <Button
                  key={range}
                  variant={selectedTimeRange === range ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setSelectedTimeRange(range)}
                >
                  {range}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metrics.map(metric => {
            const status = getMetricStatus(metric);
            const statusColor = getStatusColor(status);
            
            return (
              <div
                key={metric.name}
                className={`border rounded-lg p-4 ${statusColor}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-sm">{metric.name}</h3>
                  {getTrendIcon(metric.current, metric.previous)}
                </div>
                
                <div className="flex items-baseline space-x-2 mb-2">
                  <span className="text-2xl font-bold">
                    {formatValue(metric.current, metric.unit)}
                  </span>
                  <span className="text-sm text-gray-600">
                    {metric.current > metric.previous ? '+' : ''}
                    {formatValue(metric.current - metric.previous, metric.unit)}
                  </span>
                </div>
                
                <div className="mb-3">
                  {renderMiniChart(metric.data)}
                </div>
                
                {metric.threshold && (
                  <div className="text-xs text-gray-600">
                    <div className="flex justify-between">
                      <span>Warning: {formatValue(metric.threshold.warning, metric.unit)}</span>
                      <span>Critical: {formatValue(metric.threshold.critical, metric.unit)}</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {/* Summary Statistics */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="font-medium text-gray-900 mb-4 flex items-center space-x-2">
            <ClockIcon className="h-4 w-4" />
            <span>System Summary</span>
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {metrics.filter(m => getMetricStatus(m) === 'normal').length}
              </div>
              <div className="text-gray-600">Healthy Metrics</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {metrics.filter(m => getMetricStatus(m) === 'warning').length}
              </div>
              <div className="text-gray-600">Warning Metrics</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {metrics.filter(m => getMetricStatus(m) === 'critical').length}
              </div>
              <div className="text-gray-600">Critical Metrics</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {Math.round(metrics.reduce((acc, m) => acc + (m.current > m.previous ? 1 : 0), 0) / metrics.length * 100)}%
              </div>
              <div className="text-gray-600">Improving</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PerformanceMetrics;