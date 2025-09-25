import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import Button from '../components/ui/Button';
import { AlertSummary } from '../components/alerts';
import { PipelineStatus, PerformanceMetrics } from '../components/monitoring';
import { 
  ChartBarIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  VideoCameraIcon,
  ShieldExclamationIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

const Dashboard = () => {
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-gray-900">Dashboard</h2>
          <p className="text-gray-600">Overview of your security monitoring system</p>
        </div>
        
        <Button
          onClick={handleRefresh}
          variant="outline"
          className="flex items-center space-x-2"
        >
          <ChartBarIcon className="h-4 w-4" />
          <span>Refresh All</span>
        </Button>
      </div>

      {/* Alert Summary */}
      <AlertSummary key={`alerts-${refreshKey}`} />

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <CpuChipIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">Healthy</div>
            <p className="text-xs text-muted-foreground">All pipelines operational</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Rate</CardTitle>
            <ChartBarIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">15.2</div>
            <p className="text-xs text-muted-foreground">alerts/min average</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Uptime</CardTitle>
            <ClockIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">99.8%</div>
            <p className="text-xs text-muted-foreground">Last 30 days</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Errors</CardTitle>
            <ExclamationTriangleIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">2</div>
            <p className="text-xs text-muted-foreground">Require attention</p>
          </CardContent>
        </Card>
      </div>

      {/* Pipeline Status */}
      <PipelineStatus key={`pipeline-${refreshKey}`} onRefresh={handleRefresh} />

      {/* Performance Metrics */}
      <PerformanceMetrics key={`metrics-${refreshKey}`} />

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="h-5 w-5" />
              <span>Recent Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <ShieldExclamationIcon className="h-5 w-5 text-red-500" />
                  <div>
                    <p className="font-medium text-sm">High Risk IOC Detected</p>
                    <p className="text-xs text-gray-600">Malicious IP: 192.168.1.100</p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="danger">High</Badge>
                  <p className="text-xs text-gray-500 mt-1">2m ago</p>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <VideoCameraIcon className="h-5 w-5 text-blue-500" />
                  <div>
                    <p className="font-medium text-sm">Loitering Detected</p>
                    <p className="text-xs text-gray-600">Camera 3 - Person #47</p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="warning">Medium</Badge>
                  <p className="text-xs text-gray-500 mt-1">5m ago</p>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                  <div>
                    <p className="font-medium text-sm">Border Anomaly</p>
                    <p className="text-xs text-gray-600">Unusual movement pattern</p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="info">Low</Badge>
                  <p className="text-xs text-gray-500 mt-1">12m ago</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CpuChipIcon className="h-5 w-5" />
              <span>System Events</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <div>
                    <p className="font-medium text-sm">Pipeline Restarted</p>
                    <p className="text-xs text-gray-600">Video surveillance pipeline</p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">1h ago</p>
              </div>
              
              <div className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <div>
                    <p className="font-medium text-sm">Model Updated</p>
                    <p className="text-xs text-gray-600">Threat classification model v2.1</p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">3h ago</p>
              </div>
              
              <div className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  <div>
                    <p className="font-medium text-sm">High CPU Usage</p>
                    <p className="text-xs text-gray-600">Border anomaly pipeline: 87%</p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">4h ago</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;