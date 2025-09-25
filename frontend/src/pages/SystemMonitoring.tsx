import React, { useState } from 'react';
import { 
  PipelineStatus, 
  PerformanceMetrics, 
  SystemConfiguration, 
  ErrorLogViewer 
} from '../components/monitoring';
import Button from '../components/ui/Button';
import { 
  CpuChipIcon,
  ChartBarIcon,
  CogIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

const SystemMonitoring = () => {
  const [activeTab, setActiveTab] = useState<'status' | 'metrics' | 'config' | 'logs'>('status');
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  const tabs = [
    { id: 'status', name: 'Pipeline Status', icon: CpuChipIcon },
    { id: 'metrics', name: 'Performance', icon: ChartBarIcon },
    { id: 'config', name: 'Configuration', icon: CogIcon },
    { id: 'logs', name: 'Error Logs', icon: ExclamationTriangleIcon }
  ] as const;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-gray-900">System Monitoring</h2>
          <p className="text-gray-600">Monitor system health, performance, and configuration</p>
        </div>
        
        <Button
          onClick={handleRefresh}
          variant="outline"
          className="flex items-center space-x-2"
        >
          <ArrowPathIcon className="h-4 w-4" />
          <span>Refresh</span>
        </Button>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-5 w-5" />
                <span>{tab.name}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'status' && (
          <PipelineStatus 
            key={`pipeline-${refreshKey}`}
            onRefresh={handleRefresh}
          />
        )}
        
        {activeTab === 'metrics' && (
          <PerformanceMetrics 
            key={`metrics-${refreshKey}`}
          />
        )}
        
        {activeTab === 'config' && (
          <SystemConfiguration 
            onSave={(settings) => {
              console.log('Settings saved:', settings);
              // In real implementation, save to API
            }}
          />
        )}
        
        {activeTab === 'logs' && (
          <ErrorLogViewer 
            key={`logs-${refreshKey}`}
          />
        )}
      </div>
    </div>
  );
};

export default SystemMonitoring;