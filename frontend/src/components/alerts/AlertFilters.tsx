import React, { useState } from 'react';
import type { AlertFilters as AlertFiltersType } from '../../services/alertService';
import Button from '../ui/Button';
import { FunnelIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface AlertFiltersProps {
  filters: AlertFiltersType;
  onFilterChange: (filters: Partial<AlertFiltersType>) => void;
}

const AlertFilters: React.FC<AlertFiltersProps> = ({ filters, onFilterChange }) => {
  const [showFilters, setShowFilters] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  const pipelineOptions = [
    { value: '', label: 'All Pipelines' },
    { value: 'threat_intelligence', label: 'Threat Intelligence' },
    { value: 'video_surveillance', label: 'Video Surveillance' },
    { value: 'border_anomaly', label: 'Border Anomaly' }
  ];

  const statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: 'active', label: 'Active' },
    { value: 'reviewed', label: 'Reviewed' },
    { value: 'dismissed', label: 'Dismissed' }
  ];

  const typeOptions = [
    { value: '', label: 'All Types' },
    { value: 'threat', label: 'Threat' },
    { value: 'video', label: 'Video' },
    { value: 'anomaly', label: 'Anomaly' }
  ];

  const handleFilterChange = (key: keyof AlertFiltersType, value: string) => {
    onFilterChange({ [key]: value || undefined });
  };

  const handleSearchChange = (value: string) => {
    setSearchTerm(value);
    // Debounce search to avoid too many API calls
    const timeoutId = setTimeout(() => {
      onFilterChange({ search: value || undefined });
    }, 500);
    
    return () => clearTimeout(timeoutId);
  };

  const handleDateRangeChange = (startTime?: string, endTime?: string) => {
    onFilterChange({ 
      start_time: startTime || undefined, 
      end_time: endTime || undefined 
    });
  };

  const clearFilters = () => {
    onFilterChange({
      type: undefined,
      pipeline: undefined,
      status: undefined,
      start_time: undefined,
      end_time: undefined,
      search: undefined
    });
    setSearchTerm('');
  };

  const hasActiveFilters = filters.type || filters.pipeline || filters.status || filters.start_time || filters.end_time || filters.search;

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center space-x-2"
          >
            <FunnelIcon className="h-4 w-4" />
            <span>Filters</span>
          </Button>
          
          {hasActiveFilters && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearFilters}
              className="flex items-center space-x-2 text-gray-600"
            >
              <XMarkIcon className="h-4 w-4" />
              <span>Clear Filters</span>
            </Button>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <input
            type="text"
            placeholder="Search alerts..."
            value={searchTerm}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {showFilters && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-4 border-t border-gray-200">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Pipeline
            </label>
            <select
              value={filters.pipeline || ''}
              onChange={(e) => handleFilterChange('pipeline', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {pipelineOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Status
            </label>
            <select
              value={filters.status || ''}
              onChange={(e) => handleFilterChange('status', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {statusOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Type
            </label>
            <select
              value={filters.type || ''}
              onChange={(e) => handleFilterChange('type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {typeOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Date Range
            </label>
            <div className="space-y-2">
              <input
                type="datetime-local"
                value={filters.start_time || ''}
                onChange={(e) => handleDateRangeChange(e.target.value, filters.end_time)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Start time"
              />
              <input
                type="datetime-local"
                value={filters.end_time || ''}
                onChange={(e) => handleDateRangeChange(filters.start_time, e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="End time"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertFilters;