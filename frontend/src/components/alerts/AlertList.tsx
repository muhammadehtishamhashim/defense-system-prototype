import React, { useState, useEffect } from 'react';
import type { BaseAlert } from '../../types';
import { alertService, type AlertFilters } from '../../services/alertService';
import AlertItem from './AlertItem';
import AlertFiltersComponent from './AlertFilters';
import Pagination from './Pagination';
import LoadingSpinner from '../ui/LoadingSpinner';

interface AlertListProps {
  onAlertSelect?: (alert: BaseAlert) => void;
}

const AlertList: React.FC<AlertListProps> = ({ onAlertSelect }) => {
  const [alerts, setAlerts] = useState<BaseAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<AlertFilters>({
    page: 1,
    limit: 20
  });
  const [totalPages, setTotalPages] = useState(0);
  const [total, setTotal] = useState(0);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await alertService.getAlerts(filters);
      
      // Transform the response to match our types
      const transformedAlerts = response.alerts.map(alert => ({
        id: alert.id,
        timestamp: alert.timestamp,
        confidence: alert.confidence,
        source_pipeline: alert.pipeline,
        status: alert.status,
        ...alert.data
      })) as BaseAlert[];
      
      setAlerts(transformedAlerts);
      setTotal(response.total);
      setTotalPages(Math.ceil(response.total / (filters.limit || 20)));
    } catch (err) {
      setError('Failed to fetch alerts');
      console.error('Error fetching alerts:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, [filters]);

  const handleFilterChange = (newFilters: Partial<AlertFilters>) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters,
      page: 1 // Reset to first page when filters change
    }));
  };

  const handlePageChange = (page: number) => {
    setFilters(prev => ({ ...prev, page }));
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

  if (loading && alerts.length === 0) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <AlertFiltersComponent 
        filters={filters}
        onFilterChange={handleFilterChange}
      />
      
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}
      
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">
            Alerts ({total})
          </h3>
        </div>
        
        <div className="divide-y divide-gray-200">
          {alerts.length === 0 ? (
            <div className="px-6 py-8 text-center text-gray-500">
              No alerts found
            </div>
          ) : (
            alerts.map(alert => (
              <AlertItem
                key={alert.id}
                alert={alert}
                onSelect={onAlertSelect}
                onStatusUpdate={handleStatusUpdate}
              />
            ))
          )}
        </div>
        
        {totalPages > 1 && (
          <div className="px-6 py-4 border-t border-gray-200">
            <Pagination
              currentPage={filters.page || 1}
              totalPages={totalPages}
              onPageChange={handlePageChange}
            />
          </div>
        )}
      </div>
      
      {loading && alerts.length > 0 && (
        <div className="flex justify-center py-4">
          <LoadingSpinner />
        </div>
      )}
    </div>
  );
};

export default AlertList;