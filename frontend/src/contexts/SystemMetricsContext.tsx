import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { dataService } from '../services/dataService';

interface SystemMetricsData {
  system_status: string;
  total_alerts: number;
  alerts_today: number;
  pipelines: Record<string, any>;
  uptime: string;
  last_updated: string;
}

interface SystemMetricsContextType {
  data: SystemMetricsData | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  lastRefresh: Date | null;
  isStale: boolean;
  connectionStatus: 'connected' | 'disconnected' | 'reconnecting';
  retryCount: number;
}

const SystemMetricsContext = createContext<SystemMetricsContextType | undefined>(undefined);

interface SystemMetricsProviderProps {
  children: React.ReactNode;
  refreshInterval?: number;
}

export const SystemMetricsProvider: React.FC<SystemMetricsProviderProps> = ({
  children,
  refreshInterval = 30000 // 30 seconds
}) => {
  const [data, setData] = useState<SystemMetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [isStale, setIsStale] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  const [retryCount, setRetryCount] = useState(0);
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isActiveRef = useRef(true);
  const lastSuccessfulFetchRef = useRef<Date | null>(null);

  const refresh = useCallback(async (forceRefresh: boolean = false) => {
    if (!isActiveRef.current) return;

    try {
      setLoading(true);
      setError(null);
      setConnectionStatus('reconnecting');
      
      const systemData = forceRefresh 
        ? await dataService.refreshSystemMetrics()
        : await dataService.getSystemMetrics();
      
      setData(systemData);
      setLastRefresh(new Date());
      lastSuccessfulFetchRef.current = new Date();
      setConnectionStatus('connected');
      setRetryCount(0);
      setIsStale(false);
      
    } catch (err) {
      console.error('Failed to fetch system metrics:', err);
      setError('Failed to load system metrics');
      setConnectionStatus('disconnected');
      setRetryCount(prev => prev + 1);
      
      // Check if data is stale (older than 2 minutes)
      if (lastSuccessfulFetchRef.current) {
        const timeSinceLastSuccess = Date.now() - lastSuccessfulFetchRef.current.getTime();
        setIsStale(timeSinceLastSuccess > 120000); // 2 minutes
      }
      
    } finally {
      setLoading(false);
    }
  }, []);

  // Smart refresh that uses cached data when appropriate
  const smartRefresh = useCallback(async () => {
    if (!isActiveRef.current) return;

    try {
      // Don't show loading for background refreshes
      const systemData = await dataService.getSystemMetrics();
      setData(systemData);
      setLastRefresh(new Date());
      lastSuccessfulFetchRef.current = new Date();
      setConnectionStatus('connected');
      setRetryCount(0);
      setIsStale(false);
      
      if (error) {
        setError(null);
      }
      
    } catch (err) {
      console.error('Background refresh failed:', err);
      setRetryCount(prev => prev + 1);
      
      // Only update error state if we don't have any data
      if (!data) {
        setError('Failed to load system metrics');
        setConnectionStatus('disconnected');
      }
      
      // Check if data is stale
      if (lastSuccessfulFetchRef.current) {
        const timeSinceLastSuccess = Date.now() - lastSuccessfulFetchRef.current.getTime();
        setIsStale(timeSinceLastSuccess > 120000); // 2 minutes
      }
    }
  }, [data, error]);

  useEffect(() => {
    isActiveRef.current = true;
    
    // Initial load with loading state
    refresh(false);
    
    // Set up smart refresh interval
    intervalRef.current = setInterval(() => {
      smartRefresh();
    }, refreshInterval);
    
    // Listen for visibility changes to pause/resume polling
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, clear interval
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } else {
        // Page is visible, resume polling
        if (!intervalRef.current && isActiveRef.current) {
          smartRefresh(); // Immediate refresh when page becomes visible
          intervalRef.current = setInterval(smartRefresh, refreshInterval);
        }
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Listen for online/offline events
    const handleOnline = () => {
      setConnectionStatus('reconnecting');
      refresh(false);
    };
    
    const handleOffline = () => {
      setConnectionStatus('disconnected');
    };
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      isActiveRef.current = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [refresh, smartRefresh, refreshInterval]);

  const value: SystemMetricsContextType = {
    data,
    loading,
    error,
    refresh: () => refresh(true), // Force refresh when manually called
    lastRefresh,
    isStale,
    connectionStatus,
    retryCount
  };

  return (
    <SystemMetricsContext.Provider value={value}>
      {children}
    </SystemMetricsContext.Provider>
  );
};

export const useSystemMetrics = (): SystemMetricsContextType => {
  const context = useContext(SystemMetricsContext);
  if (context === undefined) {
    throw new Error('useSystemMetrics must be used within a SystemMetricsProvider');
  }
  return context;
};