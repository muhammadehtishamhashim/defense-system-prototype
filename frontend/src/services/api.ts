import axios, { AxiosError } from 'axios';
import type { AxiosRequestConfig, AxiosResponse } from 'axios';

// API Configuration
const API_CONFIG = {
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000, // Increased timeout for AI processing
  retryAttempts: 3,
  retryDelay: 1000,
};

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Retry configuration
interface RetryConfig extends AxiosRequestConfig {
  _retry?: boolean;
  _retryCount?: number;
}

// Exponential backoff delay
const getRetryDelay = (retryCount: number): number => {
  return API_CONFIG.retryDelay * Math.pow(2, retryCount);
};

// Check if error is retryable
const isRetryableError = (error: AxiosError): boolean => {
  if (!error.response) {
    // Network errors are retryable
    return true;
  }
  
  const status = error.response.status;
  // Retry on server errors (5xx) and some client errors
  return status >= 500 || status === 408 || status === 429;
};

// Request interceptor for adding auth tokens and request ID
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Add request ID for tracking
    config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Add timestamp
    config.headers['X-Request-Time'] = new Date().toISOString();
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors and retries
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log successful requests in development
    if (import.meta.env.DEV) {
      console.log(`✅ API Success: ${response.config.method?.toUpperCase()} ${response.config.url}`, {
        status: response.status,
        requestId: response.config.headers['X-Request-ID'],
        duration: Date.now() - new Date(response.config.headers['X-Request-Time'] as string).getTime()
      });
    }
    return response;
  },
  async (error: AxiosError) => {
    const config = error.config as RetryConfig;
    
    // Log errors in development
    if (import.meta.env.DEV) {
      console.error(`❌ API Error: ${config?.method?.toUpperCase()} ${config?.url}`, {
        status: error.response?.status,
        message: error.message,
        requestId: config?.headers?.['X-Request-ID']
      });
    }
    
    // Handle specific error cases
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      window.dispatchEvent(new CustomEvent('auth:logout'));
      return Promise.reject(error);
    }
    
    // Implement retry logic
    if (config && isRetryableError(error) && !config._retry) {
      config._retryCount = config._retryCount || 0;
      
      if (config._retryCount < API_CONFIG.retryAttempts) {
        config._retryCount++;
        config._retry = true;
        
        const delay = getRetryDelay(config._retryCount);
        
        if (import.meta.env.DEV) {
          console.log(`🔄 Retrying request (${config._retryCount}/${API_CONFIG.retryAttempts}) after ${delay}ms`);
        }
        
        await new Promise(resolve => setTimeout(resolve, delay));
        return api(config);
      }
    }
    
    return Promise.reject(error);
  }
);

// API Health Check
export const checkAPIHealth = async (): Promise<boolean> => {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};

// Connection status monitoring
let connectionStatus = true;
let statusCheckInterval: number | null = null;

export const startConnectionMonitoring = (intervalMs: number = 30000) => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval);
  }
  
  statusCheckInterval = setInterval(async () => {
    const isHealthy = await checkAPIHealth();
    
    if (isHealthy !== connectionStatus) {
      connectionStatus = isHealthy;
      window.dispatchEvent(new CustomEvent('api:connection', { 
        detail: { connected: isHealthy } 
      }));
    }
  }, intervalMs);
};

export const stopConnectionMonitoring = () => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval);
    statusCheckInterval = null;
  }
};

export const getConnectionStatus = (): boolean => connectionStatus;

// Enhanced error handling
export interface APIError {
  message: string;
  status?: number;
  code?: string;
  details?: any;
  requestId?: string;
}

export const handleAPIError = (error: AxiosError): APIError => {
  const apiError: APIError = {
    message: 'An unexpected error occurred',
    requestId: error.config?.headers?.['X-Request-ID'] as string
  };
  
  if (error.response) {
    // Server responded with error status
    apiError.status = error.response.status;
    const responseData = error.response.data as any;
    apiError.message = responseData?.message || error.response.statusText;
    apiError.code = responseData?.code;
    apiError.details = responseData?.details;
  } else if (error.request) {
    // Request was made but no response received
    apiError.message = 'Network error - please check your connection';
    apiError.code = 'NETWORK_ERROR';
  } else {
    // Something else happened
    apiError.message = error.message;
    apiError.code = 'REQUEST_ERROR';
  }
  
  return apiError;
};

// Loading state management
const loadingStates = new Map<string, boolean>();

export const setLoading = (key: string, loading: boolean) => {
  loadingStates.set(key, loading);
  window.dispatchEvent(new CustomEvent('api:loading', { 
    detail: { key, loading, totalLoading: Array.from(loadingStates.values()).some(Boolean) } 
  }));
};

export const isLoading = (key?: string): boolean => {
  if (key) {
    return loadingStates.get(key) || false;
  }
  return Array.from(loadingStates.values()).some(Boolean);
};

// Request wrapper with loading states
export const apiRequest = async <T>(
  requestFn: () => Promise<AxiosResponse<T>>,
  loadingKey?: string
): Promise<T> => {
  if (loadingKey) {
    setLoading(loadingKey, true);
  }
  
  try {
    const response = await requestFn();
    return response.data;
  } catch (error) {
    throw handleAPIError(error as AxiosError);
  } finally {
    if (loadingKey) {
      setLoading(loadingKey, false);
    }
  }
};

export default api;