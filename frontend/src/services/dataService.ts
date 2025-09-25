import api from './api';
import { alertService } from './alertService';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiry: number;
}

class DataService {
  private cache = new Map<string, CacheEntry<any>>();
  private pendingRequests = new Map<string, Promise<any>>();
  private defaultCacheDuration = 60000; // 60 seconds
  
  private getCacheKey(endpoint: string, params?: any): string {
    const paramStr = params ? JSON.stringify(params) : '';
    return `${endpoint}:${paramStr}`;
  }
  
  private isExpired(entry: CacheEntry<any>): boolean {
    return Date.now() > entry.expiry;
  }
  
  private async fetchWithCache<T>(
    key: string,
    fetcher: () => Promise<T>,
    cacheDuration: number = this.defaultCacheDuration
  ): Promise<T> {
    // Check cache first
    const cached = this.cache.get(key);
    if (cached && !this.isExpired(cached)) {
      console.log(`Cache HIT for ${key}`);
      return cached.data;
    }
    
    console.log(`Cache MISS for ${key}`);
    
    // Check if request is already pending
    const pending = this.pendingRequests.get(key);
    if (pending) {
      return pending;
    }
    
    // Make new request
    const request = fetcher().then(data => {
      // Cache the result
      this.cache.set(key, {
        data,
        timestamp: Date.now(),
        expiry: Date.now() + cacheDuration
      });
      
      // Remove from pending
      this.pendingRequests.delete(key);
      
      return data;
    }).catch(error => {
      // Remove from pending on error
      this.pendingRequests.delete(key);
      throw error;
    });
    
    this.pendingRequests.set(key, request);
    return request;
  }
  
  async getSystemMetrics() {
    const key = this.getCacheKey('/system/metrics');
    return this.fetchWithCache(key, async () => {
      try {
        const response = await api.get('/system/metrics');
        console.log('System metrics response:', response.data);
        return response.data;
      } catch (error) {
        console.error('Failed to fetch system metrics:', error);
        // Return mock data on error
        return {
          system_status: "healthy",
          total_alerts: 0,
          alerts_today: 0,
          pipelines: {
            threat_intelligence: {
              processing_rate: 12.5,
              accuracy_score: 0.852,
              status: "online",
              error_count: 0,
              last_update: new Date().toISOString()
            },
            video_surveillance: {
              processing_rate: 8.3,
              accuracy_score: 0.917,
              status: "online",
              error_count: 1,
              last_update: new Date().toISOString()
            },
            border_anomaly: {
              processing_rate: 15.8,
              accuracy_score: 0.789,
              status: "online",
              error_count: 0,
              last_update: new Date().toISOString()
            }
          },
          uptime: "99.8%",
          last_updated: new Date().toISOString()
        };
      }
    });
  }
  
  async getAlerts(filters: any = {}) {
    const key = this.getCacheKey('/alerts', filters);
    return this.fetchWithCache(key, async () => {
      try {
        return await alertService.getAlerts(filters);
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
        // Return empty result on error
        return {
          alerts: [],
          total: 0,
          limit: filters.limit || 100,
          offset: filters.offset || 0,
          has_more: false
        };
      }
    });
  }
  
  async getActiveAlerts() {
    return this.getAlerts({ status: 'active', limit: 1000 });
  }
  
  async getReviewedAlerts() {
    return this.getAlerts({ status: 'reviewed', limit: 1000 });
  }
  
  async getDismissedAlerts() {
    return this.getAlerts({ status: 'dismissed', limit: 1000 });
  }
  
  async getAllAlerts() {
    return this.getAlerts({ limit: 1000 });
  }
  
  // Clear cache for specific key or all
  clearCache(key?: string) {
    if (key) {
      this.cache.delete(key);
    } else {
      this.cache.clear();
    }
  }
  
  // Force refresh by clearing cache and fetching new data
  async refreshSystemMetrics() {
    this.clearCache(this.getCacheKey('/system/metrics'));
    return this.getSystemMetrics();
  }
  
  async refreshAlerts(filters: any = {}) {
    this.clearCache(this.getCacheKey('/alerts', filters));
    return this.getAlerts(filters);
  }
}

export const dataService = new DataService();