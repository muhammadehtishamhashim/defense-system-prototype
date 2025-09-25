import type { BaseAlert } from '../types';

export interface SSEMessage {
  type: 'alert' | 'status_update' | 'system_status' | 'heartbeat';
  data: any;
  timestamp: string;
}

export interface AlertSSEMessage extends SSEMessage {
  type: 'alert';
  data: BaseAlert;
}

export interface StatusUpdateMessage extends SSEMessage {
  type: 'status_update';
  data: {
    alert_id: string;
    status: string;
  };
}

export interface SystemStatusMessage extends SSEMessage {
  type: 'system_status';
  data: {
    pipeline: string;
    status: 'online' | 'offline' | 'error';
    metrics?: any;
  };
}

class SSEService {
  private eventSource: EventSource | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 5000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private url: string;
  private connected = false;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.url = `${baseUrl}/events`;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Close existing connection
        this.disconnect();

        this.eventSource = new EventSource(this.url);

        this.eventSource.onopen = () => {
          console.log('SSE connected');
          this.connected = true;
          this.reconnectAttempts = 0;
          this.notifyConnectionChange(true);
          resolve();
        };

        this.eventSource.onmessage = (event) => {
          try {
            const message: SSEMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing SSE message:', error);
          }
        };

        this.eventSource.onerror = (error) => {
          console.error('SSE error:', error);
          this.connected = false;
          this.notifyConnectionChange(false);
          
          if (this.eventSource?.readyState === EventSource.CLOSED) {
            this.handleReconnect();
          }
        };

        // Set up custom event listeners
        this.setupEventListeners();

      } catch (error) {
        reject(error);
      }
    });
  }

  private setupEventListeners(): void {
    if (!this.eventSource) return;

    // Listen for specific event types
    this.eventSource.addEventListener('alert', (event) => {
      try {
        const alertData = JSON.parse(event.data);
        this.handleMessage({
          type: 'alert',
          data: alertData,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Error parsing alert event:', error);
      }
    });

    this.eventSource.addEventListener('status_update', (event) => {
      try {
        const statusData = JSON.parse(event.data);
        this.handleMessage({
          type: 'status_update',
          data: statusData,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Error parsing status update event:', error);
      }
    });

    this.eventSource.addEventListener('system_status', (event) => {
      try {
        const systemData = JSON.parse(event.data);
        this.handleMessage({
          type: 'system_status',
          data: systemData,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Error parsing system status event:', error);
      }
    });

    this.eventSource.addEventListener('heartbeat', (event) => {
      // Handle heartbeat to maintain connection
      this.handleMessage({
        type: 'heartbeat',
        data: { timestamp: event.data },
        timestamp: new Date().toISOString()
      });
    });
  }

  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.connected = false;
      this.notifyConnectionChange(false);
    }
  }

  private handleMessage(message: SSEMessage): void {
    const listeners = this.listeners.get(message.type);
    if (listeners) {
      listeners.forEach(callback => callback(message.data));
    }

    // Also notify global listeners
    const globalListeners = this.listeners.get('*');
    if (globalListeners) {
      globalListeners.forEach(callback => callback(message));
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * this.reconnectAttempts; // Linear backoff instead of exponential
      
      console.log(`Attempting to reconnect SSE in ${delay}ms (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connect().catch(error => {
          console.error('SSE reconnection failed:', error);
          if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.warn('SSE reconnection disabled after max attempts. Manual reconnection required.');
          }
        });
      }, delay);
    } else {
      console.error('Max SSE reconnection attempts reached. SSE connection disabled.');
    }
  }

  private notifyConnectionChange(connected: boolean): void {
    window.dispatchEvent(new CustomEvent('sse:connection', { 
      detail: { connected } 
    }));
  }

  subscribe(eventType: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    
    this.listeners.get(eventType)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(eventType);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.listeners.delete(eventType);
        }
      }
    };
  }

  // Convenience methods for specific event types
  onNewAlert(callback: (alert: BaseAlert) => void): () => void {
    return this.subscribe('alert', callback);
  }

  onStatusUpdate(callback: (update: { alert_id: string; status: string }) => void): () => void {
    return this.subscribe('status_update', callback);
  }

  onSystemStatus(callback: (status: { pipeline: string; status: string; metrics?: any }) => void): () => void {
    return this.subscribe('system_status', callback);
  }

  onHeartbeat(callback: (data: { timestamp: string }) => void): () => void {
    return this.subscribe('heartbeat', callback);
  }

  // Subscribe to all events
  onAnyEvent(callback: (message: SSEMessage) => void): () => void {
    return this.subscribe('*', callback);
  }

  isConnected(): boolean {
    return this.connected && this.eventSource?.readyState === EventSource.OPEN;
  }

  getReadyState(): number {
    return this.eventSource?.readyState ?? EventSource.CLOSED;
  }
}

// Create singleton instance
export const sseService = new SSEService();

// Auto-connect when service is imported (optional)
// sseService.connect().catch(console.error);