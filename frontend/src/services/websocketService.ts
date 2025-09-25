import type { BaseAlert } from '../types';

export interface WebSocketMessage {
    type: 'alert' | 'status_update' | 'system_status';
    data: any;
}

export interface AlertWebSocketMessage extends WebSocketMessage {
    type: 'alert';
    data: BaseAlert;
}

export interface StatusUpdateMessage extends WebSocketMessage {
    type: 'status_update';
    data: {
        alert_id: string;
        status: string;
    };
}

class WebSocketService {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;
    private listeners: Map<string, Set<(data: any) => void>> = new Map();
    private url: string;

    constructor(url: string = 'ws://localhost:8000/ws') {
        this.url = url;
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.url);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const message: WebSocketMessage = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };

                this.ws.onclose = (event) => {
                    console.log('WebSocket disconnected:', event.code, event.reason);
                    this.handleReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };
            } catch (error) {
                reject(error);
            }
        });
    }

    disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    private handleMessage(message: WebSocketMessage): void {
        const listeners = this.listeners.get(message.type);
        if (listeners) {
            listeners.forEach(callback => callback(message.data));
        }
    }

    private handleReconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

            console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);

            setTimeout(() => {
                this.connect().catch(error => {
                    console.error('Reconnection failed:', error);
                });
            }, delay);
        } else {
            console.error('Max reconnection attempts reached');
        }
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

    onNewAlert(callback: (alert: BaseAlert) => void): () => void {
        return this.subscribe('alert', callback);
    }

    onStatusUpdate(callback: (update: { alert_id: string; status: string }) => void): () => void {
        return this.subscribe('status_update', callback);
    }

    onSystemStatus(callback: (status: any) => void): () => void {
        return this.subscribe('system_status', callback);
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// Auto-connect when service is imported (optional)
// websocketService.connect().catch(console.error);