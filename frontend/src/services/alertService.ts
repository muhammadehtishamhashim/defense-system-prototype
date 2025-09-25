import api from './api';

export interface Alert {
  id: string;
  timestamp: string;
  pipeline: string;
  type: string;
  confidence: number;
  data: any;
  status: 'active' | 'reviewed' | 'dismissed';
  created_at: string;
}

export interface AlertsResponse {
  alerts: Alert[];
  total: number;
  page: number;
  limit: number;
}

export interface AlertFilters {
  type?: string;
  pipeline?: string;
  status?: string;
  start_time?: string;
  end_time?: string;
  search?: string;
  page?: number;
  limit?: number;
}

class AlertService {
  async getAlerts(filters: AlertFilters = {}): Promise<AlertsResponse> {
    const response = await api.get('/alerts', { params: filters });
    return response.data;
  }

  async getAlert(id: string): Promise<Alert> {
    const response = await api.get(`/alerts/${id}`);
    return response.data;
  }

  async updateAlertStatus(id: string, status: Alert['status']): Promise<Alert> {
    const response = await api.put(`/alerts/${id}/status`, { status });
    return response.data;
  }

  async createThreatAlert(alertData: any): Promise<Alert> {
    const response = await api.post('/alerts/threat', alertData);
    return response.data;
  }

  async createVideoAlert(alertData: any): Promise<Alert> {
    const response = await api.post('/alerts/video', alertData);
    return response.data;
  }

  async createAnomalyAlert(alertData: any): Promise<Alert> {
    const response = await api.post('/alerts/anomaly', alertData);
    return response.data;
  }
}

export const alertService = new AlertService();