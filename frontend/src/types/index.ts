// Alert Types
export interface BaseAlert {
  id: string;
  timestamp: string;
  confidence: number;
  source_pipeline: string;
  status: 'active' | 'reviewed' | 'dismissed';
}

export interface ThreatAlert extends BaseAlert {
  ioc_type: string;
  ioc_value: string;
  risk_level: 'High' | 'Medium' | 'Low';
  evidence_text: string;
  source_feed: string;
}

export interface VideoAlert extends BaseAlert {
  event_type: 'loitering' | 'zone_violation' | 'abandoned_object';
  bounding_box: [number, number, number, number];
  track_id: number;
  snapshot_path: string;
  video_timestamp: number;
}

export interface AnomalyAlert extends BaseAlert {
  anomaly_type: string;
  severity_score: number;
  trajectory_points: [number, number][];
  feature_vector: number[];
  supporting_frames: string[];
}

// System Types
export interface SystemMetrics {
  pipeline_name: string;
  processing_rate: number;
  accuracy_score?: number;
  last_update: string;
  status: 'healthy' | 'warning' | 'error';
}

export interface PipelineStatus {
  name: string;
  status: 'online' | 'offline' | 'error';
  last_heartbeat: string;
  processed_count: number;
  error_count: number;
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  has_next: boolean;
  has_prev: boolean;
}