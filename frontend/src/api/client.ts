import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8080';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// 自动附加 JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 401 时跳转登录
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  },
);

// ── 类型定义 ──

export interface User {
  id: number;
  username: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface TrainingTask {
  id: number;
  name: string | null;
  gnn_type: string;
  dataset: string;
  num_domain: number;
  target_domain: number;
  status: string;
  progress: number;
  best_hr5: number | null;
  best_hr10: number | null;
  best_ndcg5: number | null;
  best_ndcg10: number | null;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  created_at: string;
}

export interface TrainingConfig {
  name?: string;
  gnn_type: string;
  dataset?: string;
  num_domain: number;
  target_domain: number;
  embedding_size?: number;
  round_gat?: number;
  round_ft?: number;
  lr_gnn?: number;
  lr_mf?: number;
  dp?: boolean;
  eps?: number;
  random_seed?: number;
  local_epoch?: number;
  user_batch?: number;
}

export interface MetricPoint {
  step: number;
  stage: string;
  domain: string;
  round: number;
  hr_5: number | null;
  ndcg_5: number | null;
  hr_10: number | null;
  ndcg_10: number | null;
}

export interface Checkpoint {
  dir_name: string;
  stage: string;
  gnn_type: string;
  dataset: string;
  num_domain: number;
  target_domain: number | null;
  best_hr: number | null;
  best_ndcg: number | null;
  created_at: string | null;
  file_count: number;
  size_bytes: number;
}

export interface LogEntry {
  id: number;
  level: string;
  message: string;
  timestamp: string;
}

// ── 推荐查询类型 ──

export interface RecoSnapshot {
  gnn_type: string;
  num_domain: number;
  id: string;
  target_domain_name: string;
  timestamp: string;
  before_source: string;
  after_source: string;
  global_user_count: number;
}

export interface SnapshotListResponse {
  snapshots: RecoSnapshot[];
  message?: string;
}

export interface Top10Item {
  item_id: number;
  rank: number;
}

export interface Top10Success {
  found: true;
  snapshot_id: string;
  global_user_index: number;
  local_user_index: number;
  target_domain: string;
  top10_before: number[];
  top10_after: number[];
}

export interface Top10NotFound {
  found: false;
  message: string;
  valid_users_sample: number[];
  total_valid: number;
}

export type Top10Response = Top10Success | Top10NotFound;

// ── API 函数 ──

export const authApi = {
  login: (username: string, password: string) =>
    api.post('/api/auth/login', { username, password }).then((r) => r.data),

  register: (username: string, password: string, role = 'viewer') =>
    api.post('/api/auth/register', { username, password, role }).then((r) => r.data),

  me: () => api.get<User>('/api/auth/me').then((r) => r.data),
};

export const trainingApi = {
  list: (params?: { status?: string; gnn_type?: string; limit?: number }) =>
    api.get<TrainingTask[]>('/api/training/', { params }).then((r) => r.data),

  get: (id: number) =>
    api.get(`/api/training/${id}`).then((r) => r.data),

  create: (config: TrainingConfig) =>
    api.post<TrainingTask>('/api/training/', config).then((r) => r.data),

  cancel: (id: number) =>
    api.post(`/api/training/${id}/cancel`).then((r) => r.data),

  delete: (id: number) =>
    api.delete(`/api/training/${id}`).then((r) => r.data),

  metrics: (id: number) =>
    api.get<MetricPoint[]>(`/api/training/${id}/metrics`).then((r) => r.data),

  logs: (id: number, afterId = 0) =>
    api.get<LogEntry[]>(`/api/training/${id}/logs`, { params: { after_id: afterId } }).then((r) => r.data),
};

export const checkpointApi = {
  list: (params?: { stage?: string; gnn_type?: string }) =>
    api.get<Checkpoint[]>('/api/checkpoints/', { params }).then((r) => r.data),

  delete: (dirName: string) =>
    api.delete(`/api/checkpoints/${encodeURIComponent(dirName)}`).then((r) => r.data),
};

// ── 推荐查询 API ──

export const recoApi = {
  /** 列出可用推荐快照 */
  snapshots: (params?: { gnn_type?: string; num_domain?: number }) =>
    api.get<SnapshotListResponse>('/api/recommendations/snapshots', { params }).then((r) => r.data),

  /** 查询用户在指定快照中的 Top10（跨域前/跨域后对比） */
  top10: (snapshotId: string, userIndex: number) =>
    api.get<Top10Response>(`/api/recommendations/top10/${encodeURIComponent(snapshotId)}/${userIndex}`).then((r) => r.data),

  /** 训练结果摘要 */
  resultsSummary: () =>
    api.get('/api/recommendations/results-summary').then((r) => r.data),
};

export default api;
