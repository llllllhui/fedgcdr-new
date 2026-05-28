import { useState, useEffect, useCallback, useRef } from 'react';
import { trainingApi } from '../api/client';
import type { TrainingTask, TrainingConfig, MetricPoint, LogEntry } from '../api/client';
import { useTrainingWS } from '../hooks/useTrainingWS';
import type { WSMessage } from '../hooks/useTrainingWS';

const GNN_OPTIONS = [
  { value: 'gat', label: 'GAT' },
  { value: 'lightgcn', label: 'LightGCN' },
  { value: 'graphsage', label: 'GraphSAGE' },
  { value: 'gcn', label: 'GCN' },
];

const DEFAULT_CONFIG: TrainingConfig = {
  gnn_type: 'lightgcn',
  num_domain: 4,
  target_domain: 1,
  dataset: 'amazon',
  embedding_size: 16,
  round_gat: 30,
  round_ft: 60,
  lr_gnn: 0.001,
  lr_mf: 0.005,
  dp: true,
  eps: 8.0,
  random_seed: 42,
  local_epoch: 3,
  user_batch: 16,
};

export default function TrainingPage() {
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [creating, setCreating] = useState(false);
  const [selectedTask, setSelectedTask] = useState<TrainingTask | null>(null);
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  // 加载任务列表
  const loadTasks = useCallback(() => {
    trainingApi.list({ limit: 50 }).then(setTasks);
  }, []);

  useEffect(() => {
    loadTasks();
    const interval = setInterval(loadTasks, 5000);
    return () => clearInterval(interval);
  }, [loadTasks]);

  // 选择任务后加载指标和日志
  const selectTask = useCallback((task: TrainingTask) => {
    setSelectedTask(task);
    if (task.status === 'running' || task.status === 'completed') {
      trainingApi.metrics(task.id).then(setMetrics);
      trainingApi.logs(task.id).then(setLogs);
    }
  }, []);

  // WebSocket 实时更新
  const handleWSMessage = useCallback((msg: WSMessage) => {
    if (msg.type === 'metric' && msg.task_id === selectedTask?.id) {
      setMetrics((prev) => [...prev, msg.data as unknown as MetricPoint]);
    }
    if (msg.type === 'log' && msg.task_id === selectedTask?.id) {
      setLogs((prev) => [...prev, { id: prev.length + 1, ...msg.data } as LogEntry]);
    }
    if (msg.type === 'status' && msg.task_id === selectedTask?.id) {
      setSelectedTask((prev) => prev ? { ...prev, status: msg.status, progress: msg.progress } : prev);
    }
  }, [selectedTask?.id]);

  useTrainingWS(selectedTask?.id || 0, handleWSMessage);

  // 自动滚动日志到底部
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // 创建任务
  const handleCreate = async () => {
    setCreating(true);
    try {
      await trainingApi.create(config);
      setShowForm(false);
      loadTasks();
    } catch (err) {
      alert('创建失败');
    } finally {
      setCreating(false);
    }
  };

  // 取消任务
  const handleCancel = async (id: number) => {
    await trainingApi.cancel(id);
    loadTasks();
  };

  return (
    <div style={{ display: 'flex', gap: '20px', minHeight: 'calc(100vh - 100px)' }}>
      {/* 左侧：任务列表 + 创建按钮 */}
      <div style={{ width: '340px', flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h2 style={{ fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400, margin: 0 }}>
            训练任务
          </h2>
          <button onClick={() => setShowForm(!showForm)} style={{
            padding: '8px 16px',
            borderRadius: '10px',
            border: 'none',
            background: showForm ? '#6b7280' : 'linear-gradient(140deg, #be4a2f, #d1682f)',
            color: '#fff',
            fontWeight: 700,
            cursor: 'pointer',
            fontSize: '0.9rem',
          }}>
            {showForm ? '关闭' : '+ 新任务'}
          </button>
        </div>

        {/* 创建表单 */}
        {showForm && (
          <div style={{
            background: 'rgba(255,250,239,0.92)',
            border: '1px solid rgba(28,26,23,0.14)',
            borderRadius: '14px',
            padding: '16px',
            marginBottom: '12px',
          }}>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>GNN 模型</label>
              <select value={config.gnn_type} onChange={(e) => setConfig({ ...config, gnn_type: e.target.value })}
                style={{ width: '100%', height: '36px', borderRadius: '8px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 8px' }}>
                {GNN_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </div>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>源域数量</label>
              <select value={config.num_domain} onChange={(e) => setConfig({ ...config, num_domain: Number(e.target.value) })}
                style={{ width: '100%', height: '36px', borderRadius: '8px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 8px' }}>
                {[2, 4, 8, 16].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>目标域索引</label>
              <input type="number" value={config.target_domain} onChange={(e) => setConfig({ ...config, target_domain: Number(e.target.value) })}
                style={{ width: '100%', height: '36px', borderRadius: '8px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 8px' }} />
            </div>
            <details>
              <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: '#5d5850' }}>高级参数</summary>
              <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', display: 'block' }}>KG 轮数</label>
                  <input type="number" value={config.round_gat} onChange={(e) => setConfig({ ...config, round_gat: Number(e.target.value) })}
                    style={{ width: '100%', height: '32px', borderRadius: '6px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 6px' }} />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', display: 'block' }}>FT 轮数</label>
                  <input type="number" value={config.round_ft} onChange={(e) => setConfig({ ...config, round_ft: Number(e.target.value) })}
                    style={{ width: '100%', height: '32px', borderRadius: '6px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 6px' }} />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', display: 'block' }}>学习率 GNN</label>
                  <input type="number" step="0.0001" value={config.lr_gnn} onChange={(e) => setConfig({ ...config, lr_gnn: Number(e.target.value) })}
                    style={{ width: '100%', height: '32px', borderRadius: '6px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 6px' }} />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', display: 'block' }}>DP</label>
                  <select value={String(config.dp)} onChange={(e) => setConfig({ ...config, dp: e.target.value === 'true' })}
                    style={{ width: '100%', height: '32px', borderRadius: '6px', border: '1px solid rgba(28,26,23,0.14)', background: '#fff9ee', padding: '0 6px' }}>
                    <option value="true">开启</option>
                    <option value="false">关闭</option>
                  </select>
                </div>
              </div>
            </details>
            <button onClick={handleCreate} disabled={creating} style={{
              width: '100%', marginTop: '12px', height: '40px', borderRadius: '10px', border: 'none',
              background: creating ? '#9ca3af' : 'linear-gradient(140deg, #0d6e6e, #1f5d5d)',
              color: '#fff', fontWeight: 700, cursor: creating ? 'not-allowed' : 'pointer',
            }}>
              {creating ? '创建中...' : '创建训练任务'}
            </button>
          </div>
        )}

        {/* 任务列表 */}
        <div style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 200px)' }}>
          {tasks.map((task) => (
            <div
              key={task.id}
              onClick={() => selectTask(task)}
              style={{
                padding: '12px',
                marginBottom: '6px',
                borderRadius: '10px',
                border: `1px solid ${selectedTask?.id === task.id ? '#be4a2f' : 'rgba(28,26,23,0.14)'}`,
                background: selectedTask?.id === task.id ? 'rgba(190,74,47,0.06)' : 'rgba(255,250,239,0.88)',
                cursor: 'pointer',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong style={{ fontSize: '0.9rem' }}>
                  {task.name || `${task.gnn_type.toUpperCase()} ${task.num_domain}domains`}
                </strong>
                <span style={{
                  fontSize: '0.75rem',
                  padding: '2px 8px',
                  borderRadius: '6px',
                  fontWeight: 600,
                  background: task.status === 'running' ? '#3b82f620' : task.status === 'completed' ? '#22c55e20' : task.status === 'failed' ? '#ef444420' : '#f59e0b20',
                  color: task.status === 'running' ? '#3b82f6' : task.status === 'completed' ? '#22c55e' : task.status === 'failed' ? '#ef4444' : '#f59e0b',
                }}>
                  {task.status}
                </span>
              </div>
              {task.best_hr10 != null && (
                <div style={{ fontSize: '0.85rem', color: '#5d5850', marginTop: '4px' }}>
                  HR@10: <strong>{(task.best_hr10 * 100).toFixed(2)}%</strong>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* 右侧：任务详情 */}
      <div style={{ flex: 1 }}>
        {selectedTask ? (
          <>
            {/* 任务信息 */}
            <div style={{
              background: 'rgba(255,250,239,0.88)',
              border: '1px solid rgba(28,26,23,0.14)',
              borderRadius: '14px',
              padding: '16px',
              marginBottom: '12px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <div>
                <h3 style={{ margin: 0, fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400 }}>
                  #{selectedTask.id} {selectedTask.name || ''}
                </h3>
                <div style={{ color: '#5d5850', fontSize: '0.85rem', marginTop: '4px' }}>
                  {selectedTask.gnn_type.toUpperCase()} · {selectedTask.num_domain} domains · {selectedTask.dataset}
                </div>
              </div>
              {selectedTask.status === 'running' && (
                <button onClick={() => handleCancel(selectedTask.id)} style={{
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: '1px solid #ef4444',
                  background: '#fff',
                  color: '#ef4444',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}>
                  取消任务
                </button>
              )}
            </div>

            {/* 实时指标图表 */}
            {metrics.length > 0 && (
              <div style={{
                background: 'rgba(255,250,239,0.88)',
                border: '1px solid rgba(28,26,23,0.14)',
                borderRadius: '14px',
                padding: '16px',
                marginBottom: '12px',
              }}>
                <h4 style={{ margin: '0 0 10px', fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400 }}>
                  指标曲线
                </h4>
                <div style={{ position: 'relative', height: '200px', background: '#fff9ee', borderRadius: '10px', overflow: 'hidden' }}>
                  <MiniChart metrics={metrics} />
                </div>
              </div>
            )}

            {/* 实时日志 */}
            <div style={{
              background: '#1c1a17',
              border: '1px solid rgba(28,26,23,0.14)',
              borderRadius: '14px',
              padding: '16px',
              maxHeight: '400px',
              overflow: 'auto',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <h4 style={{ margin: 0, color: '#fff', fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400 }}>
                  训练日志
                </h4>
              </div>
              <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: '#a8a29e', lineHeight: 1.6 }}>
                {logs.length === 0 && <span style={{ color: '#6b7280' }}>等待日志输出...</span>}
                {logs.map((log, i) => (
                  <div key={i}>{log.message}</div>
                ))}
                <div ref={logEndRef} />
              </div>
            </div>
          </>
        ) : (
          <div style={{
            background: 'rgba(255,250,239,0.88)',
            border: '1px dashed rgba(28,26,23,0.14)',
            borderRadius: '14px',
            padding: '40px',
            textAlign: 'center',
            color: '#5d5850',
          }}>
            选择左侧任务查看详情
          </div>
        )}
      </div>
    </div>
  );
}

/** 迷你折线图组件 */
function MiniChart({ metrics }: { metrics: MetricPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length < 2) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const hr10Values = metrics.map((m) => m.hr_10 ?? 0);
    const maxVal = Math.max(...hr10Values, 0.001);
    const minVal = Math.min(...hr10Values, 0);
    const range = maxVal - minVal || 0.001;

    ctx.strokeStyle = '#be4a2f';
    ctx.lineWidth = 2;
    ctx.beginPath();
    hr10Values.forEach((v, i) => {
      const x = (i / (hr10Values.length - 1)) * w;
      const y = h - ((v - minVal) / range) * h * 0.9 - h * 0.05;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [metrics]);

  return (
    <canvas ref={canvasRef} width={600} height={200} style={{ width: '100%', height: '200px' }} />
  );
}
