import { useEffect, useState } from 'react';
import { trainingApi } from '../api/client';
import type { TrainingTask } from '../api/client';

const STATUS_COLORS: Record<string, string> = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#22c55e',
  failed: '#ef4444',
  cancelled: '#6b7280',
};

const GNN_COLORS: Record<string, string> = {
  gat: '#be4a2f',
  lightgcn: '#0d6e6e',
  graphsage: '#7c3aed',
  gcn: '#2563eb',
};

export default function DashboardPage() {
  const [tasks, setTasks] = useState<TrainingTask[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    trainingApi.list({ limit: 50 }).then(setTasks).finally(() => setLoading(false));
  }, []);

  const runningTasks = tasks.filter((t) => t.status === 'running');
  const recentTasks = tasks.slice(0, 10);

  // 统计
  const stats = {
    total: tasks.length,
    running: runningTasks.length,
    completed: tasks.filter((t) => t.status === 'completed').length,
    failed: tasks.filter((t) => t.status === 'failed').length,
  };

  // 各模型最佳 HR@10
  const bestByGnn: Record<string, { best: number; task: TrainingTask }> = {};
  for (const t of tasks) {
    if (t.best_hr10 != null && (!bestByGnn[t.gnn_type] || t.best_hr10 > bestByGnn[t.gnn_type].best)) {
      bestByGnn[t.gnn_type] = { best: t.best_hr10, task: t };
    }
  }

  if (loading) {
    return <div style={{ padding: '40px', color: '#5d5850' }}>加载中...</div>;
  }

  return (
    <div>
      <h1 style={{ fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400, marginBottom: '8px' }}>
        训练看板
      </h1>
      <p style={{ color: '#5d5850', marginBottom: '28px' }}>
        概览全部训练任务的状态与性能对比
      </p>

      {/* 统计卡片 */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
        gap: '12px',
        marginBottom: '28px',
      }}>
        {[
          ['总任务', stats.total, '#1c1a17'],
          ['运行中', stats.running, '#3b82f6'],
          ['已完成', stats.completed, '#22c55e'],
          ['失败', stats.failed, '#ef4444'],
        ].map(([label, value, color]) => (
          <div key={String(label)} style={{
            background: 'rgba(255,250,239,0.88)',
            border: '1px solid rgba(28,26,23,0.14)',
            borderRadius: '14px',
            padding: '16px',
          }}>
            <div style={{ color: '#5d5850', fontSize: '0.86rem' }}>{String(label)}</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 700, color: String(color), marginTop: '4px' }}>
              {String(value)}
            </div>
          </div>
        ))}
      </div>

      {/* 各模型最佳对比 */}
      {Object.keys(bestByGnn).length > 0 && (
        <div style={{ marginBottom: '28px' }}>
          <h2 style={{ fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400, fontSize: '1.2rem', marginBottom: '12px' }}>
            各模型最佳 HR@10
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '12px',
          }}>
            {Object.entries(bestByGnn).map(([gnn, { best, task }]) => (
              <div key={gnn} style={{
                background: 'rgba(255,250,239,0.88)',
                border: '1px solid rgba(28,26,23,0.14)',
                borderRadius: '14px',
                padding: '14px',
                borderLeft: `4px solid ${GNN_COLORS[gnn] || '#666'}`,
              }}>
                <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{gnn.toUpperCase()}</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: GNN_COLORS[gnn] || '#666', margin: '4px 0' }}>
                  {(best * 100).toFixed(2)}%
                </div>
                <div style={{ fontSize: '0.8rem', color: '#5d5850' }}>
                  {task.num_domain} domains · #{task.id}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 运行中的任务 */}
      {runningTasks.length > 0 && (
        <div style={{ marginBottom: '28px' }}>
          <h2 style={{ fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400, fontSize: '1.2rem', marginBottom: '12px' }}>
            运行中
          </h2>
          {runningTasks.map((task) => (
            <div key={task.id} style={{
              background: 'rgba(255,250,239,0.88)',
              border: '1px solid rgba(28,26,23,0.14)',
              borderRadius: '14px',
              padding: '14px',
              marginBottom: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <div>
                <strong>{task.name || `${task.gnn_type.toUpperCase()} ${task.num_domain}domains`}</strong>
                <span style={{ marginLeft: '10px', color: '#3b82f6', fontSize: '0.85rem' }}>● 运行中</span>
              </div>
              <div style={{ color: '#5d5850', fontSize: '0.9rem' }}>
                {task.started_at ? new Date(task.started_at).toLocaleString() : ''}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 最近任务列表 */}
      <div>
        <h2 style={{ fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400, fontSize: '1.2rem', marginBottom: '12px' }}>
          最近任务
        </h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            background: 'rgba(255,250,239,0.88)',
            borderRadius: '14px',
            border: '1px solid rgba(28,26,23,0.14)',
            fontSize: '0.9rem',
          }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(28,26,23,0.14)' }}>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>ID</th>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>名称</th>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>模型</th>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>域数</th>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>状态</th>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>HR@10</th>
                <th style={{ padding: '12px 10px', textAlign: 'left' }}>时间</th>
              </tr>
            </thead>
            <tbody>
              {recentTasks.map((task) => (
                <tr key={task.id} style={{ borderBottom: '1px solid rgba(28,26,23,0.08)' }}>
                  <td style={{ padding: '10px', color: '#5d5850' }}>#{task.id}</td>
                  <td style={{ padding: '10px', fontWeight: 600 }}>
                    {task.name || `${task.gnn_type.toUpperCase()} ${task.num_domain}domains`}
                  </td>
                  <td style={{ padding: '10px' }}>
                    <span style={{
                      background: `${GNN_COLORS[task.gnn_type] || '#666'}20`,
                      color: GNN_COLORS[task.gnn_type] || '#666',
                      padding: '2px 8px',
                      borderRadius: '6px',
                      fontWeight: 600,
                      fontSize: '0.85rem',
                    }}>
                      {task.gnn_type.toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: '10px' }}>{task.num_domain}</td>
                  <td style={{ padding: '10px' }}>
                    <span style={{ color: STATUS_COLORS[task.status] || '#666', fontWeight: 600 }}>
                      ● {task.status}
                    </span>
                  </td>
                  <td style={{ padding: '10px', fontWeight: 700 }}>
                    {task.best_hr10 != null ? `${(task.best_hr10 * 100).toFixed(2)}%` : '-'}
                  </td>
                  <td style={{ padding: '10px', color: '#5d5850', fontSize: '0.85rem' }}>
                    {new Date(task.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
