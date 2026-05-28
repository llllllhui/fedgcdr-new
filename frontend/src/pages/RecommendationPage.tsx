import { useEffect, useState, useCallback } from 'react';
import { recoApi } from '../api/client';
import type {
  RecoSnapshot,
  Top10Response,
  Top10Success,
  Top10NotFound,
} from '../api/client';

const GNN_COLORS: Record<string, string> = {
  gat: '#be4a2f',
  lightgcn: '#0d6e6e',
  graphsage: '#7c3aed',
  gcn: '#2563eb',
};

interface SnapshotGroup {
  gnn_type: string;
  num_domain: number;
  snapshots: RecoSnapshot[];
}

/** 从快照列表构建分组索引 */
function groupSnapshots(snapshots: RecoSnapshot[]): SnapshotGroup[] {
  const map = new Map<string, SnapshotGroup>();
  for (const s of snapshots) {
    const key = `${s.gnn_type}|${s.num_domain}`;
    if (!map.has(key)) {
      map.set(key, { gnn_type: s.gnn_type, num_domain: s.num_domain, snapshots: [] });
    }
    map.get(key)!.snapshots.push(s);
  }
  return Array.from(map.values()).sort((a, b) => {
    if (a.gnn_type !== b.gnn_type) return a.gnn_type.localeCompare(b.gnn_type);
    return a.num_domain - b.num_domain;
  });
}

export default function RecommendationPage() {
  const [allSnapshots, setAllSnapshots] = useState<RecoSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // 筛选状态
  const [selectedGroup, setSelectedGroup] = useState<string>('');
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string>('');

  // 用户查询
  const [userIndex, setUserIndex] = useState('');
  const [querying, setQuerying] = useState(false);
  const [queryResult, setQueryResult] = useState<Top10Success | null>(null);
  const [queryError, setQueryError] = useState('');
  const [suggestedUsers, setSuggestedUsers] = useState<number[]>([]);

  // 加载快照列表
  useEffect(() => {
    recoApi
      .snapshots()
      .then((data) => {
        setAllSnapshots(data.snapshots || []);
        if (data.message) setError(data.message);
      })
      .catch((err) => setError(err?.response?.data?.detail || '加载推荐数据失败'))
      .finally(() => setLoading(false));
  }, []);

  const groups = groupSnapshots(allSnapshots);

  // 当前选中的分组
  const currentGroup = groups.find((g) => `${g.gnn_type}|${g.num_domain}` === selectedGroup);
  const currentSnapshots = currentGroup?.snapshots || [];

  // 当前选中的快照
  const currentSnapshot = currentSnapshots.find((s) => s.id === selectedSnapshotId);

  // 当分组改变时，自动选第一个快照
  useEffect(() => {
    if (currentSnapshots.length > 0 && !currentSnapshots.find((s) => s.id === selectedSnapshotId)) {
      setSelectedSnapshotId(currentSnapshots[0].id);
      clearResult();
    }
  }, [selectedGroup, currentSnapshots]);

  // 如果有快照但未选分组，自动选中第一个
  useEffect(() => {
    if (!selectedGroup && groups.length > 0) {
      const first = groups[0];
      setSelectedGroup(`${first.gnn_type}|${first.num_domain}`);
    }
  }, [groups]);

  const clearResult = () => {
    setQueryResult(null);
    setQueryError('');
    setSuggestedUsers([]);
  };

  // 查询 Top10
  const queryTop10 = useCallback(
    async (targetUserIndex?: number) => {
      const idx = targetUserIndex ?? Number(userIndex);
      if (!selectedSnapshotId || isNaN(idx) || idx < 0) {
        setQueryError('请输入有效的非负整数用户索引');
        return;
      }

      setQuerying(true);
      setQueryResult(null);
      setQueryError('');
      setSuggestedUsers([]);

      try {
        const res: Top10Response = await recoApi.top10(selectedSnapshotId, idx);
        if (res.found) {
          setQueryResult(res);
        } else {
          const notFound = res as Top10NotFound;
          setQueryError(notFound.message);
          setSuggestedUsers(notFound.valid_users_sample || []);
        }
      } catch (err: unknown) {
        const detail =
          (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
          '查询失败';
        setQueryError(detail);
      } finally {
        setQuerying(false);
      }
    },
    [selectedSnapshotId, userIndex],
  );

  // 随机有效用户
  const pickRandomUser = useCallback(() => {
    if (!currentSnapshot) return;
    const maxUser = currentSnapshot.global_user_count;
    if (maxUser <= 0) {
      setQueryError('当前快照没有可用用户');
      return;
    }
    // 先尝试随机，用户大概率在目标域中
    const rand = Math.floor(Math.random() * maxUser);
    setUserIndex(String(rand));
    queryTop10(rand);
  }, [currentSnapshot, queryTop10]);

  //  渲染
  if (loading) {
    return <div style={{ padding: '40px', color: '#5d5850' }}>加载推荐数据...</div>;
  }

  return (
    <div>
      <h1 style={{ fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400, marginBottom: '8px' }}>
        推荐查询
      </h1>
      <p style={{ color: '#5d5850', marginBottom: '24px' }}>
        选择推荐快照，查看跨域知识迁移前/后的 Top10 物品推荐对比
      </p>

      {/* 错误横幅 */}
      {error && (
        <div
          style={{
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '10px',
            padding: '12px 16px',
            color: '#b91c1c',
            marginBottom: '16px',
          }}
        >
          {error}
        </div>
      )}

      {/* ── 筛选栏 ── */}
      <div
        style={{
          background: 'rgba(255,250,239,0.88)',
          border: '1px solid rgba(28,26,23,0.14)',
          borderRadius: '14px',
          padding: '16px',
          marginBottom: '16px',
          display: 'flex',
          gap: '16px',
          flexWrap: 'wrap',
          alignItems: 'flex-end',
        }}
      >
        <div style={{ minWidth: '180px', flex: 1 }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
            模型 + 域数
          </label>
          <select
            value={selectedGroup}
            onChange={(e) => {
              setSelectedGroup(e.target.value);
              clearResult();
            }}
            style={{
              width: '100%',
              height: '38px',
              borderRadius: '8px',
              border: '1px solid rgba(28,26,23,0.14)',
              background: '#fff9ee',
              padding: '0 8px',
            }}
          >
            {groups.map((g) => (
              <option key={`${g.gnn_type}|${g.num_domain}`} value={`${g.gnn_type}|${g.num_domain}`}>
                {g.gnn_type.toUpperCase()} · {g.num_domain} domains
              </option>
            ))}
          </select>
        </div>

        <div style={{ minWidth: '240px', flex: 2 }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
            推荐快照
          </label>
          <select
            value={selectedSnapshotId}
            onChange={(e) => {
              setSelectedSnapshotId(e.target.value);
              clearResult();
            }}
            style={{
              width: '100%',
              height: '38px',
              borderRadius: '8px',
              border: '1px solid rgba(28,26,23,0.14)',
              background: '#fff9ee',
              padding: '0 8px',
            }}
          >
            {currentSnapshots.map((s) => (
              <option key={s.id} value={s.id}>
                [{s.target_domain_name}] {s.timestamp.replace('T', ' ')}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── 快照元信息 + 用户查询 ── */}
      {currentSnapshot && (
        <div
          style={{
            background: 'rgba(255,250,239,0.88)',
            border: '1px solid rgba(28,26,23,0.14)',
            borderRadius: '14px',
            padding: '16px',
            marginBottom: '16px',
          }}
        >
          {/* 元信息 */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: '8px',
              marginBottom: '16px',
              paddingBottom: '16px',
              borderBottom: '1px solid rgba(28,26,23,0.08)',
            }}
          >
            {[
              ['目标域', currentSnapshot.target_domain_name],
              ['模型', currentSnapshot.gnn_type.toUpperCase()],
              ['源域数', String(currentSnapshot.num_domain)],
              ['目标域用户', String(currentSnapshot.global_user_count)],
              ['时间', currentSnapshot.timestamp.replace('T', ' ')],
            ].map(([k, v]) => (
              <div key={k}>
                <div style={{ fontSize: '0.8rem', color: '#5d5850' }}>{k}</div>
                <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{v}</div>
              </div>
            ))}
          </div>

          {/* 查询行 */}
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div style={{ minWidth: '160px' }}>
              <label
                style={{
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  display: 'block',
                  marginBottom: '4px',
                }}
              >
                全局用户索引
              </label>
              <input
                type="number"
                min={0}
                value={userIndex}
                onChange={(e) => {
                  setUserIndex(e.target.value);
                  clearResult();
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') queryTop10();
                }}
                placeholder="输入用户索引"
                style={{
                  width: '100%',
                  height: '38px',
                  borderRadius: '8px',
                  border: '1px solid rgba(28,26,23,0.14)',
                  background: '#fff9ee',
                  padding: '0 10px',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            <button
              onClick={() => queryTop10()}
              disabled={querying || !selectedSnapshotId}
              style={{
                height: '38px',
                padding: '0 20px',
                borderRadius: '8px',
                border: 'none',
                background:
                  querying || !selectedSnapshotId
                    ? '#9ca3af'
                    : 'linear-gradient(140deg, #0d6e6e, #1f5d5d)',
                color: '#fff',
                fontWeight: 700,
                cursor: querying || !selectedSnapshotId ? 'not-allowed' : 'pointer',
                fontSize: '0.9rem',
              }}
            >
              {querying ? '查询中...' : '查询 Top10'}
            </button>

            <button
              onClick={pickRandomUser}
              disabled={querying || !currentSnapshot}
              style={{
                height: '38px',
                padding: '0 16px',
                borderRadius: '8px',
                border: '1px solid rgba(28,26,23,0.14)',
                background: '#fff9ee',
                color: '#1c1a17',
                fontWeight: 600,
                cursor: querying || !currentSnapshot ? 'not-allowed' : 'pointer',
                fontSize: '0.9rem',
              }}
            >
              随机有效用户
            </button>
          </div>

          {/* 查询状态 / 建议用户 */}
          {queryError && (
            <div
              style={{
                marginTop: '12px',
                padding: '10px 14px',
                background: '#fef2f2',
                borderRadius: '8px',
                color: '#b91c1c',
                fontSize: '0.9rem',
              }}
            >
              {queryError}
              {suggestedUsers.length > 0 && (
                <div style={{ marginTop: '6px' }}>
                  可尝试用户：
                  {suggestedUsers.map((u) => (
                    <button
                      key={u}
                      onClick={() => {
                        setUserIndex(String(u));
                        queryTop10(u);
                      }}
                      style={{
                        margin: '0 4px',
                        padding: '2px 10px',
                        borderRadius: '6px',
                        border: '1px solid rgba(28,26,23,0.14)',
                        background: '#fff',
                        cursor: 'pointer',
                        fontWeight: 600,
                        color: '#be4a2f',
                      }}
                    >
                      {u}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 查询成功提示 */}
          {queryResult && (
            <div
              style={{
                marginTop: '12px',
                padding: '10px 14px',
                background: '#f0fdf4',
                borderRadius: '8px',
                color: '#166534',
                fontSize: '0.9rem',
              }}
            >
              用户 {queryResult.global_user_index} → 目标域局部索引 {queryResult.local_user_index}
            </div>
          )}
        </div>
      )}

      {/* ── 对比展示 ── */}
      {queryResult && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          {/* 跨域前 */}
          <div
            style={{
              background: 'rgba(255,250,239,0.88)',
              border: '1px solid rgba(28,26,23,0.14)',
              borderRadius: '14px',
              padding: '16px',
            }}
          >
            <h3
              style={{
                fontFamily: '"ZCOOL XiaoWei", serif',
                fontWeight: 400,
                margin: '0 0 12px',
                color: '#9ca3af',
                fontSize: '1rem',
              }}
            >
              跨域前（KG 阶段）
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {queryResult.top10_before.length === 0 ? (
                <div style={{ color: '#5d5850' }}>无数据</div>
              ) : (
                queryResult.top10_before.map((itemId, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '8px 12px',
                      background: 'rgba(28,26,23,0.03)',
                      borderRadius: '8px',
                    }}
                  >
                    <span
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '50%',
                        background: '#e5e0d6',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 700,
                        fontSize: '0.85rem',
                        color: '#5d5850',
                        flexShrink: 0,
                      }}
                    >
                      {i + 1}
                    </span>
                    <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'monospace' }}>
                      Item #{itemId}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* 跨域后 */}
          <div
            style={{
              background: 'rgba(255,250,239,0.88)',
              border: `1px solid ${GNN_COLORS[currentSnapshot?.gnn_type || 'gat']}40`,
              borderRadius: '14px',
              padding: '16px',
              borderLeft: `4px solid ${GNN_COLORS[currentSnapshot?.gnn_type || 'gat']}`,
            }}
          >
            <h3
              style={{
                fontFamily: '"ZCOOL XiaoWei", serif',
                fontWeight: 400,
                margin: '0 0 12px',
                color: GNN_COLORS[currentSnapshot?.gnn_type || 'gat'],
                fontSize: '1rem',
              }}
            >
              跨域后（KT + FT 阶段）
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {queryResult.top10_after.length === 0 ? (
                <div style={{ color: '#5d5850' }}>无数据</div>
              ) : (
                queryResult.top10_after.map((itemId, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '8px 12px',
                      background: `${GNN_COLORS[currentSnapshot?.gnn_type || 'gat']}08`,
                      borderRadius: '8px',
                    }}
                  >
                    <span
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '50%',
                        background: GNN_COLORS[currentSnapshot?.gnn_type || 'gat'],
                        color: '#fff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 700,
                        fontSize: '0.85rem',
                        flexShrink: 0,
                      }}
                    >
                      {i + 1}
                    </span>
                    <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'monospace' }}>
                      Item #{itemId}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* 无快照提示 */}
      {!loading && allSnapshots.length === 0 && !error && (
        <div
          style={{
            background: 'rgba(255,250,239,0.88)',
            border: '1px dashed rgba(28,26,23,0.14)',
            borderRadius: '14px',
            padding: '40px',
            textAlign: 'center',
            color: '#5d5850',
          }}
        >
          暂无推荐数据。请先运行训练任务并执行
          <code style={{ background: '#e5e0d6', padding: '2px 8px', borderRadius: '4px', margin: '0 4px' }}>
            python training-results-web/scripts/build_recommendation_data.py
          </code>
          生成推荐快照。
        </div>
      )}
    </div>
  );
}
