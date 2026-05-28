import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { authApi } from '../api/client';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const data = await authApi.login(username, password);
      localStorage.setItem('token', data.access_token);
      navigate('/');
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || '登录失败';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(160deg, #f5f0e7, #efe4d2)',
    }}>
      <form onSubmit={handleSubmit} style={{
        background: 'rgba(255,250,239,0.92)',
        padding: '40px',
        borderRadius: '18px',
        boxShadow: '0 22px 36px rgba(41,30,16,0.14)',
        border: '1px solid rgba(28,26,23,0.14)',
        width: '380px',
        maxWidth: '90vw',
      }}>
        <h1 style={{ margin: '0 0 8px', fontFamily: '"ZCOOL XiaoWei", serif', fontWeight: 400 }}>
          FedGCDR
        </h1>
        <p style={{ margin: '0 0 28px', color: '#5d5850' }}>
          联邦跨域推荐 · 训练管理平台
        </p>

        {error && (
          <div style={{
            padding: '10px 14px',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '10px',
            color: '#991b1b',
            marginBottom: '16px',
            fontSize: '0.9rem',
          }}>{error}</div>
        )}

        <div style={{ marginBottom: '16px' }}>
          <label style={{ display: 'block', marginBottom: '6px', fontWeight: 600, color: '#3a3834' }}>
            用户名
          </label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="admin"
            style={{
              width: '100%',
              height: '42px',
              borderRadius: '10px',
              border: '1px solid rgba(28,26,23,0.14)',
              padding: '0 12px',
              fontSize: '0.96rem',
              background: '#fff9ee',
              boxSizing: 'border-box',
            }}
            required
          />
        </div>
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', marginBottom: '6px', fontWeight: 600, color: '#3a3834' }}>
            密码
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••"
            style={{
              width: '100%',
              height: '42px',
              borderRadius: '10px',
              border: '1px solid rgba(28,26,23,0.14)',
              padding: '0 12px',
              fontSize: '0.96rem',
              background: '#fff9ee',
              boxSizing: 'border-box',
            }}
            required
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%',
            height: '44px',
            borderRadius: '10px',
            border: 'none',
            background: 'linear-gradient(140deg, #be4a2f, #d1682f)',
            color: '#fff',
            fontSize: '1rem',
            fontWeight: 700,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? '登录中...' : '登录'}
        </button>
      </form>
    </div>
  );
}
