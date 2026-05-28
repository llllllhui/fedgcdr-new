import { BrowserRouter, Routes, Route, Navigate, Link, useLocation, useNavigate } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage';
import TrainingPage from './pages/TrainingPage';
import RecommendationPage from './pages/RecommendationPage';
import LoginPage from './pages/LoginPage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(160deg, #f5f0e7, #efe4d2)' }}>
      <nav style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(245,240,231,0.92)', backdropFilter: 'blur(6px)',
        borderBottom: '1px solid rgba(28,26,23,0.14)',
        padding: '0 20px', height: '56px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <Link to="/" style={{
            fontFamily: '"ZCOOL XiaoWei", serif', fontSize: '1.2rem', fontWeight: 600,
            color: '#1c1a17', textDecoration: 'none',
          }}>
            FedGCDR
          </Link>
          <Link to="/" style={{
            color: isActive('/') ? '#be4a2f' : '#5d5850', textDecoration: 'none',
            fontWeight: isActive('/') ? 700 : 500, fontSize: '0.9rem',
          }}>
            看板
          </Link>
          <Link to="/training" style={{
            color: isActive('/training') ? '#be4a2f' : '#5d5850', textDecoration: 'none',
            fontWeight: isActive('/training') ? 700 : 500, fontSize: '0.9rem',
          }}>
            训练管理
          </Link>
          <Link to="/recommendations" style={{
            color: isActive('/recommendations') ? '#be4a2f' : '#5d5850', textDecoration: 'none',
            fontWeight: isActive('/recommendations') ? 700 : 500, fontSize: '0.9rem',
          }}>
            推荐查询
          </Link>
        </div>
        <button onClick={handleLogout} style={{
          padding: '6px 14px', borderRadius: '8px', border: '1px solid rgba(28,26,23,0.14)',
          background: 'transparent', cursor: 'pointer', color: '#5d5850', fontSize: '0.85rem',
        }}>
          退出登录
        </button>
      </nav>
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px 20px', position: 'relative' }}>
        {children}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<ProtectedRoute><Layout><DashboardPage /></Layout></ProtectedRoute>} />
        <Route path="/training" element={<ProtectedRoute><Layout><TrainingPage /></Layout></ProtectedRoute>} />
        <Route path="/recommendations" element={<ProtectedRoute><Layout><RecommendationPage /></Layout></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
