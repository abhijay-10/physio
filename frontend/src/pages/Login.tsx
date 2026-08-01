import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Lock, Mail, ArrowRight, X } from 'lucide-react';

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const [error, setError] = useState('');

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in all fields.');
      return;
    }
    
    const existingUsers = JSON.parse(localStorage.getItem('physio_users_db') || '[]');
    const user = existingUsers.find((u: any) => u.email === email && u.password === password);
    
    if (user) {
      login(user.name);
      navigate('/');
    } else {
      setError('Invalid email or password. Please try again or sign up.');
    }
  };

  return (
    <div className="landing-page-wrapper" style={{ minHeight: 'calc(100vh - 70px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="glass-panel" style={{ padding: '40px', maxWidth: '400px', width: '100%', position: 'relative' }}>
        <button 
          onClick={() => navigate('/')}
          style={{ position: 'absolute', top: '20px', right: '20px', background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4px', borderRadius: '50%', transition: 'background 0.2s' }}
          onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
          onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
        >
          <X size={20} />
        </button>
        <h2 style={{ textAlign: 'center', marginBottom: '8px', fontSize: '1.8rem' }}>Student Login</h2>
        <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '32px' }}>Access your clinical diagnostic portal</p>
        
        {error && (
          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.5)', padding: '12px', borderRadius: '8px', color: '#ef4444', marginBottom: '20px', fontSize: '0.9rem', textAlign: 'center' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>University Email</label>
            <div style={{ position: 'relative' }}>
              <Mail size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-secondary)' }} />
              <input 
                type="email" 
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{ width: '100%', padding: '12px 12px 12px 40px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', outline: 'none' }} 
                placeholder="student@university.edu"
              />
            </div>
          </div>
          
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Password</label>
            <div style={{ position: 'relative' }}>
              <Lock size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-secondary)' }} />
              <input 
                type="password" 
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ width: '100%', padding: '12px 12px 12px 40px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', outline: 'none' }} 
                placeholder="••••••••"
              />
            </div>
          </div>
          
          <button type="submit" className="btn btn-primary" style={{ padding: '14px', marginTop: '10px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            Login <ArrowRight size={18} style={{ marginLeft: '8px' }} />
          </button>
        </form>
        
        <div style={{ textAlign: 'center', marginTop: '24px', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Don't have an account? <span onClick={() => navigate('/signup')} style={{ color: 'var(--accent-color)', cursor: 'pointer', fontWeight: 'bold' }}>Sign up</span>
        </div>
      </div>
    </div>
  );
};

export default Login;
