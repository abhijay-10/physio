import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Lock, Mail, User, ArrowRight, X } from 'lucide-react';

const Signup: React.FC = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const [error, setError] = useState('');

  const handleSignup = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !email || !password) {
      setError('Please fill in all fields.');
      return;
    }
    
    // Simulated Database
    const existingUsers = JSON.parse(localStorage.getItem('physio_users_db') || '[]');
    if (existingUsers.some((u: any) => u.email === email)) {
      setError('An account with this email already exists.');
      return;
    }
    
    existingUsers.push({ name, email, password });
    localStorage.setItem('physio_users_db', JSON.stringify(existingUsers));
    
    login(name);
    navigate('/');
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
        <h2 style={{ textAlign: 'center', marginBottom: '8px', fontSize: '1.8rem' }}>Create Account</h2>
        <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '32px' }}>Join the clinical diagnostic portal</p>
        
        {error && (
          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.5)', padding: '12px', borderRadius: '8px', color: '#ef4444', marginBottom: '20px', fontSize: '0.9rem', textAlign: 'center' }}>
            {error}
          </div>
        )}
        
        <form onSubmit={handleSignup} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Full Name</label>
            <div style={{ position: 'relative' }}>
              <User size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-secondary)' }} />
              <input 
                type="text" 
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                style={{ width: '100%', padding: '12px 12px 12px 40px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', outline: 'none' }} 
                placeholder="Student Name"
              />
            </div>
          </div>

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
            Sign Up <ArrowRight size={18} style={{ marginLeft: '8px' }} />
          </button>
        </form>
        
        <div style={{ textAlign: 'center', marginTop: '24px', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Already have an account? <span onClick={() => navigate('/login')} style={{ color: 'var(--accent-color)', cursor: 'pointer', fontWeight: 'bold' }}>Login</span>
        </div>
      </div>
    </div>
  );
};

export default Signup;
