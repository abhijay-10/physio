import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const TopNav: React.FC = () => {
  const { isLoggedIn, logout, userName } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark');
  }, []);

  return (
    <div className="top-nav-light">
      <div className="nav-container-light">
        <Link to="/" className="nav-logo-light">
          PHYSIOMASTER
        </Link>
        
        <div className="nav-links-light">
          <a href="#" className="nav-link-light">ABOUT</a>
          <a href="#" className="nav-link-light">DIAGNOSTICS <span style={{fontSize: '10px'}}>▼</span></a>
          <a href="#" className="nav-link-light">CLINICS</a>
          <Link to={isLoggedIn ? "/portfolio" : "/login"} className="nav-link-light">PORTFOLIO</Link>
          <a href="#" className="nav-link-light">BLOG</a>
        </div>

        <div className="nav-auth-light">
          {isLoggedIn ? (
            <>
              <span className="nav-link-light" style={{ textTransform: 'none', fontWeight: 500, cursor: 'default' }}>Welcome, {userName}</span>
              <button 
                className="btn-outline-light" 
                onClick={() => {
                  logout();
                  navigate('/');
                }}
                style={{ color: 'var(--text-primary)', borderColor: 'var(--border-color)' }}
              >
                LOGOUT
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="nav-link-light" style={{ textTransform: 'none', fontWeight: 500 }}>Sign In</Link>
              <Link to="/signup" className="btn-outline-light" style={{ color: 'var(--text-primary)', borderColor: 'var(--border-color)', textDecoration: 'none', padding: '8px 16px', display: 'inline-flex', alignItems: 'center' }}>SIGN UP</Link>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default TopNav;
