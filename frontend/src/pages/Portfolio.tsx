import React, { useEffect, useState } from 'react';
import { Activity, Download, Trash2, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Portfolio: React.FC = () => {
  const [images, setImages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const { userName } = useAuth();

  useEffect(() => {
    window.scrollTo(0, 0);
    fetchImages();
  }, []);

  const fetchImages = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/captures_list');
      const data = await res.json();
      setImages(data.images || []);
    } catch (e) {
      console.error("Failed to fetch captures", e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="category-view-container-light">
      <div className="category-header-light">
        <button className="back-button-light" onClick={() => navigate(-1)}>
          <ArrowLeft size={24} />
        </button>
        <div className="header-content-light">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <Activity className="text-accent" size={32} />
            <h1 className="category-title-light">Student Portfolio</h1>
          </div>
          <p className="category-subtitle-light">
            Review your autonomous diagnostic captures and AI grades, {userName}.
          </p>
        </div>
      </div>

      <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '100px 0', color: 'var(--text-secondary)' }}>
            <Activity className="spinner" size={32} style={{ margin: '0 auto 16px' }} />
            <p>Loading portfolio...</p>
          </div>
        ) : images.length === 0 ? (
          <div className="glass-panel" style={{ textAlign: 'center', padding: '100px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
            <Activity size={48} color="var(--text-secondary)" opacity={0.5} />
            <h3 style={{ fontSize: '1.5rem' }}>No Diagnostic Captures Yet</h3>
            <p style={{ color: 'var(--text-secondary)' }}>Start a diagnostic module and hold the correct posture for 4.5 seconds to capture your first clinical report.</p>
            <button className="btn btn-primary" onClick={() => navigate('/')} style={{ marginTop: '16px', padding: '12px 24px' }}>
              Go to Dashboard
            </button>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '30px' }}>
            {images.map((imgUrl, i) => (
              <div key={i} className="module-card-light" style={{ padding: '0', overflow: 'hidden', cursor: 'default' }}>
                <div style={{ width: '100%', height: '260px', overflow: 'hidden', background: '#000', position: 'relative' }}>
                  <img src={imgUrl} alt={`Capture ${i}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  <div style={{ position: 'absolute', top: '12px', right: '12px', background: 'rgba(16, 185, 129, 0.9)', color: '#000', padding: '4px 12px', borderRadius: '20px', fontWeight: 'bold', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Activity size={14} /> AI Verified
                  </div>
                </div>
                <div style={{ padding: '24px' }}>
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>Diagnostic Report #{images.length - i}</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '20px' }}>
                    Automatically captured clinical alignment snapshot.
                  </p>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    <a href={imgUrl} target="_blank" rel="noreferrer" className="btn btn-primary" style={{ flex: 1, textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', padding: '10px', textDecoration: 'none' }}>
                      <Download size={16} /> Download
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Portfolio;
