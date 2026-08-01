import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Target, Hexagon, Crosshair, Shield, Maximize, ArrowRight, Zap, Eye, BarChart, Check, Bone, FileText, X, Info, Play, ChevronLeft, ChevronRight } from 'lucide-react';
import { 
  CHEST_MODULES, 
  HAND_MODULES, 
  SPINE_MODULES, 
  ELBOW_MODULES, 
  KNEE_MODULES, 
  FOOT_MODULES,
  LOWERBACK_MODULES,
  ARM_MODULES
} from '../config/modules';

const DashboardHome: React.FC = () => {
  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();

  // Walkthrough slide state
  const [isWalkthroughOpen, setIsWalkthroughOpen] = useState(false);
  const [walkthroughStep, setWalkthroughStep] = useState(1);

  // Walkthrough dynamic selection state
  const [selectedCategory, setSelectedCategory] = useState<'chest' | 'spine' | 'hand' | 'knee' | 'elbow' | 'foot' | 'lowerback'>('chest');
  const [selectedModule, setSelectedModule] = useState<any>(CHEST_MODULES[0]);
  const [isPostureCorrected, setIsPostureCorrected] = useState(false);

  const categoriesMap: Record<string, { name: string; modules: any[]; icon: string }> = {
    chest: { name: 'Chest', modules: CHEST_MODULES, icon: '🫁' },
    spine: { name: 'Spine', modules: SPINE_MODULES, icon: '🦴' },
    hand: { name: 'Hand', modules: HAND_MODULES, icon: '🖐️' },
    elbow: { name: 'Elbow', modules: ELBOW_MODULES, icon: '💪' },
    knee: { name: 'Knee', modules: KNEE_MODULES, icon: '🦵' },
    foot: { name: 'Foot', modules: FOOT_MODULES, icon: '👣' },
    lowerback: { name: 'Lower Back', modules: LOWERBACK_MODULES, icon: '🧘' },
    arm: { name: 'Arm', modules: ARM_MODULES, icon: '💪' }
  };

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isWalkthroughOpen) return;
      if (e.key === 'ArrowRight') {
        setWalkthroughStep(prev => Math.min(4, prev + 1));
      } else if (e.key === 'ArrowLeft') {
        setWalkthroughStep(prev => Math.max(1, prev - 1));
      } else if (e.key === 'Escape') {
        setIsWalkthroughOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isWalkthroughOpen]);




  // Generate randomized floating background icons
  const floatIcons = Array.from({ length: 15 }).map((_, i) => {
    const isBone = i % 2 === 0;
    const size = Math.random() * 24 + 16;
    const left = `${Math.random() * 100}%`;
    const duration = `${Math.random() * 15 + 15}s`;
    const delay = `${Math.random() * -30}s`; // negative delay so they spawn already on screen
    return (
      <div 
        key={i} 
        className="bg-float-icon" 
        style={{ left, width: size, height: size, animationDuration: duration, animationDelay: delay }}
      >
        {isBone ? <Bone size={size} /> : <FileText size={size} />}
      </div>
    );
  });

  return (
    <div className="landing-page-wrapper">
      <section className="hero-section-light">
        <div className="bg-floating-icons-container">
          {floatIcons}
        </div>
        <div className="hero-bg-glow"></div>
        <div className="hero-container-light">
          
          <div className="hero-text-content">
            <div className="hero-label">PhysioMaster OS v2.0</div>
            <h1 className="hero-heading">
              Clinical Intelligence<br/>for the Modern Practice
            </h1>
            <p className="hero-subtext">
              Transform your diagnostic workflow with real-time, AI-driven computer vision. Accurately measure joint alignment and track structural health in milliseconds.
            </p>
            <div style={{ display: 'flex', gap: '16px' }}>
              <button 
                className="btn btn-primary"
                onClick={() => {
                  if (isLoggedIn) {
                    document.getElementById('modules-section-light')?.scrollIntoView({ behavior: 'smooth' });
                  } else {
                    navigate('/login');
                  }
                }}
                style={{ padding: '16px 32px', fontSize: '1rem' }}
              >
                {isLoggedIn ? 'START DIAGNOSTICS' : 'STUDENT LOGIN'}
              </button>
              <button 
                className="btn-outline-light btn-play-pulse" 
                onClick={() => { setIsWalkthroughOpen(true); setWalkthroughStep(1); }}
                style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
              >
                <Play size={16} fill="var(--text-primary)" /> Watch Walkthrough
              </button>
            </div>
          </div>

          <div className="hero-image-content" style={{ position: 'relative' }}>
            {/* Floating Badges */}
            <div className="floating-badge" style={{ top: '20px', left: '-40px' }}>
              <div className="badge-icon"><Check size={16} strokeWidth={3} /></div>
              <div className="badge-content">
                <h4>99.9% Accuracy</h4>
                <p>MediaPipe Core</p>
              </div>
            </div>
            
            <div className="floating-badge" style={{ bottom: '120px', right: '-20px' }}>
              <div className="badge-icon" style={{ background: 'var(--accent-glow)', color: 'var(--accent-color)' }}><Zap size={16} /></div>
              <div className="badge-content">
                <h4><span style={{fontFamily: 'monospace'}}>32ms</span> Latency</h4>
                <p>Real-time tracking</p>
              </div>
            </div>

            <div className="hero-circle-mask" style={{ width: '500px', height: '500px' }}>
              <div className="mock-room-bg" style={{ background: 'var(--bg-card)' }}>
                <div className="mock-laptop" style={{ marginTop: '80px', width: '360px', height: '220px' }}>
                  <div className="mock-screen" style={{ width: '344px', height: '200px', background: 'var(--bg-primary)' }}>
                    <div className="mock-ui-header" style={{ background: 'var(--bg-card)', color: 'var(--text-primary)' }}>
                      PhysioMaster Live Feed
                    </div>
                    <div className="mock-ui-body">
                      <div className="mock-wireframe-img" style={{ border: '1px solid var(--border-color)' }}>
                        {/* Wireframe skeleton UI inside laptop */}
                        <div style={{ position: 'absolute', inset: 12, border: '1px dashed var(--accent-color)', borderRadius: 8, opacity: 0.5, display: 'flex', alignItems: 'center', justify: 'center' }}>
                          <Crosshair className="text-accent" style={{ opacity: 0.5 }} size={48} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* About Section */}
      <section className="marketing-section" id="about">
        <div className="marketing-container about-grid">
          <div className="about-image-placeholder">
            <div className="about-glow"></div>
            <Activity size={120} color="var(--accent-color)" strokeWidth={0.5} style={{ opacity: 0.5, position: 'relative', zIndex: 1 }} />
            {/* Overlay grid lines for aesthetic */}
            <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(var(--border-color) 1px, transparent 1px), linear-gradient(90deg, var(--border-color) 1px, transparent 1px)', backgroundSize: '40px 40px', opacity: 0.2 }}></div>
          </div>
          <div>
            <span className="section-tag">About PhysioMaster</span>
            <h2 className="section-title">Redefining Structural Assessments</h2>
            <p className="section-subtitle">
              PhysioMaster is an advanced clinical platform that replaces outdated manual goniometers with state-of-the-art computer vision. 
              By utilizing your device's camera, we map 33 3D body landmarks instantly.
            </p>
            <ul className="about-list">
              <li>
                <div className="about-list-icon"><Check size={16} strokeWidth={3} /></div>
                <div>
                  <div className="about-list-title">Zero Hardware Required</div>
                  <div className="about-list-text">Runs entirely in the browser using standard webcams. No costly depth sensors or wearables needed.</div>
                </div>
              </li>
              <li>
                <div className="about-list-icon"><Check size={16} strokeWidth={3} /></div>
                <div>
                  <div className="about-list-title">Enterprise Security</div>
                  <div className="about-list-text">All video processing happens locally on the client-side. No patient video feeds are stored or transmitted.</div>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="marketing-section" style={{ background: 'var(--bg-card)' }}>
        <div className="marketing-container" style={{ textAlign: 'center' }}>
          <span className="section-tag">How It Works</span>
          <h2 className="section-title" style={{ margin: '0 auto', maxWidth: '800px' }}>From Setup to Clinical Insight in Seconds</h2>
          
          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-icon-wrapper">
                <Target size={24} />
              </div>
              <h3 className="feature-title">1. Select Target Region</h3>
              <p className="feature-desc">Choose from our specialized diagnostic modules tailored for specific joints including knee, spine, and chest.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon-wrapper">
                <Eye size={24} />
              </div>
              <h3 className="feature-title">2. Real-Time Tracking</h3>
              <p className="feature-desc">The AI establishes a holographic tracking overlay on the patient, instantly mapping structural landmarks.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon-wrapper">
                <BarChart size={24} />
              </div>
              <h3 className="feature-title">3. Actionable Metrics</h3>
              <p className="feature-desc">Receive live joint angles, alignment deviations, and confidence scores directly on your clinical dashboard.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Modules Grid (Hidden for guests) */}
      {isLoggedIn && (
        <section id="modules-section-light" className="marketing-section">
          <div className="marketing-container">
            <div style={{ textAlign: 'center', marginBottom: '64px' }}>
              <span className="section-tag">Diagnostics</span>
              <h2 className="section-title">Select a Diagnostic Module</h2>
            </div>

            <div className="grid-cols-3-light">
              {[
                { name: 'Chest', count: CHEST_MODULES.length, id: 'chest', icon: <Activity size={24} strokeWidth={1.5} /> },
                { name: 'Hand', count: HAND_MODULES.length, id: 'hand', icon: <Target size={24} strokeWidth={1.5} /> },
                { name: 'Spine', count: SPINE_MODULES.length, id: 'spine', icon: <Hexagon size={24} strokeWidth={1.5} /> },
                { name: 'Elbow', count: ELBOW_MODULES.length, id: 'elbow', icon: <Crosshair size={24} strokeWidth={1.5} /> },
                { name: 'Knee', count: KNEE_MODULES.length, id: 'knee', icon: <Shield size={24} strokeWidth={1.5} /> },
                { name: 'Foot', count: FOOT_MODULES.length, id: 'foot', icon: <Maximize size={24} strokeWidth={1.5} /> },
                { name: 'Lower Back', count: LOWERBACK_MODULES.length, id: 'lowerback', icon: <Activity size={24} strokeWidth={1.5} /> },
                { name: 'Arm', count: ARM_MODULES.length, id: 'arm', icon: <Target size={24} strokeWidth={1.5} /> },
              ].map(cat => (
                <div 
                  key={cat.id} 
                  className="module-card-light" 
                  onClick={() => navigate(`/category/${cat.id}`)} 
                >
                  <div className="module-card-icon">{cat.icon}</div>
                  <h3 style={{fontFamily: 'Inter', fontWeight: 600, fontSize: '1.2rem'}}>{cat.name} Region</h3>
                  <p>AI-assisted structural analysis and joint alignment diagnostics for the {cat.name.toLowerCase()}.</p>
                  <div className="module-card-action" style={{marginTop: 'auto'}}>
                    Initialize <ArrowRight size={14} style={{ marginLeft: '4px' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Interactive App Walkthrough Modal */}
      {isWalkthroughOpen && (
        <div className="walkthrough-overlay" onClick={() => setIsWalkthroughOpen(false)}>
          <div className="walkthrough-modal" onClick={(e) => e.stopPropagation()}>
            <div className="walkthrough-header">
              <div>
                <h2 style={{ fontSize: '1.4rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                  <Play size={18} fill="var(--accent-color)" color="var(--accent-color)" />
                  How PhysioMaster Works: Guided Tour
                </h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px', margin: 0 }}>
                  Learn how our geometric rules engine simplifies diagnostic posture calibration.
                </p>
              </div>
              <button 
                onClick={() => setIsWalkthroughOpen(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <X size={24} />
              </button>
            </div>

            <div className="walkthrough-body">
              {/* Left Column: Premium Visual Mockups */}
              <div className="walkthrough-visual-container">
                {walkthroughStep === 1 && (
                  <div className="mock-window-frame">
                    <div className="mock-window-header">
                      <div className="mock-window-dot" style={{ backgroundColor: '#ef4444' }}></div>
                      <div className="mock-window-dot" style={{ backgroundColor: '#fbbf24' }}></div>
                      <div className="mock-window-dot" style={{ backgroundColor: '#10b981' }}></div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace', marginLeft: 'auto' }}>physiomaster.io/dashboard</span>
                    </div>
                    <div className="mock-window-body" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '12px', justifyContent: 'flex-start', width: '100%' }}>
                      {/* Horizontal Category Tabs */}
                      <div className="mock-category-tabs">
                        {Object.entries(categoriesMap).map(([key, cat]) => (
                          <button 
                            key={key} 
                            className={`mock-tab-btn ${selectedCategory === key ? 'active' : ''}`}
                            onClick={() => {
                              setSelectedCategory(key as any);
                              setSelectedModule(cat.modules[0]);
                              setIsPostureCorrected(false);
                            }}
                          >
                            <span>{cat.icon}</span>
                            <span>{cat.name}</span>
                          </button>
                        ))}
                      </div>
                      
                      {/* Grid of specific modules */}
                      <div className="mock-module-grid">
                        {categoriesMap[selectedCategory].modules.map((mod: any) => (
                          <div 
                            key={mod.folder}
                            className={`mock-module-card ${selectedModule.folder === mod.folder ? 'active' : ''}`}
                            onClick={() => {
                              setSelectedModule(mod);
                              setIsPostureCorrected(false);
                            }}
                          >
                            <span style={{ fontSize: '1.2rem' }}>{mod.icon}</span>
                            <div style={{ textAlign: 'left' }}>
                              <div className="mock-module-card-title">{mod.name}</div>
                              <div className="mock-module-card-subtitle">{mod.analysisMetrics?.targetAngle || '0°'}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {walkthroughStep === 2 && (
                  <div className="mock-window-frame">
                    <div className="mock-window-header">
                      <div className="mock-window-dot"></div>
                      <div className="mock-window-dot"></div>
                      <div className="mock-window-dot"></div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace', marginLeft: 'auto' }}>AI Live Feed ({categoriesMap[selectedCategory].name})</span>
                    </div>
                    <div className="mock-window-body" style={{ padding: '8px', width: '100%' }}>
                      <div className="mock-feed-box" style={{ flexDirection: 'column' }}>
                        <div className="mock-laser-sweep"></div>
                        
                        {/* Dynamic category-based SVG skeleton */}
                        {selectedCategory === 'chest' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <line x1="60" y1="10" x2="60" y2="110" stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
                            <line x1="10" y1="60" x2="110" y2="60" stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
                            <g style={{ 
                              transform: `rotate(${isPostureCorrected ? 0 : 8}deg)`, 
                              transformOrigin: '60px 60px', 
                              transition: 'transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)' 
                            }}>
                              <circle cx="60" cy="30" r="10" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <line x1="60" y1="40" x2="60" y2="50" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <line x1="30" y1="50" x2="90" y2="50" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2.5" />
                              <circle cx="30" cy="50" r="3" fill="white" />
                              <circle cx="90" cy="50" r="3" fill="white" />
                              <line x1="60" y1="50" x2="60" y2="90" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <line x1="40" y1="90" x2="80" y2="90" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                            </g>
                          </svg>
                        )}

                        {selectedCategory === 'spine' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <line x1="60" y1="10" x2="60" y2="110" stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
                            <circle cx={isPostureCorrected ? 60 : 70} cy="25" r="9" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" style={{ transition: 'all 0.8s ease' }} />
                            <path d={isPostureCorrected ? "M 60,34 Q 57,50 60,65 Q 63,80 60,95" : "M 70,34 Q 50,50 63,65 Q 73,80 60,95"} fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2.5" style={{ transition: 'all 0.8s ease' }} />
                            <circle cx="60" cy="95" r="4" fill="white" />
                          </svg>
                        )}

                        {selectedCategory === 'hand' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <rect x="25" y="25" width="70" height="70" rx="8" fill="none" stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" />
                            <g style={{ 
                              transform: `rotate(${isPostureCorrected ? 0 : -15}deg) scale(${isPostureCorrected ? 1 : 0.85})`,
                              transformOrigin: '60px 85px',
                              transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
                            }}>
                              <line x1="50" y1="90" x2="70" y2="90" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="3" />
                              <path d="M 50,90 L 45,65 L 75,65 L 70,90 Z" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <path d="M 45,75 L 30,65 L 25,58" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <path d="M 48,65 L 45,40 L 43,30" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <path d="M 57,65 L 57,35 L 57,23" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <path d="M 66,65 L 68,38 L 69,27" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <path d="M 74,70 L 79,48 L 81,38" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                            </g>
                          </svg>
                        )}

                        {selectedCategory === 'elbow' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <path d={isPostureCorrected ? "M 60,60 A 30,30 0 0,1 60,90" : "M 40,75 A 25,25 0 0,1 60,90"} fill="none" stroke="rgba(56, 189, 248, 0.4)" strokeWidth="8" style={{ transition: 'all 0.8s ease' }} />
                            <circle cx="60" cy="25" r="4" fill="white" />
                            <line x1="60" y1="25" x2="60" y2="60" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="3" />
                            <circle cx="60" cy="60" r="5" fill="white" stroke="var(--accent-color)" strokeWidth="1.5" />
                            <line x1="60" y1="60" x2={isPostureCorrected ? 60 : 25} y2={isPostureCorrected ? 95 : 75} stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="3" style={{ transition: 'all 0.8s ease' }} />
                            <circle cx={isPostureCorrected ? 60 : 25} cy={isPostureCorrected ? 95 : 75} r="4" fill="white" style={{ transition: 'all 0.8s ease' }} />
                          </svg>
                        )}

                        {selectedCategory === 'knee' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <path d={isPostureCorrected ? "M 60,60 A 30,30 0 0,1 60,95" : "M 45,75 A 25,25 0 0,1 60,95"} fill="none" stroke="rgba(56, 189, 248, 0.4)" strokeWidth="8" style={{ transition: 'all 0.8s ease' }} />
                            <circle cx="60" cy="25" r="4" fill="white" />
                            <line x1="60" y1="25" x2="60" y2="60" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="3" />
                            <circle cx="60" cy="60" r="5" fill="white" stroke="var(--accent-color)" strokeWidth="1.5" />
                            <line x1="60" y1="60" x2={isPostureCorrected ? 60 : 35} y2={isPostureCorrected ? 95 : 85} stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="3" style={{ transition: 'all 0.8s ease' }} />
                            <circle cx={isPostureCorrected ? 60 : 35} cy={isPostureCorrected ? 95 : 85} r="4" fill="white" style={{ transition: 'all 0.8s ease' }} />
                          </svg>
                        )}

                        {selectedCategory === 'foot' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <g style={{ 
                              transform: `rotate(${isPostureCorrected ? 0 : 10}deg)`,
                              transformOrigin: '60px 60px',
                              transition: 'all 0.8s ease'
                            }}>
                              <path d="M 50,20 Q 30,45 40,80 Q 42,95 60,95 Q 78,95 80,80 Q 90,45 70,20 Z" fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2" />
                              <circle cx="60" cy="80" r="10" fill={isPostureCorrected ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)'} stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="1" />
                              <ellipse cx="60" cy="35" rx="15" ry="8" fill={isPostureCorrected ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)'} stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="1" />
                            </g>
                          </svg>
                        )}

                        {selectedCategory === 'lowerback' && (
                          <svg viewBox="0 0 120 120" style={{ width: '100px', height: '100px', opacity: 0.85 }}>
                            <line x1="60" y1="10" x2="60" y2="110" stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
                            <circle cx="60" cy="30" r="3" fill="white" />
                            <path d={isPostureCorrected ? "M 60,30 Q 52,55 60,80 L 60,95" : "M 60,30 Q 40,55 60,80 L 60,95"} fill="none" stroke={isPostureCorrected ? 'var(--success-color)' : 'var(--danger-color)'} strokeWidth="2.5" style={{ transition: 'all 0.8s ease' }} />
                            <circle cx="60" cy="95" r="4" fill="white" />
                          </svg>
                        )}

                        <div className={`mock-status-banner ${isPostureCorrected ? 'good' : 'bad'}`} style={{ transition: 'all 0.3s ease' }}>
                          {!isPostureCorrected ? (
                            selectedCategory === 'chest' ? '⚠️ TILTED: LEVEL YOUR SHOULDERS' :
                            selectedCategory === 'spine' ? '⚠️ FORWARD HEAD: RETRACT BACK' :
                            selectedCategory === 'hand' ? '⚠️ ROTATED: ALIGN HAND WITH FRAME' :
                            selectedCategory === 'elbow' ? '⚠️ ANGLE ERROR: EXTEND ELBOW TO 180°' :
                            selectedCategory === 'knee' ? '⚠️ KNEE BENT: STRAIGHTEN LEG' :
                            selectedCategory === 'foot' ? '⚠️ WEIGHT SHIFTED: CENTER FEET' :
                            '⚠️ HYPER-LORDOSIS: FLATTEN LOWER BACK'
                          ) : (
                            '✅ GOOD ALIGNMENT: HOLD STILL'
                          )}
                        </div>
                      </div>
                      
                      {/* Interactive Fix Button */}
                      <button 
                        className={`btn ${isPostureCorrected ? 'btn-outline' : 'btn-primary'}`}
                        onClick={() => setIsPostureCorrected(prev => !prev)}
                        style={{ marginTop: '12px', fontSize: '0.8rem', padding: '8px 16px', width: '100%' }}
                      >
                        {isPostureCorrected ? "🔄 Reset Posture to Tilted/Incorrect" : "💡 Click to Adjust Patient Posture"}
                      </button>
                    </div>
                  </div>
                )}

                {walkthroughStep === 3 && (
                  <div className="mock-window-frame">
                    <div className="mock-window-header">
                      <div className="mock-window-dot"></div>
                      <div className="mock-window-dot"></div>
                      <div className="mock-window-dot"></div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace', marginLeft: 'auto' }}>AI Live Feed ({categoriesMap[selectedCategory].name})</span>
                    </div>
                    <div className="mock-window-body" style={{ width: '100%' }}>
                      <div className="mock-countdown-circle">
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.7rem', color: 'var(--success-color)', letterSpacing: '1px', fontWeight: 'bold' }}>STABLE</span>
                          <span style={{ fontSize: '1.4rem', fontWeight: 'bold', color: 'white', fontFamily: 'monospace', margin: '4px 0' }}>2.0s</span>
                          <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>HOLD STILL</span>
                        </div>
                      </div>
                      
                      {/* Mini green aligned SVG skeleton at the bottom-right corner of the stream */}
                      <div style={{ position: 'absolute', bottom: '12px', right: '12px', width: '50px', height: '50px', border: '1px solid rgba(16, 185, 129, 0.3)', background: 'rgba(16, 185, 129, 0.05)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {selectedCategory === 'chest' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <circle cx="60" cy="30" r="10" fill="none" stroke="var(--success-color)" strokeWidth="3.5" />
                            <line x1="60" y1="40" x2="60" y2="50" stroke="var(--success-color)" strokeWidth="3.5" />
                            <line x1="30" y1="50" x2="90" y2="50" stroke="var(--success-color)" strokeWidth="4" />
                            <line x1="60" y1="50" x2="60" y2="90" stroke="var(--success-color)" strokeWidth="3.5" />
                          </svg>
                        )}
                        {selectedCategory === 'spine' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <circle cx="60" cy="25" r="9" fill="none" stroke="var(--success-color)" strokeWidth="3.5" />
                            <path d="M 60,34 Q 57,50 60,65 Q 63,80 60,95" fill="none" stroke="var(--success-color)" strokeWidth="4" />
                          </svg>
                        )}
                        {selectedCategory === 'hand' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <line x1="50" y1="90" x2="70" y2="90" stroke="var(--success-color)" strokeWidth="4" />
                            <path d="M 50,90 L 45,65 L 75,65 L 70,90 Z" fill="none" stroke="var(--success-color)" strokeWidth="3" />
                          </svg>
                        )}
                        {selectedCategory === 'elbow' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <line x1="60" y1="25" x2="60" y2="60" stroke="var(--success-color)" strokeWidth="4" />
                            <line x1="60" y1="60" x2="60" y2="95" stroke="var(--success-color)" strokeWidth="4" />
                          </svg>
                        )}
                        {selectedCategory === 'knee' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <line x1="60" y1="25" x2="60" y2="60" stroke="var(--success-color)" strokeWidth="4" />
                            <line x1="60" y1="60" x2="60" y2="95" stroke="var(--success-color)" strokeWidth="4" />
                          </svg>
                        )}
                        {selectedCategory === 'foot' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <path d="M 50,20 Q 30,45 40,80 Q 42,95 60,95 Q 78,95 80,80 Q 90,45 70,20 Z" fill="none" stroke="var(--success-color)" strokeWidth="3" />
                          </svg>
                        )}
                        {selectedCategory === 'lowerback' && (
                          <svg viewBox="0 0 120 120" style={{ width: '40px', height: '40px' }}>
                            <path d="M 60,30 Q 52,55 60,80 L 60,95" fill="none" stroke="var(--success-color)" strokeWidth="4" />
                          </svg>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {walkthroughStep === 4 && (
                  <div className="mock-window-frame">
                    <div className="mock-window-header">
                      <div className="mock-window-dot"></div>
                      <div className="mock-window-dot"></div>
                      <div className="mock-window-dot"></div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace', marginLeft: 'auto' }}>Report Engine v2.0</span>
                    </div>
                    <div className="mock-window-body" style={{ padding: '8px', width: '100%' }}>
                      <div className="mock-report-sheet">
                        <div className="mock-report-header">
                          DIAGNOSTIC REPORT // ID #{Math.floor(1000 + Math.random() * 9000)}
                        </div>
                        <div className="mock-report-row">
                          <span style={{ color: 'var(--text-secondary)' }}>Region Checked</span>
                          <span style={{ color: 'white', fontWeight: 'bold' }}>{selectedCategory.toUpperCase()}</span>
                        </div>
                        <div className="mock-report-row">
                          <span style={{ color: 'var(--text-secondary)' }}>Protocol View</span>
                          <span style={{ color: 'var(--accent-color)', fontWeight: 'bold' }}>{selectedModule.name}</span>
                        </div>
                        <div className="mock-report-row">
                          <span style={{ color: 'var(--text-secondary)' }}>Target Angle</span>
                          <span style={{ color: 'white' }}>{selectedModule.analysisMetrics?.targetAngle || '0°'}</span>
                        </div>
                        <div className="mock-report-row">
                          <span style={{ color: 'var(--text-secondary)' }}>Keypoints Logged</span>
                          <span style={{ color: 'var(--success-color)' }}>{selectedModule.analysisMetrics?.keyPoints || 10} Joints</span>
                        </div>
                        <div className="mock-report-row" style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '6px', marginTop: '4px' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Accuracy Score</span>
                          <span style={{ color: 'var(--success-color)', fontWeight: 'bold' }}>98.9%</span>
                        </div>
                        <div style={{ marginTop: '12px', background: 'rgba(16, 185, 129, 0.1)', padding: '6px', borderRadius: '6px', fontSize: '0.65rem', color: 'var(--success-color)', textAlign: 'center', fontWeight: 'bold' }}>
                          SECURE LOCAL DB TRANSACTION SUCCESSFUL
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Step Explanations */}
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                {walkthroughStep === 1 && (
                  <div style={{ maxHeight: '380px', overflowY: 'auto', paddingRight: '8px' }}>
                    <span className="walkthrough-step-badge">Step 1: Choose Pose</span>
                    <h3 className="walkthrough-title" style={{ fontSize: '1.6rem', marginBottom: '8px' }}>{selectedModule.name} Protocol</h3>
                    <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--accent-color)', fontWeight: 'bold', marginBottom: '12px' }}>
                      Region: {selectedCategory.toUpperCase()} View
                    </div>
                    <p className="walkthrough-desc" style={{ fontSize: '0.9rem', marginBottom: '16px' }}>
                      {selectedModule.description}
                    </p>
                    
                    <h4 style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Instructions for Patient:</h4>
                    <div className="walkthrough-features-list" style={{ marginBottom: '16px' }}>
                      {selectedModule.instructions?.map((inst: string, i: number) => (
                        <div key={i} className="walkthrough-feature-item" style={{ fontSize: '0.85rem' }}>
                          <span className="walkthrough-feature-icon">✓</span>
                          <span>{inst}</span>
                        </div>
                      ))}
                    </div>

                    {selectedModule.benefits && (
                      <div style={{ background: 'rgba(56, 189, 248, 0.05)', border: '1px solid rgba(56, 189, 248, 0.1)', padding: '10px 14px', borderRadius: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        <span style={{ fontWeight: 'bold', color: 'var(--accent-color)' }}>Clinical Benefit: </span>
                        {selectedModule.benefits}
                      </div>
                    )}
                  </div>
                )}

                {walkthroughStep === 2 && (
                  <div>
                    <span className="walkthrough-step-badge">Step 2: Calibrate</span>
                    <h3 className="walkthrough-title">Real-Time Guidance</h3>
                    <p className="walkthrough-desc">
                      The AI goniometer works by establishing key dynamic landmarks on the patient's body joints. It evaluates posture using real-time geometric rule parameters.
                    </p>
                    <div className="walkthrough-features-list" style={{ gap: '16px' }}>
                      <div className="walkthrough-feature-item">
                        <span className="walkthrough-feature-icon">✓</span>
                        <div>
                          <span style={{ fontWeight: '600', color: 'white' }}>Feedback Loop:</span>
                          <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.85rem', marginTop: '2px' }}>
                            If a joint line or angle is tilted (e.g. shoulders out of level, bent knees), the dashboard overlays red alerts and guideline advice.
                          </span>
                        </div>
                      </div>
                      <div className="walkthrough-feature-item">
                        <span className="walkthrough-feature-icon">✓</span>
                        <div>
                          <span style={{ fontWeight: '600', color: 'white' }}>Target Angle Verification:</span>
                          <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.85rem', marginTop: '2px' }}>
                            Currently evaluating against target alignment of <strong style={{ color: 'var(--accent-color)' }}>{selectedModule.analysisMetrics?.targetAngle || '0°'}</strong> for this module.
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {walkthroughStep === 3 && (
                  <div>
                    <span className="walkthrough-step-badge">Step 3: Auto-Capture</span>
                    <h3 className="walkthrough-title">Hands-Free Automation</h3>
                    <p className="walkthrough-desc">
                      PhysioMaster eliminates the need for manual shutter button clicks or patient interruptions by implementing our <strong>Continuous Stability Metric</strong>.
                    </p>
                    <div className="walkthrough-features-list" style={{ gap: '16px' }}>
                      <div className="walkthrough-feature-item">
                        <span className="walkthrough-feature-icon">✓</span>
                        <div>
                          <span style={{ fontWeight: '600', color: 'white' }}>Multi-Frame Stability Check:</span>
                          <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.85rem', marginTop: '2px' }}>
                            The coordinate filter tracks frame-to-frame joint variance. Once alignment is held stable for a threshold window, a capture is triggered.
                          </span>
                        </div>
                      </div>
                      <div className="walkthrough-feature-item">
                        <span className="walkthrough-feature-icon">✓</span>
                        <div>
                          <span style={{ fontWeight: '600', color: 'white' }}>Reduced Distortion:</span>
                          <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.85rem', marginTop: '2px' }}>
                            Capturing only at zero-variance keeps diagnostic pictures perfectly sharp, preventing patient breathing or micro-movements from causing errors.
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {walkthroughStep === 4 && (
                  <div>
                    <span className="walkthrough-step-badge">Step 4: Report</span>
                    <h3 className="walkthrough-title">Clinical Telemetry</h3>
                    <p className="walkthrough-desc">
                      All computed angular deviations, coordinate records, and posture scores are logged instantly to generate a structured diagnostic report.
                    </p>
                    <div className="walkthrough-features-list" style={{ gap: '16px' }}>
                      <div className="walkthrough-feature-item">
                        <span className="walkthrough-feature-icon">✓</span>
                        <div>
                          <span style={{ fontWeight: '600', color: 'white' }}>HIPAA Compliance by Design:</span>
                          <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.85rem', marginTop: '2px' }}>
                            Patient diagnostic feeds are computed entirely client-side. No raw video feed is uploaded, keeping health data strictly private.
                          </span>
                        </div>
                      </div>
                      <div className="walkthrough-feature-item">
                        <span className="walkthrough-feature-icon">✓</span>
                        <div>
                          <span style={{ fontWeight: '600', color: 'white' }}>Digital EMR Integration:</span>
                          <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.85rem', marginTop: '2px' }}>
                            Report outputs can be printed or copied directly to medical EMR sheets for structured diagnostic records.
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Walkthrough Footer Controls */}
            <div className="walkthrough-footer">
              <div className="walkthrough-progress">
                {[1, 2, 3, 4].map(idx => (
                  <div 
                    key={idx} 
                    className={`walkthrough-dot ${walkthroughStep === idx ? 'active' : ''}`}
                    onClick={() => setWalkthroughStep(idx)}
                  />
                ))}
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                {walkthroughStep > 1 && (
                  <button 
                    className="btn btn-outline" 
                    onClick={() => setWalkthroughStep(prev => prev - 1)}
                    style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    <ChevronLeft size={16} /> Back
                  </button>
                )}

                {walkthroughStep < 4 ? (
                  <button 
                    className="btn btn-primary" 
                    onClick={() => setWalkthroughStep(prev => prev + 1)}
                    style={{ padding: '8px 20px', display: 'flex', alignItems: 'center', gap: '6px' }}
                  >
                    Next <ChevronRight size={16} />
                  </button>
                ) : (
                  <button 
                    className="btn btn-primary" 
                    onClick={() => setIsWalkthroughOpen(false)}
                    style={{ padding: '8px 20px' }}
                  >
                    Done
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};


export default DashboardHome;
