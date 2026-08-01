import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Camera, Activity } from 'lucide-react';
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
import type { ModuleInfo } from '../config/modules';
import HologramViewer from '../components/HologramViewer';

const PoseDetailView: React.FC = () => {
  const { category, poseId } = useParams<{ category: string, poseId: string }>();
  const navigate = useNavigate();
  const [isScanning, setIsScanning] = useState(false);
  const [camIndex, setCamIndex] = useState<number>(2);
  const [loading, setLoading] = useState(false);
  const [moduleInfo, setModuleInfo] = useState<ModuleInfo | null>(null);
  const [isStreamLoading, setIsStreamLoading] = useState(true);
  const [streamId, setStreamId] = useState<number>(Date.now());
  const [targetLeg, setTargetLeg] = useState<string>('');

  // Demo Mode Simulation State
  const [simAccuracy, setSimAccuracy] = useState<number>(0);
  const [simMessage, setSimMessage] = useState<string>('');
  const [simStatus, setSimStatus] = useState<'calibrating' | 'good' | 'bad'>('calibrating');
  
  // Autonomous Clinical Reporting State
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [reportAccuracy, setReportAccuracy] = useState<number>(0);

  // Component-level speech synthesis helper with GC and timing workarounds
  const speakTimeoutRef = useRef<any>(null);

  const speak = (text: string) => {
    // Clear any pending speak timeout to avoid duplicate speak attempts
    if (speakTimeoutRef.current) {
      clearTimeout(speakTimeoutRef.current);
      speakTimeoutRef.current = null;
    }

    // Cancel currently playing speech
    window.speechSynthesis.cancel();

    // Store utterances globally to prevent garbage collection in Chrome
    (window as any)._activeUtterances = (window as any)._activeUtterances || [];

    speakTimeoutRef.current = setTimeout(() => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95; // Slightly faster to sound natural and finish sooner

      // Keep a strong reference to the utterance to prevent garbage collection
      (window as any)._activeUtterances.push(utterance);
      utterance.onend = () => {
        (window as any)._activeUtterances = ((window as any)._activeUtterances || []).filter((u: any) => u !== utterance);
      };
      utterance.onerror = () => {
        (window as any)._activeUtterances = ((window as any)._activeUtterances || []).filter((u: any) => u !== utterance);
      };

      window.speechSynthesis.speak(utterance);
    }, 150);
  };

  const takeSnapshot = () => {
    speak("Capturing diagnostic snapshot now.");
    fetch('http://127.0.0.1:8000/capture', { method: 'POST' }).catch((e) => {
      console.error("Failed to initiate capture:", e);
    });
  };

  const getBodyPartName = (cat?: string) => {
    switch (cat) {
      case 'hand': return 'hand';
      case 'elbow': return 'elbow';
      case 'knee': return 'knee';
      case 'foot': return 'foot';
      case 'spine': return 'spine';
      case 'chest': return 'chest';
      case 'lowerback': return 'lower back';
      case 'arm': return 'arm';
      default: return 'target body part';
    }
  };

  const handleCameraChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setIsStreamLoading(true);
    setCamIndex(parseInt(e.target.value));
    setStreamId(Date.now());
    setTimeout(() => setIsStreamLoading(false), 1500);
  };

  useEffect(() => {
    let modulesList: ModuleInfo[] = [];
    switch (category) {
      case 'chest': modulesList = CHEST_MODULES; break;
      case 'hand': modulesList = HAND_MODULES; break;
      case 'spine': modulesList = SPINE_MODULES; break;
      case 'elbow': modulesList = ELBOW_MODULES; break;
      case 'knee': modulesList = KNEE_MODULES; break;
      case 'foot': modulesList = FOOT_MODULES; break;
      case 'lowerback': modulesList = LOWERBACK_MODULES; break;
      case 'arm': modulesList = ARM_MODULES; break;
    }
    const found = modulesList.find(m => m.folder === `${category}/${poseId}` || m.folder === poseId);
    setModuleInfo(found || null);

    // Cleanup: Force backend to stop stream when leaving this page
    return () => {
      fetch('http://127.0.0.1:8000/stop_stream').catch(() => {});
      window.speechSynthesis.cancel();
    };
  }, [category, poseId]);

  // Demo Simulation Effect
  useEffect(() => {
    let isActive = true;
    let pollInterval: any = null;
    
    if (isScanning && !isStreamLoading) {
      // REAL TELEMETRY MODE FOR ALL MODULES
      let lastMessage = '';
      let lastSpokeTimestamp = 0;
      let lastReportUrl = '';
      
      const fetchTelemetry = async () => {
        if (!isActive) return;
        try {
          const res = await fetch(`http://127.0.0.1:8000/telemetry?t=${Date.now()}`, {
            cache: 'no-store'
          });
          if (!res.ok) return;
          const data = await res.json();
          
          setSimStatus(data.status);
          setSimMessage(data.message);
          setSimAccuracy(data.accuracy);
          
          const timeSinceLastSpoke = Date.now() - lastSpokeTimestamp;
          if (data.message && data.message !== "Analyzing...") {
            // Speak if the message has changed, or if it's the same warning but they've been stuck for > 7 seconds
            if (data.message !== lastMessage || timeSinceLastSpoke > 7000) {
              speak(data.message);
              lastMessage = data.message;
              lastSpokeTimestamp = Date.now();
            }
          }
          
          // Received capture
          if (data.last_capture_url && data.last_capture_url !== lastReportUrl) {
            lastReportUrl = data.last_capture_url;
            setReportUrl(data.last_capture_url);
            setReportAccuracy(data.accuracy);
            setIsScanning(false);
            fetch('http://127.0.0.1:8000/stop_stream').catch(() => {});
          }
        } catch (e) {
          console.error("Telemetry polling error:", e);
        }
      };
      
      pollInterval = setInterval(fetchTelemetry, 500);

    } else {
      setSimAccuracy(0);
      setSimMessage('');
      setSimStatus('calibrating');
      window.speechSynthesis.cancel();
    }
    
    return () => {
      isActive = false;
      if (pollInterval) clearInterval(pollInterval);
      if (speakTimeoutRef.current) {
        clearTimeout(speakTimeoutRef.current);
        speakTimeoutRef.current = null;
      }
      window.speechSynthesis.cancel();
    };
  }, [isScanning, isStreamLoading, category]);

  if (!moduleInfo) {
    return (
      <div style={{ padding: '20px' }}>
        <button className="btn" onClick={() => navigate(-1)} style={{ marginBottom: '20px' }}>
          <ArrowLeft size={16} /> Back
        </button>
        <div>Pose not found or category not supported yet.</div>
      </div>
    );
  }

  const startScanner = () => {
    setLoading(true);
    setTimeout(() => {
      setIsScanning(true);
      setIsStreamLoading(true);
      setStreamId(Date.now());
      // Fallback timeout because onLoad rarely fires for MJPEG streams
      setTimeout(() => setIsStreamLoading(false), 1500);
      setLoading(false);
    }, 500);
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <button 
        className="btn" 
        onClick={() => navigate(-1)} 
        style={{ marginBottom: '24px', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', display: 'flex', alignItems: 'center' }}
      >
        <ArrowLeft size={16} style={{ marginRight: '8px' }} /> Back to Category
      </button>

      <h1 style={{ fontSize: '2.5rem', marginBottom: '16px', display: 'flex', alignItems: 'center' }}>
        <span style={{ marginRight: '16px', fontSize: '3rem' }}>{moduleInfo.icon}</span> 
        {moduleInfo.name} Detection
      </h1>

      {isScanning ? (
        <div className="glass-panel" style={{ padding: '24px', position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ background: 'rgba(0,0,0,0.6)', padding: '6px 16px', borderRadius: '12px', fontSize: '0.9rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '8px', height: '8px', background: '#10b981', borderRadius: '50%', boxShadow: '0 0 8px #10b981' }}></div>
                LIVE AI SCANNER
              </span>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: '20px' }}>
                <Camera size={18} color="rgba(255,255,255,0.7)" />
                <select 
                  value={camIndex} 
                  onChange={handleCameraChange}
                  style={{
                    background: 'rgba(255,255,255,0.1)',
                    color: 'white',
                    border: '1px solid rgba(255,255,255,0.2)',
                    padding: '6px 12px',
                    borderRadius: '6px',
                    outline: 'none',
                    fontSize: '0.9rem'
                  }}
                >
                  <option value={0} style={{ color: 'black' }}>Laptop Camera</option>
                  <option value={2} style={{ color: 'black' }}>Droid Camera</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '12px' }}>
              <button 
                className="btn btn-primary" 
                style={{ 
                  width: 'auto', 
                  background: 'var(--accent-color, #10b981)', 
                  color: 'white', 
                  padding: '8px 20px', 
                  border: 'none', 
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(16, 185, 129, 0.2)'
                }}
                onClick={takeSnapshot}
              >
                <Camera size={16} />
                Take Snapshot
              </button>

              <button 
                className="btn" 
                style={{ width: 'auto', background: 'rgba(239, 68, 68, 0.8)', color: 'white', padding: '8px 16px', border: 'none', borderRadius: '8px' }}
                onClick={() => {
                  setIsScanning(false);
                  fetch('http://127.0.0.1:8000/stop_stream').catch(() => {});
                }}
              >
                Stop Scanning
              </button>
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '24px' }}>
            {/* Left Column: Camera Feed */}
            <div style={{ borderRadius: '12px', overflow: 'hidden', background: '#000', border: '1px solid rgba(255,255,255,0.1)', position: 'relative', minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {isStreamLoading && (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.8)', zIndex: 10 }}>
                  <Activity size={32} color="var(--accent-color)" style={{ marginBottom: '16px' }} />
                  <span style={{ color: 'white', letterSpacing: '1px', fontSize: '0.9rem' }}>INITIALIZING CAMERA...</span>
                </div>
              )}
              <img 
                key={`cam-${camIndex}-${streamId}-${targetLeg}`}
                src={`http://127.0.0.1:8000/video_feed?module=${category}/${poseId}&cam=${camIndex}&target=${targetLeg}&t=${streamId}`} 
                onLoad={() => setIsStreamLoading(false)}
                onError={() => setIsStreamLoading(false)}
                alt="Live Diagnostic Stream" 
                style={{ width: '100%', height: 'auto', maxHeight: '70vh', objectFit: 'contain', display: 'block', opacity: isStreamLoading ? 0 : 1, transition: 'opacity 0.3s' }} 
              />
            </div>

            {/* Right Column: Telemetry */}
            {isScanning && !isStreamLoading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                    <Activity size={16} color="var(--accent-color)" />
                    Live Telemetry
                  </h3>
                  
                  {/* Accuracy Meter */}
                  <div style={{ 
                      background: 'rgba(0,0,0,0.3)',
                      padding: '24px',
                      borderRadius: '12px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      border: `2px solid ${simStatus === 'good' ? '#10b981' : simStatus === 'bad' ? '#ef4444' : 'rgba(255,255,255,0.1)'}`,
                      boxShadow: simStatus === 'good' ? '0 0 20px rgba(16, 185, 129, 0.2)' : simStatus === 'bad' ? '0 0 20px rgba(239, 68, 68, 0.2)' : 'none',
                      transition: 'all 0.3s ease'
                    }}>
                      <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.8rem', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '8px' }}>ACCURACY</span>
                      <div style={{ fontSize: '3.5rem', fontWeight: '900', color: simStatus === 'good' ? '#10b981' : simStatus === 'bad' ? '#ef4444' : 'white', lineHeight: '1' }}>
                        {simAccuracy}%
                      </div>
                  </div>
                </div>

                {/* Message Banner */}
                {simMessage && (
                  <div style={{ 
                    background: simStatus === 'good' ? 'rgba(16, 185, 129, 0.1)' : simStatus === 'bad' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(255,255,255,0.05)',
                    border: `1px solid ${simStatus === 'good' ? 'rgba(16, 185, 129, 0.3)' : simStatus === 'bad' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(255,255,255,0.1)'}`,
                    color: 'white',
                    padding: '20px',
                    borderRadius: '16px',
                    fontWeight: 'bold',
                    fontSize: '1.1rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px',
                    transition: 'all 0.4s ease-in-out',
                  }}>
                    <span style={{ fontSize: '2rem', marginBottom: '4px' }}>
                      {simStatus === 'good' ? '✅' : simStatus === 'bad' ? '⚠️' : '⏳'}
                    </span>
                    <div style={{ lineHeight: '1.4', color: simStatus === 'good' ? '#10b981' : simStatus === 'bad' ? '#ef4444' : 'white' }}>
                      {simMessage}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px', marginTop: '30px' }}>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '1.5rem', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '12px' }}>
              Pose Details
            </h2>
            
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '1.1rem' }}>Description</h3>
              <p style={{ lineHeight: '1.6' }}>
                {moduleInfo.description || `This is the standard diagnostic procedure for ${moduleInfo.name}. It helps in accurately capturing the joint alignments and detecting structural abnormalities using our AI vision model.`}
              </p>
            </div>
            
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '1.1rem' }}>Instructions</h3>
              <ul style={{ lineHeight: '1.8', paddingLeft: '20px' }}>
                {moduleInfo.instructions ? (
                  moduleInfo.instructions.map((instruction, idx) => (
                    <li key={idx}>{instruction}</li>
                  ))
                ) : (
                  <>
                    <li>Ensure the area is well-lit and the {getBodyPartName(category)} is clearly visible in the frame.</li>
                    <li>Position the {getBodyPartName(category)} according to the specific {moduleInfo.name} protocol.</li>
                    <li>Hold the pose steady until the AI indicators turn green.</li>
                    <li>If the system prompts for rotation or straightening, adjust accordingly.</li>
                  </>
                )}
              </ul>
            </div>

            <div>
              <h3 style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '1.1rem' }}>Benefits</h3>
              <p style={{ lineHeight: '1.6' }}>
                {moduleInfo.benefits || `Correct positioning ensures high diagnostic accuracy and reduces the need for manual retakes, providing instant and precise clinical feedback.`}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Holographic 3D Viewer */}
            <HologramViewer moduleInfo={moduleInfo} />

            <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.02)' }}>
              <div style={{ color: 'rgba(255,255,255,0.4)', textAlign: 'center', marginBottom: '16px' }}>
                <Activity size={32} style={{ margin: '0 auto', marginBottom: '12px', opacity: 0.8, color: 'var(--accent-color)' }} />
                <p style={{ fontSize: '0.9rem' }}>Initialize Live Feed</p>
              </div>
              {poseId === 'patella_lateral' ? (
                <div style={{ display: 'flex', gap: '16px', width: '100%' }}>
                  <button 
                    className="btn btn-primary" 
                    onClick={() => { setTargetLeg('Left'); startScanner(); }}
                    disabled={loading}
                    style={{ flex: 1, padding: '16px', fontSize: '1.1rem', fontWeight: 'bold', display: 'flex', justifyContent: 'center', alignItems: 'center' }}
                  >
                    {loading && targetLeg === 'Left' ? 'INITIALIZING...' : 'Left Leg'}
                  </button>
                  <button 
                    className="btn btn-primary" 
                    onClick={() => { setTargetLeg('Right'); startScanner(); }}
                    disabled={loading}
                    style={{ flex: 1, padding: '16px', fontSize: '1.1rem', fontWeight: 'bold', display: 'flex', justifyContent: 'center', alignItems: 'center' }}
                  >
                    {loading && targetLeg === 'Right' ? 'INITIALIZING...' : 'Right Leg'}
                  </button>
                </div>
              ) : (
                <button 
                  className="btn btn-primary" 
                  onClick={startScanner}
                  disabled={loading}
                  style={{ width: '100%', padding: '16px', fontSize: '1.1rem', fontWeight: 'bold', display: 'flex', justifyContent: 'center', alignItems: 'center', letterSpacing: '1px' }}
                >
                  {loading ? 'Initializing AI Engine...' : (
                    <>
                      <Play size={20} style={{ marginRight: '8px', fill: 'white' }} />
                      START SCANNING
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Autonomous Diagnostic Report Modal */}
      {reportUrl && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1000, 
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(10px)'
        }}>
          <div className="glass-panel" style={{
            padding: '40px', maxWidth: '800px', width: '100%', 
            border: '2px solid rgba(16, 185, 129, 0.5)',
            boxShadow: '0 0 40px rgba(16, 185, 129, 0.2)',
            display: 'flex', flexDirection: 'column', gap: '20px'
          }}>
            <h2 style={{ fontSize: '2rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '12px', margin: 0 }}>
              <Activity size={32} />
              CLINICAL DIAGNOSTIC REPORT
            </h2>
            <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '1.1rem', letterSpacing: '1px' }}>
              AUTONOMOUS AI CAPTURE // {moduleInfo.name.toUpperCase()}
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '30px', marginTop: '20px' }}>
              <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.2)' }}>
                <img src={reportUrl} alt="Clinical Capture" style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '24px', borderRadius: '12px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                  <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.9rem', letterSpacing: '2px', marginBottom: '8px' }}>FINAL ACCURACY SCORE</div>
                  <div style={{ fontSize: '4rem', fontWeight: 'bold', color: '#10b981', lineHeight: '1' }}>{reportAccuracy}%</div>
                </div>
                
                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '24px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }}>
                  <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.9rem', letterSpacing: '2px', marginBottom: '12px' }}>AI VERIFICATION</div>
                  <div style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <span style={{ color: '#10b981' }}>✓</span> Joint Alignment
                  </div>
                  <div style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <span style={{ color: '#10b981' }}>✓</span> Posture Depth
                  </div>
                  <div style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: '#10b981' }}>✓</span> Motion Stability
                  </div>
                </div>
              </div>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
              <button className="btn btn-primary" onClick={() => setReportUrl(null)} style={{ padding: '12px 32px', fontSize: '1.1rem' }}>
                Save & Close Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PoseDetailView;
