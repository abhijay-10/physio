import React, { useEffect, useState, useRef } from 'react';
import type { ModuleInfo } from '../config/modules';

interface HologramViewerProps {
  moduleInfo: ModuleInfo;
}

const HologramViewer: React.FC<HologramViewerProps> = ({ moduleInfo }) => {
  const [scanPosition, setScanPosition] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setScanPosition(prev => (prev > 100 ? 0 : prev + 1));
    }, 40);
    return () => clearInterval(interval);
  }, []);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    // Calculate rotation (-15 to 15 degrees)
    const rotateX = ((y - centerY) / centerY) * -15;
    const rotateY = ((x - centerX) / centerX) * 15;
    
    setRotation({ x: rotateX, y: rotateY });
  };

  return (
    <div 
      className="hologram-container glass-panel"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => {
        setIsHovering(false);
        setRotation({ x: 0, y: 0 });
      }}
      style={{
        perspective: '1000px',
        overflow: 'visible'
      }}
    >
      <div 
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
          transformStyle: 'preserve-3d',
          transition: isHovering ? 'transform 0.1s ease-out' : 'transform 0.5s ease-in-out',
          position: 'relative'
        }}
      >
        <div className="hologram-header" style={{ transform: 'translateZ(30px)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div className="status-dot pulsing"></div>
            <span style={{ color: '#10b981', fontSize: '0.9rem', fontWeight: 600, letterSpacing: '1px' }}>AI MODEL LOADED</span>
          </div>
          <div style={{ color: 'var(--accent-color)', fontSize: '0.8rem', fontFamily: 'monospace' }}>
            SYS.v3.1 // {moduleInfo.name.toUpperCase()}
          </div>
        </div>

        <div className="hologram-display" style={{ transformStyle: 'preserve-3d', position: 'relative' }}>
          
          {/* Hologram Image with floating 3D effect */}
          <div className="hologram-wrapper" style={{ transform: 'translateZ(60px)', position: 'relative' }}>
            <img 
              src={moduleInfo.hologramUrl || '/holograms/hologram_pa_hand.png'} 
              alt={`${moduleInfo.name} Hologram`} 
              className="hologram-image"
              style={{
                filter: 'drop-shadow(0 0 20px var(--accent-color)) hue-rotate(15deg) brightness(1.2) contrast(1.2)',
                opacity: 0.85
              }}
            />
            {/* Scanner Line */}
            <div 
              className="scanner-line"
              style={{ top: `${scanPosition}%`, transform: 'translateZ(20px)' }}
            ></div>
          </div>

          {/* HUD Elements */}
          {moduleInfo.analysisMetrics && (
            <>
              <div className="hud-panel hud-left" style={{ transform: 'translateZ(40px)' }}>
                <div className="hud-label">TARGET ANGLE</div>
                <div className="hud-value text-glow">{moduleInfo.analysisMetrics.targetAngle}</div>
                
                <div className="hud-label" style={{ marginTop: '16px' }}>KEY POINTS</div>
                <div className="hud-value text-glow">{moduleInfo.analysisMetrics.keyPoints} Tracked</div>
              </div>

              <div className="hud-panel hud-right" style={{ transform: 'translateZ(40px)' }}>
                <div className="hud-label">FOCUS AREAS</div>
                {moduleInfo.analysisMetrics.focusAreas?.map((area, idx) => (
                  <div key={idx} className="hud-area-item">
                    <span className="hud-crosshair">+</span> {area}
                  </div>
                ))}
                <div className="confidence-meter" style={{ marginTop: '16px' }}>
                  <div className="hud-label">MODEL CONFIDENCE</div>
                  <div className="confidence-value text-accent text-glow">98.5%</div>
                  <div className="confidence-bar">
                    <div className="confidence-fill"></div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="hologram-footer" style={{ transform: 'translateZ(30px)' }}>
          <span className="typing-text">AWAITING CAMERA ACTIVATION...</span>
          <span style={{ fontFamily: 'monospace' }}>DATA: SECURE</span>
        </div>
      </div>
    </div>
  );
};

export default HologramViewer;
