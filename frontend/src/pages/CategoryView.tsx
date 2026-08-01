import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowRight, ChevronLeft } from 'lucide-react';
import { 
  CHEST_MODULES, 
  HAND_MODULES, 
  SPINE_MODULES, 
  ELBOW_MODULES, 
  KNEE_MODULES, 
  FOOT_MODULES,
  LOWERBACK_MODULES,
  ARM_MODULES,
  CATEGORIES
} from '../config/modules';
import type { ModuleInfo } from '../config/modules';

const CategoryView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [modules, setModules] = useState<ModuleInfo[]>([]);

  const categoryInfo = CATEGORIES.find(c => c.id === id);

  useEffect(() => {
    switch(id) {
      case 'chest': setModules(CHEST_MODULES); break;
      case 'hand': setModules(HAND_MODULES); break;
      case 'spine': setModules(SPINE_MODULES); break;
      case 'elbow': setModules(ELBOW_MODULES); break;
      case 'knee': setModules(KNEE_MODULES); break;
      case 'foot': setModules(FOOT_MODULES); break;
      case 'lowerback': setModules(LOWERBACK_MODULES); break;
      case 'arm': setModules(ARM_MODULES); break;
      default: setModules([]);
    }
  }, [id]);

  const handleModuleClick = (folder: string) => {
    const [category, poseId] = folder.split('/');
    if (category && poseId) {
      navigate(`/pose/${category}/${poseId}`);
    } else {
      // Fallback if folder structure is different
      navigate(`/pose/${id}/${folder}`);
    }
  };

  if (!categoryInfo) return <div>Category not found</div>;

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <button 
        className="btn" 
        onClick={() => navigate('/')} 
        style={{ marginBottom: '32px', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}
      >
        <ChevronLeft size={16} /> Dashboard
      </button>

      <div style={{ marginBottom: '40px' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontSize: '3rem' }}>{categoryInfo.icon}</span> 
          {categoryInfo.name} Diagnostic Suite
        </h1>
        <p style={{ fontSize: '1.1rem', color: 'var(--text-secondary)', maxWidth: '600px' }}>
          Select a clinical module below to load the appropriate AI model and launch the live diagnostic scanner.
        </p>
      </div>

      <div className="grid-cols-3">
        {modules.map((mod, index) => (
          <div key={index} className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <div style={{ background: 'rgba(56, 189, 248, 0.1)', padding: '12px', borderRadius: '12px' }}>
                <span style={{ fontSize: '1.5rem' }}>{mod.icon || categoryInfo.icon}</span>
              </div>
              <h3 style={{ fontSize: '1.2rem', margin: 0, color: 'var(--text-primary)' }}>
                {mod.name}
              </h3>
            </div>
            
            <p style={{ fontSize: '0.9rem', marginBottom: '24px', flex: 1, color: 'var(--text-muted)' }}>
              Standard operating procedure for {mod.name.toLowerCase()} structural assessment.
            </p>

            <button 
              className="btn btn-primary" 
              onClick={() => handleModuleClick(mod.folder)}
              style={{ width: '100%', justifyContent: 'space-between', padding: '12px 20px' }}
            >
              Enter <ArrowRight size={18} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CategoryView;
