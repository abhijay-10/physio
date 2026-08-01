import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import TopNav from './components/TopNav';
import DashboardHome from './pages/DashboardHome';
import CategoryView from './pages/CategoryView';
import PoseDetailView from './pages/PoseDetailView';
import Chatbot from './components/Chatbot';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Portfolio from './pages/Portfolio';
import { useAuth } from './context/AuthContext';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isLoggedIn } = useAuth();
  if (!isLoggedIn) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
};

const App: React.FC = () => {
  return (
    <div className="app-container">
      <TopNav />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<DashboardHome />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/portfolio" element={<ProtectedRoute><Portfolio /></ProtectedRoute>} />
          <Route path="/category/:id" element={<ProtectedRoute><CategoryView /></ProtectedRoute>} />
          <Route path="/pose/:category/:poseId" element={<ProtectedRoute><PoseDetailView /></ProtectedRoute>} />
        </Routes>
      </main>
      <Chatbot />
    </div>
  );
};

export default App;
