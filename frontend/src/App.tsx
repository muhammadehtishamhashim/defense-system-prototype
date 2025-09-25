import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import './App.css';
import DashboardLayout from './components/layout/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Alerts from './pages/Alerts';
import VideoAnalysis from './pages/VideoAnalysis';
import SystemStatus from './pages/SystemStatus';
import Settings from './pages/Settings';
import { startConnectionMonitoring } from './services/api';
import { sseService } from './services/sseService';

function App() {
  useEffect(() => {
    // Initialize connection monitoring
    startConnectionMonitoring();
    
    // Initialize SSE connection
    sseService.connect().catch(error => {
      console.warn('SSE connection failed, will use polling fallback:', error);
    });
    
    // Cleanup on unmount
    return () => {
      sseService.disconnect();
    };
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="alerts" element={<Alerts />} />
          <Route path="video" element={<VideoAnalysis />} />
          <Route path="system" element={<SystemStatus />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
