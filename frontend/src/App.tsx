import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import DashboardLayout from './components/layout/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Alerts from './pages/Alerts';
import VideoAnalysis from './pages/VideoAnalysis';
import SystemStatus from './pages/SystemStatus';
import Settings from './pages/Settings';

function App() {
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
