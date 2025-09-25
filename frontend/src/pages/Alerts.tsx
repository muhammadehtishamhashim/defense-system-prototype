import { useState } from 'react';
import type { BaseAlert } from '../types';
import AlertList from '../components/alerts/AlertList';
import RealTimeAlertFeed from '../components/alerts/RealTimeAlertFeed';
import AlertDetail from '../components/alerts/AlertDetail';
import AlertSummary from '../components/alerts/AlertSummary';
import Modal from '../components/ui/Modal';

const Alerts = () => {
  const [selectedAlert, setSelectedAlert] = useState<BaseAlert | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  const handleAlertSelect = (alert: BaseAlert) => {
    setSelectedAlert(alert);
    setShowDetailModal(true);
  };

  const handleCloseDetail = () => {
    setShowDetailModal(false);
    setSelectedAlert(null);
  };

  const handleStatusUpdate = (alertId: string, status: BaseAlert['status']) => {
    if (selectedAlert && selectedAlert.id === alertId) {
      setSelectedAlert({ ...selectedAlert, status });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">Alerts</h2>
        <p className="text-gray-600">Monitor and manage security alerts from all pipelines</p>
      </div>

      {/* Alert Summary Statistics */}
      <AlertSummary />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Alert Feed */}
        <div className="lg:col-span-1">
          <RealTimeAlertFeed 
            onAlertSelect={handleAlertSelect}
            maxAlerts={20}
            refreshInterval={5000}
            useSSE={true}
          />
        </div>

        {/* Main Alert List */}
        <div className="lg:col-span-2">
          <AlertList onAlertSelect={handleAlertSelect} />
        </div>
      </div>

      {/* Alert Detail Modal */}
      <Modal
        isOpen={showDetailModal}
        onClose={handleCloseDetail}
        title="Alert Details"
        size="xl"
      >
        {selectedAlert && (
          <AlertDetail
            alert={selectedAlert}
            onStatusUpdate={handleStatusUpdate}
            onClose={handleCloseDetail}
          />
        )}
      </Modal>
    </div>
  );
};

export default Alerts;