import React, { useState } from 'react';
import { dataService } from '../../services/dataService';
import api from '../../services/api';

const ApiTest: React.FC = () => {
  const [testResults, setTestResults] = useState<any>({});
  const [loading, setLoading] = useState(false);

  const testEndpoints = async () => {
    setLoading(true);
    const results: any = {};

    try {
      // Test health endpoint
      const healthResponse = await api.get('/health');
      results.health = { success: true, data: healthResponse.data };
    } catch (error) {
      results.health = { success: false, error: error.message };
    }

    try {
      // Test system metrics endpoint
      const metricsResponse = await api.get('/system/metrics');
      results.metrics = { success: true, data: metricsResponse.data };
    } catch (error) {
      results.metrics = { success: false, error: error.message };
    }

    try {
      // Test alerts endpoint
      const alertsResponse = await api.get('/alerts?limit=5');
      results.alerts = { success: true, data: alertsResponse.data };
    } catch (error) {
      results.alerts = { success: false, error: error.message };
    }

    try {
      // Test dataService
      const systemData = await dataService.getSystemMetrics();
      results.dataService = { success: true, data: systemData };
    } catch (error) {
      results.dataService = { success: false, error: error.message };
    }

    setTestResults(results);
    setLoading(false);
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg">
      <h3 className="text-lg font-bold mb-4">API Connection Test</h3>
      
      <button
        onClick={testEndpoints}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? 'Testing...' : 'Test API Endpoints'}
      </button>

      {Object.keys(testResults).length > 0 && (
        <div className="mt-4 space-y-2">
          {Object.entries(testResults).map(([endpoint, result]: [string, any]) => (
            <div key={endpoint} className="p-2 border rounded">
              <div className="flex items-center space-x-2">
                <span className="font-medium">{endpoint}:</span>
                <span className={result.success ? 'text-green-600' : 'text-red-600'}>
                  {result.success ? '✓ Success' : '✗ Failed'}
                </span>
              </div>
              {result.error && (
                <div className="text-red-600 text-sm mt-1">Error: {result.error}</div>
              )}
              {result.data && (
                <details className="mt-1">
                  <summary className="text-sm text-gray-600 cursor-pointer">Show data</summary>
                  <pre className="text-xs bg-gray-50 p-2 rounded mt-1 overflow-auto">
                    {JSON.stringify(result.data, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ApiTest;