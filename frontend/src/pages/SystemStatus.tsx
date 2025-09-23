import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';

const SystemStatus = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">System Status</h2>
        <p className="text-gray-600">Monitor system health and performance</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Monitoring Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">System monitoring dashboard will be implemented in task 6.4</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemStatus;