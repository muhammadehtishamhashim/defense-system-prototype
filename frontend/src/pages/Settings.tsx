import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';

const Settings = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">Settings</h2>
        <p className="text-gray-600">Configure system settings and preferences</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">Settings interface will be implemented in future tasks</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;