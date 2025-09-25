import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { SystemConfiguration } from '../components/monitoring';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import { 
  CogIcon,
  UserIcon,
  BellIcon,
  ShieldCheckIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';

const Settings = () => {
  const [activeSection, setActiveSection] = useState<'system' | 'user' | 'notifications' | 'security' | 'api'>('system');

  const sections = [
    { id: 'system', name: 'System Configuration', icon: CogIcon },
    { id: 'user', name: 'User Preferences', icon: UserIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
    { id: 'security', name: 'Security', icon: ShieldCheckIcon },
    { id: 'api', name: 'API Settings', icon: GlobeAltIcon }
  ] as const;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">Settings</h2>
        <p className="text-gray-600">Configure system settings and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <nav className="space-y-2">
                {sections.map(section => {
                  const Icon = section.icon;
                  return (
                    <button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center space-x-3 px-3 py-2 text-left rounded-lg transition-colors ${
                        activeSection === section.id
                          ? 'bg-blue-50 text-blue-700 border border-blue-200'
                          : 'text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                      <span className="font-medium">{section.name}</span>
                    </button>
                  );
                })}
              </nav>
            </CardContent>
          </Card>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          {activeSection === 'system' && (
            <SystemConfiguration 
              onSave={(settings) => {
                console.log('System settings saved:', settings);
                // In real implementation, save to API
              }}
            />
          )}

          {activeSection === 'user' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <UserIcon className="h-5 w-5" />
                  <span>User Preferences</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Display Name
                    </label>
                    <input
                      type="text"
                      defaultValue="Security Operator"
                      className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Time Zone
                    </label>
                    <select className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                      <option>UTC</option>
                      <option>America/New_York</option>
                      <option>Europe/London</option>
                      <option>Asia/Tokyo</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Dashboard Refresh Rate
                    </label>
                    <select className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                      <option>5 seconds</option>
                      <option>10 seconds</option>
                      <option>30 seconds</option>
                      <option>1 minute</option>
                    </select>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="darkMode"
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <label htmlFor="darkMode" className="text-sm font-medium text-gray-700">
                      Enable dark mode
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeSection === 'notifications' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BellIcon className="h-5 w-5" />
                  <span>Notification Settings</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-4">Alert Notifications</h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">High Priority Alerts</p>
                          <p className="text-sm text-gray-600">Immediate notifications for critical alerts</p>
                        </div>
                        <input
                          type="checkbox"
                          defaultChecked
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Medium Priority Alerts</p>
                          <p className="text-sm text-gray-600">Notifications for warning-level alerts</p>
                        </div>
                        <input
                          type="checkbox"
                          defaultChecked
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">System Status Updates</p>
                          <p className="text-sm text-gray-600">Pipeline status changes and system events</p>
                        </div>
                        <input
                          type="checkbox"
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900 mb-4">Notification Methods</h3>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="browserNotifications"
                          defaultChecked
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="browserNotifications" className="text-sm font-medium text-gray-700">
                          Browser notifications
                        </label>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="emailNotifications"
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="emailNotifications" className="text-sm font-medium text-gray-700">
                          Email notifications
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeSection === 'security' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <ShieldCheckIcon className="h-5 w-5" />
                  <span>Security Settings</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-4">Authentication</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Session Timeout
                        </label>
                        <select className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                          <option>15 minutes</option>
                          <option>30 minutes</option>
                          <option>1 hour</option>
                          <option>4 hours</option>
                        </select>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="requireMFA"
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="requireMFA" className="text-sm font-medium text-gray-700">
                          Require multi-factor authentication
                        </label>
                        <Badge variant="warning">Coming Soon</Badge>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900 mb-4">Data Protection</h3>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="encryptData"
                          defaultChecked
                          disabled
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="encryptData" className="text-sm font-medium text-gray-700">
                          Encrypt sensitive data at rest
                        </label>
                        <Badge variant="success">Enabled</Badge>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="auditLogging"
                          defaultChecked
                          disabled
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="auditLogging" className="text-sm font-medium text-gray-700">
                          Enable audit logging
                        </label>
                        <Badge variant="success">Enabled</Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeSection === 'api' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <GlobeAltIcon className="h-5 w-5" />
                  <span>API Settings</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-4">API Configuration</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          API Base URL
                        </label>
                        <input
                          type="text"
                          defaultValue="http://localhost:8000"
                          className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Request Timeout (seconds)
                        </label>
                        <input
                          type="number"
                          defaultValue="30"
                          min="5"
                          max="300"
                          className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900 mb-4">Rate Limiting</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Max Requests per Minute
                        </label>
                        <input
                          type="number"
                          defaultValue="100"
                          min="10"
                          max="1000"
                          className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Save Button */}
          <div className="mt-6">
            <Button className="w-full">
              Save Settings
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;