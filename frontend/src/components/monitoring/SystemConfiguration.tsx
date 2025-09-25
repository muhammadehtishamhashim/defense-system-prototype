import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import Button from '../ui/Button';
import Badge from '../ui/Badge';
import { 
  CogIcon,
  CheckIcon,
  XMarkIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

interface ConfigurationSetting {
  key: string;
  name: string;
  description: string;
  value: string | number | boolean;
  type: 'string' | 'number' | 'boolean' | 'select';
  options?: string[];
  min?: number;
  max?: number;
  category: string;
  requiresRestart?: boolean;
}

interface SystemConfigurationProps {
  onSave?: (settings: ConfigurationSetting[]) => void;
}

const SystemConfiguration: React.FC<SystemConfigurationProps> = ({
  onSave
}) => {
  const [settings, setSettings] = useState<ConfigurationSetting[]>([]);
  const [modifiedSettings, setModifiedSettings] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['general']));

  // Mock configuration settings
  const mockSettings: ConfigurationSetting[] = [
    // General Settings
    {
      key: 'system.name',
      name: 'System Name',
      description: 'Display name for the HifazatAI system',
      value: 'HifazatAI Security System',
      type: 'string',
      category: 'general'
    },
    {
      key: 'system.refresh_interval',
      name: 'Refresh Interval',
      description: 'How often to refresh dashboard data (seconds)',
      value: 30,
      type: 'number',
      min: 5,
      max: 300,
      category: 'general'
    },
    {
      key: 'system.debug_mode',
      name: 'Debug Mode',
      description: 'Enable detailed logging and debug information',
      value: false,
      type: 'boolean',
      category: 'general',
      requiresRestart: true
    },
    
    // Threat Intelligence Settings
    {
      key: 'threat.confidence_threshold',
      name: 'Confidence Threshold',
      description: 'Minimum confidence score for threat alerts (0-1)',
      value: 0.7,
      type: 'number',
      min: 0,
      max: 1,
      category: 'threat_intelligence'
    },
    {
      key: 'threat.risk_levels',
      name: 'Risk Level Classification',
      description: 'How to classify threat risk levels',
      value: 'automatic',
      type: 'select',
      options: ['automatic', 'manual', 'hybrid'],
      category: 'threat_intelligence'
    },
    {
      key: 'threat.feed_update_interval',
      name: 'Feed Update Interval',
      description: 'How often to check for new threat intelligence (minutes)',
      value: 15,
      type: 'number',
      min: 1,
      max: 1440,
      category: 'threat_intelligence'
    },
    
    // Video Surveillance Settings
    {
      key: 'video.detection_threshold',
      name: 'Detection Threshold',
      description: 'Minimum confidence for object detection (0-1)',
      value: 0.5,
      type: 'number',
      min: 0,
      max: 1,
      category: 'video_surveillance'
    },
    {
      key: 'video.tracking_enabled',
      name: 'Object Tracking',
      description: 'Enable multi-object tracking across frames',
      value: true,
      type: 'boolean',
      category: 'video_surveillance'
    },
    {
      key: 'video.loitering_timeout',
      name: 'Loitering Timeout',
      description: 'Time before loitering alert is triggered (seconds)',
      value: 30,
      type: 'number',
      min: 5,
      max: 300,
      category: 'video_surveillance'
    },
    {
      key: 'video.frame_skip',
      name: 'Frame Skip',
      description: 'Process every Nth frame for performance optimization',
      value: 2,
      type: 'number',
      min: 1,
      max: 10,
      category: 'video_surveillance'
    },
    
    // Border Anomaly Settings
    {
      key: 'anomaly.sensitivity',
      name: 'Anomaly Sensitivity',
      description: 'Sensitivity level for anomaly detection (0-1)',
      value: 0.8,
      type: 'number',
      min: 0,
      max: 1,
      category: 'border_anomaly'
    },
    {
      key: 'anomaly.min_trajectory_length',
      name: 'Minimum Trajectory Length',
      description: 'Minimum number of points required for trajectory analysis',
      value: 10,
      type: 'number',
      min: 3,
      max: 100,
      category: 'border_anomaly'
    },
    {
      key: 'anomaly.detection_model',
      name: 'Detection Model',
      description: 'Which anomaly detection model to use',
      value: 'isolation_forest',
      type: 'select',
      options: ['isolation_forest', 'one_class_svm', 'ensemble'],
      category: 'border_anomaly'
    }
  ];

  useEffect(() => {
    setSettings(mockSettings);
  }, []);

  const categories = Array.from(new Set(settings.map(s => s.category)));

  const formatCategoryName = (category: string) => {
    return category.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const handleSettingChange = (key: string, value: string | number | boolean) => {
    setSettings(prev => prev.map(setting => 
      setting.key === key ? { ...setting, value } : setting
    ));
    setModifiedSettings(prev => new Set(prev).add(key));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      onSave?.(settings);
      setModifiedSettings(new Set());
      
      // Show success message (in real app, use toast notification)
      console.log('Settings saved successfully');
    } catch (error) {
      console.error('Error saving settings:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setSettings(mockSettings);
    setModifiedSettings(new Set());
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  };

  const renderSettingInput = (setting: ConfigurationSetting) => {
    const isModified = modifiedSettings.has(setting.key);
    
    switch (setting.type) {
      case 'boolean':
        return (
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={setting.value as boolean}
              onChange={(e) => handleSettingChange(setting.key, e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-600">
              {setting.value ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        );
        
      case 'select':
        return (
          <select
            value={setting.value as string}
            onChange={(e) => handleSettingChange(setting.key, e.target.value)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {setting.options?.map(option => (
              <option key={option} value={option}>
                {option.charAt(0).toUpperCase() + option.slice(1).replace('_', ' ')}
              </option>
            ))}
          </select>
        );
        
      case 'number':
        return (
          <input
            type="number"
            value={setting.value as number}
            min={setting.min}
            max={setting.max}
            step={setting.min !== undefined && setting.min < 1 ? 0.1 : 1}
            onChange={(e) => handleSettingChange(setting.key, parseFloat(e.target.value))}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        );
        
      default:
        return (
          <input
            type="text"
            value={setting.value as string}
            onChange={(e) => handleSettingChange(setting.key, e.target.value)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        );
    }
  };

  const hasModifications = modifiedSettings.size > 0;
  const requiresRestart = settings.some(s => modifiedSettings.has(s.key) && s.requiresRestart);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <CogIcon className="h-5 w-5" />
            <span>System Configuration</span>
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            {requiresRestart && (
              <Badge variant="warning" className="flex items-center space-x-1">
                <ExclamationTriangleIcon className="h-3 w-3" />
                <span>Restart Required</span>
              </Badge>
            )}
            
            <Button
              variant="outline"
              size="sm"
              onClick={handleReset}
              disabled={!hasModifications || saving}
            >
              Reset
            </Button>
            
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!hasModifications || saving}
              className="flex items-center space-x-1"
            >
              <CheckIcon className="h-4 w-4" />
              <span>{saving ? 'Saving...' : 'Save Changes'}</span>
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-6">
          {categories.map(category => {
            const categorySettings = settings.filter(s => s.category === category);
            const isExpanded = expandedCategories.has(category);
            
            return (
              <div key={category} className="border rounded-lg">
                <button
                  onClick={() => toggleCategory(category)}
                  className="w-full px-4 py-3 text-left font-medium text-gray-900 hover:bg-gray-50 flex items-center justify-between"
                >
                  <span>{formatCategoryName(category)}</span>
                  <span className="text-gray-400">
                    {isExpanded ? '−' : '+'}
                  </span>
                </button>
                
                {isExpanded && (
                  <div className="px-4 pb-4 space-y-4 border-t border-gray-200">
                    {categorySettings.map(setting => (
                      <div key={setting.key} className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
                        <div>
                          <div className="flex items-center space-x-2">
                            <label className="font-medium text-gray-900">
                              {setting.name}
                            </label>
                            {modifiedSettings.has(setting.key) && (
                              <Badge variant="info" className="text-xs">Modified</Badge>
                            )}
                            {setting.requiresRestart && (
                              <Badge variant="warning" className="text-xs">Restart</Badge>
                            )}
                          </div>
                          <p className="text-sm text-gray-600 mt-1">
                            {setting.description}
                          </p>
                        </div>
                        
                        <div className="md:col-span-2">
                          {renderSettingInput(setting)}
                          {setting.type === 'number' && (setting.min !== undefined || setting.max !== undefined) && (
                            <p className="text-xs text-gray-500 mt-1">
                              Range: {setting.min ?? '−∞'} to {setting.max ?? '∞'}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {hasModifications && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="h-5 w-5 text-blue-600" />
              <div>
                <p className="font-medium text-blue-900">
                  You have unsaved changes
                </p>
                <p className="text-sm text-blue-700">
                  {modifiedSettings.size} setting{modifiedSettings.size !== 1 ? 's' : ''} modified.
                  {requiresRestart && ' Some changes require a system restart to take effect.'}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemConfiguration;