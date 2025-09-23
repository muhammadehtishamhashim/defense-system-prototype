import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';

const VideoAnalysis = () => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900">Video Analysis</h2>
        <p className="text-gray-600">Real-time video surveillance and analysis</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Video Analysis Interface</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">Video analysis interface will be implemented in task 6.3</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default VideoAnalysis;