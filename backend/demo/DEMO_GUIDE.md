# HifazatAI Demo Guide

This guide provides instructions for running comprehensive demonstrations of the HifazatAI security system.

## Quick Start

### 1. Start the Backend API
```bash
cd backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```

### 3. Run Demo Pipeline
```bash
cd backend/demo
python demo_pipeline.py --interactive
```

## Demo Scenarios

### 1. Mixed Threats (10 minutes)
**Purpose**: Demonstrate all three AI pipelines working together
**Duration**: 10 minutes
**Features**:
- Random mix of threat intelligence, video surveillance, and border anomaly alerts
- 2-5 alerts per minute
- Realistic timing and variety

```bash
python demo_pipeline.py --scenario mixed_threats
```

### 2. Security Breach Simulation
**Purpose**: Simulate a coordinated security incident
**Duration**: ~5 minutes
**Features**:
- Phase 1: Initial threat intelligence alerts (high-risk IOCs)
- Phase 2: Video surveillance detections (zone violations)
- Phase 3: Border anomaly alerts (high severity)

```bash
python demo_pipeline.py --scenario security_breach
```

### 3. Border Incident
**Purpose**: Focus on border security capabilities
**Duration**: ~4 minutes
**Features**:
- Series of escalating border anomaly alerts
- Increasing severity scores over time
- Coordinated suspicious activities

```bash
python demo_pipeline.py --scenario border_incident
```

### 4. Continuous Monitoring
**Purpose**: Long-running demonstration for exhibitions
**Duration**: Runs until stopped
**Features**:
- Continuous stream of realistic alerts
- Weighted distribution (more video alerts)
- 30-120 second intervals between alerts

```bash
python demo_pipeline.py --scenario continuous_monitoring
```

## Interactive Demo Mode

The interactive mode provides a menu-driven interface for controlling demonstrations:

```bash
python demo_pipeline.py --interactive
```

### Menu Options:
1. **Mixed Threats**: Run the 10-minute mixed scenario
2. **Security Breach**: Run the coordinated breach simulation
3. **Border Incident**: Run the border security scenario
4. **Continuous Monitoring**: Start continuous alert generation
5. **Single Alert Test**: Send one test alert of chosen type
6. **View Statistics**: Show demo runtime statistics
0. **Exit**: Stop the demo

## Demo Data Types

### Threat Intelligence Alerts
- **IOC Types**: IP addresses, domains, file hashes
- **Risk Levels**: High, Medium, Low
- **Sources**: Multiple simulated threat feeds
- **Evidence**: Realistic threat descriptions

Example:
```json
{
  "ioc_type": "ip",
  "ioc_value": "192.168.1.100",
  "risk_level": "High",
  "evidence_text": "Detected communication with known botnet C&C server",
  "source_feed": "ThreatIntel_Feed_1"
}
```

### Video Surveillance Alerts
- **Event Types**: Loitering, zone violations, abandoned objects
- **Camera Sources**: 5 simulated cameras
- **Tracking**: Realistic track IDs and bounding boxes
- **Snapshots**: Simulated snapshot paths

Example:
```json
{
  "event_type": "zone_violation",
  "bounding_box": [150, 200, 230, 350],
  "track_id": 456,
  "snapshot_path": "/snapshots/camera_2_track_456.jpg"
}
```

### Border Anomaly Alerts
- **Anomaly Types**: Erratic movement, unusual speed, suspicious loitering
- **Trajectories**: Realistic movement patterns
- **Severity**: Graduated severity scores
- **Supporting Data**: Feature vectors and frame references

Example:
```json
{
  "anomaly_type": "erratic_movement",
  "severity_score": 0.87,
  "trajectory_points": [[100, 200], [120, 180], [95, 220]],
  "supporting_frames": ["/frames/anomaly_1.jpg", "/frames/anomaly_2.jpg"]
}
```

## Frontend Features to Demonstrate

### 1. Real-time Alert Feed
- **Location**: Alerts page, right panel
- **Features**: Live updates, connection status, pause/resume
- **Demo Points**: Show alerts appearing in real-time

### 2. Alert Management
- **Location**: Alerts page, main panel
- **Features**: Filtering, search, status updates, pagination
- **Demo Points**: Filter by pipeline, update alert status

### 3. Alert Details
- **Location**: Modal when clicking any alert
- **Features**: Type-specific details, evidence display, actions
- **Demo Points**: Show different alert types, mark as reviewed

### 4. Dashboard Overview
- **Location**: Main dashboard page
- **Features**: Summary statistics, recent alerts, system status
- **Demo Points**: Overall system health, alert distribution

### 5. Video Analysis Interface
- **Location**: Video Analysis page
- **Features**: Video player, bounding boxes, snapshot gallery
- **Demo Points**: Show video alerts with overlays

### 6. System Monitoring
- **Location**: System Monitoring page (if implemented)
- **Features**: Pipeline status, performance metrics, configuration
- **Demo Points**: System health, processing rates

## Demonstration Script

### Opening (2 minutes)
1. **Introduction**: "HifazatAI is a comprehensive security AI system..."
2. **Architecture Overview**: Show the three main pipelines
3. **Dashboard Tour**: Quick overview of the interface

### Core Demonstration (8 minutes)

#### Threat Intelligence (2 minutes)
1. Start demo pipeline with threat focus
2. Show alerts appearing in real-time feed
3. Click on threat alert to show details
4. Explain IOC extraction and risk classification

#### Video Surveillance (3 minutes)
1. Switch to video analysis page
2. Show video alerts with bounding boxes
3. Demonstrate snapshot gallery
4. Explain object detection and tracking

#### Border Anomaly Detection (2 minutes)
1. Generate border anomaly alerts
2. Show trajectory visualization
3. Explain anomaly detection algorithms
4. Demonstrate severity scoring

#### System Integration (1 minute)
1. Show dashboard with mixed alerts
2. Demonstrate filtering and search
3. Show connection status and real-time updates

### Advanced Features (5 minutes)
1. **Alert Management**: Status updates, bulk operations
2. **System Monitoring**: Pipeline health, performance metrics
3. **Configuration**: Threshold adjustments, settings
4. **Evaluation**: Show evaluation results and metrics

### Closing (2 minutes)
1. **Summary**: Key capabilities demonstrated
2. **Performance**: Metrics achieved
3. **Scalability**: Deployment options
4. **Q&A**: Address questions

## Troubleshooting

### Common Issues

#### Demo Pipeline Not Connecting
```bash
# Check API is running
curl http://localhost:8000/health

# Check API logs
# Look for CORS errors or connection issues
```

#### No Alerts Appearing in Frontend
1. Check browser console for errors
2. Verify API connection status in header
3. Check network tab for failed requests
4. Ensure SSE/WebSocket connection is working

#### Slow Performance
1. Reduce demo alert frequency
2. Check system resources (CPU, memory)
3. Use smaller batch sizes in configuration
4. Close unnecessary browser tabs

### Demo Environment Setup

#### Minimum Requirements
- **CPU**: 4 cores (i5 6th gen or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **Network**: Stable internet for dependencies

#### Recommended Setup
- **Display**: 1920x1080 or higher for best visualization
- **Browser**: Chrome or Firefox (latest versions)
- **Audio**: For presentation narration
- **Backup**: Have demo data pre-loaded in case of network issues

### Performance Optimization

#### For Live Demonstrations
```bash
# Use optimized settings
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Start with CPU optimizations
python demo_pipeline.py --scenario mixed_threats --optimize-cpu
```

#### For Extended Exhibitions
```bash
# Use continuous monitoring with longer intervals
python demo_pipeline.py --scenario continuous_monitoring --min-interval 60
```

## Customization

### Adding Custom Scenarios
1. Edit `demo_pipeline.py`
2. Add new scenario function
3. Register in scenarios dictionary
4. Test with single alerts first

### Modifying Alert Frequency
```python
# In demo_pipeline.py, adjust timing
await asyncio.sleep(random.uniform(10, 30))  # 10-30 seconds between alerts
```

### Custom Alert Data
```python
# Modify generator functions to use specific data
def generate_custom_threat_alert(self):
    return {
        "ioc_type": "custom_type",
        "ioc_value": "custom_value",
        # ... other fields
    }
```

## Integration with Presentations

### PowerPoint/Keynote Integration
1. Use windowed mode for easy switching
2. Prepare screenshots for backup slides
3. Have demo scenarios match presentation flow

### Video Recording
1. Use screen recording software
2. Record individual scenarios separately
3. Prepare edited highlights reel

### Live Streaming
1. Use OBS or similar streaming software
2. Set up scene transitions
3. Include system metrics overlay

## Support and Maintenance

### Before Each Demo
- [ ] Test API connectivity
- [ ] Verify frontend builds correctly
- [ ] Run single alert test
- [ ] Check system resources
- [ ] Prepare backup scenarios

### During Demo
- [ ] Monitor system performance
- [ ] Watch for error messages
- [ ] Have fallback plans ready
- [ ] Keep demo script handy

### After Demo
- [ ] Stop demo processes
- [ ] Clear demo data if needed
- [ ] Collect feedback
- [ ] Update demo based on issues

For technical support or questions about the demo system, refer to the main project documentation or contact the development team.