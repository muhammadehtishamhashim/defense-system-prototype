# HifazatAI Troubleshooting Guide

## Quick Diagnostic Checklist

### System Health Check
1. **API Status**: `curl http://localhost:8000/health`
2. **Frontend Access**: Navigate to `http://localhost:3000`
3. **Database Connection**: Check logs for SQLite errors
4. **Pipeline Status**: Verify all three pipelines are running
5. **Resource Usage**: Monitor CPU and memory consumption

### Common Symptoms and Solutions

#### 🚨 Critical Issues

##### System Won't Start
**Symptoms**: Services fail to start, connection errors
**Quick Fix**:
```bash
# Check if ports are available
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# Restart services
docker-compose down && docker-compose up --build
```

##### No Alerts Appearing
**Symptoms**: Dashboard shows no new alerts despite activity
**Quick Fix**:
```bash
# Check pipeline status
curl http://localhost:8000/metrics/threat_intelligence
curl http://localhost:8000/metrics/video_surveillance
curl http://localhost:8000/metrics/border_anomaly

# Restart pipelines
cd backend && python demo/demo_pipeline.py --scenario mixed_threats
```

#### ⚠️ Performance Issues

##### High CPU Usage
**Symptoms**: System running slowly, high CPU utilization
**Solutions**:
1. Apply CPU optimizations:
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```
2. Reduce processing frequency in pipeline configs
3. Lower video resolution and frame rates
4. Use CPU-optimized model variants

##### Memory Leaks
**Symptoms**: Gradually increasing memory usage
**Solutions**:
1. Restart services periodically
2. Enable garbage collection optimization
3. Monitor memory usage patterns
4. Check for large file accumulation

## Detailed Troubleshooting

### Frontend Issues

#### Dashboard Not Loading
**Error**: Blank screen or loading spinner
**Diagnosis**:
```bash
# Check browser console for errors
# Verify API connectivity
curl http://localhost:8000/health

# Check frontend build
cd frontend && npm run build
```
**Solutions**:
- Clear browser cache and cookies
- Verify CORS configuration in backend
- Check network connectivity
- Try different browser or incognito mode

#### Real-time Updates Not Working
**Error**: Alerts not appearing in real-time
**Diagnosis**:
```javascript
// Check SSE connection in browser console
const eventSource = new EventSource('http://localhost:8000/events');
eventSource.onerror = (error) => console.error('SSE Error:', error);
```
**Solutions**:
- Verify SSE endpoint is accessible
- Check firewall and proxy settings
- Restart frontend service
- Validate WebSocket/SSE configuration

### Backend Issues

#### API Server Won't Start
**Error**: FastAPI fails to initialize
**Diagnosis**:
```bash
# Check Python environment
python --version
pip list | grep fastapi

# Check for port conflicts
lsof -i :8000

# Review startup logs
cd backend && python -m uvicorn api.main:app --reload
```
**Solutions**:
- Install missing dependencies: `pip install -r requirements.txt`
- Change port if 8000 is occupied
- Check Python version compatibility (3.10+)
- Verify virtual environment activation

#### Database Connection Errors
**Error**: SQLite connection failures
**Diagnosis**:
```bash
# Check database file permissions
ls -la backend/hifazat.db

# Test database connectivity
cd backend && python -c "from models.database import db_manager; print('DB OK')"
```
**Solutions**:
- Ensure database directory exists and is writable
- Initialize database: `cd backend && python init_backend.py`
- Check disk space availability
- Verify SQLite installation
#### 
Pipeline Processing Errors
**Error**: AI pipelines failing to process data
**Diagnosis**:
```bash
# Check pipeline logs
tail -f backend/logs/pipelines.log

# Test individual pipelines
cd backend && python -c "from pipelines.threat_intelligence import ThreatIntelligencePipeline; print('Threat pipeline OK')"
cd backend && python -c "from pipelines.video_surveillance import VideoSurveillancePipeline; print('Video pipeline OK')"
cd backend && python -c "from pipelines.border_anomaly import BorderAnomalyPipeline; print('Anomaly pipeline OK')"
```
**Solutions**:
- Check model file availability and permissions
- Verify input data format and quality
- Review pipeline configuration settings
- Restart individual pipeline services
- Check for missing dependencies

### AI Model Issues

#### Model Loading Failures
**Error**: AI models fail to load or initialize
**Diagnosis**:
```bash
# Check model files
ls -la backend/models/
file backend/models/yolov8n.pt

# Test model loading
cd backend && python -c "import torch; model = torch.hub.load('ultralytics/yolov8', 'yolov8n'); print('Model loaded')"
```
**Solutions**:
- Download missing model files
- Verify model file integrity
- Check PyTorch/ONNX compatibility
- Clear model cache and reload
- Use alternative model variants

#### Poor Detection Accuracy
**Error**: High false positive/negative rates
**Diagnosis**:
- Review confidence thresholds in configuration
- Analyze input data quality and format
- Check model performance metrics
- Validate ground truth data
**Solutions**:
- Adjust confidence thresholds
- Retrain models with better data
- Use ensemble methods
- Implement post-processing filters
- Collect more training data

### Performance Issues

#### Slow Processing Speed
**Error**: High latency in alert generation
**Diagnosis**:
```bash
# Monitor system resources
top -p $(pgrep -f "python.*api.main")
htop

# Check processing times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"
```
**Solutions**:
- Enable CPU optimizations
- Reduce input resolution/quality
- Implement batch processing
- Use model quantization
- Scale horizontally

#### Memory Leaks
**Error**: Gradually increasing memory usage
**Diagnosis**:
```bash
# Monitor memory usage over time
while true; do
  ps aux | grep python | grep -v grep
  sleep 60
done

# Check for memory leaks
cd backend && python -m memory_profiler api/main.py
```
**Solutions**:
- Restart services periodically
- Implement garbage collection
- Fix memory leaks in code
- Use memory pooling
- Monitor and alert on usage

### Network and Connectivity

#### API Connection Timeouts
**Error**: Frontend cannot connect to backend
**Diagnosis**:
```bash
# Test API connectivity
curl -v http://localhost:8000/health
telnet localhost 8000

# Check network configuration
netstat -tulpn | grep :8000
ss -tulpn | grep :8000
```
**Solutions**:
- Verify API server is running
- Check firewall and port settings
- Review CORS configuration
- Test with different network
- Use alternative ports

#### SSE Connection Issues
**Error**: Real-time updates not working
**Diagnosis**:
```bash
# Test SSE endpoint
curl -N http://localhost:8000/events

# Check browser network tab
# Verify SSE implementation
```
**Solutions**:
- Check SSE endpoint implementation
- Verify browser SSE support
- Review proxy/firewall settings
- Implement WebSocket fallback
- Test with different browsers

## Advanced Troubleshooting

### Log Analysis

#### Centralized Logging
```bash
# View all logs
tail -f backend/logs/*.log

# Filter by severity
grep "ERROR" backend/logs/*.log
grep "WARNING" backend/logs/*.log

# Search for specific issues
grep -i "memory" backend/logs/*.log
grep -i "timeout" backend/logs/*.log
```

#### Log Rotation and Management
```bash
# Check log sizes
du -sh backend/logs/

# Rotate logs manually
logrotate -f /etc/logrotate.d/hifazat

# Clean old logs
find backend/logs/ -name "*.log.*" -mtime +7 -delete
```

### Database Troubleshooting

#### SQLite Issues
```bash
# Check database integrity
cd backend && sqlite3 hifazat.db "PRAGMA integrity_check;"

# Analyze database size
sqlite3 hifazat.db ".dbinfo"

# Vacuum database
sqlite3 hifazat.db "VACUUM;"
```

#### Query Performance
```bash
# Enable query logging
sqlite3 hifazat.db ".timer on"
sqlite3 hifazat.db "SELECT COUNT(*) FROM alerts;"

# Analyze slow queries
sqlite3 hifazat.db "EXPLAIN QUERY PLAN SELECT * FROM alerts WHERE timestamp > datetime('now', '-1 hour');"
```

### Configuration Debugging

#### Validate Configuration Files
```bash
# Check JSON syntax
cd backend/config && python -m json.tool default_config.json

# Validate YAML files
cd backend/config && python -c "import yaml; yaml.safe_load(open('sample_config.yaml'))"

# Test configuration loading
cd backend && python -c "from utils.config import config_manager; print(config_manager.validate_config())"
```

#### Environment Variables
```bash
# Check environment setup
env | grep -E "(OMP|MKL|NUMEXPR)_NUM_THREADS"
env | grep -E "(VITE_|LOG_|DATABASE_)"

# Test environment loading
cd backend && python -c "import os; print('API URL:', os.getenv('VITE_API_URL', 'Not set'))"
```

## Performance Optimization

### CPU Optimization

#### Thread Configuration
```bash
# Optimal settings for i5 6th gen
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

#### Process Monitoring
```bash
# Monitor CPU usage by process
pidstat -u 1 -p $(pgrep -f "python.*api.main")

# Check thread usage
ps -eLf | grep python | wc -l
```

### Memory Optimization

#### Garbage Collection Tuning
```python
import gc
import os

# Optimize garbage collection
gc.set_threshold(700, 10, 10)
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
```

#### Memory Monitoring
```bash
# Monitor memory usage
watch -n 1 'free -h && ps aux --sort=-%mem | head -10'

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python api/main.py
```

## Emergency Procedures

### System Recovery

#### Complete System Reset
```bash
# Stop all services
docker-compose down

# Clear temporary data
rm -rf backend/logs/*.log
rm -rf backend/media/temp/*

# Reset database (WARNING: Data loss)
rm backend/hifazat.db
cd backend && python init_backend.py

# Restart system
docker-compose up --build
```

#### Partial Service Recovery
```bash
# Restart individual services
docker-compose restart backend
docker-compose restart frontend

# Restart specific pipeline
pkill -f "threat_intelligence"
cd backend && python -m pipelines.threat_intelligence &
```

### Data Recovery

#### Database Backup and Restore
```bash
# Create backup
sqlite3 backend/hifazat.db ".backup backup_$(date +%Y%m%d_%H%M%S).db"

# Restore from backup
cp backup_20240115_103000.db backend/hifazat.db

# Export data
sqlite3 backend/hifazat.db ".mode csv" ".output alerts_export.csv" "SELECT * FROM alerts;"
```

#### Configuration Backup
```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz backend/config/

# Restore configuration
tar -xzf config_backup_20240115.tar.gz
```

## Monitoring and Alerting

### Health Monitoring Setup

#### Automated Health Checks
```bash
#!/bin/bash
# health_check.sh
API_URL="http://localhost:8000"

# Check API health
if ! curl -f "$API_URL/health" > /dev/null 2>&1; then
    echo "API health check failed" | mail -s "HifazatAI Alert" admin@company.com
fi

# Check resource usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "High CPU usage: $CPU_USAGE%" | mail -s "HifazatAI Alert" admin@company.com
fi
```

#### Log Monitoring
```bash
# Monitor for errors
tail -f backend/logs/api.log | grep -i error | while read line; do
    echo "Error detected: $line" | mail -s "HifazatAI Error" admin@company.com
done
```

### Performance Monitoring

#### Metrics Collection
```bash
# Collect system metrics
#!/bin/bash
# metrics_collector.sh
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEM_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | cut -d'%' -f1)

echo "$TIMESTAMP,$CPU_USAGE,$MEM_USAGE,$DISK_USAGE" >> metrics.csv
```

## Getting Additional Help

### Support Resources

#### Documentation
- **User Manual**: Complete operational guide
- **API Documentation**: Technical API reference
- **System Architecture**: Detailed system design
- **Configuration Guide**: Setup and configuration help

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussion Forum**: Community Q&A and tips
- **Video Tutorials**: Step-by-step guides
- **Knowledge Base**: Common solutions and FAQs

#### Professional Support
- **Technical Support**: Direct assistance for critical issues
- **Consulting Services**: Custom implementation and optimization
- **Training Programs**: User and administrator training
- **Maintenance Contracts**: Ongoing support and updates

### Diagnostic Information Collection

When reporting issues, please collect:

#### System Information
```bash
# System details
uname -a
lsb_release -a
python --version
node --version
docker --version

# Resource information
free -h
df -h
lscpu
```

#### Application Logs
```bash
# Collect recent logs
tar -czf hifazat_logs_$(date +%Y%m%d).tar.gz backend/logs/

# Configuration files
tar -czf hifazat_config_$(date +%Y%m%d).tar.gz backend/config/
```

#### Performance Data
```bash
# System performance snapshot
top -bn1 > system_performance.txt
ps aux > process_list.txt
netstat -tulpn > network_status.txt
```

This troubleshooting guide provides comprehensive solutions for common issues and advanced diagnostic procedures. For issues not covered here, please consult the additional documentation or contact support with the diagnostic information collected using the procedures above.