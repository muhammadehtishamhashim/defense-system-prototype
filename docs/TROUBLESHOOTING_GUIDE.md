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