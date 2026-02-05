import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dictionary to hold the latest frame for each source
latest_frames: Dict[str, bytes] = {}
# Verified stats (only incremented by user action)
verified_stats: Dict[str, int] = {
    "threats": 0,
    "thefts": 0,
    "border_anomalies": 0
}

# Manager for active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass # Handle disconnection gracefully

manager = ConnectionManager()

from detection import DetectionEngine

VIDEO_DIR = "/mnt/mydata/defense-system-v2/app/videos"
VIDEO_PATHS = {
    "theft": f"{VIDEO_DIR}/theft.mp4",
    "threat": f"{VIDEO_DIR}/Is Your Grip Tight - T.REX ARMS (360p, h264).mp4",
    "border": f"{VIDEO_DIR}/theft.mp4" # Placeholder
}

# Throttling snapshots (Debounce logic)
last_alert_time = {}

# State to control which models are running inference
active_monitors: Dict[str, bool] = {
    "threat": False,
    "theft": False,
    "border": False
}

# Background Detection Tasks
async def detection_loop(source: str, model_id: str):
    print(f"Starting detection loop for {source} with model {model_id}")
    video_path = VIDEO_PATHS.get(source)
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video not found for {source} at {video_path}")
        return

    engine = DetectionEngine(source_path=video_path, model_id=model_id)
    
    # Engine yields: frame_bytes, count, confidence
    async for frame_bytes, count, confidence in engine.run():
        # Always update video feed even if inference is "off" (engine currently runs inference every stride)
        # TODO: Ideally engine should support pausing inference. 
        # For now, we just don't process the alert if monitor is off.
        latest_frames[source] = frame_bytes

        if not active_monitors.get(source, False):
            continue

        # Strict Verification Workflow:
        # 1. Detection > 0.75 confidence (handled in engine)
        # 2. Debounce: Only 1 alert every 5 seconds per source
        
        if count > 0:
            now = time.time()
            # 5-second cooldown
            if now - last_alert_time.get(source, 0) > 5.0:
                timestamp = int(now)
                # Filename: source_timestamp_confidence.jpg
                # confidence is 0.0 to 1.0, we can save it as integer percent or float string
                conf_str = f"{confidence:.2f}"
                filename = f"{source}_{timestamp}_{conf_str}.jpg"
                filepath = os.path.join("snapshots", filename)
                
                # Save snapshot
                await asyncio.to_thread(write_snapshot, filepath, frame_bytes)
                last_alert_time[source] = now
                
                # Send "PENDING" alert. Stats DO NOT update yet.
                await manager.broadcast({
                    "type": "alert",
                    "source": source,
                    "count": count,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "snapshot": f"/snapshots/{filename}",
                    "status": "pending"
                })

def write_snapshot(path, data):
    with open(path, "wb") as f:
        f.write(data)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background tasks
    tasks = [
        asyncio.create_task(detection_loop("threat", os.getenv("THREAT_MODEL_ID", "threat-model"))),
        asyncio.create_task(detection_loop("theft", os.getenv("THEFT_MODEL_ID", "theft-model"))),
        asyncio.create_task(detection_loop("border", os.getenv("BORDER_MODEL_ID", "border-model"))),
    ]
    yield
    # Shutdown: Cancel tasks
    for task in tasks:
        task.cancel()

app = FastAPI(lifespan=lifespan)

# Mount snapshots
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

from pydantic import BaseModel

@app.get("/snapshots-list")
async def get_pending_snapshots():
    # List all jpg files in snapshots/ that are NOT in verified/ (subdirectories are excluded by os.listdir usually, but let's be safe)
    files = []
    if os.path.exists("snapshots"):
        for f in os.listdir("snapshots"):
            if f.endswith(".jpg"):
                # Parse timestamp from filename: source_timestamp.jpg (e.g. theft_172345.jpg)
                try:
                    # Expected: source_timestamp_confidence.jpg
                    # But also handle old format: source_timestamp.jpg (confidence = 0)
                    name_part = f.rsplit(".", 1)[0] # remove extension
                    parts = name_part.split("_")
                    
                    if len(parts) >= 3:
                        source = parts[0]
                        timestamp = int(parts[1])
                        confidence = float(parts[2])
                    else:
                        # Old format fallback
                        source = parts[0]
                        timestamp = int(parts[1])
                        confidence = 0.0

                    files.append({
                        "source": source,
                        "timestamp": timestamp,
                        "confidence": confidence,
                        "snapshot": f"/snapshots/{f}",
                        "filename": f 
                    })
                except Exception:
                    continue
    # Sort by timestamp desc
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return files

class VerifyRequest(BaseModel):
    source: str
    count: int = 1
    filename: str

@app.post("/verify")
async def verify_alert(req: VerifyRequest):
    # 1. Move file to verified/
    source_path = os.path.join("snapshots", req.filename)
    dest_path = os.path.join("snapshots", "verified", req.filename)
    
    if os.path.exists(source_path):
        os.rename(source_path, dest_path)
    
    # 2. Increment global stats
    if req.source == "threat":
        verified_stats["threats"] += req.count
    elif req.source == "theft":
        verified_stats["thefts"] += req.count
    elif req.source == "border":
        verified_stats["border_anomalies"] += req.count
    
    return {"status": "verified", "stats": verified_stats}

class DismissRequest(BaseModel):
    filename: str

@app.post("/dismiss")
async def dismiss_alert(req: DismissRequest):
    # Delete the file
    path = os.path.join("snapshots", req.filename)
    if os.path.exists(path):
        os.remove(path)
    return {"status": "dismissed"}

# CORS Config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hifazat AI Brain is Active"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive, maybe listen for client commands
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/stats")
async def get_stats():
    # Return VERIFIED stats only
    return verified_stats

@app.post("/control/{source}/{action}")
async def control_monitor(source: str, action: str):
    if source not in active_monitors:
        return {"error": "Invalid source"}
    
    if action == "start":
        active_monitors[source] = True
    elif action == "stop":
        active_monitors[source] = False
    else:
        return {"error": "Invalid action (start/stop)"}

    return {"source": source, "active": active_monitors[source]}

@app.get("/status")
async def get_monitor_status():
    return active_monitors

# Endpoint to stream video
def generate_mjpeg(source: str):
    while True:
        frame = latest_frames.get(source)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            import time
            time.sleep(0.04) # Limit to approx 25 FPS
        else:
            # Return a blank frame or wait if no frame available yet
            import time
            time.sleep(0.1)

@app.get("/video_feed/{source}")
async def video_feed(source: str):
    return StreamingResponse(generate_mjpeg(source), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
