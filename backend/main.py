import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dictionary to hold the latest frame for each source
latest_frames: Dict[str, bytes] = {}
# Dictionary to hold the latest stats
latest_stats: Dict[str, int] = {
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

# Background Detection Tasks
async def detection_loop(source: str, model_id: str):
    print(f"Starting detection loop for {source} with model {model_id}")
    video_path = VIDEO_PATHS.get(source)
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video not found for {source} at {video_path}")
        return

    engine = DetectionEngine(source_path=video_path, model_id=model_id)
    
from fastapi.staticfiles import StaticFiles

# ... existing imports ...

# Mount snapshots


# ... existing code ...

# Throttling snapshots
last_snapshot_time = {}

# Background Detection Tasks
async def detection_loop(source: str, model_id: str):
    print(f"Starting detection loop for {source} with model {model_id}")
    video_path = VIDEO_PATHS.get(source)
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video not found for {source} at {video_path}")
        return

    engine = DetectionEngine(source_path=video_path, model_id=model_id)
    
    async for frame_bytes, count in engine.run():
        # Update global state
        latest_frames[source] = frame_bytes
        
        # Update stats...
        if source == "threat":
            latest_stats["threats"] = count
        elif source == "theft":
            latest_stats["thefts"] = count
        elif source == "border":
            latest_stats["border_anomalies"] = count
            
        # Broadcast and Snapshot
        if count > 0:
            now = asyncio.get_event_loop().time()
            # Throttle snapshot saving: once every 1.0 seconds per source
            if now - last_snapshot_time.get(source, 0) > 1.0:
                timestamp = int(now)
                filename = f"{source}_{timestamp}.jpg"
                filepath = os.path.join("snapshots", filename)
                # Run in thread to not block
                await asyncio.to_thread(write_snapshot, filepath, frame_bytes)
                last_snapshot_time[source] = now
                
                # Send alert with snapshot URL
                await manager.broadcast({
                    "type": "alert",
                    "source": source,
                    "count": count,
                    "timestamp": timestamp,
                    "snapshot": f"/snapshots/{filename}"
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
    return {"message": "Defense System v3 Brain is Active"}

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
    return latest_stats

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
