import cv2
import time
import os
import asyncio
from inference_sdk import InferenceHTTPClient
import supervision as sv
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

class DetectionEngine:
    def __init__(self, source_path: str, model_id: str, frame_stride: int = 30):
        self.source_path = source_path
        self.model_id = model_id
        self.frame_stride = frame_stride  # Process every Nth frame
        
        # Initialize Client
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    async def run(self):
        """
        Async generator that yields (processed_frame_bytes, detection_count, detected_classes)
        """
        # We use a loop to restart the video when it ends to simulate 24/7 stream
        while True:
            cap = cv2.VideoCapture(self.source_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30 # Default fallback
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Logic: Only process inference every `frame_stride` frames
                # But we want to stream VIDEO smoothly-ish.
                # If we only process 1 FPS, we should still yield the other frames, just without NEW detections?
                # Or just yield the processed frames at 1 FPS (User said "Process only 1 frame per second... but keep video stream smooth if possible")
                
                # Strategy: 
                # Run inference only if frame_count % frame_stride == 0.
                # Otherwise, display the *last known* detections on the current frame.
                
                detections = None
                
                # NOTE: For improved performance in async/FastAPI, 
                # we should run the blocking inference in a thread.
                if frame_count % self.frame_stride == 0:
                    # Run Inference
                    try:
                        # Wrapping blocking call to avoid blocking the event loop
                        result = await asyncio.to_thread(
                            self.client.infer, 
                            frame, 
                            self.model_id
                        )
                        detections = sv.Detections.from_inference(result)
                        self.last_detections = detections # Cache detections
                    except Exception as e:
                        print(f"Inference error: {e}")
                else:
                    # Use cached detections
                    detections = getattr(self, 'last_detections', None)

                # Annotate Frame
                if detections:
                    annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
                    annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)
                else:
                    annotated_frame = frame

                # Encode to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                
                # Count current detections
                count = len(detections) if detections else 0
                
                yield frame_bytes, count
                
                frame_count += 1
                
                # Control framerate (Sleep to match video FPS)
                # processing_time = time.time() - start_time
                # delay = max(0, (1.0 / fps) - processing_time)
                # await asyncio.sleep(delay) 
                
                # Since we are in an async loop and this is CPU bound, strict sleep might not be precise, 
                # but good enough for a demo.
                await asyncio.sleep(0.01)

            cap.release()
            print(f"Video {self.source_path} ended, restarting...")
