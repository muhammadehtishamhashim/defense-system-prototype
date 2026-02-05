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
        Async generator that yields (processed_frame_bytes, detection_count, max_confidence)
        """
        while True:
            cap = cv2.VideoCapture(self.source_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30 # Default fallback
            
            # User wants "1 frame of 1 sec prediction" -> Run inference once per second
            # So stride should be equal to FPS
            self.frame_stride = int(fps)
            
            frame_time = 1.0 / fps
            frame_count = 0
            
            while cap.isOpened():
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = None
                
                # Logic: Run inference once per second (approx)
                if frame_count % self.frame_stride == 0:
                    try:
                        # Wrapping blocking call
                        result = await asyncio.to_thread(
                            self.client.infer, 
                            frame, 
                            self.model_id
                        )
                        raw_detections = sv.Detections.from_inference(result)
                        # Filter by confidence > 0.75
                        self.last_detections = raw_detections[raw_detections.confidence > 0.75]
                    except Exception as e:
                        print(f"Inference error: {e}")
                else:
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
                
                # Count detections
                count = len(detections) if detections else 0
                
                # Get max confidence
                confidence = 0.0
                if count > 0:
                    try:
                        # detections.confidence is a numpy array
                        confidence = float(detections.confidence.max())
                    except:
                        confidence = 0.0
                
                yield frame_bytes, count, confidence
                
                frame_count += 1
                
                # Enforce Real-Time Playback (Prevent Fast Forward)
                processing_time = time.time() - start_time
                delay = max(0.001, frame_time - processing_time)
                await asyncio.sleep(delay)

            cap.release()
            print(f"Video {self.source_path} ended, restarting...")
