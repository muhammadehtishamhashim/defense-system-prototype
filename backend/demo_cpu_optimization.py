#!/usr/bin/env python3
"""
Demo script for CPU-optimized video surveillance pipeline.
Demonstrates YOLOv8 Nano with ONNX optimization and ByteTrack tracking on i5 6th gen CPU.
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path

from pipelines.video_surveillance.detection import (
    CPUOptimizedDetector,
    ByteTrackTracker, 
    CPUOptimizedVideoProcessor,
    CPUOptimizedVideoSource
)
from pipelines.video_surveillance.analysis import VideoAnalysisPipeline
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("cpu_demo")


def create_demo_video(output_path: str, duration: int = 10, fps: int = 15):
    """Create a demo video with moving objects for testing"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
        
        # Add moving objects (simulate people/vehicles)
        for obj_id in range(3):
            # Calculate object position (moving in different patterns)
            if obj_id == 0:  # Linear movement
                x = int((frame_num * 2) % (width - 100))
                y = height // 2
            elif obj_id == 1:  # Circular movement
                angle = (frame_num * 0.1) % (2 * np.pi)
                x = int(width // 2 + 150 * np.cos(angle))
                y = int(height // 2 + 100 * np.sin(angle))
            else:  # Diagonal movement
                x = int((frame_num * 1.5) % (width - 80))
                y = int((frame_num * 1) % (height - 80))
            
            # Draw object (rectangle to simulate person/vehicle)
            color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][obj_id]
            cv2.rectangle(frame, (x, y), (x + 60, y + 80), color, -1)
            
            # Add some noise/texture
            cv2.rectangle(frame, (x + 10, y + 10), (x + 50, y + 30), (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Demo video created: {output_path}")


def run_cpu_optimization_demo(video_source: str = None, show_video: bool = False):
    """Run CPU optimization demo"""
    
    print("="*60)
    print("CPU-OPTIMIZED VIDEO SURVEILLANCE DEMO")
    print("="*60)
    print("Optimizations for i5 6th gen CPU:")
    print("- YOLOv8 Nano model (smallest, fastest)")
    print("- ONNX Runtime for CPU inference")
    print("- Frame skipping (process every 2nd frame)")
    print("- Reduced input resolution (416x416)")
    print("- ByteTrack lightweight tracking")
    print("- Multi-threading optimization (4 cores)")
    print("="*60)
    
    # Create demo video if no source provided
    if video_source is None:
        video_source = "demo_video.mp4"
        if not Path(video_source).exists():
            print("Creating demo video...")
            create_demo_video(video_source, duration=15, fps=15)
    
    # Initialize CPU-optimized components
    print("\nInitializing CPU-optimized components...")
    
    detector_config = {
        'model_path': 'yolov8n.pt',
        'confidence_threshold': 0.5,
        'use_onnx': True,  # Enable ONNX for CPU optimization
        'input_size': 416,
        'frame_rate': 15,
        'max_resolution': (640, 480)
    }
    
    analyzer_config = {
        'loitering_threshold': 30,
        'abandoned_threshold': 60
    }
    
    # Create pipeline
    pipeline = VideoAnalysisPipeline(detector_config, analyzer_config)
    
    # Create video source
    video_source_obj = CPUOptimizedVideoSource(
        source=video_source,
        target_fps=15,
        max_resolution=(640, 480)
    )
    
    print(f"Video source: {video_source}")
    print(f"Source info: {video_source_obj.get_source_info()}")
    
    # Processing loop
    frame_count = 0
    total_processing_time = 0
    detection_count = 0
    track_count = 0
    
    print("\nStarting video processing...")
    print("Press 'q' to quit (if showing video)")
    
    try:
        while True:
            # Read frame
            frame = video_source_obj.read_frame()
            if frame is None:
                break
            
            frame_count += 1
            
            # Process frame
            start_time = time.time()
            results = pipeline.process_frame(frame)
            processing_time = time.time() - start_time
            
            total_processing_time += processing_time
            detection_count += results.get('detections', 0)
            track_count += results.get('tracks', 0)
            
            # Display results
            if frame_count % 30 == 0:  # Print stats every 30 frames
                avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
                print(f"Frame {frame_count}: {avg_fps:.1f} FPS, "
                      f"{results.get('detections', 0)} detections, "
                      f"{results.get('tracks', 0)} tracks, "
                      f"{len(results.get('alerts', []))} alerts")
            
            # Show video if requested
            if show_video:
                # Draw detection info on frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Detections: {results.get('detections', 0)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Tracks: {results.get('tracks', 0)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('CPU-Optimized Video Surveillance', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Limit demo to reasonable length
            if frame_count >= 300:  # Stop after 300 frames
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        video_source_obj.release()
        if show_video:
            cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*60)
    print("DEMO RESULTS")
    print("="*60)
    
    if total_processing_time > 0:
        avg_fps = frame_count / total_processing_time
        avg_frame_time = total_processing_time / frame_count * 1000  # ms
        
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average frame processing time: {avg_frame_time:.1f} ms")
        print(f"Total detections: {detection_count}")
        print(f"Total tracks: {track_count}")
        
        # Performance assessment
        if avg_fps >= 15:
            print("✅ EXCELLENT: Real-time performance achieved (15+ FPS)")
        elif avg_fps >= 10:
            print("✅ GOOD: Near real-time performance (10+ FPS)")
        elif avg_fps >= 5:
            print("⚠️  ACCEPTABLE: Usable performance (5+ FPS)")
        else:
            print("❌ POOR: Performance below acceptable threshold")
        
        # CPU optimization effectiveness
        pipeline_stats = pipeline.get_pipeline_status()
        print(f"\nPipeline Statistics:")
        print(f"- ONNX Runtime: {'Enabled' if detector_config.get('use_onnx') else 'Disabled'}")
        print(f"- Input Size: {detector_config.get('input_size')}x{detector_config.get('input_size')}")
        print(f"- Frame Skip Rate: {pipeline.processor.detector.frame_skip}")
        print(f"- Max Resolution: {detector_config.get('max_resolution')}")
        
    print("="*60)


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="CPU-Optimized Video Surveillance Demo")
    parser.add_argument("--video", help="Path to video file (creates demo video if not provided)")
    parser.add_argument("--show", action="store_true", help="Show video window during processing")
    parser.add_argument("--camera", type=int, help="Use camera index (e.g., 0 for default camera)")
    
    args = parser.parse_args()
    
    # Determine video source
    video_source = None
    if args.camera is not None:
        video_source = args.camera
        print(f"Using camera index: {args.camera}")
    elif args.video:
        video_source = args.video
        print(f"Using video file: {args.video}")
    
    # Run demo
    run_cpu_optimization_demo(video_source, args.show)


if __name__ == "__main__":
    main()