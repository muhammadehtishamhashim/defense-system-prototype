"""
CPU Performance Benchmark for Video Surveillance Pipeline
Tests performance on i5 6th gen CPU and provides optimization recommendations.
"""

import time
import psutil
import numpy as np
import cv2
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import threading
import queue
from pathlib import Path

from pipelines.video_surveillance.detection import (
    CPUOptimizedDetector, 
    ByteTrackTracker, 
    CPUOptimizedVideoProcessor
)
from pipelines.video_surveillance.analysis import VideoAnalysisPipeline
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("cpu_benchmark")


class CPUBenchmark:
    """Benchmark CPU performance for video surveillance pipeline"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize CPU benchmark
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load benchmark configuration"""
        default_config = {
            "test_duration": 30,  # seconds
            "test_resolutions": [(320, 240), (640, 480), (1280, 720)],
            "test_input_sizes": [320, 416, 640],
            "test_frame_rates": [10, 15, 30],
            "test_with_onnx": [True, False],
            "synthetic_video_length": 100,  # frames
            "memory_monitoring": True,
            "cpu_monitoring": True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            cpu_info = {
                "cpu_count": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available
            }
            
            # Try to get CPU model name
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['cpu_model'] = line.split(':')[1].strip()
                            break
            except:
                cpu_info['cpu_model'] = 'Unknown'
            
            return cpu_info
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {}
    
    def create_synthetic_video(self, resolution: Tuple[int, int], num_frames: int) -> List[np.ndarray]:
        """Create synthetic video frames for testing"""
        frames = []
        width, height = resolution
        
        for i in range(num_frames):
            # Create frame with moving objects
            frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
            
            # Add some moving rectangles (simulate objects)
            num_objects = np.random.randint(1, 5)
            for _ in range(num_objects):
                x = int((i * 2) % (width - 50))
                y = int((i * 1.5) % (height - 50))
                cv2.rectangle(frame, (x, y), (x + 50, y + 50), (255, 255, 255), -1)
            
            frames.append(frame)
        
        return frames
    
    def benchmark_detector(self, resolution: Tuple[int, int], input_size: int, 
                          use_onnx: bool, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Benchmark detector performance"""
        logger.info(f"Benchmarking detector: {resolution}, input_size={input_size}, onnx={use_onnx}")
        
        try:
            detector = CPUOptimizedDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.5,
                use_onnx=use_onnx,
                input_size=input_size
            )
            
            # Warm up
            for _ in range(3):
                detector.detect(frames[0])
            
            # Benchmark
            start_time = time.time()
            detection_times = []
            total_detections = 0
            
            for frame in frames:
                frame_start = time.time()
                detections = detector.detect(frame)
                frame_time = time.time() - frame_start
                
                detection_times.append(frame_time)
                total_detections += len(detections)
            
            total_time = time.time() - start_time
            
            return {
                "resolution": resolution,
                "input_size": input_size,
                "use_onnx": use_onnx,
                "total_time": total_time,
                "avg_frame_time": np.mean(detection_times),
                "min_frame_time": np.min(detection_times),
                "max_frame_time": np.max(detection_times),
                "fps": len(frames) / total_time,
                "total_detections": total_detections,
                "avg_detections_per_frame": total_detections / len(frames)
            }
            
        except Exception as e:
            logger.error(f"Detector benchmark failed: {str(e)}")
            return {"error": str(e)}
    
    def benchmark_tracker(self, frames: List[np.ndarray], frame_rate: int) -> Dict[str, Any]:
        """Benchmark tracker performance"""
        logger.info(f"Benchmarking tracker: frame_rate={frame_rate}")
        
        try:
            detector = CPUOptimizedDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.5,
                use_onnx=True,
                input_size=416
            )
            
            tracker = ByteTrackTracker(
                frame_rate=frame_rate,
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8
            )
            
            # Warm up
            for _ in range(3):
                detections = detector.detect(frames[0])
                tracker.update(detections, frames[0])
            
            # Benchmark
            start_time = time.time()
            tracking_times = []
            total_tracks = 0
            
            for frame in frames:
                detections = detector.detect(frame)
                
                track_start = time.time()
                tracked_objects = tracker.update(detections, frame)
                track_time = time.time() - track_start
                
                tracking_times.append(track_time)
                total_tracks += len(tracked_objects)
            
            total_time = time.time() - start_time
            
            return {
                "frame_rate": frame_rate,
                "total_time": total_time,
                "avg_track_time": np.mean(tracking_times),
                "fps": len(frames) / total_time,
                "total_tracks": total_tracks,
                "avg_tracks_per_frame": total_tracks / len(frames)
            }
            
        except Exception as e:
            logger.error(f"Tracker benchmark failed: {str(e)}")
            return {"error": str(e)}
    
    def benchmark_complete_pipeline(self, resolution: Tuple[int, int], 
                                  frames: List[np.ndarray]) -> Dict[str, Any]:
        """Benchmark complete video analysis pipeline"""
        logger.info(f"Benchmarking complete pipeline: {resolution}")
        
        try:
            detector_config = {
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.5,
                "use_onnx": True,
                "input_size": 416,
                "frame_rate": 15,
                "max_resolution": resolution
            }
            
            analyzer_config = {
                "loitering_threshold": 30,
                "abandoned_threshold": 60
            }
            
            pipeline = VideoAnalysisPipeline(detector_config, analyzer_config)
            
            # Warm up
            for _ in range(3):
                pipeline.process_frame(frames[0])
            
            # Benchmark
            start_time = time.time()
            processing_times = []
            total_alerts = 0
            
            for frame in frames:
                frame_start = time.time()
                results = pipeline.process_frame(frame)
                frame_time = time.time() - frame_start
                
                processing_times.append(frame_time)
                total_alerts += len(results.get('alerts', []))
            
            total_time = time.time() - start_time
            
            return {
                "resolution": resolution,
                "total_time": total_time,
                "avg_processing_time": np.mean(processing_times),
                "fps": len(frames) / total_time,
                "total_alerts": total_alerts,
                "pipeline_stats": pipeline.get_pipeline_status()
            }
            
        except Exception as e:
            logger.error(f"Pipeline benchmark failed: {str(e)}")
            return {"error": str(e)}
    
    def monitor_resources(self, duration: float) -> Dict[str, Any]:
        """Monitor CPU and memory usage during benchmark"""
        cpu_percentages = []
        memory_percentages = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percentages.append(psutil.cpu_percent(interval=0.1))
            memory_percentages.append(psutil.virtual_memory().percent)
            time.sleep(0.1)
        
        return {
            "avg_cpu_percent": np.mean(cpu_percentages),
            "max_cpu_percent": np.max(cpu_percentages),
            "avg_memory_percent": np.mean(memory_percentages),
            "max_memory_percent": np.max(memory_percentages),
            "cpu_samples": len(cpu_percentages)
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        logger.info("Starting full CPU benchmark suite")
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "config": self.config,
            "detector_benchmarks": [],
            "tracker_benchmarks": [],
            "pipeline_benchmarks": [],
            "resource_usage": {}
        }
        
        # Start resource monitoring
        resource_queue = queue.Queue()
        
        def resource_monitor():
            resource_usage = self.monitor_resources(self.config["test_duration"])
            resource_queue.put(resource_usage)
        
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.start()
        
        try:
            # Test different configurations
            for resolution in self.config["test_resolutions"]:
                logger.info(f"Testing resolution: {resolution}")
                
                # Create synthetic video for this resolution
                frames = self.create_synthetic_video(
                    resolution, 
                    self.config["synthetic_video_length"]
                )
                
                # Test detector with different settings
                for input_size in self.config["test_input_sizes"]:
                    for use_onnx in self.config["test_with_onnx"]:
                        result = self.benchmark_detector(
                            resolution, input_size, use_onnx, frames
                        )
                        benchmark_results["detector_benchmarks"].append(result)
                
                # Test tracker
                for frame_rate in self.config["test_frame_rates"]:
                    result = self.benchmark_tracker(frames, frame_rate)
                    benchmark_results["tracker_benchmarks"].append(result)
                
                # Test complete pipeline
                result = self.benchmark_complete_pipeline(resolution, frames)
                benchmark_results["pipeline_benchmarks"].append(result)
        
        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")
            benchmark_results["error"] = str(e)
        
        # Wait for resource monitoring to complete
        monitor_thread.join(timeout=5)
        
        if not resource_queue.empty():
            benchmark_results["resource_usage"] = resource_queue.get()
        
        return benchmark_results
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []
        
        # Analyze detector performance
        detector_results = results.get("detector_benchmarks", [])
        if detector_results:
            # Find best performing configuration
            valid_results = [r for r in detector_results if "error" not in r]
            if valid_results:
                best_fps = max(valid_results, key=lambda x: x.get("fps", 0))
                
                recommendations.append(
                    f"Best detector performance: {best_fps['resolution']} resolution, "
                    f"input_size={best_fps['input_size']}, ONNX={best_fps['use_onnx']} "
                    f"({best_fps['fps']:.1f} FPS)"
                )
                
                # Check if ONNX helps
                onnx_results = [r for r in valid_results if r.get("use_onnx")]
                pytorch_results = [r for r in valid_results if not r.get("use_onnx")]
                
                if onnx_results and pytorch_results:
                    avg_onnx_fps = np.mean([r["fps"] for r in onnx_results])
                    avg_pytorch_fps = np.mean([r["fps"] for r in pytorch_results])
                    
                    if avg_onnx_fps > avg_pytorch_fps * 1.1:
                        recommendations.append("ONNX runtime provides significant performance improvement")
                    else:
                        recommendations.append("ONNX runtime provides minimal benefit on this system")
        
        # Analyze resource usage
        resource_usage = results.get("resource_usage", {})
        if resource_usage:
            avg_cpu = resource_usage.get("avg_cpu_percent", 0)
            max_cpu = resource_usage.get("max_cpu_percent", 0)
            
            if max_cpu > 90:
                recommendations.append("CPU usage is very high - consider reducing resolution or frame rate")
            elif max_cpu > 70:
                recommendations.append("CPU usage is moderate - system can handle current load")
            else:
                recommendations.append("CPU usage is low - system can handle higher resolution or frame rate")
        
        # Analyze pipeline performance
        pipeline_results = results.get("pipeline_benchmarks", [])
        if pipeline_results:
            valid_pipeline_results = [r for r in pipeline_results if "error" not in r]
            if valid_pipeline_results:
                best_pipeline = max(valid_pipeline_results, key=lambda x: x.get("fps", 0))
                
                if best_pipeline["fps"] >= 15:
                    recommendations.append("System can handle real-time processing at 15+ FPS")
                elif best_pipeline["fps"] >= 10:
                    recommendations.append("System can handle near real-time processing at 10+ FPS")
                else:
                    recommendations.append("System may struggle with real-time processing - consider optimizations")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("CPU BENCHMARK SUMMARY")
        print("="*60)
        
        # System info
        system_info = results.get("system_info", {})
        print(f"CPU Model: {system_info.get('cpu_model', 'Unknown')}")
        print(f"CPU Cores: {system_info.get('cpu_count', 'Unknown')} physical, "
              f"{system_info.get('cpu_count_logical', 'Unknown')} logical")
        print(f"Memory: {system_info.get('memory_total', 0) / (1024**3):.1f} GB total")
        
        # Best performance
        detector_results = [r for r in results.get("detector_benchmarks", []) if "error" not in r]
        if detector_results:
            best_detector = max(detector_results, key=lambda x: x.get("fps", 0))
            print(f"\nBest Detector Performance: {best_detector['fps']:.1f} FPS")
            print(f"  Resolution: {best_detector['resolution']}")
            print(f"  Input Size: {best_detector['input_size']}")
            print(f"  ONNX: {best_detector['use_onnx']}")
        
        # Resource usage
        resource_usage = results.get("resource_usage", {})
        if resource_usage:
            print(f"\nResource Usage:")
            print(f"  Average CPU: {resource_usage.get('avg_cpu_percent', 0):.1f}%")
            print(f"  Peak CPU: {resource_usage.get('max_cpu_percent', 0):.1f}%")
            print(f"  Average Memory: {resource_usage.get('avg_memory_percent', 0):.1f}%")
        
        # Recommendations
        recommendations = self.generate_recommendations(results)
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*60)


def main():
    """Run CPU benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CPU Performance Benchmark for Video Surveillance")
    parser.add_argument("--config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="cpu_benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--duration", type=int, default=30,
                       help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = CPUBenchmark(args.config)
    
    if args.duration:
        benchmark.config["test_duration"] = args.duration
    
    # Run benchmark
    print("Starting CPU benchmark... This may take several minutes.")
    results = benchmark.run_full_benchmark()
    
    # Save and display results
    benchmark.save_results(results, args.output)
    benchmark.print_summary(results)


if __name__ == "__main__":
    main()