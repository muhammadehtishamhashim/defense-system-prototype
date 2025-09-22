#!/usr/bin/env python3
"""
Demo script for CPU-optimized anomaly detection models.
Showcases the enhanced anomaly detection capabilities with synthetic data.
"""

import time
import warnings
from typing import List, Dict, Any

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

from pipelines.border_anomaly.anomaly_detector import (
    IsolationForestDetector, MotionBasedDetector, PyODEnsembleDetector,
    create_synthetic_anomaly_data, AnomalyResult
)
from pipelines.border_anomaly.evaluation import AnomalyDetectionEvaluator
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("anomaly_demo")


def demo_cpu_optimized_detection():
    """Demonstrate CPU-optimized anomaly detection"""
    print("=" * 80)
    print("CPU-OPTIMIZED BORDER ANOMALY DETECTION DEMO")
    print("=" * 80)
    print()
    
    # Generate synthetic test data
    print("🔄 Generating synthetic trajectory data...")
    trajectories, labels = create_synthetic_anomaly_data(num_normal=30, num_anomalies=8)
    normal_trajectories = [t for t, l in zip(trajectories, labels) if not l]
    anomaly_trajectories = [t for t, l in zip(trajectories, labels) if l]
    
    print(f"✅ Generated {len(trajectories)} trajectories:")
    print(f"   • Normal trajectories: {len(normal_trajectories)}")
    print(f"   • Anomalous trajectories: {len(anomaly_trajectories)}")
    print()
    
    # Test each detector
    detectors = {
        "CPU-Optimized Isolation Forest": IsolationForestDetector(
            contamination=0.2,
            n_estimators=50,
            n_jobs=-1
        ),
        "Enhanced Motion-Based (Adaptive)": MotionBasedDetector(
            adaptive_thresholds=True
        ),
        "PyOD Ensemble": None  # Will be created if available
    }
    
    # Add PyOD if available
    try:
        detectors["PyOD Ensemble"] = PyODEnsembleDetector(
            contamination=0.2,
            combination_method='average'
        )
    except ImportError:
        print("⚠️  PyOD not available, skipping PyOD ensemble demo")
        del detectors["PyOD Ensemble"]
    
    results = {}
    
    for name, detector in detectors.items():
        if detector is None:
            continue
            
        print(f"🧠 Testing {name}...")
        
        # Training
        start_time = time.time()
        detector.fit(normal_trajectories)
        training_time = time.time() - start_time
        
        # Prediction on test data
        start_time = time.time()
        predictions = []
        for trajectory in trajectories:
            result = detector.predict(trajectory)
            predictions.append(result)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        true_positives = sum(1 for r, l in zip(predictions, labels) if r.is_anomaly and l)
        false_positives = sum(1 for r, l in zip(predictions, labels) if r.is_anomaly and not l)
        true_negatives = sum(1 for r, l in zip(predictions, labels) if not r.is_anomaly and not l)
        false_negatives = sum(1 for r, l in zip(predictions, labels) if not r.is_anomaly and l)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'detected_anomalies': sum(1 for r in predictions if r.is_anomaly)
        }
        
        print(f"   ✅ Training time: {training_time:.3f}s")
        print(f"   ✅ Prediction time: {prediction_time:.3f}s")
        print(f"   ✅ Detected anomalies: {results[name]['detected_anomalies']}/{len(trajectories)}")
        print(f"   ✅ Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        print()
    
    # Summary
    print("📊 PERFORMANCE SUMMARY")
    print("-" * 50)
    
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
    fastest = min(results.items(), key=lambda x: x[1]['training_time'] + x[1]['prediction_time'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    
    print(f"🏆 Best F1-Score: {best_f1[0]} ({best_f1[1]['f1_score']:.3f})")
    print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['training_time'] + fastest[1]['prediction_time']:.3f}s total)")
    print(f"🎯 Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.3f})")
    print()


def demo_anomaly_types():
    """Demonstrate detection of different anomaly types"""
    print("🔍 ANOMALY TYPE DETECTION DEMO")
    print("-" * 50)
    
    # Create detector
    detector = MotionBasedDetector(adaptive_thresholds=False)
    
    # Generate different types of anomalies
    trajectories, labels = create_synthetic_anomaly_data(num_normal=5, num_anomalies=15)
    anomaly_trajectories = [t for t, l in zip(trajectories, labels) if l]
    
    print(f"Analyzing {len(anomaly_trajectories)} anomalous trajectories...")
    print()
    
    for i, trajectory in enumerate(anomaly_trajectories[:5]):  # Show first 5
        result = detector.predict(trajectory)
        
        print(f"Trajectory {trajectory.track_id}:")
        print(f"  • Anomaly detected: {result.is_anomaly}")
        print(f"  • Confidence: {result.confidence:.3f}")
        print(f"  • Anomaly score: {result.anomaly_score:.3f}")
        
        if 'anomaly_reasons' in result.details:
            reasons = result.details['anomaly_reasons']
            if reasons:
                print(f"  • Reasons: {', '.join(reasons)}")
        
        if 'features' in result.details:
            features = result.details['features']
            print(f"  • Max speed: {features.get('max_speed', 0):.1f} px/s")
            print(f"  • Direction changes: {features.get('direction_changes', 0)}")
            print(f"  • Path curvature: {features.get('path_curvature', 0):.3f}")
        
        print()


def demo_threshold_tuning():
    """Demonstrate adaptive threshold tuning"""
    print("⚙️  ADAPTIVE THRESHOLD TUNING DEMO")
    print("-" * 50)
    
    # Generate training data
    training_trajectories, _ = create_synthetic_anomaly_data(num_normal=25, num_anomalies=0)
    
    # Create adaptive detector
    adaptive_detector = MotionBasedDetector(adaptive_thresholds=True)
    fixed_detector = MotionBasedDetector(adaptive_thresholds=False)
    
    print("Training adaptive detector on normal trajectories...")
    adaptive_detector.fit(training_trajectories)
    
    print("\nComputed adaptive thresholds:")
    for key, value in adaptive_detector.computed_thresholds.items():
        default_value = getattr(adaptive_detector, f"{key}_threshold", "N/A")
        print(f"  • {key}: {value:.3f} (default: {default_value})")
    
    print("\nFixed thresholds:")
    print(f"  • speed: {fixed_detector.speed_threshold}")
    print(f"  • direction_changes: {fixed_detector.direction_change_threshold}")
    print(f"  • curvature: {fixed_detector.curvature_threshold}")
    print(f"  • stop_duration: {fixed_detector.stop_duration_threshold}")
    
    # Test on new data
    test_trajectories, test_labels = create_synthetic_anomaly_data(num_normal=10, num_anomalies=5)
    
    adaptive_results = [adaptive_detector.predict(t) for t in test_trajectories]
    fixed_results = [fixed_detector.predict(t) for t in test_trajectories]
    
    adaptive_detected = sum(1 for r in adaptive_results if r.is_anomaly)
    fixed_detected = sum(1 for r in fixed_results if r.is_anomaly)
    
    print(f"\nTest results on {len(test_trajectories)} trajectories ({sum(test_labels)} actual anomalies):")
    print(f"  • Adaptive detector: {adaptive_detected} anomalies detected")
    print(f"  • Fixed detector: {fixed_detected} anomalies detected")
    print()


if __name__ == "__main__":
    try:
        demo_cpu_optimized_detection()
        demo_anomaly_types()
        demo_threshold_tuning()
        
        print("🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ CPU-optimized Isolation Forest with reduced estimators")
        print("✅ PyOD-based ensemble detector for baseline comparison")
        print("✅ Enhanced motion-based detection with adaptive thresholds")
        print("✅ Comprehensive anomaly scoring and threshold tuning")
        print("✅ Synthetic anomaly data generation for testing")
        print("✅ Performance evaluation and comparison metrics")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise