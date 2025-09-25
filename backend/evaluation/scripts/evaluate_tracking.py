#!/usr/bin/env python3
"""
Multi-Object Tracking Evaluation Script
Evaluates tracking performance using MOTA, MOTP, IDF1 and other MOT metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import motmetrics as mm

@dataclass
class TrackingResult:
    """Container for tracking results."""
    frame_id: int
    track_id: int
    bbox: List[float]  # [x, y, width, height]
    confidence: float

@dataclass
class TrackingMetrics:
    """Container for tracking evaluation metrics."""
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision
    idf1: float  # ID F1 Score
    mt: int      # Mostly Tracked trajectories
    ml: int      # Mostly Lost trajectories
    pt: int      # Partially Tracked trajectories
    fp: int      # False Positives
    fn: int      # False Negatives
    ids: int     # ID Switches
    frag: int    # Fragmentations
    num_frames: int
    num_objects: int

class TrackingEvaluator:
    """Evaluates multi-object tracking performance."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.distance_threshold = self.config.get('distance_threshold', 0.5)
        self.metrics_to_calculate = self.config.get('metrics', [
            'MOTA', 'MOTP', 'IDF1', 'MT', 'ML', 'FP', 'FN', 'IDs'
        ])
    
    def load_ground_truth(self, gt_file: str) -> pd.DataFrame:
        """Load ground truth tracking data in MOT format."""
        try:
            # MOT format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
            gt_data = pd.read_csv(gt_file, header=None, names=[
                'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                'conf', 'x', 'y', 'z'
            ])
            
            # Filter out non-pedestrian classes and low confidence detections
            gt_data = gt_data[gt_data['conf'] == 1]  # Only consider positive samples
            gt_data = gt_data[gt_data['id'] > 0]     # Valid track IDs
            
            return gt_data
        except Exception as e:
            print(f"Failed to load ground truth: {e}")
            return pd.DataFrame()
    
    def load_tracking_results(self, results_file: str) -> pd.DataFrame:
        """Load tracking results in MOT format."""
        try:
            # Same format as ground truth
            results_data = pd.read_csv(results_file, header=None, names=[
                'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                'conf', 'x', 'y', 'z'
            ])
            
            return results_data
        except Exception as e:
            print(f"Failed to load tracking results: {e}")
            return pd.DataFrame()
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_motmetrics_accumulator(self, gt_data: pd.DataFrame, 
                                    results_data: pd.DataFrame) -> mm.MOTAccumulator:
        """Create MOTMetrics accumulator for evaluation."""
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Get all frames
        all_frames = sorted(set(gt_data['frame'].unique()) | set(results_data['frame'].unique()))
        
        for frame in all_frames:
            # Get ground truth for this frame
            gt_frame = gt_data[gt_data['frame'] == frame]
            gt_ids = gt_frame['id'].values
            gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            
            # Get predictions for this frame
            pred_frame = results_data[results_data['frame'] == frame]
            pred_ids = pred_frame['id'].values
            pred_boxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            
            # Calculate distance matrix (1 - IoU)
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                distances = np.zeros((len(gt_boxes), len(pred_boxes)))
                for i, gt_box in enumerate(gt_boxes):
                    for j, pred_box in enumerate(pred_boxes):
                        iou = self.calculate_iou(gt_box, pred_box)
                        distances[i, j] = 1 - iou  # Convert IoU to distance
                
                # Update accumulator
                acc.update(gt_ids, pred_ids, distances)
            elif len(gt_boxes) > 0:
                # Only ground truth objects (all missed)
                acc.update(gt_ids, [], [])
            elif len(pred_boxes) > 0:
                # Only predicted objects (all false positives)
                acc.update([], pred_ids, [])
            else:
                # No objects in this frame
                acc.update([], [], [])
        
        return acc
    
    def calculate_tracking_metrics(self, gt_data: pd.DataFrame, 
                                 results_data: pd.DataFrame) -> TrackingMetrics:
        """Calculate comprehensive tracking metrics."""
        # Create MOTMetrics accumulator
        acc = self.create_motmetrics_accumulator(gt_data, results_data)
        
        # Calculate metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=self.metrics_to_calculate, 
                           name='sequence')
        
        # Extract metrics
        mota = summary['MOTA'].iloc[0] if 'MOTA' in summary.columns else 0.0
        motp = summary['MOTP'].iloc[0] if 'MOTP' in summary.columns else 0.0
        idf1 = summary['IDF1'].iloc[0] if 'IDF1' in summary.columns else 0.0
        mt = int(summary['MT'].iloc[0]) if 'MT' in summary.columns else 0
        ml = int(summary['ML'].iloc[0]) if 'ML' in summary.columns else 0
        pt = int(summary['PT'].iloc[0]) if 'PT' in summary.columns else 0
        fp = int(summary['FP'].iloc[0]) if 'FP' in summary.columns else 0
        fn = int(summary['FN'].iloc[0]) if 'FN' in summary.columns else 0
        ids = int(summary['IDs'].iloc[0]) if 'IDs' in summary.columns else 0
        frag = int(summary['Frag'].iloc[0]) if 'Frag' in summary.columns else 0
        
        # Additional statistics
        num_frames = len(set(gt_data['frame'].unique()) | set(results_data['frame'].unique()))
        num_objects = len(gt_data['id'].unique())
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            idf1=idf1,
            mt=mt,
            ml=ml,
            pt=pt,
            fp=fp,
            fn=fn,
            ids=ids,
            frag=frag,
            num_frames=num_frames,
            num_objects=num_objects
        )
    
    def evaluate_sequence(self, sequence_dir: str, results_file: str, 
                         output_dir: str) -> TrackingMetrics:
        """Evaluate tracking on a single sequence."""
        sequence_path = Path(sequence_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load ground truth
        gt_file = sequence_path / "gt" / "gt.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        gt_data = self.load_ground_truth(str(gt_file))
        if gt_data.empty:
            raise ValueError("Failed to load ground truth data")
        
        # Load tracking results
        if not Path(results_file).exists():
            # Create dummy results for demonstration
            print(f"Results file not found: {results_file}")
            print("Creating dummy tracking results for demonstration...")
            results_data = self.create_dummy_results(gt_data)
        else:
            results_data = self.load_tracking_results(results_file)
        
        # Calculate metrics
        metrics = self.calculate_tracking_metrics(gt_data, results_data)
        
        # Save results
        self.save_results(metrics, output_path, sequence_path.name)
        self.create_visualizations(metrics, gt_data, results_data, output_path)
        
        return metrics
    
    def create_dummy_results(self, gt_data: pd.DataFrame) -> pd.DataFrame:
        """Create dummy tracking results for demonstration."""
        results = []
        
        # Add some noise and ID switches to ground truth to simulate tracking results
        for _, row in gt_data.iterrows():
            # Add some noise to bounding boxes
            noise_scale = 0.1
            bb_left = row['bb_left'] + np.random.normal(0, noise_scale * row['bb_width'])
            bb_top = row['bb_top'] + np.random.normal(0, noise_scale * row['bb_height'])
            bb_width = row['bb_width'] * (1 + np.random.normal(0, noise_scale))
            bb_height = row['bb_height'] * (1 + np.random.normal(0, noise_scale))
            
            # Occasionally switch IDs or miss detections
            if np.random.random() < 0.05:  # 5% chance of ID switch
                track_id = row['id'] + np.random.randint(1, 10)
            elif np.random.random() < 0.1:  # 10% chance of missed detection
                continue
            else:
                track_id = row['id']
            
            results.append({
                'frame': row['frame'],
                'id': track_id,
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'conf': 0.8 + np.random.random() * 0.2,  # Random confidence 0.8-1.0
                'x': -1,
                'y': -1,
                'z': -1
            })
        
        # Add some false positive detections
        max_frame = gt_data['frame'].max()
        for _ in range(int(len(gt_data) * 0.05)):  # 5% false positives
            results.append({
                'frame': np.random.randint(1, max_frame + 1),
                'id': np.random.randint(1000, 2000),  # High ID to avoid conflicts
                'bb_left': np.random.randint(0, 1000),
                'bb_top': np.random.randint(0, 600),
                'bb_width': np.random.randint(50, 200),
                'bb_height': np.random.randint(100, 300),
                'conf': 0.5 + np.random.random() * 0.3,
                'x': -1,
                'y': -1,
                'z': -1
            })
        
        return pd.DataFrame(results)
    
    def save_results(self, metrics: TrackingMetrics, output_dir: Path, sequence_name: str):
        """Save tracking evaluation results."""
        results = {
            'sequence': sequence_name,
            'metrics': {
                'MOTA': metrics.mota,
                'MOTP': metrics.motp,
                'IDF1': metrics.idf1,
                'MT': metrics.mt,
                'ML': metrics.ml,
                'PT': metrics.pt,
                'FP': metrics.fp,
                'FN': metrics.fn,
                'IDs': metrics.ids,
                'Frag': metrics.frag,
                'num_frames': metrics.num_frames,
                'num_objects': metrics.num_objects
            },
            'configuration': self.config
        }
        
        with open(output_dir / 'tracking_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_dir / 'tracking_results.json'}")
    
    def create_visualizations(self, metrics: TrackingMetrics, gt_data: pd.DataFrame,
                            results_data: pd.DataFrame, output_dir: Path):
        """Create visualization plots for tracking evaluation."""
        plt.style.use('seaborn-v0_8')
        
        # Main metrics visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # MOTA, MOTP, IDF1 bar chart
        main_metrics = ['MOTA', 'MOTP', 'IDF1']
        main_values = [metrics.mota, metrics.motp, metrics.idf1]
        
        bars1 = ax1.bar(main_metrics, main_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Main Tracking Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars1, main_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Track status distribution
        track_labels = ['Mostly Tracked', 'Partially Tracked', 'Mostly Lost']
        track_values = [metrics.mt, metrics.pt, metrics.ml]
        
        colors = ['green', 'orange', 'red']
        bars2 = ax2.bar(track_labels, track_values, color=colors, alpha=0.7)
        ax2.set_title('Track Status Distribution')
        ax2.set_ylabel('Number of Tracks')
        
        for bar, value in zip(bars2, track_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Error analysis
        error_labels = ['False Positives', 'False Negatives', 'ID Switches', 'Fragmentations']
        error_values = [metrics.fp, metrics.fn, metrics.ids, metrics.frag]
        
        bars3 = ax3.bar(error_labels, error_values, color='salmon', alpha=0.7)
        ax3.set_title('Error Analysis')
        ax3.set_ylabel('Count')
        
        for bar, value in zip(bars3, error_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom')
        
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Track length distribution
        gt_track_lengths = gt_data.groupby('id').size()
        results_track_lengths = results_data.groupby('id').size()
        
        ax4.hist([gt_track_lengths, results_track_lengths], 
                bins=20, alpha=0.7, label=['Ground Truth', 'Predictions'],
                color=['blue', 'red'])
        ax4.set_title('Track Length Distribution')
        ax4.set_xlabel('Track Length (frames)')
        ax4.set_ylabel('Number of Tracks')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tracking_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Timeline visualization
        self.create_timeline_plot(gt_data, results_data, output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def create_timeline_plot(self, gt_data: pd.DataFrame, results_data: pd.DataFrame, 
                           output_dir: Path):
        """Create timeline visualization of tracks."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Ground truth timeline
        for track_id in gt_data['id'].unique():
            track_data = gt_data[gt_data['id'] == track_id]
            frames = track_data['frame'].values
            ax1.plot(frames, [track_id] * len(frames), 'o-', alpha=0.7, markersize=2)
        
        ax1.set_title('Ground Truth Tracks Timeline')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Track ID')
        ax1.grid(True, alpha=0.3)
        
        # Predictions timeline
        for track_id in results_data['id'].unique():
            track_data = results_data[results_data['id'] == track_id]
            frames = track_data['frame'].values
            ax2.plot(frames, [track_id] * len(frames), 'o-', alpha=0.7, markersize=2)
        
        ax2.set_title('Predicted Tracks Timeline')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Track ID')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tracks_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for tracking evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate multi-object tracking')
    parser.add_argument('--config', required=True, help='Path to evaluation config file')
    parser.add_argument('--sequence', required=True, help='Path to sequence directory')
    parser.add_argument('--results', help='Path to tracking results file (optional)')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TrackingEvaluator(args.config)
    
    # Run evaluation
    try:
        print("Starting tracking evaluation...")
        
        results_file = args.results if args.results else "dummy_results.txt"
        metrics = evaluator.evaluate_sequence(args.sequence, results_file, args.output)
        
        print("\n" + "="*50)
        print("TRACKING EVALUATION RESULTS")
        print("="*50)
        print(f"MOTA:           {metrics.mota:.3f}")
        print(f"MOTP:           {metrics.motp:.3f}")
        print(f"IDF1:           {metrics.idf1:.3f}")
        print(f"Mostly Tracked: {metrics.mt}")
        print(f"Mostly Lost:    {metrics.ml}")
        print(f"Partially Tracked: {metrics.pt}")
        print(f"False Positives: {metrics.fp}")
        print(f"False Negatives: {metrics.fn}")
        print(f"ID Switches:    {metrics.ids}")
        print(f"Fragmentations: {metrics.frag}")
        print(f"Total Frames:   {metrics.num_frames}")
        print(f"Total Objects:  {metrics.num_objects}")
        
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()