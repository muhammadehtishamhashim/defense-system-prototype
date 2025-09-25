#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner
Runs all evaluation scripts and generates a combined report.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class EvaluationRunner:
    """Runs comprehensive evaluation of all HifazatAI components."""
    
    def __init__(self, base_dir: str = "backend/evaluation"):
        self.base_dir = Path(base_dir)
        self.scripts_dir = self.base_dir / "scripts"
        self.configs_dir = self.base_dir / "configs"
        self.datasets_dir = self.base_dir / "datasets"
        self.results_dir = self.base_dir / "results"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"evaluation_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
    
    def run_detection_evaluation(self) -> Optional[Dict]:
        """Run object detection evaluation."""
        print("Running object detection evaluation...")
        
        config_file = self.configs_dir / "detection_eval.json"
        dataset_dir = self.datasets_dir / "visdrone"
        output_dir = self.run_dir / "detection"
        
        # Check if required files exist
        if not config_file.exists():
            print(f"Warning: Detection config not found: {config_file}")
            return None
        
        if not dataset_dir.exists():
            print(f"Warning: Detection dataset not found: {dataset_dir}")
            return None
        
        # For demonstration, we'll use a dummy model path
        model_path = "yolov8n.pt"  # This would be downloaded by ultralytics
        
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "evaluate_detection.py"),
                "--config", str(config_file),
                "--model", model_path,
                "--dataset", str(dataset_dir),
                "--output", str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Detection evaluation completed successfully")
                
                # Load results
                results_file = output_dir / "evaluation_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"❌ Detection evaluation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Detection evaluation timed out")
        except Exception as e:
            print(f"❌ Detection evaluation error: {e}")
        
        return None
    
    def run_tracking_evaluation(self) -> Optional[Dict]:
        """Run multi-object tracking evaluation."""
        print("Running tracking evaluation...")
        
        config_file = self.configs_dir / "tracking_eval.json"
        sequence_dir = self.datasets_dir / "mot" / "MOT17" / "train" / "MOT17-02-FRCNN"
        output_dir = self.run_dir / "tracking"
        
        # Check if required files exist
        if not config_file.exists():
            print(f"Warning: Tracking config not found: {config_file}")
            return None
        
        if not sequence_dir.exists():
            print(f"Warning: Tracking sequence not found: {sequence_dir}")
            return None
        
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "evaluate_tracking.py"),
                "--config", str(config_file),
                "--sequence", str(sequence_dir),
                "--output", str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Tracking evaluation completed successfully")
                
                # Load results
                results_file = output_dir / "tracking_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"❌ Tracking evaluation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Tracking evaluation timed out")
        except Exception as e:
            print(f"❌ Tracking evaluation error: {e}")
        
        return None
    
    def run_anomaly_evaluation(self) -> Optional[Dict]:
        """Run anomaly detection evaluation."""
        print("Running anomaly detection evaluation...")
        
        config_file = self.configs_dir / "anomaly_eval.json"
        data_file = self.datasets_dir / "anomaly_test.json"
        output_dir = self.run_dir / "anomaly"
        
        # Check if required files exist
        if not config_file.exists():
            print(f"Warning: Anomaly config not found: {config_file}")
            return None
        
        if not data_file.exists():
            print(f"Warning: Anomaly test data not found: {data_file}")
            return None
        
        try:
            cmd = [
                sys.executable,
                str(self.scripts_dir / "evaluate_anomaly.py"),
                "--config", str(config_file),
                "--data", str(data_file),
                "--output", str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Anomaly evaluation completed successfully")
                
                # Load results
                results_file = output_dir / "anomaly_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        return json.load(f)
            else:
                print(f"❌ Anomaly evaluation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Anomaly evaluation timed out")
        except Exception as e:
            print(f"❌ Anomaly evaluation error: {e}")
        
        return None
    
    def create_combined_report(self, detection_results: Optional[Dict],
                             tracking_results: Optional[Dict],
                             anomaly_results: Optional[Dict]):
        """Create a combined evaluation report."""
        print("Creating combined evaluation report...")
        
        report = {
            "evaluation_timestamp": self.timestamp,
            "evaluation_summary": {},
            "detailed_results": {
                "detection": detection_results,
                "tracking": tracking_results,
                "anomaly": anomaly_results
            }
        }
        
        # Extract key metrics for summary
        summary = {}
        
        if detection_results:
            overall_metrics = detection_results.get('overall_metrics', {})
            summary['detection'] = {
                'mAP_50_95': overall_metrics.get('mAP_50_95', 0.0),
                'mAP_50': overall_metrics.get('mAP_50', 0.0),
                'precision': overall_metrics.get('precision', 0.0),
                'recall': overall_metrics.get('recall', 0.0),
                'f1_score': overall_metrics.get('f1_score', 0.0)
            }
        
        if tracking_results:
            metrics = tracking_results.get('metrics', {})
            summary['tracking'] = {
                'MOTA': metrics.get('MOTA', 0.0),
                'MOTP': metrics.get('MOTP', 0.0),
                'IDF1': metrics.get('IDF1', 0.0),
                'ID_switches': metrics.get('IDs', 0)
            }
        
        if anomaly_results:
            # Get best model results
            model_results = anomaly_results.get('model_results', {})
            if model_results:
                best_model = max(model_results.keys(), 
                               key=lambda k: model_results[k].get('f1_score', 0))
                best_results = model_results[best_model]
                summary['anomaly'] = {
                    'best_model': best_model,
                    'roc_auc': best_results.get('roc_auc', 0.0),
                    'precision': best_results.get('precision', 0.0),
                    'recall': best_results.get('recall', 0.0),
                    'f1_score': best_results.get('f1_score', 0.0)
                }
        
        report['evaluation_summary'] = summary
        
        # Save combined report
        report_file = self.run_dir / "combined_evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary visualization
        self.create_summary_visualization(summary)
        
        print(f"Combined report saved to: {report_file}")
        return report
    
    def create_summary_visualization(self, summary: Dict):
        """Create summary visualization of all evaluation results."""
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Detection metrics
        if 'detection' in summary:
            detection_data = summary['detection']
            metrics = ['mAP@0.5:0.95', 'mAP@0.5', 'Precision', 'Recall', 'F1-Score']
            values = [
                detection_data.get('mAP_50_95', 0),
                detection_data.get('mAP_50', 0),
                detection_data.get('precision', 0),
                detection_data.get('recall', 0),
                detection_data.get('f1_score', 0)
            ]
            
            bars = axes[0, 0].bar(metrics, values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Object Detection Performance')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
        else:
            axes[0, 0].text(0.5, 0.5, 'Detection\nEvaluation\nNot Available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Object Detection Performance')
        
        # Tracking metrics
        if 'tracking' in summary:
            tracking_data = summary['tracking']
            metrics = ['MOTA', 'MOTP', 'IDF1']
            values = [
                tracking_data.get('MOTA', 0),
                tracking_data.get('MOTP', 0),
                tracking_data.get('IDF1', 0)
            ]
            
            bars = axes[0, 1].bar(metrics, values, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Multi-Object Tracking Performance')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[0, 1].text(0.5, 0.5, 'Tracking\nEvaluation\nNot Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Multi-Object Tracking Performance')
        
        # Anomaly detection metrics
        if 'anomaly' in summary:
            anomaly_data = summary['anomaly']
            metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
            values = [
                anomaly_data.get('roc_auc', 0),
                anomaly_data.get('precision', 0),
                anomaly_data.get('recall', 0),
                anomaly_data.get('f1_score', 0)
            ]
            
            bars = axes[1, 0].bar(metrics, values, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title(f'Anomaly Detection Performance\n({anomaly_data.get("best_model", "N/A")})')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, 'Anomaly\nEvaluation\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Anomaly Detection Performance')
        
        # Overall system performance radar chart
        categories = []
        scores = []
        
        if 'detection' in summary:
            categories.append('Detection\nF1-Score')
            scores.append(summary['detection'].get('f1_score', 0))
        
        if 'tracking' in summary:
            categories.append('Tracking\nMOTA')
            scores.append(summary['tracking'].get('MOTA', 0))
        
        if 'anomaly' in summary:
            categories.append('Anomaly\nF1-Score')
            scores.append(summary['anomaly'].get('f1_score', 0))
        
        if categories and scores:
            # Simple bar chart instead of radar for simplicity
            bars = axes[1, 1].bar(categories, scores, color='gold', alpha=0.7)
            axes[1, 1].set_title('Overall System Performance')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
            
            for bar, value in zip(bars, scores):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Evaluation\nResults Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Overall System Performance')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualization saved to: {self.run_dir / 'evaluation_summary.png'}")
    
    def run_complete_evaluation(self) -> Dict:
        """Run complete evaluation of all components."""
        print("="*60)
        print("STARTING HIFAZATAI COMPREHENSIVE EVALUATION")
        print("="*60)
        print(f"Evaluation timestamp: {self.timestamp}")
        print(f"Results directory: {self.run_dir}")
        print()
        
        # Run individual evaluations
        detection_results = self.run_detection_evaluation()
        tracking_results = self.run_tracking_evaluation()
        anomaly_results = self.run_anomaly_evaluation()
        
        # Create combined report
        report = self.create_combined_report(detection_results, tracking_results, anomaly_results)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        
        # Print summary
        summary = report.get('evaluation_summary', {})
        
        if 'detection' in summary:
            print(f"Object Detection - mAP@0.5: {summary['detection'].get('mAP_50', 0):.3f}")
        
        if 'tracking' in summary:
            print(f"Multi-Object Tracking - MOTA: {summary['tracking'].get('MOTA', 0):.3f}")
        
        if 'anomaly' in summary:
            anomaly_data = summary['anomaly']
            print(f"Anomaly Detection ({anomaly_data.get('best_model', 'N/A')}) - F1: {anomaly_data.get('f1_score', 0):.3f}")
        
        print(f"\nDetailed results available in: {self.run_dir}")
        
        return report

def main():
    """Main function for comprehensive evaluation."""
    parser = argparse.ArgumentParser(description='Run comprehensive HifazatAI evaluation')
    parser.add_argument('--base-dir', default='backend/evaluation', 
                       help='Base directory for evaluation files')
    parser.add_argument('--components', nargs='+', 
                       choices=['detection', 'tracking', 'anomaly', 'all'],
                       default=['all'],
                       help='Components to evaluate')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(args.base_dir)
    
    # Run evaluation
    try:
        if 'all' in args.components:
            report = runner.run_complete_evaluation()
        else:
            # Run individual components
            detection_results = None
            tracking_results = None
            anomaly_results = None
            
            if 'detection' in args.components:
                detection_results = runner.run_detection_evaluation()
            
            if 'tracking' in args.components:
                tracking_results = runner.run_tracking_evaluation()
            
            if 'anomaly' in args.components:
                anomaly_results = runner.run_anomaly_evaluation()
            
            report = runner.create_combined_report(detection_results, tracking_results, anomaly_results)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Report saved to: {runner.run_dir / 'combined_evaluation_report.json'}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()