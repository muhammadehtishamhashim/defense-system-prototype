#!/usr/bin/env python3
"""
HifazatAI Evaluation Setup Script
Sets up evaluation repositories and datasets for testing AI pipelines.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional

class EvaluationSetup:
    def __init__(self, base_dir: str = "backend/evaluation"):
        self.base_dir = Path(base_dir)
        self.repos_dir = self.base_dir / "repositories"
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        for dir_path in [self.repos_dir, self.datasets_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_yolov8_repo(self) -> bool:
        """Clone and setup YOLOv8 repository for object detection evaluation."""
        print("Setting up YOLOv8 repository...")
        
        yolo_dir = self.repos_dir / "ultralytics"
        
        if yolo_dir.exists():
            print("YOLOv8 repository already exists, pulling latest changes...")
            try:
                subprocess.run(["git", "pull"], cwd=yolo_dir, check=True)
            except subprocess.CalledProcessError:
                print("Failed to pull YOLOv8 updates")
                return False
        else:
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ultralytics/ultralytics.git",
                    str(yolo_dir)
                ], check=True)
            except subprocess.CalledProcessError:
                print("Failed to clone YOLOv8 repository")
                return False
        
        # Install YOLOv8 requirements
        requirements_file = yolo_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(requirements_file)
                ], check=True)
                print("YOLOv8 requirements installed successfully")
            except subprocess.CalledProcessError:
                print("Failed to install YOLOv8 requirements")
                return False
        
        return True
    
    def setup_bytetrack_repo(self) -> bool:
        """Clone and setup ByteTrack repository for tracking evaluation."""
        print("Setting up ByteTrack repository...")
        
        bytetrack_dir = self.repos_dir / "ByteTrack"
        
        if bytetrack_dir.exists():
            print("ByteTrack repository already exists, pulling latest changes...")
            try:
                subprocess.run(["git", "pull"], cwd=bytetrack_dir, check=True)
            except subprocess.CalledProcessError:
                print("Failed to pull ByteTrack updates")
                return False
        else:
            try:
                subprocess.run([
                    "git", "clone",
                    "https://github.com/ifzhang/ByteTrack.git",
                    str(bytetrack_dir)
                ], check=True)
            except subprocess.CalledProcessError:
                print("Failed to clone ByteTrack repository")
                return False
        
        # Install ByteTrack requirements
        requirements_file = bytetrack_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "-r", str(requirements_file)
                ], check=True)
                print("ByteTrack requirements installed successfully")
            except subprocess.CalledProcessError:
                print("Failed to install ByteTrack requirements")
                return False
        
        return True
    
    def setup_pyod_repo(self) -> bool:
        """Setup PyOD for anomaly detection evaluation."""
        print("Setting up PyOD for anomaly detection...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "pyod", "scikit-learn", "numpy", "pandas"
            ], check=True)
            print("PyOD and dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install PyOD")
            return False
    
    def download_visdrone_dataset(self) -> bool:
        """Download VisDrone dataset subset for testing."""
        print("Downloading VisDrone dataset subset...")
        
        visdrone_dir = self.datasets_dir / "visdrone"
        visdrone_dir.mkdir(exist_ok=True)
        
        # Create sample dataset structure (in real implementation, download actual data)
        sample_data = {
            "images": ["sample_001.jpg", "sample_002.jpg", "sample_003.jpg"],
            "annotations": ["sample_001.txt", "sample_002.txt", "sample_003.txt"],
            "classes": [
                "pedestrian", "people", "bicycle", "car", "van", 
                "truck", "tricycle", "awning-tricycle", "bus", "motor"
            ]
        }
        
        # Create sample annotation files
        annotations_dir = visdrone_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        for ann_file in sample_data["annotations"]:
            ann_path = annotations_dir / ann_file
            with open(ann_path, 'w') as f:
                # Sample YOLO format annotation
                f.write("0 0.5 0.5 0.2 0.3\n")  # class_id x_center y_center width height
                f.write("1 0.3 0.7 0.1 0.2\n")
        
        # Create dataset info file
        info_file = visdrone_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"VisDrone dataset structure created at {visdrone_dir}")
        return True
    
    def download_mot_dataset(self) -> bool:
        """Download MOT17/MOT20 dataset samples for tracking evaluation."""
        print("Setting up MOT dataset samples...")
        
        mot_dir = self.datasets_dir / "mot"
        mot_dir.mkdir(exist_ok=True)
        
        # Create sample MOT dataset structure
        for dataset in ["MOT17", "MOT20"]:
            dataset_dir = mot_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            
            # Create sample sequence
            seq_dir = dataset_dir / "train" / "MOT17-02-FRCNN"
            seq_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample ground truth file
            gt_file = seq_dir / "gt" / "gt.txt"
            gt_file.parent.mkdir(exist_ok=True)
            
            with open(gt_file, 'w') as f:
                # Sample MOT format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
                f.write("1,1,100,200,50,100,1,-1,-1,-1\n")
                f.write("2,1,105,205,50,100,1,-1,-1,-1\n")
                f.write("3,1,110,210,50,100,1,-1,-1,-1\n")
            
            # Create sequence info
            seqinfo_file = seq_dir / "seqinfo.ini"
            with open(seqinfo_file, 'w') as f:
                f.write("[Sequence]\n")
                f.write("name=MOT17-02-FRCNN\n")
                f.write("imDir=img1\n")
                f.write("frameRate=30\n")
                f.write("seqLength=600\n")
                f.write("imWidth=1920\n")
                f.write("imHeight=1080\n")
                f.write("imExt=.jpg\n")
        
        print(f"MOT dataset structure created at {mot_dir}")
        return True
    
    def setup_evaluation_tools(self) -> bool:
        """Install evaluation tools and metrics libraries."""
        print("Installing evaluation tools...")
        
        tools = [
            "pycocotools",  # For mAP calculation
            "motmetrics",   # For MOT metrics
            "scikit-learn", # For classification metrics
            "matplotlib",   # For visualization
            "seaborn",      # For advanced plotting
            "opencv-python", # For image processing
            "Pillow",       # For image handling
            "tqdm",         # For progress bars
        ]
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + tools, check=True)
            print("Evaluation tools installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install evaluation tools")
            return False
    
    def create_evaluation_configs(self) -> bool:
        """Create configuration files for evaluation."""
        print("Creating evaluation configuration files...")
        
        configs_dir = self.base_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        # Object detection evaluation config
        detection_config = {
            "model_path": "models/yolov8n.pt",
            "dataset_path": "datasets/visdrone",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "classes": [
                "pedestrian", "people", "bicycle", "car", "van",
                "truck", "tricycle", "awning-tricycle", "bus", "motor"
            ],
            "metrics": ["mAP", "mAP50", "mAP75", "precision", "recall"]
        }
        
        with open(configs_dir / "detection_eval.json", 'w') as f:
            json.dump(detection_config, f, indent=2)
        
        # Tracking evaluation config
        tracking_config = {
            "dataset_path": "datasets/mot",
            "tracker_config": {
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 30
            },
            "metrics": ["MOTA", "MOTP", "IDF1", "MT", "ML", "FP", "FN", "IDs"]
        }
        
        with open(configs_dir / "tracking_eval.json", 'w') as f:
            json.dump(tracking_config, f, indent=2)
        
        # Anomaly detection evaluation config
        anomaly_config = {
            "models": ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
            "contamination": 0.1,
            "features": ["speed", "curvature", "duration", "direction_change"],
            "metrics": ["ROC-AUC", "precision", "recall", "f1_score"],
            "test_data_path": "datasets/anomaly_test.json"
        }
        
        with open(configs_dir / "anomaly_eval.json", 'w') as f:
            json.dump(anomaly_config, f, indent=2)
        
        print("Evaluation configuration files created")
        return True
    
    def create_sample_test_data(self) -> bool:
        """Create sample test data for evaluation."""
        print("Creating sample test data...")
        
        # Create sample anomaly test data
        anomaly_data = {
            "normal_trajectories": [
                {
                    "id": "normal_001",
                    "points": [[100, 200], [105, 205], [110, 210], [115, 215]],
                    "features": {"speed": 2.5, "curvature": 0.1, "duration": 4.0},
                    "label": "normal"
                },
                {
                    "id": "normal_002", 
                    "points": [[200, 300], [205, 305], [210, 310], [215, 315]],
                    "features": {"speed": 3.0, "curvature": 0.05, "duration": 3.5},
                    "label": "normal"
                }
            ],
            "anomalous_trajectories": [
                {
                    "id": "anomaly_001",
                    "points": [[150, 250], [200, 300], [100, 400], [300, 200]],
                    "features": {"speed": 15.0, "curvature": 2.5, "duration": 2.0},
                    "label": "anomaly"
                }
            ]
        }
        
        with open(self.datasets_dir / "anomaly_test.json", 'w') as f:
            json.dump(anomaly_data, f, indent=2)
        
        # Create sample threat intelligence test data
        threat_data = {
            "high_risk_samples": [
                {
                    "ioc_type": "ip",
                    "ioc_value": "192.168.1.100",
                    "context": "Known botnet C&C server",
                    "label": "high"
                },
                {
                    "ioc_type": "domain",
                    "ioc_value": "malicious-site.com",
                    "context": "Phishing campaign infrastructure",
                    "label": "high"
                }
            ],
            "medium_risk_samples": [
                {
                    "ioc_type": "hash",
                    "ioc_value": "d41d8cd98f00b204e9800998ecf8427e",
                    "context": "Suspicious file hash",
                    "label": "medium"
                }
            ],
            "low_risk_samples": [
                {
                    "ioc_type": "ip",
                    "ioc_value": "8.8.8.8",
                    "context": "Google DNS server",
                    "label": "low"
                }
            ]
        }
        
        with open(self.datasets_dir / "threat_test.json", 'w') as f:
            json.dump(threat_data, f, indent=2)
        
        print("Sample test data created")
        return True
    
    def run_setup(self) -> bool:
        """Run the complete evaluation setup."""
        print("Starting HifazatAI Evaluation Setup...")
        print("=" * 50)
        
        setup_steps = [
            ("Setting up YOLOv8 repository", self.setup_yolov8_repo),
            ("Setting up ByteTrack repository", self.setup_bytetrack_repo),
            ("Setting up PyOD", self.setup_pyod_repo),
            ("Downloading VisDrone dataset", self.download_visdrone_dataset),
            ("Setting up MOT dataset", self.download_mot_dataset),
            ("Installing evaluation tools", self.setup_evaluation_tools),
            ("Creating evaluation configs", self.create_evaluation_configs),
            ("Creating sample test data", self.create_sample_test_data),
        ]
        
        failed_steps = []
        
        for step_name, step_func in setup_steps:
            print(f"\n{step_name}...")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    print(f"❌ {step_name} failed")
                else:
                    print(f"✅ {step_name} completed")
            except Exception as e:
                failed_steps.append(step_name)
                print(f"❌ {step_name} failed with error: {e}")
        
        print("\n" + "=" * 50)
        if failed_steps:
            print(f"Setup completed with {len(failed_steps)} failures:")
            for step in failed_steps:
                print(f"  - {step}")
            return False
        else:
            print("✅ All setup steps completed successfully!")
            print(f"Evaluation environment ready at: {self.base_dir}")
            return True

def main():
    """Main entry point for the evaluation setup script."""
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "backend/evaluation"
    
    setup = EvaluationSetup(base_dir)
    success = setup.run_setup()
    
    if success:
        print("\nNext steps:")
        print("1. Run evaluation scripts in backend/evaluation/scripts/")
        print("2. Check configuration files in backend/evaluation/configs/")
        print("3. Review sample datasets in backend/evaluation/datasets/")
        sys.exit(0)
    else:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()