#!/usr/bin/env python3
"""
Test Data Management System
Manages synthetic data generation, test dataset loading, validation, and versioning.
"""

import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import shutil
import random
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

@dataclass
class DatasetVersion:
    """Container for dataset version information."""
    version: str
    timestamp: str
    description: str
    file_count: int
    checksum: str
    metadata: Dict[str, Any]

@dataclass
class AnnotationData:
    """Container for annotation data."""
    image_id: str
    annotations: List[Dict[str, Any]]
    image_info: Dict[str, Any]

class SyntheticDataGenerator:
    """Generates synthetic data for testing AI pipelines."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_threat_intelligence_data(self, num_samples: int = 100) -> Dict[str, List[Dict]]:
        """Generate synthetic threat intelligence data."""
        print(f"Generating {num_samples} threat intelligence samples...")
        
        # IOC types and patterns
        ioc_patterns = {
            'ip': {
                'high_risk': ['192.168.{}.{}', '10.0.{}.{}', '172.16.{}.{}'],
                'medium_risk': ['203.0.113.{}', '198.51.100.{}'],
                'low_risk': ['8.8.8.8', '1.1.1.1', '208.67.222.222']
            },
            'domain': {
                'high_risk': ['malware-{}.com', 'phishing-{}.net', 'botnet-{}.org'],
                'medium_risk': ['suspicious-{}.info', 'questionable-{}.biz'],
                'low_risk': ['google.com', 'microsoft.com', 'amazon.com']
            },
            'hash': {
                'high_risk': 'malicious',
                'medium_risk': 'suspicious',
                'low_risk': 'benign'
            },
            'url': {
                'high_risk': ['http://malicious-{}.com/payload', 'https://phishing-{}.net/login'],
                'medium_risk': ['http://suspicious-{}.info/download'],
                'low_risk': ['https://www.google.com', 'https://www.github.com']
            }
        }
        
        threat_contexts = {
            'high_risk': [
                'Known botnet C&C server',
                'Malware distribution site',
                'Phishing campaign infrastructure',
                'APT group infrastructure',
                'Ransomware payment site'
            ],
            'medium_risk': [
                'Suspicious file hash',
                'Potentially unwanted program',
                'Adware distribution',
                'Suspicious network activity',
                'Unverified threat indicator'
            ],
            'low_risk': [
                'Legitimate service',
                'False positive',
                'Benign file',
                'Normal network traffic',
                'Whitelisted indicator'
            ]
        }
        
        data = {'high_risk_samples': [], 'medium_risk_samples': [], 'low_risk_samples': []}
        
        for risk_level in ['high_risk', 'medium_risk', 'low_risk']:
            samples_per_risk = num_samples // 3
            
            for _ in range(samples_per_risk):
                ioc_type = random.choice(list(ioc_patterns.keys()))
                
                # Generate IOC value
                if ioc_type == 'ip':
                    if risk_level == 'low_risk':
                        ioc_value = random.choice(ioc_patterns[ioc_type][risk_level])
                    else:
                        pattern = random.choice(ioc_patterns[ioc_type][risk_level])
                        ioc_value = pattern.format(random.randint(1, 254), random.randint(1, 254))
                elif ioc_type == 'domain':
                    if risk_level == 'low_risk':
                        ioc_value = random.choice(ioc_patterns[ioc_type][risk_level])
                    else:
                        pattern = random.choice(ioc_patterns[ioc_type][risk_level])
                        ioc_value = pattern.format(random.randint(1000, 9999))
                elif ioc_type == 'hash':
                    # Generate fake hash
                    hash_type = ioc_patterns[ioc_type][risk_level]
                    ioc_value = hashlib.md5(f"{hash_type}_{random.randint(1, 10000)}".encode()).hexdigest()
                elif ioc_type == 'url':
                    if risk_level == 'low_risk':
                        ioc_value = random.choice(ioc_patterns[ioc_type][risk_level])
                    else:
                        pattern = random.choice(ioc_patterns[ioc_type][risk_level])
                        ioc_value = pattern.format(random.randint(1000, 9999))
                
                sample = {
                    'ioc_type': ioc_type,
                    'ioc_value': ioc_value,
                    'context': random.choice(threat_contexts[risk_level]),
                    'label': risk_level.replace('_risk', ''),
                    'confidence': random.uniform(0.7, 1.0) if risk_level == 'high_risk' else 
                                 random.uniform(0.4, 0.8) if risk_level == 'medium_risk' else 
                                 random.uniform(0.1, 0.5),
                    'timestamp': datetime.now().isoformat(),
                    'source': f"synthetic_feed_{random.randint(1, 5)}"
                }
                
                data[risk_level].append(sample)
        
        # Save to file
        output_file = self.output_dir / "synthetic_threat_data.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Threat intelligence data saved to: {output_file}")
        return data
    
    def generate_video_detection_data(self, num_images: int = 50) -> Dict[str, Any]:
        """Generate synthetic video detection data with annotations."""
        print(f"Generating {num_images} synthetic video detection samples...")
        
        images_dir = self.output_dir / "video_detection" / "images"
        annotations_dir = self.output_dir / "video_detection" / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Object classes
        classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle']
        class_colors = {
            'person': (255, 0, 0),
            'car': (0, 255, 0),
            'truck': (0, 0, 255),
            'bicycle': (255, 255, 0),
            'motorcycle': (255, 0, 255)
        }
        
        dataset_info = {
            'images': [],
            'annotations': [],
            'classes': classes,
            'num_images': num_images
        }
        
        for i in range(num_images):
            # Create synthetic image
            img_width, img_height = 640, 480
            image = Image.new('RGB', (img_width, img_height), color=(128, 128, 128))
            draw = ImageDraw.Draw(image)
            
            # Generate random objects
            num_objects = random.randint(1, 5)
            image_annotations = []
            
            for obj_id in range(num_objects):
                obj_class = random.choice(classes)
                class_id = classes.index(obj_class)
                
                # Random bounding box
                x = random.randint(0, img_width - 100)
                y = random.randint(0, img_height - 100)
                w = random.randint(50, min(150, img_width - x))
                h = random.randint(50, min(150, img_height - y))
                
                # Draw bounding box
                color = class_colors[obj_class]
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                draw.text((x, y - 20), obj_class, fill=color)
                
                # YOLO format annotation (normalized coordinates)
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                image_annotations.append({
                    'class_id': class_id,
                    'class_name': obj_class,
                    'bbox': [x, y, w, h],  # Absolute coordinates
                    'bbox_normalized': [x_center, y_center, norm_width, norm_height],  # YOLO format
                    'confidence': random.uniform(0.8, 1.0)
                })
            
            # Save image
            img_filename = f"synthetic_{i:04d}.jpg"
            img_path = images_dir / img_filename
            image.save(img_path)
            
            # Save YOLO format annotation
            ann_filename = f"synthetic_{i:04d}.txt"
            ann_path = annotations_dir / ann_filename
            with open(ann_path, 'w') as f:
                for ann in image_annotations:
                    bbox_norm = ann['bbox_normalized']
                    f.write(f"{ann['class_id']} {bbox_norm[0]:.6f} {bbox_norm[1]:.6f} {bbox_norm[2]:.6f} {bbox_norm[3]:.6f}\n")
            
            dataset_info['images'].append(img_filename)
            dataset_info['annotations'].append({
                'image': img_filename,
                'objects': image_annotations
            })
        
        # Save dataset info
        info_file = self.output_dir / "video_detection" / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Video detection data saved to: {self.output_dir / 'video_detection'}")
        return dataset_info
    
    def generate_anomaly_trajectory_data(self, num_trajectories: int = 100) -> Dict[str, List[Dict]]:
        """Generate synthetic trajectory data for anomaly detection."""
        print(f"Generating {num_trajectories} synthetic trajectory samples...")
        
        def generate_normal_trajectory():
            """Generate a normal trajectory (straight line with small variations)."""
            start_x = random.randint(50, 200)
            start_y = random.randint(50, 200)
            
            points = [(start_x, start_y)]
            direction = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2, 5)
            
            for step in range(random.randint(10, 30)):
                # Small random variations
                direction += random.uniform(-0.2, 0.2)
                speed += random.uniform(-0.5, 0.5)
                speed = max(1, min(10, speed))  # Clamp speed
                
                prev_x, prev_y = points[-1]
                new_x = prev_x + speed * np.cos(direction)
                new_y = prev_y + speed * np.sin(direction)
                
                # Keep within bounds
                new_x = max(0, min(800, new_x))
                new_y = max(0, min(600, new_y))
                
                points.append((new_x, new_y))
            
            return points
        
        def generate_anomalous_trajectory():
            """Generate an anomalous trajectory (erratic movement, loops, etc.)."""
            trajectory_type = random.choice(['erratic', 'loop', 'zigzag', 'sudden_stop'])
            
            start_x = random.randint(100, 300)
            start_y = random.randint(100, 300)
            points = [(start_x, start_y)]
            
            if trajectory_type == 'erratic':
                # Highly erratic movement
                for step in range(random.randint(15, 40)):
                    prev_x, prev_y = points[-1]
                    new_x = prev_x + random.uniform(-20, 20)
                    new_y = prev_y + random.uniform(-20, 20)
                    
                    new_x = max(0, min(800, new_x))
                    new_y = max(0, min(600, new_y))
                    points.append((new_x, new_y))
            
            elif trajectory_type == 'loop':
                # Circular/loop movement
                center_x, center_y = start_x, start_y
                radius = random.uniform(30, 80)
                
                for angle in np.linspace(0, 4 * np.pi, random.randint(20, 50)):
                    new_x = center_x + radius * np.cos(angle)
                    new_y = center_y + radius * np.sin(angle)
                    points.append((new_x, new_y))
            
            elif trajectory_type == 'zigzag':
                # Zigzag pattern
                direction = 1
                for step in range(random.randint(15, 35)):
                    prev_x, prev_y = points[-1]
                    new_x = prev_x + 5
                    new_y = prev_y + direction * 15
                    
                    if step % 5 == 0:
                        direction *= -1
                    
                    new_x = max(0, min(800, new_x))
                    new_y = max(0, min(600, new_y))
                    points.append((new_x, new_y))
            
            elif trajectory_type == 'sudden_stop':
                # Normal movement then sudden stop
                direction = random.uniform(0, 2 * np.pi)
                speed = random.uniform(5, 10)
                
                # Normal movement
                for step in range(random.randint(10, 20)):
                    prev_x, prev_y = points[-1]
                    new_x = prev_x + speed * np.cos(direction)
                    new_y = prev_y + speed * np.sin(direction)
                    points.append((new_x, new_y))
                
                # Sudden stop (stay in same area)
                stop_x, stop_y = points[-1]
                for step in range(random.randint(10, 20)):
                    new_x = stop_x + random.uniform(-2, 2)
                    new_y = stop_y + random.uniform(-2, 2)
                    points.append((new_x, new_y))
            
            return points
        
        # Generate trajectories
        normal_count = int(num_trajectories * 0.7)  # 70% normal
        anomaly_count = num_trajectories - normal_count  # 30% anomalous
        
        data = {
            'normal_trajectories': [],
            'anomalous_trajectories': []
        }
        
        # Generate normal trajectories
        for i in range(normal_count):
            points = generate_normal_trajectory()
            trajectory = {
                'id': f'normal_{i:04d}',
                'points': points,
                'label': 'normal',
                'timestamp': datetime.now().isoformat()
            }
            data['normal_trajectories'].append(trajectory)
        
        # Generate anomalous trajectories
        for i in range(anomaly_count):
            points = generate_anomalous_trajectory()
            trajectory = {
                'id': f'anomaly_{i:04d}',
                'points': points,
                'label': 'anomaly',
                'timestamp': datetime.now().isoformat()
            }
            data['anomalous_trajectories'].append(trajectory)
        
        # Save to file
        output_file = self.output_dir / "synthetic_anomaly_data.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Anomaly trajectory data saved to: {output_file}")
        return data

class TestDataManager:
    """Manages test datasets including loading, validation, and versioning."""
    
    def __init__(self, base_dir: str = "backend/evaluation/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.base_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "dataset_metadata.json"
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def calculate_directory_checksum(self, dir_path: Path) -> str:
        """Calculate checksum for all files in a directory."""
        checksums = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                checksums.append(self.calculate_checksum(file_path))
        
        combined = "".join(checksums)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def create_dataset_version(self, dataset_name: str, dataset_path: str, 
                             description: str = "") -> DatasetVersion:
        """Create a new version of a dataset."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Generate version string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        
        # Calculate checksum
        if dataset_path.is_file():
            checksum = self.calculate_checksum(dataset_path)
            file_count = 1
        else:
            checksum = self.calculate_directory_checksum(dataset_path)
            file_count = len(list(dataset_path.rglob("*")))
        
        # Create version directory
        version_dir = self.versions_dir / dataset_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dataset to version directory
        if dataset_path.is_file():
            shutil.copy2(dataset_path, version_dir / dataset_path.name)
        else:
            shutil.copytree(dataset_path, version_dir / dataset_path.name, dirs_exist_ok=True)
        
        # Create version metadata
        version_info = DatasetVersion(
            version=version,
            timestamp=timestamp,
            description=description,
            file_count=file_count,
            checksum=checksum,
            metadata={
                'dataset_name': dataset_name,
                'original_path': str(dataset_path),
                'size_bytes': sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file()) if dataset_path.is_dir() else dataset_path.stat().st_size
            }
        )
        
        # Save version info
        version_file = version_dir / "version_info.json"
        with open(version_file, 'w') as f:
            json.dump(asdict(version_info), f, indent=2)
        
        # Update dataset metadata
        self.update_dataset_metadata(dataset_name, version_info)
        
        print(f"Created dataset version: {dataset_name} {version}")
        return version_info
    
    def update_dataset_metadata(self, dataset_name: str, version_info: DatasetVersion):
        """Update the global dataset metadata."""
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        if dataset_name not in metadata:
            metadata[dataset_name] = {
                'versions': [],
                'latest_version': None,
                'created': datetime.now().isoformat()
            }
        
        metadata[dataset_name]['versions'].append(asdict(version_info))
        metadata[dataset_name]['latest_version'] = version_info.version
        metadata[dataset_name]['updated'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_dataset_version(self, dataset_name: str, version: str = None) -> Optional[Path]:
        """Load a specific version of a dataset."""
        if version is None:
            # Load latest version
            metadata = self.get_dataset_metadata(dataset_name)
            if not metadata:
                return None
            version = metadata['latest_version']
        
        version_dir = self.versions_dir / dataset_name / version
        if not version_dir.exists():
            print(f"Dataset version not found: {dataset_name} {version}")
            return None
        
        return version_dir
    
    def get_dataset_metadata(self, dataset_name: str = None) -> Optional[Dict]:
        """Get metadata for a dataset or all datasets."""
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if dataset_name:
            return metadata.get(dataset_name)
        return metadata
    
    def validate_dataset(self, dataset_path: Path, dataset_type: str) -> Dict[str, Any]:
        """Validate a dataset based on its type."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if dataset_type == 'threat_intelligence':
            validation_results.update(self._validate_threat_dataset(dataset_path))
        elif dataset_type == 'video_detection':
            validation_results.update(self._validate_video_dataset(dataset_path))
        elif dataset_type == 'anomaly_detection':
            validation_results.update(self._validate_anomaly_dataset(dataset_path))
        else:
            validation_results['errors'].append(f"Unknown dataset type: {dataset_type}")
            validation_results['valid'] = False
        
        return validation_results
    
    def _validate_threat_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate threat intelligence dataset."""
        results = {'valid': True, 'errors': [], 'warnings': [], 'statistics': {}}
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            required_keys = ['high_risk_samples', 'medium_risk_samples', 'low_risk_samples']
            for key in required_keys:
                if key not in data:
                    results['errors'].append(f"Missing required key: {key}")
                    results['valid'] = False
            
            # Validate samples
            total_samples = 0
            for risk_level in required_keys:
                if risk_level in data:
                    samples = data[risk_level]
                    total_samples += len(samples)
                    
                    for i, sample in enumerate(samples):
                        required_fields = ['ioc_type', 'ioc_value', 'context', 'label']
                        for field in required_fields:
                            if field not in sample:
                                results['errors'].append(f"{risk_level}[{i}]: Missing field '{field}'")
                                results['valid'] = False
            
            results['statistics'] = {
                'total_samples': total_samples,
                'high_risk': len(data.get('high_risk_samples', [])),
                'medium_risk': len(data.get('medium_risk_samples', [])),
                'low_risk': len(data.get('low_risk_samples', []))
            }
            
        except json.JSONDecodeError as e:
            results['errors'].append(f"Invalid JSON format: {e}")
            results['valid'] = False
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
            results['valid'] = False
        
        return results
    
    def _validate_video_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate video detection dataset."""
        results = {'valid': True, 'errors': [], 'warnings': [], 'statistics': {}}
        
        try:
            # Check directory structure
            images_dir = dataset_path / "images"
            annotations_dir = dataset_path / "annotations"
            info_file = dataset_path / "dataset_info.json"
            
            if not images_dir.exists():
                results['errors'].append("Missing images directory")
                results['valid'] = False
            
            if not annotations_dir.exists():
                results['errors'].append("Missing annotations directory")
                results['valid'] = False
            
            if not info_file.exists():
                results['errors'].append("Missing dataset_info.json file")
                results['valid'] = False
            
            if results['valid']:
                # Load dataset info
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                # Check images and annotations match
                image_files = set(f.name for f in images_dir.glob("*.jpg"))
                annotation_files = set(f.stem + ".jpg" for f in annotations_dir.glob("*.txt"))
                
                missing_annotations = image_files - annotation_files
                missing_images = annotation_files - image_files
                
                if missing_annotations:
                    results['warnings'].append(f"Images without annotations: {len(missing_annotations)}")
                
                if missing_images:
                    results['warnings'].append(f"Annotations without images: {len(missing_images)}")
                
                results['statistics'] = {
                    'num_images': len(image_files),
                    'num_annotations': len(annotation_files),
                    'classes': info.get('classes', []),
                    'matched_pairs': len(image_files & annotation_files)
                }
        
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
            results['valid'] = False
        
        return results
    
    def _validate_anomaly_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate anomaly detection dataset."""
        results = {'valid': True, 'errors': [], 'warnings': [], 'statistics': {}}
        
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            required_keys = ['normal_trajectories', 'anomalous_trajectories']
            for key in required_keys:
                if key not in data:
                    results['errors'].append(f"Missing required key: {key}")
                    results['valid'] = False
            
            # Validate trajectories
            total_trajectories = 0
            for traj_type in required_keys:
                if traj_type in data:
                    trajectories = data[traj_type]
                    total_trajectories += len(trajectories)
                    
                    for i, traj in enumerate(trajectories):
                        required_fields = ['id', 'points', 'label']
                        for field in required_fields:
                            if field not in traj:
                                results['errors'].append(f"{traj_type}[{i}]: Missing field '{field}'")
                                results['valid'] = False
                        
                        # Validate points
                        if 'points' in traj:
                            points = traj['points']
                            if len(points) < 2:
                                results['warnings'].append(f"{traj_type}[{i}]: Trajectory too short ({len(points)} points)")
            
            results['statistics'] = {
                'total_trajectories': total_trajectories,
                'normal_trajectories': len(data.get('normal_trajectories', [])),
                'anomalous_trajectories': len(data.get('anomalous_trajectories', []))
            }
            
        except json.JSONDecodeError as e:
            results['errors'].append(f"Invalid JSON format: {e}")
            results['valid'] = False
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
            results['valid'] = False
        
        return results
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all managed datasets."""
        metadata = self.get_dataset_metadata()
        if not metadata:
            return {}
        
        summary = {}
        for dataset_name, dataset_info in metadata.items():
            summary[dataset_name] = {
                'latest_version': dataset_info['latest_version'],
                'num_versions': len(dataset_info['versions']),
                'created': dataset_info['created'],
                'updated': dataset_info.get('updated', dataset_info['created'])
            }
        
        return summary

def main():
    """Main function for test data management."""
    parser = argparse.ArgumentParser(description='Test Data Management System')
    parser.add_argument('--action', required=True, 
                       choices=['generate', 'validate', 'version', 'list'],
                       help='Action to perform')
    parser.add_argument('--type', 
                       choices=['threat', 'video', 'anomaly'],
                       help='Data type for generation/validation')
    parser.add_argument('--output', default='backend/evaluation/datasets',
                       help='Output directory')
    parser.add_argument('--dataset', help='Dataset name for versioning')
    parser.add_argument('--path', help='Dataset path for versioning/validation')
    parser.add_argument('--description', help='Version description')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    if args.action == 'generate':
        if not args.type:
            print("Error: --type required for generation")
            return
        
        generator = SyntheticDataGenerator(args.output)
        
        if args.type == 'threat':
            generator.generate_threat_intelligence_data(args.samples)
        elif args.type == 'video':
            generator.generate_video_detection_data(args.samples)
        elif args.type == 'anomaly':
            generator.generate_anomaly_trajectory_data(args.samples)
    
    elif args.action == 'validate':
        if not args.path or not args.type:
            print("Error: --path and --type required for validation")
            return
        
        manager = TestDataManager(args.output)
        results = manager.validate_dataset(Path(args.path), args.type)
        
        print("Validation Results:")
        print(f"Valid: {results['valid']}")
        
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['statistics']:
            print("Statistics:")
            for key, value in results['statistics'].items():
                print(f"  {key}: {value}")
    
    elif args.action == 'version':
        if not args.dataset or not args.path:
            print("Error: --dataset and --path required for versioning")
            return
        
        manager = TestDataManager(args.output)
        version_info = manager.create_dataset_version(
            args.dataset, args.path, args.description or ""
        )
        
        print(f"Created version: {version_info.version}")
        print(f"Checksum: {version_info.checksum}")
        print(f"Files: {version_info.file_count}")
    
    elif args.action == 'list':
        manager = TestDataManager(args.output)
        datasets = manager.list_datasets()
        
        if not datasets:
            print("No datasets found")
        else:
            print("Managed Datasets:")
            for name, info in datasets.items():
                print(f"  {name}:")
                print(f"    Latest Version: {info['latest_version']}")
                print(f"    Total Versions: {info['num_versions']}")
                print(f"    Created: {info['created']}")
                print(f"    Updated: {info['updated']}")

if __name__ == "__main__":
    main()