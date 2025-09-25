# HifazatAI Evaluation Framework

This directory contains the comprehensive evaluation framework for the HifazatAI security system prototype.

## Directory Structure

```
backend/evaluation/
├── setup_evaluation.py          # Setup script for evaluation environment
├── test_data_manager.py         # Test data management system
├── scripts/                     # Evaluation scripts
│   ├── evaluate_detection.py    # Object detection evaluation
│   ├── evaluate_tracking.py     # Multi-object tracking evaluation
│   ├── evaluate_anomaly.py      # Anomaly detection evaluation
│   └── run_evaluation.py        # Comprehensive evaluation runner
├── configs/                     # Configuration files
│   ├── detection_eval.json      # Detection evaluation config
│   ├── tracking_eval.json       # Tracking evaluation config
│   └── anomaly_eval.json        # Anomaly detection config
├── datasets/                    # Test datasets
│   ├── visdrone/                # VisDrone dataset subset
│   ├── mot/                     # MOT dataset samples
│   ├── anomaly_test.json        # Anomaly test data
│   └── threat_test.json         # Threat intelligence test data
├── repositories/                # External repositories
│   ├── ultralytics/             # YOLOv8 repository
│   └── ByteTrack/               # ByteTrack repository
└── results/                     # Evaluation results
    └── evaluation_YYYYMMDD_HHMMSS/
        ├── detection/
        ├── tracking/
        ├── anomaly/
        └── combined_evaluation_report.json
```

## Quick Start

### 1. Setup Evaluation Environment

```bash
cd backend/evaluation
python setup_evaluation.py
```

This will:
- Clone required repositories (YOLOv8, ByteTrack)
- Install evaluation dependencies
- Download/create sample datasets
- Generate configuration files

### 2. Generate Synthetic Test Data

```bash
# Generate threat intelligence data
python test_data_manager.py --action generate --type threat --samples 200

# Generate video detection data
python test_data_manager.py --action generate --type video --samples 100

# Generate anomaly trajectory data
python test_data_manager.py --action generate --type anomaly --samples 150
```

### 3. Run Comprehensive Evaluation

```bash
cd scripts
python run_evaluation.py
```

Or run individual evaluations:

```bash
# Object detection evaluation
python evaluate_detection.py --config ../configs/detection_eval.json --model yolov8n.pt --dataset ../datasets/visdrone --output ../results/detection

# Tracking evaluation
python evaluate_tracking.py --config ../configs/tracking_eval.json --sequence ../datasets/mot/MOT17/train/MOT17-02-FRCNN --output ../results/tracking

# Anomaly detection evaluation
python evaluate_anomaly.py --config ../configs/anomaly_eval.json --data ../datasets/anomaly_test.json --output ../results/anomaly
```

## Evaluation Metrics

### Object Detection
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

### Multi-Object Tracking
- **MOTA**: Multiple Object Tracking Accuracy
- **MOTP**: Multiple Object Tracking Precision
- **IDF1**: ID F1 Score (identity preservation)
- **MT**: Mostly Tracked trajectories
- **ML**: Mostly Lost trajectories
- **FP**: False Positives
- **FN**: False Negatives
- **IDs**: ID Switches
- **Frag**: Fragmentations

### Anomaly Detection
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Accuracy**: (True Positives + True Negatives) / Total Samples

## Test Data Management

The test data management system provides:

### Dataset Versioning
```bash
# Create a new version of a dataset
python test_data_manager.py --action version --dataset threat_intel --path datasets/threat_test.json --description "Updated threat samples"

# List all managed datasets
python test_data_manager.py --action list
```

### Dataset Validation
```bash
# Validate a threat intelligence dataset
python test_data_manager.py --action validate --type threat --path datasets/threat_test.json

# Validate a video detection dataset
python test_data_manager.py --action validate --type video --path datasets/visdrone

# Validate an anomaly detection dataset
python test_data_manager.py --action validate --type anomaly --path datasets/anomaly_test.json
```

## Configuration Files

### Detection Evaluation Config (`configs/detection_eval.json`)
```json
{
  "model_path": "models/yolov8n.pt",
  "dataset_path": "datasets/visdrone",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.5,
  "classes": ["pedestrian", "people", "bicycle", "car", "van", "truck"],
  "metrics": ["mAP", "mAP50", "mAP75", "precision", "recall"]
}
```

### Tracking Evaluation Config (`configs/tracking_eval.json`)
```json
{
  "dataset_path": "datasets/mot",
  "tracker_config": {
    "track_thresh": 0.5,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "frame_rate": 30
  },
  "metrics": ["MOTA", "MOTP", "IDF1", "MT", "ML", "FP", "FN", "IDs"]
}
```

### Anomaly Detection Config (`configs/anomaly_eval.json`)
```json
{
  "models": ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"],
  "contamination": 0.1,
  "features": ["speed", "curvature", "duration", "direction_change"],
  "metrics": ["ROC-AUC", "precision", "recall", "f1_score"]
}
```

## Results and Reporting

Evaluation results are saved in timestamped directories under `results/`. Each evaluation generates:

- **JSON reports** with detailed metrics
- **Visualization plots** (confusion matrices, performance charts)
- **Combined evaluation report** with summary metrics
- **Configuration snapshots** for reproducibility

### Example Results Structure
```
results/evaluation_20241225_143022/
├── detection/
│   ├── evaluation_results.json
│   ├── detection_metrics.png
│   └── confusion_matrix.png
├── tracking/
│   ├── tracking_results.json
│   ├── tracking_metrics.png
│   └── tracks_timeline.png
├── anomaly/
│   ├── anomaly_results.json
│   ├── anomaly_metrics.png
│   └── feature_distributions.png
├── combined_evaluation_report.json
└── evaluation_summary.png
```

## Integration with GitHub Actions

The evaluation framework can be integrated with GitHub Actions for automated testing:

```yaml
name: Model Evaluation
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        cd backend/evaluation
        python setup_evaluation.py
    - name: Run evaluation
      run: |
        cd backend/evaluation/scripts
        python run_evaluation.py
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: evaluation-results
        path: backend/evaluation/results/
```

## Performance Targets

The HifazatAI system aims to achieve the following performance targets:

- **Object Detection**: mAP@0.5 ≥ 0.85
- **Multi-Object Tracking**: MOTA ≥ 0.70, IDF1 ≥ 0.65
- **Anomaly Detection**: F1-Score ≥ 0.70, ROC-AUC ≥ 0.80
- **Threat Intelligence**: Precision ≥ 0.80 for high-risk classifications

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues**
   - The evaluation framework is designed to work on CPU
   - For GPU acceleration, ensure CUDA-compatible PyTorch is installed

3. **Dataset Loading Errors**
   - Verify dataset paths in configuration files
   - Run dataset validation to check format compliance

4. **Memory Issues**
   - Reduce batch sizes in configuration
   - Use smaller dataset subsets for testing

### Getting Help

For issues and questions:
1. Check the troubleshooting section above
2. Review configuration files for correct paths
3. Validate datasets using the test data manager
4. Check evaluation logs for detailed error messages

## Contributing

To add new evaluation metrics or datasets:

1. Follow the existing code structure
2. Add appropriate configuration options
3. Include validation and error handling
4. Update documentation and examples
5. Add unit tests for new functionality