#!/usr/bin/env python3
"""
Anomaly Detection Evaluation Script
Evaluates anomaly detection models using ROC-AUC, precision, recall, and F1-score.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnomalyMetrics:
    """Container for anomaly detection evaluation metrics."""
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    threshold: float

@dataclass
class TrajectoryFeatures:
    """Container for trajectory features."""
    speed: float
    curvature: float
    duration: float
    direction_change: float
    acceleration: float
    path_efficiency: float

class AnomalyEvaluator:
    """Evaluates anomaly detection model performance."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.models = self.config.get('models', ['IsolationForest'])
        self.contamination = self.config.get('contamination', 0.1)
        self.feature_names = self.config.get('features', [
            'speed', 'curvature', 'duration', 'direction_change'
        ])
        self.scaler = StandardScaler()
    
    def load_test_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load test data with trajectories and labels."""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        features = []
        labels = []
        trajectories = []
        
        # Process normal trajectories
        for traj in data.get('normal_trajectories', []):
            traj_features = self.extract_trajectory_features(traj['points'])
            features.append(traj_features)
            labels.append(0)  # Normal = 0
            trajectories.append(traj)
        
        # Process anomalous trajectories
        for traj in data.get('anomalous_trajectories', []):
            traj_features = self.extract_trajectory_features(traj['points'])
            features.append(traj_features)
            labels.append(1)  # Anomaly = 1
            trajectories.append(traj)
        
        return np.array(features), np.array(labels), trajectories
    
    def extract_trajectory_features(self, points: List[List[float]]) -> List[float]:
        """Extract features from trajectory points."""
        if len(points) < 2:
            return [0.0] * len(self.feature_names)
        
        points = np.array(points)
        
        # Calculate speed
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        speed = np.mean(distances) if len(distances) > 0 else 0.0
        
        # Calculate curvature
        if len(points) >= 3:
            # Use angle changes as a proxy for curvature
            vectors = np.diff(points, axis=0)
            angles = []
            for i in range(len(vectors) - 1):
                v1, v2 = vectors[i], vectors[i + 1]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            curvature = np.mean(angles) if angles else 0.0
        else:
            curvature = 0.0
        
        # Calculate duration (proxy based on number of points)
        duration = len(points)
        
        # Calculate direction changes
        if len(points) >= 3:
            direction_changes = 0
            for i in range(1, len(points) - 1):
                v1 = points[i] - points[i-1]
                v2 = points[i+1] - points[i]
                
                # Normalize vectors
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
                
                # Calculate angle change
                dot_product = np.dot(v1_norm, v2_norm)
                dot_product = np.clip(dot_product, -1, 1)
                angle_change = np.arccos(dot_product)
                
                if angle_change > np.pi / 4:  # 45 degrees threshold
                    direction_changes += 1
            
            direction_change = direction_changes / (len(points) - 2)
        else:
            direction_change = 0.0
        
        # Calculate acceleration
        if len(distances) >= 2:
            accelerations = np.diff(distances)
            acceleration = np.std(accelerations)
        else:
            acceleration = 0.0
        
        # Calculate path efficiency (straight line distance / actual path length)
        if len(points) >= 2:
            straight_distance = np.linalg.norm(points[-1] - points[0])
            actual_distance = np.sum(distances)
            path_efficiency = straight_distance / (actual_distance + 1e-8)
        else:
            path_efficiency = 1.0
        
        # Return features in the order specified in config
        feature_dict = {
            'speed': speed,
            'curvature': curvature,
            'duration': duration,
            'direction_change': direction_change,
            'acceleration': acceleration,
            'path_efficiency': path_efficiency
        }
        
        return [feature_dict.get(name, 0.0) for name in self.feature_names]
    
    def create_model(self, model_name: str) -> Any:
        """Create anomaly detection model."""
        if model_name == 'IsolationForest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif model_name == 'OneClassSVM':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif model_name == 'LocalOutlierFactor':
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def evaluate_model(self, model_name: str, X_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> AnomalyMetrics:
        """Evaluate a single anomaly detection model."""
        # Create and train model
        model = self.create_model(model_name)
        
        # Fit model on training data (assumed to be mostly normal)
        model.fit(X_train)
        
        # Get anomaly scores
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test)
        elif hasattr(model, 'score_samples'):
            scores = model.score_samples(X_test)
        else:
            # For models that only provide binary predictions
            predictions = model.predict(X_test)
            scores = predictions.astype(float)
        
        # Convert scores to probabilities (higher score = more anomalous)
        if model_name == 'IsolationForest':
            # Isolation Forest: negative scores for anomalies
            anomaly_scores = -scores
        elif model_name == 'OneClassSVM':
            # One-Class SVM: negative scores for anomalies
            anomaly_scores = -scores
        elif model_name == 'LocalOutlierFactor':
            # LOF: negative scores for anomalies
            anomaly_scores = -scores
        else:
            anomaly_scores = scores
        
        # Normalize scores to [0, 1]
        if len(np.unique(anomaly_scores)) > 1:
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Calculate ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, anomaly_scores)
        except ValueError:
            roc_auc = 0.5  # Random performance if all labels are the same
        
        # Find optimal threshold using precision-recall curve
        if len(np.unique(y_test)) > 1:
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, anomaly_scores)
            f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
            best_threshold_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        else:
            optimal_threshold = 0.5
        
        # Make binary predictions
        y_pred = (anomaly_scores >= optimal_threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = np.mean(y_test == y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        
        return AnomalyMetrics(
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            threshold=optimal_threshold
        )
    
    def evaluate_all_models(self, test_data_path: str, output_dir: str) -> Dict[str, AnomalyMetrics]:
        """Evaluate all configured models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        X, y, trajectories = self.load_test_data(test_data_path)
        
        if len(X) == 0:
            raise ValueError("No test data loaded")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (use normal data for training, all data for testing)
        normal_indices = np.where(y == 0)[0]
        X_train = X_scaled[normal_indices]
        X_test = X_scaled
        y_test = y
        
        print(f"Training on {len(X_train)} normal samples")
        print(f"Testing on {len(X_test)} samples ({np.sum(y_test)} anomalies)")
        
        results = {}
        
        for model_name in self.models:
            print(f"\nEvaluating {model_name}...")
            try:
                metrics = self.evaluate_model(model_name, X_train, X_test, y_test)
                results[model_name] = metrics
                
                print(f"  ROC-AUC: {metrics.roc_auc:.3f}")
                print(f"  Precision: {metrics.precision:.3f}")
                print(f"  Recall: {metrics.recall:.3f}")
                print(f"  F1-Score: {metrics.f1_score:.3f}")
                
            except Exception as e:
                print(f"  Failed to evaluate {model_name}: {e}")
                continue
        
        # Save results and create visualizations
        self.save_results(results, X_test, y_test, trajectories, output_path)
        self.create_visualizations(results, X_test, y_test, output_path)
        
        return results
    
    def save_results(self, results: Dict[str, AnomalyMetrics], X_test: np.ndarray,
                    y_test: np.ndarray, trajectories: List[Dict], output_path: Path):
        """Save evaluation results to JSON file."""
        results_dict = {
            'model_results': {},
            'dataset_info': {
                'num_samples': len(X_test),
                'num_anomalies': int(np.sum(y_test)),
                'num_normal': int(len(y_test) - np.sum(y_test)),
                'feature_names': self.feature_names
            },
            'configuration': self.config
        }
        
        for model_name, metrics in results.items():
            results_dict['model_results'][model_name] = {
                'roc_auc': metrics.roc_auc,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'accuracy': metrics.accuracy,
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'true_negatives': metrics.true_negatives,
                'false_negatives': metrics.false_negatives,
                'threshold': metrics.threshold
            }
        
        with open(output_path / 'anomaly_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path / 'anomaly_results.json'}")
    
    def create_visualizations(self, results: Dict[str, AnomalyMetrics], 
                            X_test: np.ndarray, y_test: np.ndarray, output_path: Path):
        """Create visualization plots for anomaly detection evaluation."""
        plt.style.use('seaborn-v0_8')
        
        # Model comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC-AUC comparison
        model_names = list(results.keys())
        roc_aucs = [results[name].roc_auc for name in model_names]
        
        bars1 = ax1.bar(model_names, roc_aucs, color='skyblue', alpha=0.7)
        ax1.set_title('ROC-AUC Comparison')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_ylim(0, 1)
        
        for bar, value in zip(bars1, roc_aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Precision, Recall, F1 comparison
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(model_names))
        width = 0.25
        
        precisions = [results[name].precision for name in model_names]
        recalls = [results[name].recall for name in model_names]
        f1_scores = [results[name].f1_score for name in model_names]
        
        ax2.bar(x - width, precisions, width, label='Precision', alpha=0.7)
        ax2.bar(x, recalls, width, label='Recall', alpha=0.7)
        ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)
        
        ax2.set_title('Precision, Recall, F1-Score Comparison')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('Model')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Confusion matrix for best model (highest F1-score)
        best_model = max(results.keys(), key=lambda k: results[k].f1_score)
        best_metrics = results[best_model]
        
        cm = np.array([[best_metrics.true_negatives, best_metrics.false_positives],
                      [best_metrics.false_negatives, best_metrics.true_positives]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        ax3.set_title(f'Confusion Matrix - {best_model}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Feature importance (if available)
        if len(self.feature_names) > 0 and len(X_test) > 0:
            # Calculate feature statistics
            normal_features = X_test[y_test == 0]
            anomaly_features = X_test[y_test == 1]
            
            if len(normal_features) > 0 and len(anomaly_features) > 0:
                feature_diffs = []
                for i in range(len(self.feature_names)):
                    normal_mean = np.mean(normal_features[:, i])
                    anomaly_mean = np.mean(anomaly_features[:, i])
                    diff = abs(anomaly_mean - normal_mean)
                    feature_diffs.append(diff)
                
                bars4 = ax4.bar(self.feature_names, feature_diffs, color='lightcoral', alpha=0.7)
                ax4.set_title('Feature Importance (Mean Difference)')
                ax4.set_ylabel('Absolute Difference')
                
                for bar, value in zip(bars4, feature_diffs):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'anomaly_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature distribution plots
        if len(self.feature_names) > 0 and len(X_test) > 0:
            self.create_feature_distributions(X_test, y_test, output_path)
        
        print(f"Visualizations saved to {output_path}")
    
    def create_feature_distributions(self, X_test: np.ndarray, y_test: np.ndarray, 
                                   output_path: Path):
        """Create feature distribution plots."""
        n_features = len(self.feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature_name in enumerate(self.feature_names):
            ax = axes[i] if n_features > 1 else axes
            
            normal_values = X_test[y_test == 0, i]
            anomaly_values = X_test[y_test == 1, i]
            
            ax.hist(normal_values, bins=20, alpha=0.7, label='Normal', color='blue')
            ax.hist(anomaly_values, bins=20, alpha=0.7, label='Anomaly', color='red')
            
            ax.set_title(f'{feature_name} Distribution')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for anomaly detection evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection models')
    parser.add_argument('--config', required=True, help='Path to evaluation config file')
    parser.add_argument('--data', required=True, help='Path to test data file')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AnomalyEvaluator(args.config)
    
    # Run evaluation
    try:
        print("Starting anomaly detection evaluation...")
        results = evaluator.evaluate_all_models(args.data, args.output)
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION EVALUATION RESULTS")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  ROC-AUC:     {metrics.roc_auc:.3f}")
            print(f"  Precision:   {metrics.precision:.3f}")
            print(f"  Recall:      {metrics.recall:.3f}")
            print(f"  F1-Score:    {metrics.f1_score:.3f}")
            print(f"  Accuracy:    {metrics.accuracy:.3f}")
            print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}")
            print(f"  TN: {metrics.true_negatives}, FN: {metrics.false_negatives}")
        
        # Find best model
        if results:
            best_model = max(results.keys(), key=lambda k: results[k].f1_score)
            print(f"\nBest Model: {best_model} (F1-Score: {results[best_model].f1_score:.3f})")
        
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()