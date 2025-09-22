"""
Evaluation utilities for border anomaly detection models.
Provides comprehensive evaluation metrics and comparison tools.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from .trajectory import Trajectory
from .anomaly_detector import (
    AnomalyDetector, AnomalyResult, IsolationForestDetector,
    MotionBasedDetector, PyODEnsembleDetector, create_synthetic_anomaly_data
)
from utils.logging import get_pipeline_logger

logger = get_pipeline_logger("border_anomaly_evaluation")


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for anomaly detection"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_score: Optional[float]
    confusion_matrix: np.ndarray
    training_time: float
    prediction_time: float
    memory_usage: Optional[float] = None
    
    def __str__(self) -> str:
        return (f"Precision: {self.precision:.3f}, Recall: {self.recall:.3f}, "
                f"F1: {self.f1_score:.3f}, Accuracy: {self.accuracy:.3f}, "
                f"Training: {self.training_time:.3f}s, Prediction: {self.prediction_time:.3f}s")


class AnomalyDetectionEvaluator:
    """Comprehensive evaluator for anomaly detection models"""
    
    def __init__(self, test_data_size: Tuple[int, int] = (100, 20)):
        """
        Initialize evaluator
        
        Args:
            test_data_size: Tuple of (num_normal, num_anomalies) for test data
        """
        self.test_data_size = test_data_size
        self.test_trajectories = None
        self.test_labels = None
        self.results = {}
    
    def generate_test_data(self) -> Tuple[List[Trajectory], List[bool]]:
        """Generate synthetic test data"""
        logger.info(f"Generating test data: {self.test_data_size[0]} normal, {self.test_data_size[1]} anomalies")
        
        self.test_trajectories, self.test_labels = create_synthetic_anomaly_data(
            num_normal=self.test_data_size[0],
            num_anomalies=self.test_data_size[1]
        )
        
        return self.test_trajectories, self.test_labels
    
    def evaluate_detector(self, detector: AnomalyDetector, name: str,
                         train_on_normal_only: bool = True) -> EvaluationMetrics:
        """
        Evaluate a single anomaly detector
        
        Args:
            detector: The anomaly detector to evaluate
            name: Name for the detector (for logging)
            train_on_normal_only: Whether to train only on normal data
        
        Returns:
            EvaluationMetrics object with comprehensive metrics
        """
        if self.test_trajectories is None:
            self.generate_test_data()
        
        logger.info(f"Evaluating {name} detector...")
        
        # Prepare training data
        if train_on_normal_only:
            training_trajectories = [
                t for t, l in zip(self.test_trajectories, self.test_labels) if not l
            ]
        else:
            training_trajectories = self.test_trajectories
        
        # Training phase
        start_time = time.time()
        try:
            detector.fit(training_trajectories)
            training_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
            return self._create_failed_metrics()
        
        # Prediction phase
        start_time = time.time()
        predictions = []
        scores = []
        
        try:
            for trajectory in self.test_trajectories:
                result = detector.predict(trajectory)
                predictions.append(result.is_anomaly)
                scores.append(result.confidence)
            
            prediction_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Prediction failed for {name}: {e}")
            return self._create_failed_metrics()
        
        # Calculate metrics
        y_true = self.test_labels
        y_pred = predictions
        y_scores = scores
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # AUC score (if we have meaningful scores)
        try:
            auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None
        except ValueError:
            auc = None
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            auc_score=auc,
            confusion_matrix=cm,
            training_time=training_time,
            prediction_time=prediction_time
        )
        
        self.results[name] = metrics
        logger.info(f"{name} evaluation completed: {metrics}")
        
        return metrics
    
    def _create_failed_metrics(self) -> EvaluationMetrics:
        """Create metrics object for failed evaluation"""
        return EvaluationMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            auc_score=None,
            confusion_matrix=np.array([[0, 0], [0, 0]]),
            training_time=0.0,
            prediction_time=0.0
        )
    
    def compare_detectors(self, detectors: Dict[str, AnomalyDetector]) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple anomaly detectors
        
        Args:
            detectors: Dictionary of {name: detector} pairs
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Comparing {len(detectors)} detectors...")
        
        results = {}
        for name, detector in detectors.items():
            try:
                metrics = self.evaluate_detector(detector, name)
                results[name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                results[name] = self._create_failed_metrics()
        
        # Log comparison summary
        self._log_comparison_summary(results)
        
        return results
    
    def _log_comparison_summary(self, results: Dict[str, EvaluationMetrics]) -> None:
        """Log a summary comparison of all detectors"""
        logger.info("\n" + "="*80)
        logger.info("ANOMALY DETECTION COMPARISON SUMMARY")
        logger.info("="*80)
        
        # Sort by F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1].f1_score, reverse=True)
        
        for name, metrics in sorted_results:
            logger.info(f"{name:20s} | {metrics}")
        
        logger.info("="*80)
    
    def plot_comparison(self, results: Optional[Dict[str, EvaluationMetrics]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Plot comparison of detector performance
        
        Args:
            results: Results to plot (uses self.results if None)
            save_path: Path to save the plot
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Anomaly Detection Performance Comparison', fontsize=16)
        
        names = list(results.keys())
        
        # Performance metrics
        metrics_data = {
            'Precision': [results[name].precision for name in names],
            'Recall': [results[name].recall for name in names],
            'F1-Score': [results[name].f1_score for name in names],
            'Accuracy': [results[name].accuracy for name in names]
        }
        
        # Plot 1: Performance metrics
        ax1 = axes[0, 0]
        x = np.arange(len(names))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax1.bar(x + i * width, values, width, label=metric)
        
        ax1.set_xlabel('Detectors')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Timing comparison
        ax2 = axes[0, 1]
        training_times = [results[name].training_time for name in names]
        prediction_times = [results[name].prediction_time for name in names]
        
        x = np.arange(len(names))
        ax2.bar(x - 0.2, training_times, 0.4, label='Training Time', alpha=0.7)
        ax2.bar(x + 0.2, prediction_times, 0.4, label='Prediction Time', alpha=0.7)
        
        ax2.set_xlabel('Detectors')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Performance Timing')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Precision vs Recall
        ax3 = axes[1, 0]
        precisions = [results[name].precision for name in names]
        recalls = [results[name].recall for name in names]
        
        ax3.scatter(recalls, precisions, s=100, alpha=0.7)
        for i, name in enumerate(names):
            ax3.annotate(name, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Confusion matrices (for best detector)
        ax4 = axes[1, 1]
        best_detector = max(results.items(), key=lambda x: x[1].f1_score)
        cm = best_detector[1].confusion_matrix
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title(f'Confusion Matrix - {best_detector[0]}')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Optional[Dict[str, EvaluationMetrics]] = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: Results to include in report (uses self.results if None)
        
        Returns:
            Formatted evaluation report as string
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No evaluation results available."
        
        report = []
        report.append("BORDER ANOMALY DETECTION EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Test data summary
        if self.test_trajectories:
            report.append(f"Test Data: {len(self.test_trajectories)} trajectories")
            report.append(f"Normal: {sum(1 for l in self.test_labels if not l)}")
            report.append(f"Anomalies: {sum(1 for l in self.test_labels if l)}")
            report.append("")
        
        # Individual detector results
        for name, metrics in results.items():
            report.append(f"Detector: {name}")
            report.append("-" * 30)
            report.append(f"Precision: {metrics.precision:.4f}")
            report.append(f"Recall: {metrics.recall:.4f}")
            report.append(f"F1-Score: {metrics.f1_score:.4f}")
            report.append(f"Accuracy: {metrics.accuracy:.4f}")
            if metrics.auc_score is not None:
                report.append(f"AUC Score: {metrics.auc_score:.4f}")
            report.append(f"Training Time: {metrics.training_time:.4f}s")
            report.append(f"Prediction Time: {metrics.prediction_time:.4f}s")
            report.append("")
        
        # Best performer summary
        best_f1 = max(results.items(), key=lambda x: x[1].f1_score)
        best_recall = max(results.items(), key=lambda x: x[1].recall)
        fastest = min(results.items(), key=lambda x: x[1].training_time + x[1].prediction_time)
        
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Best F1-Score: {best_f1[0]} ({best_f1[1].f1_score:.4f})")
        report.append(f"Best Recall: {best_recall[0]} ({best_recall[1].recall:.4f})")
        report.append(f"Fastest: {fastest[0]} ({fastest[1].training_time + fastest[1].prediction_time:.4f}s total)")
        
        return "\n".join(report)


def run_comprehensive_evaluation() -> Dict[str, EvaluationMetrics]:
    """
    Run comprehensive evaluation of all available anomaly detectors
    
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Starting comprehensive anomaly detection evaluation...")
    
    # Initialize evaluator
    evaluator = AnomalyDetectionEvaluator(test_data_size=(80, 20))
    
    # Initialize detectors
    detectors = {
        'IsolationForest_CPU': IsolationForestDetector(
            contamination=0.2,
            n_estimators=50,
            n_jobs=-1
        ),
        'MotionBased_Adaptive': MotionBasedDetector(
            adaptive_thresholds=True
        ),
        'MotionBased_Fixed': MotionBasedDetector(
            adaptive_thresholds=False,
            speed_threshold=80.0,
            direction_change_threshold=8
        )
    }
    
    # Add PyOD ensemble if available
    try:
        detectors['PyOD_Ensemble'] = PyODEnsembleDetector(
            contamination=0.2,
            combination_method='average'
        )
    except ImportError:
        logger.warning("PyOD not available, skipping PyOD ensemble evaluation")
    
    # Run evaluation
    results = evaluator.compare_detectors(detectors)
    
    # Generate and log report
    report = evaluator.generate_evaluation_report(results)
    logger.info(f"\n{report}")
    
    # Plot comparison
    try:
        evaluator.plot_comparison(results, save_path="anomaly_detection_comparison.png")
    except Exception as e:
        logger.warning(f"Failed to generate comparison plot: {e}")
    
    return results


if __name__ == "__main__":
    # Run evaluation when script is executed directly
    results = run_comprehensive_evaluation()