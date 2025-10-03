"""
evaluation.py

Comprehensive evaluation metrics for novelty detection
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, auc, 
    classification_report, confusion_matrix, roc_curve, 
    accuracy_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NoveltyDetectionEvaluator:
    """Comprehensive evaluation for novelty detection models"""
    
    def __init__(self):
        self.results = {}
    
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_scores: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', pos_label=1),
            'recall': recall_score(y_true, y_pred, average='binary', pos_label=1),
            'f1_score': f1_score(y_true, y_pred, average='binary', pos_label=1),
            'roc_auc': roc_auc_score(y_true, y_scores)
        }
        
        # Compute specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def compute_advanced_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Compute advanced detection metrics"""
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fpr_at_95_tpr = self._compute_fpr_at_tpr(fpr, tpr, target_tpr=0.95)
        
        # Youden's Index (Sensitivity + Specificity - 1)
        youdens_index = np.max(tpr - fpr)
        
        # AUSUC (Area Under Seen-Unseen Curve) - approximation using ROC AUC
        ausuc = roc_auc_score(y_true, y_scores)
        
        return {
            'pr_auc': pr_auc,
            'fpr_at_95_tpr': fpr_at_95_tpr,
            'tnr_at_95_tpr': 1 - fpr_at_95_tpr,
            'youdens_index': youdens_index,
            'ausuc': ausuc
        }
    
    def compute_calibration_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Compute calibration metrics"""
        # Convert scores to probabilities (sigmoid-like transformation)
        y_prob = 1 / (1 + np.exp(-y_scores))
        
        # Brier Score
        brier_score = np.mean((y_true - y_prob) ** 2)
        
        # Expected Calibration Error (simplified)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'brier_score': brier_score,
            'ece': ece
        }
    
    def compute_zero_shot_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 class_labels: Optional[np.ndarray] = None,
                                 is_seen_class: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute zero-shot learning specific metrics
        
        Args:
            y_true: True binary labels (0=seen/normal, 1=novel/unseen)
            y_pred: Predicted binary labels
            class_labels: Class indices for each sample
            is_seen_class: Boolean array indicating if each class is seen (True) or unseen (False)
        """
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        if class_labels is None:
            return {
                'seen_accuracy': 0.0,
                'unseen_accuracy': 0.0, 
                'harmonic_mean': 0.0,
                'overall_accuracy': overall_accuracy,
                'mean_per_class_accuracy': 0.0,
                'std_per_class_accuracy': 0.0
            }
        
        # Compute per-class accuracies
        unique_classes = np.unique(class_labels)
        per_class_acc = []
        seen_class_accs = []
        unseen_class_accs = []
        
        for cls in unique_classes:
            mask = class_labels == cls
            if mask.sum() > 0:
                cls_acc = accuracy_score(y_true[mask], y_pred[mask])
                per_class_acc.append(cls_acc)
                
                # Determine if this is a seen or unseen class
                if is_seen_class is not None and len(is_seen_class) > cls:
                    if is_seen_class[cls]:
                        seen_class_accs.append(cls_acc)
                    else:
                        unseen_class_accs.append(cls_acc)
                else:
                    # Fallback: use y_true to determine seen vs unseen
                    # If majority of samples in this class are labeled as seen (0), it's a seen class
                    if np.mean(y_true[mask]) < 0.5:
                        seen_class_accs.append(cls_acc)
                    else:
                        unseen_class_accs.append(cls_acc)
        
        # Calculate metrics
        mean_acc = np.mean(per_class_acc) if per_class_acc else 0.0
        std_acc = np.std(per_class_acc) if per_class_acc else 0.0
        
        seen_accuracy = np.mean(seen_class_accs) if seen_class_accs else 0.0
        unseen_accuracy = np.mean(unseen_class_accs) if unseen_class_accs else 0.0
        
        # Harmonic mean (standard zero-shot learning metric)
        if seen_accuracy + unseen_accuracy > 0:
            harmonic_mean = 2 * (seen_accuracy * unseen_accuracy) / (seen_accuracy + unseen_accuracy)
        else:
            harmonic_mean = 0.0
        
        return {
            'seen_accuracy': seen_accuracy,
            'unseen_accuracy': unseen_accuracy,
            'harmonic_mean': harmonic_mean,
            'overall_accuracy': overall_accuracy,
            'mean_per_class_accuracy': mean_acc,
            'std_per_class_accuracy': std_acc,
            'n_seen_classes': len(seen_class_accs),
            'n_unseen_classes': len(unseen_class_accs)
        }
    
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_scores: np.ndarray, class_labels: Optional[np.ndarray] = None,
                              is_seen_class: Optional[np.ndarray] = None) -> Dict:
        """Perform comprehensive evaluation"""
        
        logger.info("Computing comprehensive evaluation metrics...")
        
        results = {}
        
        # Basic metrics
        results.update(self.compute_basic_metrics(y_true, y_pred, y_scores))
        
        # Advanced metrics
        results.update(self.compute_advanced_metrics(y_true, y_scores))
        
        # Calibration metrics
        results.update(self.compute_calibration_metrics(y_true, y_scores))
        
        # Zero-shot metrics with proper class information
        results.update(self.compute_zero_shot_metrics(y_true, y_pred, class_labels, is_seen_class))
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        self.results = results
        return results
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print detailed evaluation report"""
        if not self.results:
            logger.error("No results available. Run evaluate_comprehensive first.")
            return
        
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*50)
        
        print("\nBINARY CLASSIFICATION METRICS:")
        print(f"- ROC AUC: {self.results['roc_auc']:.4f}")
        print(f"- PR AUC: {self.results['pr_auc']:.4f}")
        print(f"- Accuracy: {self.results['accuracy']:.4f}")
        print(f"- Precision: {self.results['precision']:.4f}")
        print(f"- Recall: {self.results['recall']:.4f}")
        print(f"- F1-Score: {self.results['f1_score']:.4f}")
        print(f"- Specificity: {self.results['specificity']:.4f}")
        
        print("\nADVANCED DETECTION METRICS:")
        print(f"- FPR@95% TPR: {self.results['fpr_at_95_tpr']:.4f}")
        print(f"- TNR@95% TPR: {self.results['tnr_at_95_tpr']:.4f}")
        print(f"- Youden's Index: {self.results['youdens_index']:.4f}")
        
        print("\nZERO-SHOT LEARNING METRICS:")
        print(f"- Seen Accuracy (Acc_S): {self.results['seen_accuracy']:.4f}")
        print(f"- Unseen Accuracy (Acc_U): {self.results['unseen_accuracy']:.4f}")
        print(f"- Harmonic Mean (H): {self.results['harmonic_mean']:.4f}")
        print(f"- Overall Accuracy: {self.results['overall_accuracy']:.4f}")
        
        print("\nCALIBRATION METRICS:")
        print(f"- Expected Calibration Error (ECE): {self.results['ece']:.4f}")
        print(f"- Brier Score: {self.results['brier_score']:.4f}")
        
        print("\nPER-CLASS ANALYSIS:")
        print(f"- Mean Per-Class Accuracy: {self.results['mean_per_class_accuracy']:.4f} ± {self.results['std_per_class_accuracy']:.4f}")
        
        print("\nCONFUSION MATRIX:")
        print(self.results['confusion_matrix'])
        
        print(f"\nAUSUC (Area Under Seen-Unseen Accuracy Curve): {self.results['ausuc']:.4f}")
        
        # Classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, 
                                  target_names=["Seen (Normal)", "Novel (Anomaly)"]))
    
    def _compute_fpr_at_tpr(self, fpr: np.ndarray, tpr: np.ndarray, 
                           target_tpr: float = 0.95) -> float:
        """Compute FPR at specific TPR threshold"""
        if len(tpr) == 0:
            return 1.0
        
        # Find the index where TPR is closest to target
        idx = np.argmax(tpr >= target_tpr)
        if tpr[idx] < target_tpr:
            return 1.0  # If we can't reach target TPR
        
        return fpr[idx]


class VisualizationUtils:
    """Visualization utilities for novelty detection"""
    
    @staticmethod
    def plot_performance_curves(y_true: np.ndarray, y_scores: np.ndarray, 
                               title: str = "PCA + LOF Performance", 
                               save_path: Optional[str] = None):
        """Plot ROC and Precision-Recall curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{title} - ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})', linewidth=2)
        ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, 
                   label='Random baseline')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{title} - Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_score_distribution(y_true: np.ndarray, y_scores: np.ndarray, 
                               title: str = "Score Distribution",
                               save_path: Optional[str] = None):
        """Plot distribution of novelty scores"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate scores by class
        normal_scores = y_scores[y_true == 0]
        novel_scores = y_scores[y_true == 1]
        
        # Plot histograms
        ax.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', 
                color='blue', density=True)
        ax.hist(novel_scores, bins=50, alpha=0.7, label=f'Novel (n={len(novel_scores)})', 
                color='red', density=True)
        
        ax.set_xlabel('Novelty Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: list = ["Normal", "Novel"],
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_pca_components(pca_model, n_components: int = 10,
                           title: str = "PCA Explained Variance",
                           save_path: Optional[str] = None):
        """Plot PCA explained variance"""
        explained_var = pca_model.explained_variance_ratio_[:n_components]
        cumulative_var = np.cumsum(explained_var)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Individual component variance
        ax1.bar(range(1, len(explained_var) + 1), explained_var)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Component Variance')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()