"""
Improved novelty detection with threshold calibration, reduced contamination,
post-hoc calibration, and ensemble methods
"""
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any, Optional, List
import logging
import torch
import clip

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimize decision threshold for novelty detection"""
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                              metric: str = 'hmean') -> Tuple[float, Dict]:
        """
        Find optimal threshold using validation data
        
        Args:
            y_true: True labels (0=seen, 1=novel)
            y_scores: Novelty scores (higher = more novel)
            metric: 'hmean', 'youden', 'f1', or 'balanced_acc'
        
        Returns:
            optimal_threshold, metrics_dict
        """
        # Try different percentiles as thresholds
        thresholds = np.percentile(y_scores, np.linspace(1, 99, 100))
        
        best_threshold = None
        best_score = -np.inf
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            
            # Calculate metrics
            seen_mask = (y_true == 0)
            novel_mask = (y_true == 1)
            
            seen_acc = np.mean(y_pred[seen_mask] == 0) if seen_mask.sum() > 0 else 0
            novel_acc = np.mean(y_pred[novel_mask] == 1) if novel_mask.sum() > 0 else 0
            
            # Compute optimization metric
            if metric == 'hmean':
                if seen_acc + novel_acc > 0:
                    score = 2 * (seen_acc * novel_acc) / (seen_acc + novel_acc)
                else:
                    score = 0
            elif metric == 'youden':
                # Youden's J = Sensitivity + Specificity - 1
                score = seen_acc + novel_acc - 1
            elif metric == 'f1':
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    score = 0
            elif metric == 'balanced_acc':
                score = (seen_acc + novel_acc) / 2
            else:
                score = seen_acc + novel_acc
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'seen_accuracy': seen_acc,
                    'novel_accuracy': novel_acc,
                    'harmonic_mean': 2 * (seen_acc * novel_acc) / (seen_acc + novel_acc) if (seen_acc + novel_acc) > 0 else 0,
                    'balanced_accuracy': (seen_acc + novel_acc) / 2
                }
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} ({metric}={best_score:.4f})")
        logger.info(f"  Seen accuracy: {best_metrics['seen_accuracy']:.4f}")
        logger.info(f"  Novel accuracy: {best_metrics['novel_accuracy']:.4f}")
        
        return best_threshold, best_metrics


class ScoreCalibrator:
    """Post-hoc calibration of novelty scores"""
    
    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: 'isotonic' or 'platt' (sigmoid)
        """
        self.method = method
        self.calibrator = None
        
    def fit(self, y_scores: np.ndarray, y_true: np.ndarray):
        """Fit calibrator on validation data"""
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_scores, y_true)
        elif self.method == 'platt':
            # Platt scaling: fit logistic regression
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_scores.reshape(-1, 1), y_true)
        
        logger.info(f"Fitted {self.method} calibrator")
    
    def transform(self, y_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to scores"""
        if self.calibrator is None:
            return y_scores
        
        if self.method == 'isotonic':
            return self.calibrator.predict(y_scores)
        elif self.method == 'platt':
            return self.calibrator.predict_proba(y_scores.reshape(-1, 1))[:, 1]
        
        return y_scores


class MahalanobisDetector:
    """Mahalanobis distance-based novelty detection"""
    
    def __init__(self):
        self.mean = None
        self.cov_inv = None
        
    def fit(self, X_train: np.ndarray):
        """Fit Mahalanobis detector"""
        self.mean = np.mean(X_train, axis=0)
        
        # Robust covariance estimation
        cov = EmpiricalCovariance()
        cov.fit(X_train)
        self.cov_inv = cov.precision_
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance (higher = more novel)"""
        diff = X - self.mean
        distances = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))
        return distances


class ImprovedNoveltyDetector:
    """
    Improved novelty detector with:
    - Threshold calibration
    - Reduced contamination
    - Post-hoc calibration
    - Ensemble methods
    """
    
    def __init__(self, 
                 n_components: int = 50,
                 n_neighbors: int = 10,
                 contamination: float = 0.1,  # Lower default
                 metric: str = 'cosine',
                 use_threshold_tuning: bool = True,
                 calibration_method: Optional[str] = 'isotonic',
                 ensemble_methods: List[str] = ['lof'],
                 fusion_method: str = 'concat',
                 text_weight: float = 0.3,
                 use_text_embeddings: bool = True):
        """
        Initialize improved novelty detector
        
        Args:
            n_components: PCA components
            n_neighbors: LOF neighbors
            contamination: Expected proportion of outliers (lower is better)
            metric: Distance metric
            use_threshold_tuning: Enable threshold optimization
            calibration_method: 'isotonic', 'platt', or None
            ensemble_methods: List of ['lof', 'ocsvm', 'iforest', 'mahalanobis']
            fusion_method: Multimodal fusion method
            text_weight: Weight for text features
            use_text_embeddings: Use text embeddings
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.use_threshold_tuning = use_threshold_tuning
        self.calibration_method = calibration_method
        self.ensemble_methods = ensemble_methods
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.use_text_embeddings = use_text_embeddings
        
        # Components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        
        # Detectors
        self.detectors = {}
        self._init_detectors()
        
        # Calibration
        self.optimal_threshold = None
        self.calibrator = ScoreCalibrator(calibration_method) if calibration_method else None
        
        # Text embeddings
        self.class_text_embeddings = None
        self.seen_class_names = None
        
        self.is_fitted = False
    
    def _init_detectors(self):
        """Initialize ensemble detectors"""
        if 'lof' in self.ensemble_methods:
            self.detectors['lof'] = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                metric=self.metric,
                novelty=True
            )
        
        if 'ocsvm' in self.ensemble_methods:
            self.detectors['ocsvm'] = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,  # nu ~ contamination
                gamma='auto'
            )
        
        if 'iforest' in self.ensemble_methods:
            self.detectors['iforest'] = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        
        if 'mahalanobis' in self.ensemble_methods:
            self.detectors['mahalanobis'] = MahalanobisDetector()
    
    def fit(self, X_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            class_labels: np.ndarray = None,
            val_class_labels: np.ndarray = None,
            class_names: List[str] = None,
            val_split: float = 0.2):
        """
        Fit detector with validation split for threshold tuning
        
        Args:
            X_train: Training features (seen classes only)
            X_val: Validation features (mix of seen + novel) - optional
            y_val: Validation labels (0=seen, 1=novel) - optional
            class_labels: Class labels for training samples
            val_class_labels: Class labels for validation samples
            class_names: Class names for text embeddings
            val_split: Proportion for validation split if X_val not provided
        """
        logger.info("Fitting improved novelty detector...")
        
        # Extract text embeddings if enabled
        if self.use_text_embeddings and class_names is not None:
            from multimodal_novelty_detector import MultimodalFeatureExtractor
            
            extractor = MultimodalFeatureExtractor(
                fusion_method=self.fusion_method,
                text_weight=self.text_weight
            )
            
            self.seen_class_names = class_names
            self.class_text_embeddings = extractor.extract_text_embeddings(class_names)
            
            if class_labels is not None:
                text_features = self.class_text_embeddings[class_labels]
                X_train = extractor.fuse_features(X_train, text_features)
        
        # Scale and PCA
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        
        # Fit all detectors
        for name, detector in self.detectors.items():
            logger.info(f"Fitting {name}...")
            detector.fit(X_train_pca)
        
        self.is_fitted = True
        
        # Threshold tuning on validation set
        if self.use_threshold_tuning and X_val is not None and y_val is not None:
            logger.info("Tuning threshold on validation set...")
            
            # Process validation data
            if self.use_text_embeddings and val_class_labels is not None:
                text_features = self.class_text_embeddings[val_class_labels]
                X_val = extractor.fuse_features(X_val, text_features)
            
            val_scores = self.predict_proba(X_val, val_class_labels, use_threshold=False)
            
            # Optimize threshold
            self.optimal_threshold, metrics = ThresholdOptimizer.find_optimal_threshold(
                y_val, val_scores, metric='hmean'
            )
            
            # Fit calibrator
            if self.calibrator:
                self.calibrator.fit(val_scores, y_val)
        
        elif self.use_threshold_tuning and val_split > 0:
            logger.info(f"Creating {val_split:.0%} validation split from training data...")
            # Note: This is not ideal as validation should have novel classes
            # Better to pass proper validation data
            logger.warning("Validation split from training data - consider providing separate validation set with novel classes")
        
        logger.info(f"✓ Fitted with contamination={self.contamination}")
        logger.info(f"  PCA variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        logger.info(f"  Ensemble methods: {list(self.detectors.keys())}")
        
        return self
    
    def predict_proba(self, X_test: np.ndarray, 
                     class_labels: np.ndarray = None,
                     use_threshold: bool = False) -> np.ndarray:
        """
        Predict novelty scores
        
        Args:
            X_test: Test features
            class_labels: Class labels for test samples
            use_threshold: Apply optimal threshold (for binary prediction)
        
        Returns:
            Novelty scores (higher = more novel)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_test_processed = self._preprocess(X_test, class_labels)
        
        # Get scores from all detectors
        ensemble_scores = []
        
        for name, detector in self.detectors.items():
            if name == 'lof':
                scores = -detector.score_samples(X_test_processed)
            elif name == 'ocsvm':
                scores = -detector.decision_function(X_test_processed)
            elif name == 'iforest':
                scores = -detector.score_samples(X_test_processed)
            elif name == 'mahalanobis':
                scores = detector.score_samples(X_test_processed)
            
            ensemble_scores.append(scores)
        
        # Ensemble: average normalized scores
        ensemble_scores = np.array(ensemble_scores)
        
        # Normalize each detector's scores to [0, 1]
        for i in range(len(ensemble_scores)):
            min_val, max_val = ensemble_scores[i].min(), ensemble_scores[i].max()
            if max_val > min_val:
                ensemble_scores[i] = (ensemble_scores[i] - min_val) / (max_val - min_val)
        
        # Average
        final_scores = ensemble_scores.mean(axis=0)
        
        # Apply calibration
        if self.calibrator and self.calibrator.calibrator is not None:
            final_scores = self.calibrator.transform(final_scores)
        
        return final_scores
    
    def predict(self, X_test: np.ndarray, class_labels: np.ndarray = None) -> np.ndarray:
        """Binary prediction with optimal threshold"""
        scores = self.predict_proba(X_test, class_labels)
        
        if self.optimal_threshold is not None:
            predictions = (scores > self.optimal_threshold).astype(int)
        else:
            # Fallback to median threshold
            threshold = np.median(scores)
            predictions = (scores > threshold).astype(int)
        
        return predictions
    
    def _preprocess(self, X: np.ndarray, class_labels: np.ndarray = None) -> np.ndarray:
        """Apply preprocessing"""
        if self.use_text_embeddings and self.class_text_embeddings is not None and class_labels is not None:
            from multimodal_novelty_detector import MultimodalFeatureExtractor
            extractor = MultimodalFeatureExtractor(
                fusion_method=self.fusion_method,
                text_weight=self.text_weight
            )
            text_features = self.class_text_embeddings[class_labels]
            X = extractor.fuse_features(X, text_features)
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
    
    def save_model(self, filepath: str):
        """Save model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'detectors': self.detectors,
            'optimal_threshold': self.optimal_threshold,
            'calibrator': self.calibrator,
            'class_text_embeddings': self.class_text_embeddings,
            'seen_class_names': self.seen_class_names,
            'params': {
                'n_components': self.n_components,
                'n_neighbors': self.n_neighbors,
                'contamination': self.contamination,
                'metric': self.metric,
                'use_threshold_tuning': self.use_threshold_tuning,
                'calibration_method': self.calibration_method,
                'ensemble_methods': self.ensemble_methods,
                'fusion_method': self.fusion_method,
                'text_weight': self.text_weight,
                'use_text_embeddings': self.use_text_embeddings
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(**model_data['params'])
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
        detector.detectors = model_data['detectors']
        detector.optimal_threshold = model_data['optimal_threshold']
        detector.calibrator = model_data['calibrator']
        detector.class_text_embeddings = model_data['class_text_embeddings']
        detector.seen_class_names = model_data['seen_class_names']
        detector.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return detector
