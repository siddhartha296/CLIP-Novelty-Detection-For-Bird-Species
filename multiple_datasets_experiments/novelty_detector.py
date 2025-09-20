"""
Novelty detection using PCA + Local Outlier Factor (LOF)
"""
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PCALOFNoveltyDetector:
    """PCA + LOF based novelty detection model"""
    
    def __init__(self, n_components: int = 50, n_neighbors: int = 10, 
                 contamination: float = 0.35, metric: str = 'cosine'):
        """
        Initialize the novelty detector
        
        Args:
            n_components: Number of PCA components
            n_neighbors: Number of neighbors for LOF
            contamination: Contamination parameter for LOF
            metric: Distance metric for LOF
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        
        # Components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True
        )
        
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray) -> 'PCALOFNoveltyDetector':
        """
        Fit the novelty detector
        
        Args:
            X_train: Training data (only seen/normal samples)
        """
        logger.info("Fitting PCA + LOF novelty detector...")
        
        # Step 1: Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Step 2: Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        
        # Step 3: Fit LOF
        self.lof.fit(X_train_pca)
        
        self.is_fitted = True
        
        logger.info(f"✅ Model fitted with {self.n_components} PCA components")
        logger.info(f"   Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict novelty labels (0=normal, 1=novel)
        
        Args:
            X_test: Test data
            
        Returns:
            Binary predictions (0=normal, 1=novel)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_processed = self._preprocess(X_test)
        predictions = (self.lof.predict(X_test_processed) == -1).astype(int)
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict novelty scores (higher = more novel)
        
        Args:
            X_test: Test data
            
        Returns:
            Novelty scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_processed = self._preprocess(X_test)
        scores = -self.lof.score_samples(X_test_processed)
        return scores
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling and PCA transformation"""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model
        
        Args:
            X_test: Test features
            y_test: Test labels (0=normal, 1=novel)
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_scores = self.predict_proba(X_test)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_scores)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
        
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        
        results = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1,
            'variance_explained': self.pca.explained_variance_ratio_.sum(),
            'y_pred': y_pred,
            'y_scores': y_scores
        }
        
        return results
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'lof': self.lof,
            'params': {
                'n_components': self.n_components,
                'n_neighbors': self.n_neighbors,
                'contamination': self.contamination,
                'metric': self.metric
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'PCALOFNoveltyDetector':
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        params = model_data['params']
        detector = cls(**params)
        
        # Restore fitted components
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
        detector.lof = model_data['lof']
        detector.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return detector


class HyperparameterOptimizer:
    """Hyperparameter optimization for PCA + LOF"""
    
    def __init__(self, search_space: Dict[str, list]):
        """
        Initialize optimizer with search space
        
        Args:
            search_space: Dictionary with parameter ranges
                Example: {
                    'n_components': [30, 40, 50],
                    'n_neighbors': [5, 10, 15],
                    'contamination': [0.3, 0.35, 0.4]
                }
        """
        self.search_space = search_space
        self.best_params = None
        self.best_score = 0
        self.results = []
    
    def optimize(self, X_train: np.ndarray, X_test: np.ndarray, 
                y_test: np.ndarray, metric: str = 'roc_auc') -> Tuple[Dict, Dict]:
        """
        Perform hyperparameter optimization
        
        Args:
            X_train: Training data
            X_test: Test data  
            y_test: Test labels
            metric: Optimization metric
            
        Returns:
            best_params, best_results
        """
        logger.info("Starting hyperparameter optimization...")
        
        from itertools import product
        
        # Get parameter combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        total_combinations = np.prod([len(vals) for vals in param_values])
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            try:
                # Train model with current parameters
                detector = PCALOFNoveltyDetector(**params)
                detector.fit(X_train)
                
                # Evaluate
                results = detector.evaluate(X_test, y_test)
                score = results[metric]
                
                # Store results
                result_entry = {**params, **results, 'score': score}
                self.results.append(result_entry)
                
                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    self.best_results = results.copy()
                
                if i % 10 == 0:
                    logger.info(f"Completed {i+1}/{total_combinations} combinations")
                    
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue
        
        logger.info(f"🏆 Best {metric}: {self.best_score:.4f}")
        logger.info(f"🏆 Best parameters: {self.best_params}")
        
        return self.best_params, self.best_results
    
    def get_results_summary(self) -> Dict:
        """Get summary of optimization results"""
        if not self.results:
            return {}
        
        scores = [r['score'] for r in self.results]
        return {
            'n_trials': len(self.results),
            'best_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'best_params': self.best_params
        }
