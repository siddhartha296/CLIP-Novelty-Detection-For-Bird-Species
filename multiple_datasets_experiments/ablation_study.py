"""
Comprehensive ablation study comparing all improvements
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AblationStudy:
    """Run comprehensive ablation study"""
    
    def __init__(self, dataset_name: str, data: Dict):
        self.dataset_name = dataset_name
        self.data = data
        self.results = []
        
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all ablation experiments"""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ABLATION STUDY: {self.dataset_name}")
        print(f"{'='*80}\n")
        
        # Create validation split for threshold tuning
        X_val, y_val, val_labels = self._create_validation_split()
        
        experiments = [
            # Baseline
            {
                'name': 'Baseline (Original)',
                'contamination': 0.35,
                'ensemble': ['lof'],
                'threshold_tuning': False,
                'calibration': None,
                'use_text': False
            },
            
            # Effect of contamination
            {
                'name': 'Low Contamination (0.05)',
                'contamination': 0.05,
                'ensemble': ['lof'],
                'threshold_tuning': False,
                'calibration': None,
                'use_text': False
            },
            {
                'name': 'Low Contamination (0.1)',
                'contamination': 0.1,
                'ensemble': ['lof'],
                'threshold_tuning': False,
                'calibration': None,
                'use_text': False
            },
            {
                'name': 'Low Contamination (0.2)',
                'contamination': 0.2,
                'ensemble': ['lof'],
                'threshold_tuning': False,
                'calibration': None,
                'use_text': False
            },
            
            # Threshold tuning
            {
                'name': 'With Threshold Tuning',
                'contamination': 0.1,
                'ensemble': ['lof'],
                'threshold_tuning': True,
                'calibration': None,
                'use_text': False
            },
            
            # Calibration methods
            {
                'name': 'Isotonic Calibration',
                'contamination': 0.1,
                'ensemble': ['lof'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            {
                'name': 'Platt Calibration',
                'contamination': 0.1,
                'ensemble': ['lof'],
                'threshold_tuning': True,
                'calibration': 'platt',
                'use_text': False
            },
            
            # Different detectors
            {
                'name': 'One-Class SVM',
                'contamination': 0.1,
                'ensemble': ['ocsvm'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            {
                'name': 'Isolation Forest',
                'contamination': 0.1,
                'ensemble': ['iforest'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            {
                'name': 'Mahalanobis Distance',
                'contamination': 0.1,
                'ensemble': ['mahalanobis'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            
            # Ensemble combinations
            {
                'name': 'Ensemble: LOF + OC-SVM',
                'contamination': 0.1,
                'ensemble': ['lof', 'ocsvm'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            {
                'name': 'Ensemble: LOF + Mahalanobis',
                'contamination': 0.1,
                'ensemble': ['lof', 'mahalanobis'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            {
                'name': 'Ensemble: All Methods',
                'contamination': 0.1,
                'ensemble': ['lof', 'ocsvm', 'iforest', 'mahalanobis'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': False
            },
            
            # Text embeddings
            {
                'name': 'With Text Embeddings',
                'contamination': 0.1,
                'ensemble': ['lof'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': True
            },
            
            # Best configuration
            {
                'name': 'Best Configuration',
                'contamination': 0.1,
                'ensemble': ['lof', 'mahalanobis'],
                'threshold_tuning': True,
                'calibration': 'isotonic',
                'use_text': True
            },
        ]
        
        for exp in experiments:
            print(f"\n{'─'*80}")
            print(f"Running: {exp['name']}")
            print(f"{'─'*80}")
            
            result = self._run_single_experiment(exp, X_val, y_val, val_labels)
            self.results.append(result)
        
        # Create results dataframe
        df = pd.DataFrame(self.results)
        
        # Print summary
        self._print_summary(df)
        
        # Generate plots
        self._generate_plots(df)
        
        return df
    
    def _create_validation_split(self) -> Tuple:
        """Create validation split with mix of seen and novel classes"""
        # Take 20% of test data for validation
        from sklearn.model_selection import train_test_split
        
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        test_labels = self.data.get('test_class_labels')
        
        if test_labels is not None:
            X_val, _, y_val, _, val_labels, _ = train_test_split(
                X_test, y_test, test_labels,
                test_size=0.8, stratify=y_test, random_state=42
            )
        else:
            X_val, _, y_val, _ = train_test_split(
                X_test, y_test,
                test_size=0.8, stratify=y_test, random_state=42
            )
            val_labels = None
        
        logger.info(f"Validation split: {len(X_val)} samples")
        logger.info(f"  Seen: {np.sum(y_val == 0)}, Novel: {np.sum(y_val == 1)}")
        
        return X_val, y_val, val_labels
    
    def _run_single_experiment(self, config: Dict, X_val, y_val, val_labels) -> Dict:
        """Run single experiment configuration"""
        from improved_novelty_detector import ImprovedNoveltyDetector
        
        # Create detector
        detector = ImprovedNoveltyDetector(
            n_components=50,
            n_neighbors=10,
            contamination=config['contamination'],
            use_threshold_tuning=config['threshold_tuning'],
            calibration_method=config['calibration'],
            ensemble_methods=config['ensemble'],
            use_text_embeddings=config['use_text']
        )
        
        # Fit
        if config['threshold_tuning']:
            detector.fit(
                self.data['X_seen_train'],
                X_val, y_val,
                self.data.get('X_seen_train_labels'),
                val_labels,
                self.data.get('seen_class_names')
            )
        else:
            detector.fit(
                self.data['X_seen_train'],
                class_labels=self.data.get('X_seen_train_labels'),
                class_names=self.data.get('seen_class_names')
            )
        
        # Predict on test set
        y_pred = detector.predict(
            self.data['X_test'],
            self.data.get('test_class_labels')
        )
        y_scores = detector.predict_proba(
            self.data['X_test'],
            self.data.get('test_class_labels')
        )
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        
        y_true = self.data['y_test']
        seen_mask = (y_true == 0)
        novel_mask = (y_true == 1)
        
        seen_acc = accuracy_score(y_true[seen_mask], y_pred[seen_mask])
        novel_acc = accuracy_score(y_true[novel_mask], y_pred[novel_mask])
        hmean = 2 * (seen_acc * novel_acc) / (seen_acc + novel_acc) if (seen_acc + novel_acc) > 0 else 0
        
        result = {
            'experiment': config['name'],
            'contamination': config['contamination'],
            'ensemble': '+'.join(config['ensemble']),
            'threshold_tuning': config['threshold_tuning'],
            'calibration': config['calibration'] or 'None',
            'use_text': config['use_text'],
            'roc_auc': roc_auc_score(y_true, y_scores),
            'f1_score': f1_score(y_true, y_pred),
            'seen_accuracy': seen_acc,
            'novel_accuracy': novel_acc,
            'harmonic_mean': hmean,
            'balanced_accuracy': (seen_acc + novel_acc) / 2,
            'optimal_threshold': detector.optimal_threshold
        }
        
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        print(f"  H-Mean: {result['harmonic_mean']:.4f}")
        print(f"  Seen Acc: {result['seen_accuracy']:.4f}, Novel Acc: {result['novel_accuracy']:.4f}")
        
        return result
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary table"""
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY: {self.dataset_name}")
        print(f"{'='*80}\n")
        
        # Sort by harmonic mean
        df_sorted = df.sort_values('harmonic_mean', ascending=False)
        
        print(df_sorted[['experiment', 'roc_auc', 'harmonic_mean', 'seen_accuracy', 'novel_accuracy']].
