"""
Utility functions and helpers for the novelty detection pipeline
"""
import os
import json
import pickle
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def ensure_directories(*dirs):
    """Ensure directories exist"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict:
    """Load JSON file as dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str):
    """Save object using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object using pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_device_info():
    """Get device information"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    info = {
        'device': device,
        'device_name': None,
        'memory_total': None,
        'memory_available': None
    }
    
    if device == "cuda":
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['memory_available'] = torch.cuda.memory_allocated(0)
    
    return info


def format_results_for_paper(results: Dict, model_name: str = "PCA+LOF") -> str:
    """Format results for academic paper reporting"""
    
    report = f"""
========================================
COMPREHENSIVE EVALUATION REPORT
========================================

MODEL: {model_name}
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BINARY CLASSIFICATION METRICS:
- ROC AUC: {results.get('roc_auc', 0):.4f}
- PR AUC: {results.get('pr_auc', 0):.4f}
- Accuracy: {results.get('accuracy', 0):.4f}
- Precision: {results.get('precision', 0):.4f}
- Recall: {results.get('recall', 0):.4f}
- F1-Score: {results.get('f1_score', 0):.4f}
- Specificity: {results.get('specificity', 0):.4f}

ADVANCED DETECTION METRICS:
- FPR@95% TPR: {results.get('fpr_at_95_tpr', 0):.4f}
- TNR@95% TPR: {results.get('tnr_at_95_tpr', 0):.4f}
- Youden's Index: {results.get('youdens_index', 0):.4f}

ZERO-SHOT LEARNING METRICS:
- Seen Accuracy (Acc_S): {results.get('seen_accuracy', 0):.4f}
- Unseen Accuracy (Acc_U): {results.get('unseen_accuracy', 0):.4f}
- Harmonic Mean (H): {results.get('harmonic_mean', 0):.4f}
- Overall Accuracy: {results.get('overall_accuracy', 0):.4f}

CALIBRATION METRICS:
- Expected Calibration Error (ECE): {results.get('ece', 0):.4f}
- Brier Score: {results.get('brier_score', 0):.4f}

PER-CLASS ANALYSIS:
- Mean Per-Class Accuracy: {results.get('mean_per_class_accuracy', 0):.4f} ± {results.get('std_per_class_accuracy', 0):.4f}

CONFUSION MATRIX:
{results.get('confusion_matrix', 'N/A')}

AUSUC (Area Under Seen-Unseen Accuracy Curve): {results.get('ausuc', 0):.4f}
    """
    
    return report.strip()


class ExperimentTracker:
    """Track and save experiment results"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        ensure_directories(self.experiment_dir)
        
        self.results = []
        self.metadata = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'device_info': get_device_info()
        }
    
    def log_experiment(self, config: Dict, results: Dict, model_path: Optional[str] = None):
        """Log a single experiment"""
        experiment_entry = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results,
            'model_path': model_path
        }
        
        self.results.append(experiment_entry)
        
        # Save immediately
        self.save_results()
    
    def save_results(self):
        """Save all results to file"""
        results_file = os.path.join(self.experiment_dir, 'results.json')
        
        data = {
            'metadata': self.metadata,
            'experiments': self.results
        }
        
        save_json(data, results_file)
    
    def get_best_experiment(self, metric: str = 'roc_auc'):
        """Get best experiment based on metric"""
        if not self.results:
            return None
        
        best_exp = max(self.results, key=lambda x: x['results'].get(metric, 0))
        return best_exp
    
    def print_summary(self):
        """Print experiment summary"""
        if not self.results:
            print("No experiments logged yet.")
            return
        
        print(f"\n{'='*50}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*50}")
        print(f"Total experiments: {len(self.results)}")
        
        # Get best results
        best_exp = self.get_best_experiment('roc_auc')
        if best_exp:
            print(f"Best ROC AUC: {best_exp['results'].get('roc_auc', 0):.4f}")
            print(f"Best config: {best_exp['config']}")


def validate_dataset_structure(dataset_path: str) -> bool:
    """Validate dataset directory structure"""
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Check if it has subdirectories (classes)
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(subdirs) == 0:
        logging.error("No class subdirectories found in dataset")
        return False
    
    # Check if subdirectories contain image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    valid_classes = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        files = os.listdir(subdir_path)
        
        # Check for image files
        image_files = [f for f in files 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if image_files:
            valid_classes += 1
    
    if valid_classes == 0:
        logging.error("No image files found in class directories")
        return False
    
    logging.info(f"Dataset validation passed: {valid_classes} valid classes found")
    return True


def memory_usage_mb():
    """Get current memory usage in MB"""
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class ProgressTimer:
    """Simple progress timer context manager"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logging.info(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logging.info(f"Completed {self.description} in {format_time(elapsed)}")


def get_class_statistics(dataset_path: str) -> Dict[str, Any]:
    """Get statistics about dataset classes"""
    stats = {
        'total_classes': 0,
        'total_images': 0,
        'images_per_class': [],
        'class_names': []
    }
    
    if not os.path.exists(dataset_path):
        return stats
    
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        files = os.listdir(subdir_path)
        
        image_files = [f for f in files 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if image_files:
            stats['class_names'].append(subdir)
            stats['images_per_class'].append(len(image_files))
            stats['total_images'] += len(image_files)
    
    stats['total_classes'] = len(stats['class_names'])
    stats['mean_images_per_class'] = np.mean(stats['images_per_class']) if stats['images_per_class'] else 0
    stats['std_images_per_class'] = np.std(stats['images_per_class']) if stats['images_per_class'] else 0
    stats['min_images_per_class'] = min(stats['images_per_class']) if stats['images_per_class'] else 0
    stats['max_images_per_class'] = max(stats['images_per_class']) if stats['images_per_class'] else 0
    
    return stats


def print_dataset_info(dataset_path: str):
    """Print comprehensive dataset information"""
    stats = get_class_statistics(dataset_path)
    
    print(f"\n{'='*50}")
    print("DATASET INFORMATION")
    print(f"{'='*50}")
    print(f"Dataset path: {dataset_path}")
    print(f"Total classes: {stats['total_classes']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Images per class: {stats['mean_images_per_class']:.1f} ± {stats['std_images_per_class']:.1f}")
    print(f"Range: [{stats['min_images_per_class']}, {stats['max_images_per_class']}]")
    print(f"First 5 classes: {stats['class_names'][:5]}")
    print(f"{'='*50}")


def create_experiment_config(
    dataset_path: str,
    seen_classes: int = 150,
    max_images_per_class: int = 100,
    n_components: int = 50,
    n_neighbors: int = 10,
    contamination: float = 0.35,
    metric: str = 'cosine',
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Create standardized experiment configuration"""
    
    return {
        'dataset': {
            'path': dataset_path,
            'seen_classes': seen_classes,
            'max_images_per_class': max_images_per_class,
            'test_size': test_size,
            'random_state': random_state
        },
        'model': {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'metric': metric
        },
        'preprocessing': {
            'standardize': True,
            'clip_model': 'ViT-B/32'
        },
        'experiment': {
            'timestamp': datetime.now().isoformat(),
            'device_info': get_device_info()
        }
    }
