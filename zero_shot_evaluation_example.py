"""
Example of proper zero-shot learning evaluation
This demonstrates how to correctly compute and interpret ZSL metrics
"""
import numpy as np
from evaluation import NoveltyDetectionEvaluator
from data_loader import load_and_process_data
from novelty_detector import PCALOFNoveltyDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_zsl_evaluation():
    """
    Demonstrate proper zero-shot learning evaluation
    """
    
    print("="*60)
    print("ZERO-SHOT LEARNING EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Load data with proper class tracking
    logger.info("Loading data with class information...")
    data = load_and_process_data(
        dataset_path="CUB_200_2011/images",
        seen_classes=150,
        max_images_per_class=50,  # Smaller for demo
        test_size=0.3
    )
    
    # Train model
    logger.info("Training PCA+LOF model...")
    detector = PCALOFNoveltyDetector(n_components=30, n_neighbors=10)
    detector.fit(data['X_seen_train'])
    
    # Make predictions
    y_pred = detector.predict(data['X_test'])
    y_scores = detector.predict_proba(data['X_test'])
    
    # Comprehensive evaluation with ZSL metrics
    evaluator = NoveltyDetectionEvaluator()
    results = evaluator.evaluate_comprehensive(
        y_true=data['y_test'],
        y_pred=y_pred,
        y_scores=y_scores,
        class_labels=data['test_class_labels'],
        is_seen_class=data['is_seen_class']
    )
    
    # Print ZSL-specific results
    print("\n" + "="*50)
    print("ZERO-SHOT LEARNING METRICS")
    print("="*50)
    
    print(f"Dataset Information:")
    print(f"  - Total classes: {len(data['is_seen_class'])}")
    print(f"  - Seen classes: {results['n_seen_classes']}")
    print(f"  - Unseen classes: {results['n_unseen_classes']}")
    print(f"  - Test samples: {len(data['y_test'])}")
    print(f"  - Seen samples in test: {np.sum(data['y_test'] == 0)}")
    print(f"  - Unseen samples in test: {np.sum(data['y_test'] == 1)}")
    
    print(f"\nZero-Shot Learning Results:")
    print(f"  - Seen Class Accuracy (Acc_S): {results['seen_accuracy']:.4f}")
    print(f"  - Unseen Class Accuracy (Acc_U): {results['unseen_accuracy']:.4f}")
    print(f"  - Harmonic Mean (H): {results['harmonic_mean']:.4f}")
    print(f"  - Overall Accuracy: {results['overall_accuracy']:.4f}")
    
    print(f"\nPer-Class Analysis:")
    print(f"  - Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.4f}")
    print(f"  - Std Per-Class Accuracy: {results['std_per_class_accuracy']:.4f}")
    
    # Interpretation
    print(f"\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    if results['harmonic_mean'] > 0.5:
        print("✅ Good balance between seen and unseen class performance")
    elif results['seen_accuracy'] > results['unseen_accuracy']:
        print("⚠️  Model is biased toward seen classes (typical in novelty detection)")
    else:
        print("⚠️  Model performs better on unseen classes (unusual)")
    
    print(f"\nRecommendations:")
    if results['seen_accuracy'] < 0.7:
        print("- Consider increasing contamination parameter")
        print("- Try different PCA components")
    if results['unseen_accuracy'] < 0.7:
        print("- Consider decreasing contamination parameter") 
        print("- Try different distance metrics")
    if results['harmonic_mean'] < 0.3:
        print("- Significant class imbalance in performance")
        print("- Consider ensemble methods or threshold tuning")
    
    return results


def analyze_per_class_performance(data, y_pred, y_scores):
    """
    Analyze performance for each class individually
    """
    
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("="*60)
    
    unique_classes = np.unique(data['test_class_labels'])
    
    print(f"{'Class':<15} {'Type':<8} {'Samples':<8} {'Accuracy':<10} {'Avg Score':<10}")
    print("-" * 60)
    
    for cls in unique_classes[:20]:  # Show first 20 classes
        mask = data['test_class_labels'] == cls
        if mask.sum() == 0:
            continue
            
        cls_true = data['y_test'][mask]
        cls_pred = y_pred[mask]
        cls_scores = y_scores[mask]
        
        cls_accuracy = np.mean(cls_true == cls_pred)
        avg_score = np.mean(cls_scores)
        
        cls_type = "Seen" if data['is_seen_class'][cls] else "Unseen"
        n_samples = mask.sum()
        
        print(f"{cls:<15d} {cls_type:<8} {n_samples:<8d} {cls_accuracy:<10.3f} {avg_score:<10.3f}")


def compare_zsl_methods():
    """
    Compare different novelty detection methods for ZSL
    """
    
    print("\n" + "="*60) 
    print("COMPARING DIFFERENT METHODS FOR ZSL")
    print("="*60)
    
    # Load data
    data = load_and_process_data(
        dataset_path="CUB_200_2011/images",
        seen_classes=100,  # Smaller for faster demo
        max_images_per_class=30
    )
    
    methods = [
        ("PCA+LOF (cosine)", {"n_components": 50, "metric": "cosine"}),
        ("PCA+LOF (euclidean)", {"n_components": 50, "metric": "euclidean"}),
        ("PCA+LOF (small)", {"n_components": 20, "metric": "cosine"}),
        ("PCA+LOF (large)", {"n_components": 80, "metric": "cosine"}),
    ]
    
    evaluator = NoveltyDetectionEvaluator()
    results_comparison = []
    
    for method_name, params in methods:
        logger.info(f"Testing {method_name}...")
        
        # Train model
        detector = PCALOFNoveltyDetector(**params)
        detector.fit(data['X_seen_train'])
        
        # Evaluate
        y_pred = detector.predict(data['X_test'])
        y_scores = detector.predict_proba(data['X_test'])
        
        results = evaluator.evaluate_comprehensive(
            data['y_test'], y_pred, y_scores,
            data['test_class_labels'], data['is_seen_class']
        )
        
        results_comparison.append({
            'method': method_name,
            'seen_acc': results['seen_accuracy'],
            'unseen_acc': results['unseen_accuracy'],
            'harmonic_mean': results['harmonic_mean'],
            'roc_auc': results['roc_auc']
        })
    
    # Print comparison
    print(f"\n{'Method':<20} {'Seen Acc':<10} {'Unseen Acc':<12} {'H-Mean':<10} {'ROC AUC':<10}")
    print("-" * 70)
    
    for result in results_comparison:
        print(f"{result['method']:<20} "
              f"{result['seen_acc']:<10.3f} "
              f"{result['unseen_acc']:<12.3f} "
              f"{result['harmonic_mean']:<10.3f} "
              f"{result['roc_auc']:<10.3f}")
    
    # Find best method
    best_method = max(results_comparison, key=lambda x: x['harmonic_mean'])
    print(f"\n🏆 Best method by Harmonic Mean: {best_method['method']}")


if __name__ == "__main__":
    try:
        # Demonstrate proper ZSL evaluation
        results = demonstrate_zsl_evaluation()
        
        # Analyze per-class performance 
        # analyze_per_class_performance(data, y_pred, y_scores)  # Uncomment if needed
        
        # Compare different methods
        # compare_zsl_methods()  # Uncomment for method comparison
        
        print("\n✅ Zero-shot learning evaluation completed!")
        
    except Exception as e:
        logger.error(f"Error in ZSL evaluation: {e}")
        raise