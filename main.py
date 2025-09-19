"""
Main pipeline for PCA + LOF novelty detection
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Import our modules
from config import *
from data_loader import load_and_process_data
from novelty_detector import PCALOFNoveltyDetector, HyperparameterOptimizer
from evaluation import NoveltyDetectionEvaluator, VisualizationUtils
from utils import (
    setup_logging, ensure_directories, save_json, 
    ExperimentTracker, ProgressTimer, validate_dataset_structure,
    print_dataset_info, create_experiment_config, format_results_for_paper
)


def run_baseline_experiment(config: dict, experiment_tracker: ExperimentTracker = None):
    """Run baseline PCA + LOF experiment"""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting baseline PCA + LOF experiment")
    
    # Validate dataset
    if not validate_dataset_structure(config['dataset']['path']):
        raise ValueError("Invalid dataset structure")
    
    print_dataset_info(config['dataset']['path'])
    
    # Load and process data
    with ProgressTimer("Data loading and preprocessing"):
        data = load_and_process_data(
            dataset_path=config['dataset']['path'],
            seen_classes=config['dataset']['seen_classes'],
            max_images_per_class=config['dataset']['max_images_per_class'],
            clip_model=config['preprocessing']['clip_model'],
            test_size=config['dataset']['test_size'],
            random_state=config['dataset']['random_state']
        )
    
    # Initialize and train model
    with ProgressTimer("Model training"):
        detector = PCALOFNoveltyDetector(
            n_components=config['model']['n_components'],
            n_neighbors=config['model']['n_neighbors'],
            contamination=config['model']['contamination'],
            metric=config['model']['metric']
        )
        
        detector.fit(data['X_seen_train'])
    
    # Evaluate model
    with ProgressTimer("Model evaluation"):
        evaluator = NoveltyDetectionEvaluator()
        
        # Get predictions
        y_pred = detector.predict(data['X_test'])
        y_scores = detector.predict_proba(data['X_test'])
        
        # Comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            data['y_test'], y_pred, y_scores
        )
        
        # Print detailed report
        evaluator.print_report(data['y_test'], y_pred)
    
    # Save model
    model_filename = f"pca_lof_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    detector.save_model(model_path)
    
    # Visualizations
    plot_dir = os.path.join(PLOTS_DIR, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_directories(plot_dir)
    
    VisualizationUtils.plot_performance_curves(
        data['y_test'], y_scores, 
        title="PCA + LOF Performance",
        save_path=os.path.join(plot_dir, "performance_curves.png")
    )
    
    VisualizationUtils.plot_score_distribution(
        data['y_test'], y_scores,
        save_path=os.path.join(plot_dir, "score_distribution.png")
    )
    
    VisualizationUtils.plot_confusion_matrix(
        data['y_test'], y_pred,
        save_path=os.path.join(plot_dir, "confusion_matrix.png")
    )
    
    VisualizationUtils.plot_pca_components(
        detector.pca,
        save_path=os.path.join(plot_dir, "pca_components.png")
    )
    
    # Log experiment
    if experiment_tracker:
        experiment_tracker.log_experiment(config, results, model_path)
    
    # Print academic report
    print("\n" + "="*60)
    print("ACADEMIC PAPER REPORT")
    print("="*60)
    print(format_results_for_paper(results, "PCA+LOF"))
    
    logger.info("✅ Baseline experiment completed successfully!")
    return results, detector, data


def run_hyperparameter_optimization(config: dict, experiment_tracker: ExperimentTracker = None):
    """Run hyperparameter optimization"""
    
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting hyperparameter optimization")
    
    # Load data (reuse from baseline if needed)
    data = load_and_process_data(
        dataset_path=config['dataset']['path'],
        seen_classes=config['dataset']['seen_classes'],
        max_images_per_class=config['dataset']['max_images_per_class'],
        clip_model=config['preprocessing']['clip_model'],
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state']
    )
    
    # Setup hyperparameter search
    search_space = {
        'n_components': HYPERPARAMETER_SEARCH['pca_dims'],
        'n_neighbors': HYPERPARAMETER_SEARCH['n_neighbors_list'],
        'contamination': HYPERPARAMETER_SEARCH['contaminations'],
        'metric': HYPERPARAMETER_SEARCH['metrics']
    }
    
    optimizer = HyperparameterOptimizer(search_space)
    
    # Run optimization
    with ProgressTimer("Hyperparameter optimization"):
        best_params, best_results = optimizer.optimize(
            data['X_seen_train'], data['X_test'], data['y_test']
        )
    
    # Train final model with best parameters
    logger.info("🏆 Training final model with best parameters")
    best_detector = PCALOFNoveltyDetector(**best_params)
    best_detector.fit(data['X_seen_train'])
    
    # Save best model
    model_filename = f"best_pca_lof_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    best_detector.save_model(model_path)
    
    # Update config with best parameters
    optimized_config = config.copy()
    optimized_config['model'].update(best_params)
    
    # Log best experiment
    if experiment_tracker:
        experiment_tracker.log_experiment(optimized_config, best_results, model_path)
    
    # Print optimization summary
    summary = optimizer.get_results_summary()
    print(f"\n🎉 OPTIMIZATION SUMMARY:")
    print(f"   Total trials: {summary['n_trials']}")
    print(f"   Best ROC AUC: {summary['best_score']:.4f}")
    print(f"   Mean ROC AUC: {summary['mean_score']:.4f} ± {summary['std_score']:.4f}")
    print(f"   Best parameters: {summary['best_params']}")
    
    return best_params, best_results, best_detector


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="PCA + LOF Novelty Detection Pipeline")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                       help="Path to dataset directory")
    parser.add_argument("--seen_classes", type=int, default=SEEN_CLASSES,
                       help="Number of seen classes")
    parser.add_argument("--max_images", type=int, default=MAX_IMAGES_PER_CLASS,
                       help="Maximum images per class")
    parser.add_argument("--optimize", action="store_true",
                       help="Run hyperparameter optimization")
    parser.add_argument("--experiment_name", type=str, default=f"novelty_detection_{datetime.now().strftime('%Y%m%d')}",
                       help="Name for experiment tracking")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, f"{args.experiment_name}.log")
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Ensure output directories exist
    ensure_directories(OUTPUT_DIR, MODEL_DIR, PLOTS_DIR)
    
    logger.info("🎯 Starting Novelty Detection Pipeline")
    logger.info(f"   Dataset: {args.dataset_path}")
    logger.info(f"   Seen classes: {args.seen_classes}")
    logger.info(f"   Experiment: {args.experiment_name}")
    
    # Create experiment configuration
    config = create_experiment_config(
        dataset_path=args.dataset_path,
        seen_classes=args.seen_classes,
        max_images_per_class=args.max_images,
        n_components=DEFAULT_PCA_COMPONENTS,
        n_neighbors=DEFAULT_N_NEIGHBORS,
        contamination=DEFAULT_CONTAMINATION,
        metric=DEFAULT_METRIC
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(args.experiment_name)
    
    try:
        if args.optimize:
            # Run hyperparameter optimization
            best_params, best_results, best_model = run_hyperparameter_optimization(config, tracker)
            
            # Also run baseline for comparison
            logger.info("Running baseline experiment for comparison...")
            baseline_results, baseline_model, data = run_baseline_experiment(config, tracker)
            
            # Compare results
            print(f"\n📊 COMPARISON:")
            print(f"   Baseline ROC AUC: {baseline_results['roc_auc']:.4f}")
            print(f"   Optimized ROC AUC: {best_results['roc_auc']:.4f}")
            print(f"   Improvement: {best_results['roc_auc'] - baseline_results['roc_auc']:.4f}")
            
        else:
            # Run baseline experiment only
            results, model, data = run_baseline_experiment(config, tracker)
        
        # Print final summary
        tracker.print_summary()
        
        # Save final configuration
        config_file = os.path.join(OUTPUT_DIR, f"{args.experiment_name}_config.json")
        save_json(config, config_file)
        
        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"   Results saved to: {OUTPUT_DIR}")
        logger.info(f"   Models saved to: {MODEL_DIR}")
        logger.info(f"   Plots saved to: {PLOTS_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        raise
    
    return True


def run_inference_example():
    """Example of how to use trained model for inference"""
    
    logger = logging.getLogger(__name__)
    logger.info("🔮 Running inference example")
    
    # This would typically load a saved model
    # For demo purposes, we'll create a simple example
    
    try:
        # Load the most recent model (this is just an example)
        import glob
        model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
        
        if not model_files:
            logger.warning("No saved models found. Run training first.")
            return
        
        latest_model = max(model_files, key=os.path.getctime)
        logger.info(f"Loading model: {latest_model}")
        
        # Load model
        detector = PCALOFNoveltyDetector.load_model(latest_model)
        
        # Load some test data (you would replace this with your actual data)
        logger.info("Loading test data for inference...")
        
        # Example: Load and process a few images for inference
        # This would be replaced with your actual inference data
        config = create_experiment_config(DATASET_PATH)
        data = load_and_process_data(
            dataset_path=config['dataset']['path'],
            seen_classes=config['dataset']['seen_classes'],
            max_images_per_class=5,  # Just a few samples for demo
            clip_model=config['preprocessing']['clip_model']
        )
        
        # Make predictions on a small subset
        sample_data = data['X_test'][:10]  # Just 10 samples
        
        predictions = detector.predict(sample_data)
        scores = detector.predict_proba(sample_data)
        
        # Display results
        print(f"\n🔍 INFERENCE RESULTS (10 samples):")
        print(f"   Predictions: {predictions}")
        print(f"   Average novelty score: {scores.mean():.4f}")
        print(f"   Novel samples detected: {predictions.sum()}")
        
        logger.info("✅ Inference example completed")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")


if __name__ == "__main__":
    main()
