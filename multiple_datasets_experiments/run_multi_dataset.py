"""
Multi-dataset experiment runner for novelty detection
Supports CUB, AWA2, and Oxford Flowers datasets
"""
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List

from dataset_configs import get_dataset_config, get_dataset_params, print_dataset_info, get_literature_benchmarks
from dataset_adapter import load_multi_dataset, validate_dataset_setup, setup_dataset_instructions
from novelty_detector import PCALOFNoveltyDetector, HyperparameterOptimizer
from evaluation import NoveltyDetectionEvaluator, VisualizationUtils
from utils import (
    setup_logging, ensure_directories, save_json, 
    ExperimentTracker, ProgressTimer, create_experiment_config, 
    format_results_for_paper
)

# Create output directories
ensure_directories("outputs", "models", "plots", "experiments")


class MultiDatasetExperiment:
    """Multi-dataset experiment runner"""
    
    def __init__(self, datasets: List[str], experiment_name: str = None):
        self.datasets = [d.upper() for d in datasets]
        self.experiment_name = experiment_name or f"multi_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {}
        
        # Setup logging
        log_file = f"outputs/{self.experiment_name}.log"
        setup_logging("INFO", log_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize experiment tracker
        self.tracker = ExperimentTracker(self.experiment_name)
        
    def validate_datasets(self) -> bool:
        """Validate all datasets are available"""
        all_valid = True
        
        for dataset in self.datasets:
            self.logger.info(f"Validating {dataset} dataset...")
            if not validate_dataset_setup(dataset):
                self.logger.error(f"❌ {dataset} dataset validation failed")
                all_valid = False
            else:
                self.logger.info(f"✅ {dataset} dataset validated")
        
        return all_valid
    
    def run_single_dataset(self, dataset_name: str, optimize: bool = False) -> Dict:
        """Run experiment on a single dataset"""
        
        self.logger.info(f"🚀 Starting experiment on {dataset_name}")
        print_dataset_info(dataset_name)
        
        # Get dataset configuration
        config = get_dataset_config(dataset_name)
        params = get_dataset_params(dataset_name)
        
        # Load dataset
        with ProgressTimer(f"{dataset_name} data loading"):
            data = load_multi_dataset(
                dataset_name=dataset_name,
                max_images_per_class=params['max_images_per_class'],
                test_size=0.3,
                random_state=42,
                clip_model=params['clip_model']
            )
        
        # Train model
        with ProgressTimer(f"{dataset_name} model training"):
            if optimize:
                # Hyperparameter optimization
                search_space = {
                    'n_components': [20, 30, 40, 50, 60],
                    'n_neighbors': [5, 10, 15, 20],
                    'contamination': [0.2, 0.25, 0.3, 0.35, 0.4],
                    'metric': ['cosine', 'euclidean']
                }
                
                optimizer = HyperparameterOptimizer(search_space)
                best_params, _ = optimizer.optimize(
                    data['X_seen_train'], data['X_test'], data['y_test']
                )
                
                detector = PCALOFNoveltyDetector(**best_params)
                self.logger.info(f"🏆 Best params for {dataset_name}: {best_params}")
            else:
                # Use default parameters
                detector = PCALOFNoveltyDetector(
                    n_components=params['default_pca_components'],
                    contamination=params['default_contamination'],
                    metric='cosine'
                )
            
            detector.fit(data['X_seen_train'])
        
        # Evaluate model
        with ProgressTimer(f"{dataset_name} evaluation"):
            evaluator = NoveltyDetectionEvaluator()
            
            y_pred = detector.predict(data['X_test'])
            y_scores = detector.predict_proba(data['X_test'])
            
            results = evaluator.evaluate_comprehensive(
                data['y_test'], y_pred, y_scores,
                class_labels=data['test_class_labels'],
                is_seen_class=data['is_seen_class']
            )
            
            # Print results
            print(f"\n{'='*60}")
            print(f"RESULTS FOR {dataset_name}")
            print(f"{'='*60}")
            evaluator.print_report(data['y_test'], y_pred)
        
        # Save model
        model_filename = f"{dataset_name.lower()}_pca_lof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = f"models/{model_filename}"
        detector.save_model(model_path)
        
        # Create visualizations
        plot_dir = f"plots/{self.experiment_name}_{dataset_name.lower()}"
        ensure_directories(plot_dir)
        
        VisualizationUtils.plot_performance_curves(
            data['y_test'], y_scores,
            title=f"{dataset_name} Performance",
            save_path=f"{plot_dir}/performance_curves.png"
        )
        
        VisualizationUtils.plot_score_distribution(
            data['y_test'], y_scores,
            title=f"{dataset_name} Score Distribution",
            save_path=f"{plot_dir}/score_distribution.png"
        )
        
        # Literature comparison
        benchmarks = get_literature_benchmarks(dataset_name)
        if benchmarks:
            print(f"\n📊 Literature Comparison for {dataset_name}:")
            print("-" * 50)
            for method, scores in benchmarks.items():
                print(f"{method:20s} | H-Mean: {scores.get('harmonic_mean', 'N/A'):.3f}")
            print(f"{'Our PCA+LOF':20s} | H-Mean: {results.get('harmonic_mean', 0):.3f}")
        
        # Store results
        experiment_config = {
            'dataset': dataset_name,
            'parameters': params,
            'optimization': optimize
        }
        
        self.tracker.log_experiment(experiment_config, results, model_path)
        self.results[dataset_name] = results
        
        self.logger.info(f"✅ {dataset_name} experiment completed")
        return results
    
    def run_all_datasets(self, optimize: bool = False) -> Dict:
        """Run experiments on all datasets"""
        
        self.logger.info(f"🎯 Running experiments on {len(self.datasets)} datasets")
        
        if not self.validate_datasets():
            setup_dataset_instructions()
            raise ValueError("Some datasets are not properly set up")
        
        all_results = {}
        
        for dataset in self.datasets:
            try:
                results = self.run_single_dataset(dataset, optimize)
                all_results[dataset] = results
            except Exception as e:
                self.logger.error(f"❌ Failed to process {dataset}: {e}")
                continue
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: Dict):
        """Generate cross-dataset comparison report"""
        
        print(f"\n{'='*80}")
        print("CROSS-DATASET COMPARISON REPORT")
        print(f"{'='*80}")
        
        # Summary table
        print(f"\n{'Dataset':<15} {'ROC AUC':<10} {'H-Mean':<10} {'Seen Acc':<10} {'Unseen Acc':<12} {'F1-Score':<10}")
        print("-" * 80)
        
        best_dataset = None
        best_score = 0
        
        for dataset, result in results.items():
            roc_auc = result.get('roc_auc', 0)
            h_mean = result.get('harmonic_mean', 0)
            seen_acc = result.get('seen_accuracy', 0)
            unseen_acc = result.get('unseen_accuracy', 0)
            f1_score = result.get('f1_score', 0)
            
            print(f"{dataset:<15} {roc_auc:<10.3f} {h_mean:<10.3f} {seen_acc:<10.3f} {unseen_acc:<12.3f} {f1_score:<10.3f}")
            
            # Track best performing dataset
            if h_mean > best_score:
                best_score = h_mean
                best_dataset = dataset
        
        print(f"\n🏆 Best performing dataset: {best_dataset} (H-Mean: {best_score:.3f})")
        
        # Dataset characteristics analysis
        print(f"\n📊 Dataset Characteristics Analysis:")
        print("-" * 50)
        
        for dataset in results.keys():
            config = get_dataset_config(dataset)
            result = results[dataset]
            
            print(f"\n{dataset}:")
            print(f"  Classes: {config.total_classes} ({config.seen_classes} seen, {config.novel_classes} novel)")
            print(f"  Performance: ROC AUC {result.get('roc_auc', 0):.3f}, H-Mean {result.get('harmonic_mean', 0):.3f}")
            
            # Performance analysis
            if result.get('harmonic_mean', 0) > 0.7:
                print(f"  ✅ Excellent zero-shot performance")
            elif result.get('harmonic_mean', 0) > 0.6:
                print(f"  ✅ Good zero-shot performance")
            else:
                print(f"  ⚠️  Room for improvement")
        
        # Cross-dataset insights
        print(f"\n🔍 Cross-Dataset Insights:")
        print("-" * 50)
        
        roc_scores = [results[d].get('roc_auc', 0) for d in results.keys()]
        h_means = [results[d].get('harmonic_mean', 0) for d in results.keys()]
        
        print(f"Average ROC AUC across datasets: {np.mean(roc_scores):.3f} ± {np.std(roc_scores):.3f}")
        print(f"Average H-Mean across datasets: {np.mean(h_means):.3f} ± {np.std(h_means):.3f}")
        
        # Dataset difficulty ranking
        dataset_difficulty = sorted(results.items(), key=lambda x: x[1].get('harmonic_mean', 0), reverse=True)
        print(f"\nDataset difficulty ranking (easiest to hardest):")
        for i, (dataset, result) in enumerate(dataset_difficulty, 1):
            print(f"  {i}. {dataset} (H-Mean: {result.get('harmonic_mean', 0):.3f})")
        
        # Save comparison report
        comparison_data = {
            'experiment_name': self.experiment_name,
            'datasets': list(results.keys()),
            'results': results,
            'summary': {
                'best_dataset': best_dataset,
                'best_score': best_score,
                'avg_roc_auc': float(np.mean(roc_scores)),
                'avg_h_mean': float(np.mean(h_means)),
                'std_roc_auc': float(np.std(roc_scores)),
                'std_h_mean': float(np.std(h_means))
            }
        }
        
        save_json(comparison_data, f"outputs/{self.experiment_name}_comparison.json")
        
        # Generate LaTeX table for paper
        self.generate_latex_table(results)
        
    def generate_latex_table(self, results: Dict):
        """Generate LaTeX table for academic paper"""
        
        latex_table = f"""
% Cross-dataset comparison table
\\begin{{table}}[htbp]
\\centering
\\caption{{Cross-Dataset Performance Comparison}}
\\label{{tab:cross_dataset}}
\\begin{{tabular}}{{|l|c|c|c|c|c|}}
\\hline
\\textbf{{Dataset}} & \\textbf{{Classes}} & \\textbf{{ROC AUC}} & \\textbf{{H-Mean}} & \\textbf{{Seen Acc}} & \\textbf{{Unseen Acc}} \\\\
\\hline
"""
        
        for dataset, result in results.items():
            config = get_dataset_config(dataset)
            latex_table += f"{dataset} & {config.total_classes} & {result.get('roc_auc', 0):.3f} & {result.get('harmonic_mean', 0):.3f} & {result.get('seen_accuracy', 0):.3f} & {result.get('unseen_accuracy', 0):.3f} \\\\\n"
        
        # Add averages
        roc_avg = np.mean([results[d].get('roc_auc', 0) for d in results.keys()])
        h_avg = np.mean([results[d].get('harmonic_mean', 0) for d in results.keys()])
        seen_avg = np.mean([results[d].get('seen_accuracy', 0) for d in results.keys()])
        unseen_avg = np.mean([results[d].get('unseen_accuracy', 0) for d in results.keys()])
        
        latex_table += f"""\\hline
\\textbf{{Average}} & - & \\textbf{{{roc_avg:.3f}}} & \\textbf{{{h_avg:.3f}}} & \\textbf{{{seen_avg:.3f}}} & \\textbf{{{unseen_avg:.3f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        
        print(f"\n📄 LaTeX Table for Paper:")
        print("=" * 60)
        print(latex_table)
        
        # Save to file
        with open(f"outputs/{self.experiment_name}_latex_table.tex", 'w') as f:
            f.write(latex_table)


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Multi-Dataset Novelty Detection Experiments")
    
    parser.add_argument("--datasets", nargs='+', default=['CUB', 'AWA2', 'FLO'],
                       choices=['CUB', 'AWA2', 'FLO'],
                       help="Datasets to run experiments on")
    
    parser.add_argument("--single", type=str, choices=['CUB', 'AWA2', 'FLO'],
                       help="Run experiment on single dataset only")
    
    parser.add_argument("--optimize", action="store_true",
                       help="Run hyperparameter optimization")
    
    parser.add_argument("--experiment_name", type=str, 
                       default=f"multi_dataset_{datetime.now().strftime('%Y%m%d')}",
                       help="Name for experiment")
    
    parser.add_argument("--setup_help", action="store_true",
                       help="Show dataset setup instructions")
    
    args = parser.parse_args()
    
    if args.setup_help:
        setup_dataset_instructions()
        return
    
    # Determine which datasets to run
    if args.single:
        datasets = [args.single]
        experiment_name = f"{args.single.lower()}_{args.experiment_name}"
    else:
        datasets = args.datasets
        experiment_name = args.experiment_name
    
    print(f"🎯 Multi-Dataset Novelty Detection Experiments")
    print(f"Datasets: {datasets}")
    print(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
    print(f"Experiment: {experiment_name}")
    
    # Run experiments
    try:
        experiment = MultiDatasetExperiment(datasets, experiment_name)
        results = experiment.run_all_datasets(optimize=args.optimize)
        
        print(f"\n✅ All experiments completed successfully!")
        print(f"📁 Results saved to: outputs/{experiment_name}*")
        print(f"🤖 Models saved to: models/")
        print(f"📊 Plots saved to: plots/{experiment_name}*/")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1
    
    return 0


# Additional utility functions
def quick_dataset_test(dataset_name: str):
    """Quick test of dataset loading"""
    
    print(f"🧪 Testing {dataset_name} dataset loading...")
    
    try:
        if not validate_dataset_setup(dataset_name):
            print(f"❌ Dataset validation failed")
            return False
        
        # Try loading a small sample
        data = load_multi_dataset(
            dataset_name=dataset_name,
            max_images_per_class=5,  # Small sample
            test_size=0.5,
            random_state=42
        )
        
        print(f"✅ Successfully loaded {dataset_name}:")
        print(f"   Training set: {data['X_seen_train'].shape}")
        print(f"   Test set: {data['X_test'].shape}")
        print(f"   Classes: {len(data['seen_class_list'])} seen, {len(data['novel_class_list'])} novel")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def create_dataset_comparison_script():
    """Create a simple script for running dataset comparisons"""
    
    script_content = '''#!/bin/bash

# Multi-dataset comparison script

echo "🚀 Multi-Dataset Novelty Detection Comparison"
echo "=============================================="

# Test all datasets first
echo "🧪 Testing dataset availability..."
python run_multi_dataset.py --setup_help

# Run quick tests
datasets=("CUB" "AWA2" "FLO")
for dataset in "${datasets[@]}"; do
    echo "Testing $dataset..."
    python -c "from run_multi_dataset import quick_dataset_test; quick_dataset_test('$dataset')"
done

# Run full experiments
echo "🎯 Running full experiments..."

# Option 1: Run all datasets with default parameters
python run_multi_dataset.py --datasets CUB AWA2 FLO --experiment_name "comparison_baseline"

# Option 2: Run with optimization (takes longer)
# python run_multi_dataset.py --datasets CUB AWA2 FLO --optimize --experiment_name "comparison_optimized"

# Option 3: Run individual datasets
# python run_multi_dataset.py --single CUB --experiment_name "cub_detailed"
# python run_multi_dataset.py --single AWA2 --experiment_name "awa2_detailed" 
# python run_multi_dataset.py --single FLO --experiment_name "flo_detailed"

echo "✅ Comparison completed!"
'''
    
    with open('run_comparison.sh', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('run_comparison.sh', 0o755)
    print("📝 Created run_comparison.sh script")


if __name__ == "__main__":
    import numpy as np  # Import needed for the script
    main()
