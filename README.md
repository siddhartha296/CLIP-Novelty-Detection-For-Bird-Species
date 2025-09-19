# CLIP-Novelty-Detection-For-Bird-Species
This repo is a project that is trying to make AI models that are capable of detecting unseen categories (specifically we are using cub -200-2011 dataset of 200 bird species and training the model only on 150 bird species and 50 bird species are novel )that they never seen in the training .



# PCA + LOF Novelty Detection Pipeline

A modular, production-ready implementation of novelty detection using Principal Component Analysis (PCA) and Local Outlier Factor (LOF) with CLIP embeddings.

## 🎯 Overview

This pipeline implements a comprehensive novelty detection system that:
- Uses CLIP embeddings as feature representations
- Applies PCA for dimensionality reduction 
- Employs LOF for novelty detection
- Provides extensive evaluation metrics
- Supports hyperparameter optimization
- Includes visualization utilities

## 📁 Project Structure

```
├── config.py                 # Configuration settings
├── data_loader.py            # Data loading and CLIP embedding extraction
├── novelty_detector.py       # PCA + LOF model implementation
├── evaluation.py             # Comprehensive evaluation metrics
├── utils.py                  # Utility functions and experiment tracking
├── main.py                   # Main pipeline execution
├── requirements.txt          # Python dependencies
├── run_experiment.sh         # Bash script for running experiments
├── README.md                 # This file
├── outputs/                  # Experiment outputs and logs
├── models/                   # Saved model files
├── plots/                    # Generated visualizations
└── experiments/              # Experiment tracking data
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd novelty-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your dataset in the following structure:
```
dataset/
├── class_001/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_002/
│   ├── image1.jpg
│   └── ...
└── ...
```

Update the `DATASET_PATH` in `config.py` or pass it as argument.

### 3. Running Experiments

#### Option A: Using the bash script (recommended)

```bash
# Make script executable
chmod +x run_experiment.sh

# Run baseline experiment
./run_experiment.sh baseline

# Run hyperparameter optimization
./run_experiment.sh optimize

# Run both baseline and optimization
./run_experiment.sh both
```

#### Option B: Using Python directly

```bash
# Baseline experiment
python main.py --dataset_path "path/to/dataset" --seen_classes 150

# With hyperparameter optimization
python main.py --dataset_path "path/to/dataset" --optimize

# Custom configuration
python main.py \
    --dataset_path "CUB_200_2011/images" \
    --seen_classes 150 \
    --max_images 100 \
    --experiment_name "my_experiment" \
    --log_level INFO
```

## 🔧 Configuration

### Key Parameters (config.py)

```python
# Dataset Configuration
DATASET_PATH = "CUB_200_2011/images"
SEEN_CLASSES = 150
MAX_IMAGES_PER_CLASS = 100

# Model Parameters
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_N_NEIGHBORS = 10
DEFAULT_CONTAMINATION = 0.35
DEFAULT_METRIC = 'cosine'

# Hyperparameter Search Space
HYPERPARAMETER_SEARCH = {
    'pca_dims': [30, 40, 50, 60, 70],
    'n_neighbors_list': [5, 10, 15, 20],
    'contaminations': [0.3, 0.35, 0.4],
    'metrics': ['cosine', 'euclidean']
}
```

## 📊 Evaluation Metrics

The pipeline provides comprehensive evaluation including:

### Binary Classification Metrics
- ROC AUC
- Precision-Recall AUC
- Accuracy, Precision, Recall, F1-Score
- Specificity

### Advanced Detection Metrics
- FPR@95% TPR
- TNR@95% TPR
- Youden's Index
- AUSUC (Area Under Seen-Unseen Curve)

### Calibration Metrics
- Expected Calibration Error (ECE)
- Brier Score

### Visualizations
- ROC and Precision-Recall curves
- Score distributions
- Confusion matrices
- PCA component analysis

## 🧩 Module Details

### 1. `data_loader.py`
- **CLIPEmbeddingExtractor**: Extracts CLIP embeddings from images
- **DatasetProcessor**: Handles dataset splitting and preprocessing
- **load_and_process_data()**: Complete data processing pipeline

### 2. `novelty_detector.py`
- **PCALOFNoveltyDetector**: Main model class with fit/predict interface
- **HyperparameterOptimizer**: Automated hyperparameter search
- Model saving/loading capabilities

### 3. `evaluation.py`
- **NoveltyDetectionEvaluator**: Comprehensive evaluation metrics
- **VisualizationUtils**: Plotting utilities
- Academic paper-ready reporting

### 4. `utils.py`
- **ExperimentTracker**: Track and save experiment results
- **ProgressTimer**: Timing utilities
- Dataset validation and statistics
- Configuration management

## 📈 Usage Examples

### Basic Usage

```python
from data_loader import load_and_process_data
from novelty_detector import PCALOFNoveltyDetector
from evaluation import NoveltyDetectionEvaluator

# Load data
data = load_and_process_data("path/to/dataset")

# Train model
detector = PCALOFNoveltyDetector(n_components=50, n_neighbors=10)
detector.fit(data['X_seen_train'])

# Make predictions
y_pred = detector.predict(data['X_test'])
y_scores = detector.predict_proba(data['X_test'])

# Evaluate
evaluator = NoveltyDetectionEvaluator()
results = evaluator.evaluate_comprehensive(data['y_test'], y_pred, y_scores)
evaluator.print_report(data['y_test'], y_pred)
```

### Hyperparameter Optimization

```python
from novelty_detector import HyperparameterOptimizer

search_space = {
    'n_components': [30, 50, 70],
    'n_neighbors': [5, 10, 15],
    'contamination': [0.3, 0.35, 0.4]
}

optimizer = HyperparameterOptimizer(search_space)
best_params, best_results = optimizer.optimize(X_train, X_test, y_test)
```

### Model Persistence

```python
# Save model
detector.save_model("models/my_model.pkl")

# Load model
detector = PCALOFNoveltyDetector.load_model("models/my_model.pkl")
```

## 🔍 Improving Performance

Based on your current results (ROC AUC: 0.7393), here are suggestions for improvement:

### 1. **Feature Engineering**
- Try different CLIP models (ViT-B/16, ViT-L/14)
- Experiment with feature fusion (multiple CLIP models)
- Add data augmentation during embedding extraction

### 2. **Advanced Preprocessing**
- Apply feature selection before PCA
- Try different normalization techniques
- Experiment with whitening transformations

### 3. **Model Variations**
- **Isolation Forest** instead of LOF
- **One-Class SVM** with different kernels
- **Autoencoder-based** approaches
- **Ensemble methods** combining multiple detectors

### 4. **Architecture Improvements**
- **Deep SVDD** (Support Vector Data Description)
- **CLIP-based few-shot learning**
- **Contrastive learning** approaches
- **Self-supervised pretraining**

### 5. **Hyperparameter Tuning**
- Expand search space for contamination (0.1-0.5)
- Try different distance metrics (manhattan, chebyshev)
- Optimize PCA components based on variance threshold
- Use Bayesian optimization instead of grid search

### Example Enhancement:

```python
# Multi-model ensemble
from sklearn.ensemble import VotingClassifier

detectors = [
    ('pca_lof', PCALOFNoveltyDetector(n_components=50)),
    ('pca_lof_euclidean', PCALOFNoveltyDetector(metric='euclidean')),
    ('isolation_forest', IsolationForest())
]

ensemble = EnsembleNoveltyDetector(detectors)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in CLIP extraction
   MAX_IMAGES_PER_CLASS = 50
   ```

2. **Dataset Not Found**
   ```bash
   # Check path and create symlink if needed
   ln -s /path/to/actual/dataset CUB_200_2011
   ```

3. **Poor Performance**
   ```python
   # Try different contamination values
   contamination = len(novel_samples) / len(total_samples)
   ```

4. **Memory Issues**
   ```python
   # Process data in batches
   for batch in data_loader.get_batches(batch_size=1000):
       embeddings.extend(extract_embeddings(batch))
   ```

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests if applicable
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to branch (`git push origin feature/improvement`)
7. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Siddhartha Pittala[https://www.linkedin.com/in/siddhartha-pittala-036001254/]
- OpenAI for the CLIP model
- scikit-learn community for excellent ML tools
- Contributors and testers