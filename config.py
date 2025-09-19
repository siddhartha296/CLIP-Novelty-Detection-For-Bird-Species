"""
Configuration file for novelty detection pipeline
"""
import os

# Dataset Configuration
DATASET_PATH = "CUB_200_2011/images"
SEEN_CLASSES = 150
MAX_IMAGES_PER_CLASS = 100

# Model Configuration
CLIP_MODEL = "ViT-B/32"
RANDOM_STATE = 42
TEST_SIZE = 0.3

# PCA + LOF Parameters
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

# Device Configuration
def get_device():
    """Get the best available device"""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
PLOTS_DIR = "plots"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
