"""
Multi-dataset configuration for novelty detection experiments
Supports CUB-200-2011, AWA2, and Oxford-102 Flowers
"""
import os
from typing import Dict, List, Tuple

class DatasetConfig:
    """Base dataset configuration"""
    
    def __init__(self, name: str, path: str, total_classes: int, seen_classes: int):
        self.name = name
        self.path = path
        self.total_classes = total_classes
        self.seen_classes = seen_classes
        self.novel_classes = total_classes - seen_classes
        
    def get_class_splits(self) -> Tuple[int, int]:
        """Get seen and novel class counts"""
        return self.seen_classes, self.novel_classes


# Dataset configurations
DATASET_CONFIGS = {
    'CUB': DatasetConfig(
        name='CUB-200-2011',
        path='CUB_200_2011/images',
        total_classes=200,
        seen_classes=150
    ),
    
    'AWA2': DatasetConfig(
        name='Animals with Attributes 2',
        path='AWA2/JPEGImages',
        total_classes=50,
        seen_classes=40
    ),
    
    'FLO': DatasetConfig(
        name='Oxford-102 Flowers', 
        path='oxford-102-flowers/jpg',
        total_classes=102,
        seen_classes=80
    )
}

# Dataset-specific parameters
DATASET_PARAMS = {
    'CUB': {
        'max_images_per_class': 100,
        'image_extensions': ['.jpg', '.jpeg'],
        'default_pca_components': 50,
        'default_contamination': 0.35,
        'clip_model': 'ViT-B/32'
    },
    
    'AWA2': {
        'max_images_per_class': 200,  # AWA2 has more images per class
        'image_extensions': ['.jpg', '.jpeg'],
        'default_pca_components': 40,  # Fewer classes, fewer components
        'default_contamination': 0.30,
        'clip_model': 'ViT-B/32'
    },
    
    'FLO': {
        'max_images_per_class': 80,   # Flowers dataset is smaller
        'image_extensions': ['.jpg', '.jpeg'],
        'default_pca_components': 45,
        'default_contamination': 0.25,  # Flowers might be more separable
        'clip_model': 'ViT-B/32'
    }
}

# Standard zero-shot learning splits (from literature)
ZSL_SPLITS = {
    'CUB': {
        'seen_classes': list(range(150)),
        'unseen_classes': list(range(150, 200))
    },
    
    'AWA2': {
        # Standard AWA2 zero-shot split
        'seen_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        'unseen_classes': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    },
    
    'FLO': {
        'seen_classes': list(range(80)),
        'unseen_classes': list(range(80, 102))
    }
}

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for specific dataset"""
    if dataset_name.upper() not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name.upper()]

def get_dataset_params(dataset_name: str) -> Dict:
    """Get parameters for specific dataset"""
    if dataset_name.upper() not in DATASET_PARAMS:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(DATASET_PARAMS.keys())}")
    
    return DATASET_PARAMS[dataset_name.upper()]

def get_zsl_split(dataset_name: str) -> Dict[str, List[int]]:
    """Get zero-shot learning split for specific dataset"""
    if dataset_name.upper() not in ZSL_SPLITS:
        raise ValueError(f"ZSL split for {dataset_name} not defined")
    
    return ZSL_SPLITS[dataset_name.upper()]

def print_dataset_info(dataset_name: str):
    """Print information about a dataset"""
    config = get_dataset_config(dataset_name)
    params = get_dataset_params(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"DATASET: {config.name}")
    print(f"{'='*60}")
    print(f"Path: {config.path}")
    print(f"Total classes: {config.total_classes}")
    print(f"Seen classes: {config.seen_classes}")
    print(f"Novel classes: {config.novel_classes}")
    print(f"Max images per class: {params['max_images_per_class']}")
    print(f"PCA components: {params['default_pca_components']}")
    print(f"Contamination: {params['default_contamination']}")
    print(f"CLIP model: {params['clip_model']}")
    print(f"{'='*60}")

# Benchmark results from literature for comparison
LITERATURE_BENCHMARKS = {
    'CUB': {
        'CLIP': {'seen_acc': 0.780, 'unseen_acc': 0.514, 'harmonic_mean': 0.620},
        'DUET': {'seen_acc': 0.892, 'unseen_acc': 0.658, 'harmonic_mean': 0.757},
        'SJE': {'seen_acc': 0.655, 'unseen_acc': 0.538, 'harmonic_mean': 0.591},
    },
    
    'AWA2': {
        'CLIP': {'seen_acc': 0.685, 'unseen_acc': 0.596, 'harmonic_mean': 0.638},
        'DUET': {'seen_acc': 0.823, 'unseen_acc': 0.672, 'harmonic_mean': 0.740},
        'SJE': {'seen_acc': 0.617, 'unseen_acc': 0.565, 'harmonic_mean': 0.590},
    },
    
    'FLO': {
        'CLIP': {'seen_acc': 0.823, 'unseen_acc': 0.678, 'harmonic_mean': 0.744},
        'Visual-Semantic': {'seen_acc': 0.742, 'unseen_acc': 0.631, 'harmonic_mean': 0.682},
        'Attribute-based': {'seen_acc': 0.698, 'unseen_acc': 0.587, 'harmonic_mean': 0.638},
    }
}

def get_literature_benchmarks(dataset_name: str) -> Dict:
    """Get literature benchmarks for comparison"""
    return LITERATURE_BENCHMARKS.get(dataset_name.upper(), {})
