"""
Data loading and CLIP embedding extraction utilities
"""
import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPEmbeddingExtractor:
    """CLIP embedding extraction for images"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        logger.info(f"✅ CLIP model {model_name} loaded on {self.device}")
    
    def extract_embeddings(self, class_list: List[str], dataset_path: str, 
                          label_type: str = "seen", max_per_class: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract CLIP embeddings from images
        
        Args:
            class_list: List of class names
            dataset_path: Path to dataset
            label_type: "seen" or "novel" 
            max_per_class: Maximum images per class
            
        Returns:
            embeddings: CLIP embeddings
            labels: Binary labels (0=seen, 1=novel)
            class_labels: Class indices
        """
        embeddings, labels, class_labels = [], [], []
        
        for cls_idx, cls in enumerate(tqdm(class_list, desc=f"Processing {label_type} classes")):
            cls_path = os.path.join(dataset_path, cls)
            if not os.path.exists(cls_path):
                logger.warning(f"Path not found: {cls_path}")
                continue
                
            img_files = [f for f in os.listdir(cls_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for i, img_file in enumerate(img_files):
                if max_per_class and i >= max_per_class:
                    break
                    
                try:
                    img_path = os.path.join(cls_path, img_file)
                    image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        emb = self.model.encode_image(image)
                        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
                        
                    embeddings.append(emb.cpu().numpy().flatten())
                    labels.append(0 if label_type == "seen" else 1)
                    class_labels.append(cls_idx)
                        
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        logger.info(f"Extracted {len(embeddings)} embeddings for {label_type} classes")
        return np.array(embeddings), np.array(labels), np.array(class_labels)


class DatasetProcessor:
    """Dataset processing and splitting utilities"""
    
    def __init__(self, dataset_path: str, seen_classes: int = 150):
        self.dataset_path = dataset_path
        self.seen_classes = seen_classes
        
    def get_class_splits(self) -> Tuple[List[str], List[str]]:
        """Get seen and novel class splits"""
        all_classes = sorted(os.listdir(self.dataset_path))
        seen_classes = all_classes[:self.seen_classes]
        novel_classes = all_classes[self.seen_classes:]
        
        logger.info(f"Dataset split: {len(seen_classes)} seen | {len(novel_classes)} novel classes")
        return seen_classes, novel_classes
    
    def create_train_val_split(self, X: np.ndarray, y: np.ndarray, class_labels: np.ndarray,
                              test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Create train/validation split with stratification"""
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=class_labels
        )
    
    def prepare_test_set(self, X_seen_val: np.ndarray, y_seen_val: np.ndarray,
                        X_novel: np.ndarray, y_novel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Combine validation seen + novel for final evaluation"""
        X_test = np.vstack([X_seen_val, X_novel])
        y_test = np.hstack([y_seen_val, y_novel])
        
        logger.info(f"Test set: {X_test.shape} with {np.sum(y_test == 0)} seen, {np.sum(y_test == 1)} novel")
        return X_test, y_test


def load_and_process_data(dataset_path: str, seen_classes: int = 150, 
                         max_images_per_class: int = 100, clip_model: str = "ViT-B/32",
                         test_size: float = 0.3, random_state: int = 42):
    """
    Complete data loading and processing pipeline
    
    Returns:
        Dictionary containing all processed data splits and components
    """
    # Initialize components
    extractor = CLIPEmbeddingExtractor(clip_model)
    processor = DatasetProcessor(dataset_path, seen_classes)
    
    # Get class splits
    seen_class_list, novel_class_list = processor.get_class_splits()
    
    # Extract embeddings
    logger.info("Extracting CLIP embeddings...")
    X_seen, y_seen, seen_class_labels = extractor.extract_embeddings(
        seen_class_list, dataset_path, "seen", max_images_per_class
    )
    X_novel, y_novel, novel_class_labels = extractor.extract_embeddings(
        novel_class_list, dataset_path, "novel", max_images_per_class
    )
    
    # Create train/val split for seen data
    X_seen_train, X_seen_val, y_seen_train, y_seen_val = processor.create_train_val_split(
        X_seen, y_seen, seen_class_labels, test_size, random_state
    )
    
    # Prepare final test set
    X_test, y_test = processor.prepare_test_set(X_seen_val, y_seen_val, X_novel, y_novel)
    
    return {
        'X_seen_train': X_seen_train,
        'X_seen_val': X_seen_val,
        'X_novel': X_novel,
        'X_test': X_test,
        'y_seen_train': y_seen_train,
        'y_seen_val': y_seen_val,
        'y_novel': y_novel,
        'y_test': y_test,
        'seen_class_labels': seen_class_labels,
        'extractor': extractor,
        'processor': processor
    }
