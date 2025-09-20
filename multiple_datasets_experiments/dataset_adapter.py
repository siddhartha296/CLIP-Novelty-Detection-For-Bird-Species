"""
Dataset adapter for handling multiple datasets (CUB, AWA2, FLO)
with different directory structures and formats
"""
import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch
import clip
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset_configs import get_dataset_config, get_dataset_params, get_zsl_split

logger = logging.getLogger(__name__)


class MultiDatasetAdapter:
    """Adapter to handle different dataset structures"""
    
    def __init__(self, dataset_name: str, clip_model: str = "ViT-B/32", device: str = None):
        self.dataset_name = dataset_name.upper()
        self.config = get_dataset_config(self.dataset_name)
        self.params = get_dataset_params(self.dataset_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        logger.info(f"✅ CLIP model {clip_model} loaded for {self.config.name}")
        
    def get_class_structure(self) -> Tuple[List[str], List[str]]:
        """Get class structure based on dataset type"""
        
        if self.dataset_name == 'CUB':
            return self._get_cub_classes()
        elif self.dataset_name == 'AWA2':
            return self._get_awa2_classes()
        elif self.dataset_name == 'FLO':
            return self._get_flo_classes()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _get_cub_classes(self) -> Tuple[List[str], List[str]]:
        """Get CUB-200-2011 class structure"""
        all_classes = sorted(os.listdir(self.config.path))
        seen_classes = all_classes[:self.config.seen_classes]
        novel_classes = all_classes[self.config.seen_classes:]
        return seen_classes, novel_classes
    
    def _get_awa2_classes(self) -> Tuple[List[str], List[str]]:
        """Get AWA2 class structure"""
        # AWA2 has a specific class list
        all_classes = sorted(os.listdir(self.config.path))
        
        # Filter out any non-directory items
        all_classes = [cls for cls in all_classes 
                      if os.path.isdir(os.path.join(self.config.path, cls))]
        
        if len(all_classes) != self.config.total_classes:
            logger.warning(f"Expected {self.config.total_classes} classes, found {len(all_classes)}")
        
        seen_classes = all_classes[:self.config.seen_classes]
        novel_classes = all_classes[self.config.seen_classes:]
        
        logger.info(f"AWA2: {len(seen_classes)} seen classes, {len(novel_classes)} novel classes")
        return seen_classes, novel_classes
    
    def _get_flo_classes(self) -> Tuple[List[str], List[str]]:
        """Get Oxford-102 Flowers class structure"""
        
        # Oxford Flowers might have numbered directories
        if os.path.exists(self.config.path):
            all_classes = sorted(os.listdir(self.config.path))
            all_classes = [cls for cls in all_classes 
                          if os.path.isdir(os.path.join(self.config.path, cls))]
        else:
            # Create numbered class directories if they don't exist
            all_classes = [f"class_{i:03d}" for i in range(1, self.config.total_classes + 1)]
        
        seen_classes = all_classes[:self.config.seen_classes]
        novel_classes = all_classes[self.config.seen_classes:]
        
        logger.info(f"FLO: {len(seen_classes)} seen classes, {len(novel_classes)} novel classes")
        return seen_classes, novel_classes
    
    def extract_embeddings(self, class_list: List[str], label_type: str = "seen", 
                          max_per_class: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract CLIP embeddings from images"""
        
        if max_per_class is None:
            max_per_class = self.params['max_images_per_class']
        
        embeddings, labels, class_labels = [], [], []
        valid_extensions = set(self.params['image_extensions'])
        
        for cls_idx, cls in enumerate(tqdm(class_list, desc=f"Processing {label_type} classes ({self.dataset_name})")):
            cls_path = os.path.join(self.config.path, cls)
            
            if not os.path.exists(cls_path):
                logger.warning(f"Class directory not found: {cls_path}")
                continue
            
            # Get all image files
            try:
                all_files = os.listdir(cls_path)
            except PermissionError:
                logger.warning(f"Permission denied: {cls_path}")
                continue
            
            img_files = [f for f in all_files 
                        if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            if not img_files:
                logger.warning(f"No images found in {cls_path}")
                continue
            
            # Process images
            processed_count = 0
            for img_file in img_files:
                if processed_count >= max_per_class:
                    break
                
                img_path = os.path.join(cls_path, img_file)
                
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    # Extract CLIP embedding
                    with torch.no_grad():
                        embedding = self.model.encode_image(image_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
                    
                    embeddings.append(embedding.cpu().numpy().flatten())
                    labels.append(0 if label_type == "seen" else 1)
                    class_labels.append(cls_idx)
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            if processed_count > 0:
                logger.debug(f"Processed {processed_count} images from class {cls}")
        
        logger.info(f"Extracted {len(embeddings)} embeddings for {label_type} classes ({self.dataset_name})")
        return np.array(embeddings), np.array(labels), np.array(class_labels)


def load_multi_dataset(dataset_name: str, max_images_per_class: int = None, 
                      test_size: float = 0.3, random_state: int = 42,
                      clip_model: str = "ViT-B/32") -> Dict:
    """
    Load and process any supported dataset
    
    Args:
        dataset_name: 'CUB', 'AWA2', or 'FLO'
        max_images_per_class: Maximum images to use per class
        test_size: Train/validation split ratio
        random_state: Random seed
        clip_model: CLIP model variant
    
    Returns:
        Dictionary with processed data splits
    """
    
    logger.info(f"🔄 Loading {dataset_name} dataset...")
    
    # Initialize adapter
    adapter = MultiDatasetAdapter(dataset_name, clip_model)
    config = adapter.config
    params = adapter.params
    
    if max_images_per_class is None:
        max_images_per_class = params['max_images_per_class']
    
    # Get class structure
    seen_classes, novel_classes = adapter.get_class_structure()
    
    logger.info(f"📊 {config.name}: {len(seen_classes)} seen | {len(novel_classes)} novel classes")
    
    # Extract embeddings
    X_seen, y_seen, seen_class_labels = adapter.extract_embeddings(
        seen_classes, "seen", max_images_per_class
    )
    
    X_novel, y_novel, novel_class_labels = adapter.extract_embeddings(
        novel_classes, "novel", max_images_per_class
    )
    
    # Adjust novel class labels to global indices
    novel_class_labels = novel_class_labels + len(seen_classes)
    
    # Create train/val split for seen data
    X_seen_train, X_seen_val, y_seen_train, y_seen_val, seen_train_cls, seen_val_cls = train_test_split(
        X_seen, y_seen, seen_class_labels, 
        test_size=test_size, random_state=random_state, 
        stratify=seen_class_labels
    )
    
    # Prepare test set
    X_test = np.vstack([X_seen_val, X_novel])
    y_test = np.hstack([y_seen_val, y_novel])
    test_class_labels = np.hstack([seen_val_cls, novel_class_labels])
    
    # Create is_seen_class array
    total_classes = len(seen_classes) + len(novel_classes)
    is_seen_class = np.zeros(total_classes, dtype=bool)
    is_seen_class[:len(seen_classes)] = True
    
    logger.info(f"📊 Final test set: {X_test.shape} with {np.sum(y_test == 0)} seen, {np.sum(y_test == 1)} novel")
    
    return {
        'dataset_name': dataset_name.upper(),
        'dataset_config': config,
        'dataset_params': params,
        'X_seen_train': X_seen_train,
        'X_seen_val': X_seen_val,
        'X_novel': X_novel,
        'X_test': X_test,
        'y_seen_train': y_seen_train,
        'y_seen_val': y_seen_val,
        'y_novel': y_novel,
        'y_test': y_test,
        'seen_class_labels': seen_class_labels,
        'test_class_labels': test_class_labels,
        'is_seen_class': is_seen_class,
        'seen_class_list': seen_classes,
        'novel_class_list': novel_classes,
        'adapter': adapter
    }


def validate_dataset_setup(dataset_name: str) -> bool:
    """Validate that a dataset is properly set up"""
    
    config = get_dataset_config(dataset_name)
    
    if not os.path.exists(config.path):
        logger.error(f"❌ Dataset path not found: {config.path}")
        return False
    
    # Check for class directories
    if not os.path.isdir(config.path):
        logger.error(f"❌ Dataset path is not a directory: {config.path}")
        return False
    
    # Count class directories
    class_dirs = [d for d in os.listdir(config.path) 
                 if os.path.isdir(os.path.join(config.path, d))]
    
    if len(class_dirs) == 0:
        logger.error(f"❌ No class directories found in {config.path}")
        return False
    
    logger.info(f"✅ {config.name} dataset validated: {len(class_dirs)} classes found")
    return True


def setup_dataset_instructions():
    """Print setup instructions for different datasets"""
    
    instructions = """
    
🔧 DATASET SETUP INSTRUCTIONS:

1. CUB-200-2011:
   - Download from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
   - Extract to: CUB_200_2011/images/
   - Structure: CUB_200_2011/images/001.Black_footed_Albatross/...

2. AWA2 (Animals with Attributes 2):
   - Download from: https://cvml.ist.ac.at/AwA2/
   - Extract to: AWA2/JPEGImages/
   - Structure: AWA2/JPEGImages/antelope/...

3. Oxford-102 Flowers:
   - Download from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
   - Extract to: oxford-102-flowers/jpg/
   - Structure: oxford-102-flowers/jpg/image_00001.jpg (or organized by class)

📁 Expected directory structure:
your_project/
├── CUB_200_2011/images/          # CUB dataset
├── AWA2/JPEGImages/              # AWA2 dataset  
├── oxford-102-flowers/jpg/       # Flowers dataset
└── [your code files]

🔗 Quick setup commands:
# Create symbolic links if datasets are elsewhere
ln -s /path/to/CUB_200_2011 ./CUB_200_2011
ln -s /path/to/AWA2 ./AWA2
ln -s /path/to/oxford-102-flowers ./oxford-102-flowers
    """
    
    print(instructions)
