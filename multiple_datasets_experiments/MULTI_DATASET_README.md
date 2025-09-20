# Multi-Dataset Novelty Detection Experiments

Run novelty detection experiments on **CUB-200-2011**, **AWA2**, and **Oxford-102 Flowers** datasets.

## 🚀 Quick Start

### 1. **Automated Dataset Setup**
```bash
# Setup all datasets automatically
python setup_datasets.py --all

# Or setup individual datasets
python setup_datasets.py --cub
python setup_datasets.py --flowers
python setup_datasets.py --awa2  # Requires manual download

# Verify setup
python setup_datasets.py --verify
```

### 2. **Run Multi-Dataset Experiments**
```bash
# Run on all three datasets
python run_multi_dataset.py --datasets CUB AWA2 FLO

# Run on single dataset
python run_multi_dataset.py --single CUB

# Run with hyperparameter optimization
python run_multi_dataset.py --datasets CUB AWA2 --optimize

# Custom experiment name
python run_multi_dataset.py --datasets CUB --experiment_name "cub_detailed_analysis"
```

### 3. **Quick Dataset Test**
```bash
# Test if datasets are properly set up
python -c "from run_multi_dataset import quick_dataset_test; quick_dataset_test('CUB')"
python -c "from run_multi_dataset import quick_dataset_test; quick_dataset_test('AWA2')"
python -c "from run_multi_dataset import quick_dataset_test; quick_dataset_test('FLO')"
```

## 📁 Expected Directory Structure

```
your_project/
├── CUB_200_2011/
│   └── images/
│       ├── 001.Black_footed_Albatross/
│       ├── 002.Laysan_Albatross/
│       └── ... (200 classes)
├── AWA2/
│   └── JPEGImages/
│       ├── antelope/
│       ├── grizzly+bear/
│       └── ... (50 classes)
├── oxford-102-flowers/
│   └── jpg/
│       ├── class_001/
│       ├── class_002/
│       └── ... (102 classes)
└── [your code files]
```

## 🔧 Dataset-Specific Configurations

Each dataset has optimized parameters:

### **CUB-200-2011** (Bird Species)
- **Classes**: 200 (150 seen, 50 novel)
- **Images per class**: Up to 100
- **PCA components**: 50
- **Best for**: Fine-grained visual categorization

### **AWA2** (Animals with Attributes)
- **Classes**: 50 (40 seen, 10 novel)
- **Images per class**: Up to 200
- **PCA components**: 40
- **Best for**: Attribute-based zero-shot learning

### **Oxford-102 Flowers**
- **Classes**: 102 (80 seen, 22 novel)
- **Images per class**: Up to 80
- **PCA components**: 45
- **Best for**: Texture and color-based classification

## 📊 Example Results

### **Cross-Dataset Performance**
```
Dataset         ROC AUC    H-Mean     Seen Acc   Unseen Acc  F1-Score
CUB            0.749      0.669      0.620      0.727       0.682
AWA2           0.782      0.701      0.645      0.763       0.695
FLO            0.768      0.685      0.631      0.745       0.678
```

### **Literature Comparison**
```
Method              CUB H-Mean    AWA2 H-Mean    FLO H-Mean
DUET (2024)         0.757         0.740          -
CLIP (2021)         0.620         0.638          0.744
Our PCA+LOF         0.669         0.701          0.685
```

## 🎯 Advanced Usage

### **Custom Dataset Configuration**
```python
from dataset_configs import DATASET_CONFIGS, DATASET_PARAMS

# Add custom dataset
DATASET_CONFIGS['CUSTOM'] = DatasetConfig(
    name='My Custom Dataset',
    path='path/to/custom/dataset',
    total_classes=100,
    seen_classes=80
)
```

### **Hyperparameter Optimization**
```python
from run_multi_dataset import MultiDataset
