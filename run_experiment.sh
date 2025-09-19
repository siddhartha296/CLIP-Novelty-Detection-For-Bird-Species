#!/bin/bash

# PCA + LOF Novelty Detection Experiment Runner
# Usage: ./run_experiment.sh [baseline|optimize|both]

# Set default values
DATASET_PATH="CUB_200_2011/images"
SEEN_CLASSES=150
MAX_IMAGES=100
LOG_LEVEL="INFO"

# Parse command line argument
MODE=${1:-baseline}

# Create timestamp for experiment naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="novelty_detection_${TIMESTAMP}"

echo "=================================================="
echo "🚀 PCA + LOF Novelty Detection Pipeline"
echo "=================================================="
echo "Mode: $MODE"
echo "Dataset: $DATASET_PATH"
echo "Seen classes: $SEEN_CLASSES"
echo "Experiment: $EXPERIMENT_NAME"
echo "=================================================="

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset directory not found: $DATASET_PATH"
    echo "Please update DATASET_PATH in the script or create symlink"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "🔧 Activating existing virtual environment..."
    source venv/bin/activate
fi

# Check GPU availability
if python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"; then
    echo "🎯 GPU setup verified"
else
    echo "⚠️  Warning: CUDA not available, using CPU"
fi

# Run the appropriate experiment
case $MODE in
    "baseline")
        echo "🎯 Running baseline experiment..."
        python main.py \
            --dataset_path "$DATASET_PATH" \
            --seen_classes $SEEN_CLASSES \
            --max_images $MAX_IMAGES \
            --experiment_name "$EXPERIMENT_NAME" \
            --log_level "$LOG_LEVEL"
        ;;
    
    "optimize")
        echo "🔍 Running hyperparameter optimization..."
        python main.py \
            --dataset_path "$DATASET_PATH" \
            --seen_classes $SEEN_CLASSES \
            --max_images $MAX_IMAGES \
            --optimize \
            --experiment_name "${EXPERIMENT_NAME}_opt" \
            --log_level "$LOG_LEVEL"
        ;;
    
    "both")
        echo "🎯 Running both baseline and optimization..."
        
        # Run baseline first
        echo "📊 Step 1: Baseline experiment..."
        python main.py \
            --dataset_path "$DATASET_PATH" \
            --seen_classes $SEEN_CLASSES \
            --max_images $MAX_IMAGES \
            --experiment_name "${EXPERIMENT_NAME}_baseline" \
            --log_level "$LOG_LEVEL"
        
        echo "🔍 Step 2: Hyperparameter optimization..."
        python main.py \
            --dataset_path "$DATASET_PATH" \
            --seen_classes $SEEN_CLASSES \
            --max_images $MAX_IMAGES \
            --optimize \
            --experiment_name "${EXPERIMENT_NAME}_optimized" \
            --log_level "$LOG_LEVEL"
        ;;
    
    *)
        echo "❌ Error: Invalid mode '$MODE'"
        echo "Usage: $0 [baseline|optimize|both]"
        exit 1
        ;;
esac

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Experiment completed successfully!"
    echo "📁 Results directory: outputs/"
    echo "🤖 Models directory: models/"
    echo "📊 Plots directory: plots/"
    echo ""
    echo "📝 Key files:"
    echo "   - Log: outputs/${EXPERIMENT_NAME}.log"
    echo "   - Config: outputs/${EXPERIMENT_NAME}_config.json" 
    echo "   - Results: experiments/${EXPERIMENT_NAME}/results.json"
    echo ""
    echo "🔍 To view results:"
    echo "   tail -f outputs/${EXPERIMENT_NAME}.log"
    echo "   python -c \"from utils import load_json; print(load_json('experiments/${EXPERIMENT_NAME}/results.json'))\""
else
    echo ""
    echo "❌ Experiment failed!"
    echo "📝 Check log file: outputs/${EXPERIMENT_NAME}.log"
    exit 1
fi

# Optional: Run inference example
read -p "🔮 Would you like to run inference example? (y/N): " run_inference
if [[ $run_inference =~ ^[Yy]$ ]]; then
    echo "Running inference example..."
    python -c "from main import run_inference_example; run_inference_example()"
fi

echo "🎉 All done!"
