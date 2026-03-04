#!/bin/bash
#
# Train SuiT model on SUNRGBD with GeoLexels superpixels
#
# Usage:
#   bash run_suit_training.sh
#   bash run_suit_training.sh --batch-size 16 --epochs 100
#

set -e

# Default parameters
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_ROOT="${PROJECT_ROOT}/datasets/SUNRGBD"
GEOLEXELS_CACHE="${DATASET_ROOT}/.geolexels_cache"

# Training parameters (can be overridden)
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
MODEL="${MODEL:-suit_tiny_224}"
INPUT_SIZE="${INPUT_SIZE:-224}"
N_SPIX_SEGMENTS="${N_SPIX_SEGMENTS:-196}"
DOWNSAMPLE="${DOWNSAMPLE:-2}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Verify GeoLexels cache exists
if [ ! -d "$GEOLEXELS_CACHE" ]; then
    echo "ERROR: GeoLexels cache not found at: $GEOLEXELS_CACHE"
    echo "Please run the preprocessing script first:"
    echo "  cd $SCRIPT_DIR && bash run_geolexels_preprocessing.sh"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${SCRIPT_DIR}/outputs/sunrgbd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Create checkpoint directory
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/train.log"

echo "=========================================="
echo "Training SuiT on SUNRGBD with GeoLexels"
echo "=========================================="
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Dataset: $DATASET_ROOT"
echo "GeoLexels cache: $GEOLEXELS_CACHE"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Change to SuiT directory
cd "$SCRIPT_DIR"

# Run training
python3 main.py \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --data-path "$DATASET_ROOT" \
    --data-set SUNRGBD \
    --geolexels-cache-dir "$GEOLEXELS_CACHE" \
    --n-spix-segments "$N_SPIX_SEGMENTS" \
    --downsample "$DOWNSAMPLE" \
    --num-workers "$WORKERS" \
    --device "$DEVICE" \
    --input-size "$INPUT_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --log-file "$LOG_FILE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
