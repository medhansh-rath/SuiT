#!/bin/bash
#
# Complete workflow script for GeoLexels preprocessing and SuiT training
# Run this for a complete end-to-end setup
#
# Usage:
#   bash run_complete_workflow.sh
#   bash run_complete_workflow.sh --max-images 100  # Test with 100 images
#   bash run_complete_workflow.sh --num-workers 8   # Use 8 parallel workers for GPU
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Parse arguments
MAX_IMAGES=""
NUM_WORKERS="4"
SKIP_PREPROCESSING=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-images)
            MAX_IMAGES="--max-images $2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_section() {
    echo -e "\n${BLUE}========== $1 ==========${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}\n"
}

# Step 0: Verify Prerequisites
print_section "Step 0: Verifying Prerequisites"

FAST_CLOUD="${PROJECT_ROOT}/pointcloud/build/fast_cloud"
if [ ! -f "$FAST_CLOUD" ]; then
    echo "ERROR: fast_cloud not found at $FAST_CLOUD"
    echo "Building fast_cloud..."
    cd "$PROJECT_ROOT/pointcloud"
    mkdir -p build
    cd build
    cmake .. > /dev/null 2>&1
    make -j$(nproc) > /dev/null 2>&1
    print_success "Built fast_cloud"
else
    print_success "Found fast_cloud at $FAST_CLOUD"
fi

DATASET="${PROJECT_ROOT}/datasets/SUNRGBD"
if [ ! -d "$DATASET" ]; then
    echo "ERROR: SUNRGBD dataset not found at $DATASET"
    exit 1
fi
print_success "Found SUNRGBD dataset"

# Step 1: Precompute GeoLexels
if [ $SKIP_PREPROCESSING -eq 0 ]; then
    print_section "Step 1: Preprocessing SUNRGBD with GeoLexels"
    echo "This generates cached GeoLexels superpixel assignments for all RGB/Depth pairs."
    echo "Time ~5-10 seconds per image, 17188 images total (~24-48h on single CPU)"
    echo ""
    
    cd "$SCRIPT_DIR"
    bash run_geolexels_preprocessing.sh --num-workers $NUM_WORKERS $MAX_IMAGES
    print_success "GeoLexels preprocessing complete!"
else
    print_section "Step 1: Skipping preprocessing (--skip-preprocessing flag)"
fi

# Step 2: Verify cache exists
print_section "Step 2: Verifying Cache"

CACHE_DIR="${DATASET}/.geolexels_cache"
CACHE_COUNT=$(find "$CACHE_DIR" -name "*.npy" 2>/dev/null | wc -l || echo 0)

if [ $CACHE_COUNT -eq 0 ]; then
    echo "ERROR: No cached GeoLexels found. Preprocessing may have failed."
    exit 1
fi

print_success "Found $CACHE_COUNT cached GeoLexels files"

# Step 3: Summary
print_section "Step 3: Ready for Training"

echo "To train the SuiT model, run:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  bash run_suit_training.sh"
echo ""
echo "Or with custom parameters:"
echo "  bash run_suit_training.sh --batch-size 32 --epochs 50 --lr 0.0005"
echo ""
echo "Configuration:"
echo "  - Dataset: $DATASET"
echo "  - Cache: $CACHE_DIR"
echo "  - Cached images: $CACHE_COUNT"
echo ""
