#!/bin/bash
# Preprocess SUNRGBD dataset with GeoLexels superpixel segmentation
# This script runs fast_cloud + GeoLexels segmentation to generate superpixel labels

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
DATASET_ROOT="${PROJECT_ROOT}/datasets/SUNRGBD"
OUTPUT_DIR=""  # Will default to $DATASET_ROOT/.geolexels_cache
FAST_CLOUD_EXE="${PROJECT_ROOT}/pointcloud/build/fast_cloud"
NUM_WORKERS=4
SKIP_EXISTING=false

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print usage
print_usage() {
    echo "Usage: bash preprocess_geolexels.sh [OPTIONS]"
    echo ""
    echo "Preprocess SUNRGBD dataset with GeoLexels superpixel segmentation."
    echo ""
    echo "Options:"
    echo "  --dataset-root DIR     Path to SUNRGBD dataset (default: $DATASET_ROOT)"
    echo "  --output-dir DIR       Output directory for cache (default: <dataset-root>/.geolexels_cache)"
    echo "  --fast-cloud-exe PATH  Path to fast_cloud executable (default: $FAST_CLOUD_EXE)"
    echo "  --num-workers N        Number of parallel workers (default: $NUM_WORKERS)"
    echo "  --skip-existing        Skip already processed files"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "GeoLexels Parameters (hardcoded in precompute_geolexels.py):"
    echo "  - Mode: 3 (Color + Depth + Normals with Cosine distance)"
    echo "  - Threshold: 60.0"
    echo "  - Weight Depth: 1.0"
    echo "  - Weight Normals: 2.0"
    echo "  - Focal Length: 1.0"
    echo ""
    echo "Example:"
    echo "  bash preprocess_geolexels.sh --num-workers 8 --skip-existing"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fast-cloud-exe)
            FAST_CLOUD_EXE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Set default output dir if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${DATASET_ROOT}/.geolexels_cache"
fi

# Print configuration
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}GeoLexels Preprocessing for SUNRGBD${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Configuration:"
echo "  Dataset root:     $DATASET_ROOT"
echo "  Output directory: $OUTPUT_DIR"
echo "  fast_cloud exe:   $FAST_CLOUD_EXE"
echo "  Workers:          $NUM_WORKERS"
echo "  Skip existing:    $SKIP_EXISTING"
echo ""

# Validate paths
if [ ! -d "$DATASET_ROOT" ]; then
    echo -e "${RED}Error: Dataset root not found: $DATASET_ROOT${NC}"
    exit 1
fi

if [ ! -f "$FAST_CLOUD_EXE" ]; then
    echo -e "${RED}Error: fast_cloud executable not found: $FAST_CLOUD_EXE${NC}"
    echo -e "${YELLOW}Did you build it? Run: cd pointcloud/build && cmake .. && make${NC}"
    exit 1
fi

# Check if GeoLexels library is compiled
GEOLEXELS_LIB="${PROJECT_ROOT}/GeoLexels/_geolexels.*.so"
if ! ls $GEOLEXELS_LIB 1> /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: GeoLexels library not found${NC}"
    echo -e "${YELLOW}Compiling GeoLexels library...${NC}"
    cd "${PROJECT_ROOT}/GeoLexels"
    python compile_geolexels_lib.py
    cd "$SCRIPT_DIR"
    echo -e "${GREEN}GeoLexels library compiled successfully${NC}"
    echo ""
fi

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate suit

# Build command
CMD="python precompute_geolexels.py \
    --dataset-root \"$DATASET_ROOT\" \
    --output-dir \"$OUTPUT_DIR\" \
    --fast-cloud-exe \"$FAST_CLOUD_EXE\" \
    --num-workers $NUM_WORKERS"

if [ "$SKIP_EXISTING" = true ]; then
    CMD="$CMD --skip-existing"
fi

echo -e "${GREEN}======================================${NC}"
echo "Starting preprocessing..."
echo ""

# Run preprocessing
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Preprocessing completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Cache location: $OUTPUT_DIR"
    echo ""
    echo "You can now train with:"
    echo "  bash run_suit_training.sh --num-workers 4 --trial-name my_trial"
else
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Preprocessing failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}======================================${NC}"
    exit $EXIT_CODE
fi
