#!/bin/bash
#
# Batch process SUNRGBD dataset with GeoLexels
# This script runs the Python preprocessing script to generate GeoLexels for all RGB/Depth pairs
#
# Usage:
#   bash run_geolexels_preprocessing.sh [--max-images N] [--resume-from IDX] [--skip-existing] [--num-workers N]
#
# For GPU optimization:
#   - Use --num-workers equal to number of GPU cores (e.g., 8-16 for RTX 5060)
#   - Higher values = better GPU utilization but more CPU overhead
#   Example for RTX 5060: bash run_geolexels_preprocessing.sh --num-workers 8
#

set -e

# Default paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
DATASET_ROOT="${PROJECT_ROOT}/datasets/SUNRGBD"
FAST_CLOUD_EXE="${PROJECT_ROOT}/pointcloud/build/fast_cloud"
PYTHON_SCRIPT="${SCRIPT_DIR}/precompute_geolexels.py"

# Parse arguments
MAX_IMAGES=""
START_IDX=""
SKIP_EXISTING=""
NUM_WORKERS="4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-images)
            MAX_IMAGES="--max-images $2"
            shift 2
            ;;
        --resume-from)
            START_IDX="--start-idx $2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Verify files exist
if [ ! -f "$FAST_CLOUD_EXE" ]; then
    echo "ERROR: fast_cloud executable not found at: $FAST_CLOUD_EXE"
    echo "Please build the pointcloud project first:"
    echo "  cd $PROJECT_ROOT/pointcloud && mkdir build && cd build && cmake .. && make"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: SUNRGBD dataset not found at: $DATASET_ROOT"
    exit 1
fi

# Create log directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Create timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/preprocessing_${TIMESTAMP}.log"

echo "=========================================="
echo "GeoLexels Preprocessing for SUNRGBD"
echo "=========================================="
echo "Dataset root: $DATASET_ROOT"
echo "Output cache: $DATASET_ROOT/.geolexels_cache"
echo "fast_cloud executable: $FAST_CLOUD_EXE"
echo "Parallel workers: $NUM_WORKERS"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

# Run the preprocessing script
python3 "$PYTHON_SCRIPT" \
    --dataset-root "$DATASET_ROOT" \
    --fast-cloud-exe "$FAST_CLOUD_EXE" \
    --temp-dir /tmp \
    --num-workers $NUM_WORKERS \
    $MAX_IMAGES \
    $START_IDX \
    $SKIP_EXISTING \
    2>&1 | tee "$LOG_FILE"
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
