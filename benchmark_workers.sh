#!/bin/bash
#
# Benchmark script to find optimal number of workers for your GPU
# Tests different worker counts and measures processing speed
#
# Usage:
#   bash benchmark_workers.sh
#
# Requirements:
#   - Anaconda with suit environment activated
#   - nvidia-smi (optional, for GPU memory monitoring)
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Test parameters
TEST_IMAGES=200  # Test with 200 images for thorough results
WORKERS_TO_TEST=(1 2 4 8 12 16 32 64 128)  # Test these worker counts

echo "=========================================="
echo "GPU Worker Optimization Benchmark"
echo "=========================================="
echo ""
echo "This will test different worker counts to find optimal GPU utilization."
echo "Each test processes $TEST_IMAGES images and measures:"
echo "  - Processing time"
echo "  - Speed (images/second)"
echo ""

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA=1
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU detected"
    echo ""
else
    HAS_NVIDIA=0
    echo "WARNING: nvidia-smi not found. GPU monitoring will be skipped."
    echo ""
fi

# Create results file
RESULTS_FILE="${SCRIPT_DIR}/benchmark_results.txt"
> "$RESULTS_FILE"  # Clear file

echo "Starting benchmark tests..."
echo ""

# Track best configuration
BEST_WORKERS=1
BEST_SPEED=0

for WORKERS in "${WORKERS_TO_TEST[@]}"; do
    echo "=========================================="
    echo "Testing with $WORKERS workers"
    echo "=========================================="
    
    # Record start time (simple seconds)
    START_TIME=$(date +%s)
    
    # Run preprocessing
    python3 "$SCRIPT_DIR/precompute_geolexels.py" \
        --max-images $TEST_IMAGES \
        --num-workers $WORKERS \
        --fast-cloud-exe "$PROJECT_ROOT/pointcloud/build/fast_cloud" \
        --dataset-root "$PROJECT_ROOT/datasets/SUNRGBD" \
        2>&1 | tail -15
    
    # Record end time
    END_TIME=$(date +%s)
    DURATION_SEC=$((END_TIME - START_TIME))
    
    # Ensure minimum duration
    if [ $DURATION_SEC -lt 1 ]; then DURATION_SEC=1; fi
    
    # Calculate speed (simple division)
    SPEED=$(echo "$TEST_IMAGES / $DURATION_SEC" | bc 2>/dev/null || echo "0")
    
    # Get GPU memory if available
    if [ $HAS_NVIDIA -eq 1 ]; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
        GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
        if [ "$GPU_MEM" != "N/A" ] && [ "$GPU_TOTAL" != "N/A" ]; then
            echo "GPU Memory: ${GPU_MEM}MB / ${GPU_TOTAL}MB"
        fi
    fi
    
    echo "Time: ${DURATION_SEC}s | Speed: ${SPEED} img/s"
    echo ""
    
    # Save results
    echo "Workers: $WORKERS | Speed: ${SPEED} img/s | Time: ${DURATION_SEC}s" >> "$RESULTS_FILE"
    
    # Track best (simple integer comparison)
    SPEED_INT=$(echo "$SPEED" | cut -d. -f1 2>/dev/null || echo "0")
    BEST_INT=$(echo "$BEST_SPEED" | cut -d. -f1 2>/dev/null || echo "0")
    if [ -z "$SPEED_INT" ]; then SPEED_INT=0; fi
    if [ -z "$BEST_INT" ]; then BEST_INT=0; fi
    
    if [ "$SPEED_INT" -gt "$BEST_INT" ] 2>/dev/null; then
        BEST_WORKERS=$WORKERS
        BEST_SPEED=$SPEED
    fi
done

echo "=========================================="
echo "Benchmark Results"
echo "=========================================="
cat "$RESULTS_FILE"
echo ""
echo "=========================================="
echo "RECOMMENDATION"
echo "=========================================="
echo "Optimal workers: $BEST_WORKERS"
echo "Peak speed: ${BEST_SPEED} images/second"
echo ""
echo "Use this command for production:"
echo "  bash run_geolexels_preprocessing.sh --num-workers $BEST_WORKERS"
echo ""
echo "For full 17,188 images:"
TOTAL_IMAGES=17188
if [ "$BEST_SPEED" != "0" ] && [ "$BEST_SPEED" != "N/A" ]; then
    TOTAL_SECONDS=$(echo "scale=0; $TOTAL_IMAGES / $BEST_SPEED" | bc 2>/dev/null || echo "N/A")
    if [ "$TOTAL_SECONDS" != "N/A" ] && [ "$TOTAL_SECONDS" != "0" ]; then
        TOTAL_HOURS=$(echo "scale=1; $TOTAL_SECONDS / 3600" | bc 2>/dev/null || echo "N/A")
    else
        TOTAL_HOURS="N/A"
    fi
else
    TOTAL_SECONDS="N/A"
    TOTAL_HOURS="N/A"
fi
echo "  Estimated time: ${TOTAL_HOURS} hours"
echo ""

# Save benchmark configuration
echo "num_workers=$BEST_WORKERS" > "${SCRIPT_DIR}/.benchmark_config"
echo "speed=${BEST_SPEED}" >> "${SCRIPT_DIR}/.benchmark_config"
echo "estimated_hours=${TOTAL_HOURS}" >> "${SCRIPT_DIR}/.benchmark_config"

echo "Benchmark configuration saved to: ${SCRIPT_DIR}/.benchmark_config"
echo ""
