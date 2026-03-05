#!/bin/bash
#
# Run superpixel algorithm comparison on SUNRGBD dataset.
#
# Usage:
#   bash compare_algorithms.sh [--max-images N] [--algorithms algo1 algo2 ...]
#
# Examples:
#   bash compare_algorithms.sh --max-images 50
#   bash compare_algorithms.sh --max-images 50 --algorithms geolexels
#

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

DATASET_ROOT="${PROJECT_ROOT}/datasets/SUNRGBD"
FAST_CLOUD_EXE="${PROJECT_ROOT}/pointcloud/build/fast_cloud"
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate_algorithms.py"
DEFAULT_OUTPUT_DIR="${DATASET_ROOT}/.superpixel_comparison"

# Find Python
ACTIVE_VENV_PYTHON=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    ACTIVE_VENV_PYTHON="${VIRTUAL_ENV}/bin/python"
fi

if [ -n "$ACTIVE_VENV_PYTHON" ]; then
    PYTHON_BIN="$ACTIVE_VENV_PYTHON"
elif [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif [ -x "${PROJECT_ROOT}/GeoLexels/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/GeoLexels/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: No Python interpreter found."
    exit 1
fi

# Validate
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Evaluator script not found at: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: SUNRGBD dataset root not found at: $DATASET_ROOT"
    exit 1
fi

if [ ! -f "$FAST_CLOUD_EXE" ]; then
    echo "ERROR: fast_cloud executable not found at: $FAST_CLOUD_EXE"
    echo "Build command: cd $PROJECT_ROOT/pointcloud/build && cmake .. && make"
    exit 1
fi

# Ensure GeoLexels is compiled
if ! ls "${PROJECT_ROOT}/GeoLexels"/_geolexels*.so >/dev/null 2>&1; then
    echo "GeoLexels shared library not found, compiling..."
    (
        cd "${PROJECT_ROOT}/GeoLexels"
        "$PYTHON_BIN" compile_geolexels_lib.py
    )
fi

# Create log directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/comparison_${TIMESTAMP}.log"

# Build command
CMD=("$PYTHON_BIN" "$PYTHON_SCRIPT")
CMD+=("--dataset-root" "$DATASET_ROOT")
CMD+=("--fast-cloud-exe" "$FAST_CLOUD_EXE")
CMD+=("--output-dir" "$DEFAULT_OUTPUT_DIR")

# Default to a small subset for testing
if ! echo "$@" | grep -q "\-\-max-images"; then
    CMD+=("--max-images" "10")
fi

# Algorithms
if ! echo "$@" | grep -q "\-\-algorithms"; then
    CMD+=("--algorithms" "geolexels")
fi

# Add user args
if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "Running: ${CMD[@]}"
echo "Output dir: $DEFAULT_OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
RETURN_CODE=${PIPESTATUS[0]}

echo ""
echo "Evaluation completed with exit code: $RETURN_CODE"
echo "Results saved to: $DEFAULT_OUTPUT_DIR"

exit $RETURN_CODE
