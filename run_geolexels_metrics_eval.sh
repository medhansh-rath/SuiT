#!/bin/bash
#
# Evaluate SUNRGBD with fast_cloud + GeoLexels and compute superpixel metrics.
#
# Usage:
#   bash run_geolexels_metrics_eval.sh [evaluator options]
#
# Examples:
#   bash run_geolexels_metrics_eval.sh --max-images 100
#   bash run_geolexels_metrics_eval.sh --start-idx 500 --max-images 200 --verbose
#   bash run_geolexels_metrics_eval.sh --output-dir /tmp/geolexels_eval --ue-threshold 0.1
#

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

DATASET_ROOT="${PROJECT_ROOT}/datasets/SUNRGBD"
FAST_CLOUD_EXE="${PROJECT_ROOT}/pointcloud/build/fast_cloud"
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate_sunrgbd_geolexels_metrics.py"
DEFAULT_OUTPUT_DIR="${DATASET_ROOT}/.geolexels_eval"

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

has_flag() {
    local sought="$1"
    shift
    for arg in "$@"; do
        if [ "$arg" = "$sought" ]; then
            return 0
        fi
    done
    return 1
}

# Validate required files/directories.
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

# Ensure GeoLexels CFFI module exists.
if ! ls "${PROJECT_ROOT}/GeoLexels"/_geolexels*.so >/dev/null 2>&1; then
    echo "GeoLexels shared library not found, compiling..."
    (
        cd "${PROJECT_ROOT}/GeoLexels"
        "$PYTHON_BIN" compile_geolexels_lib.py
    )
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/metrics_eval_${TIMESTAMP}.log"

CMD=("$PYTHON_BIN" "$PYTHON_SCRIPT")

if ! has_flag "--dataset-root" "$@"; then
    CMD+=("--dataset-root" "$DATASET_ROOT")
fi

if ! has_flag "--fast-cloud-exe" "$@"; then
    CMD+=("--fast-cloud-exe" "$FAST_CLOUD_EXE")
fi

if ! has_flag "--output-dir" "$@"; then
    CMD+=("--output-dir" "$DEFAULT_OUTPUT_DIR")
fi

# Forward user-provided evaluator options.
if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "=========================================="
echo "GeoLexels Metrics Evaluation (SUNRGBD)"
echo "=========================================="
echo "Python:            $PYTHON_BIN"
echo "Dataset root:      $DATASET_ROOT"
echo "fast_cloud:        $FAST_CLOUD_EXE"
echo "Default output:    $DEFAULT_OUTPUT_DIR"
echo "Log file:          $LOG_FILE"
echo "------------------------------------------"
echo "GeoLexels settings used by evaluator:"
echo "  mode=3, threshold=0.25, focal_length=1.0"
echo "  weight_depth=0.45, weight_normals=0.1, color~0.45"
echo "  depth_norm=sensor_max, color/depth=Laplace, normals=vMF"
echo "  convert_to_cielab=True"
echo "=========================================="
echo ""

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Evaluation complete"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
