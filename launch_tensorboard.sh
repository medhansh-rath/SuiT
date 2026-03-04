#!/bin/bash
# Launch TensorBoard to monitor training progress

# Default trial name
TRIAL_NAME="${1:-}"
PORT="${2:-6006}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# If no trial name provided, show all trials
if [ -z "$TRIAL_NAME" ]; then
    LOGDIR="$LOG_DIR"
    if [ ! -d "$LOGDIR" ]; then
        echo "Error: No logs directory found at: $LOGDIR"
        echo "Have you started training yet?"
        exit 1
    fi
    
    echo "Available trials:"
    ls -1 "$LOG_DIR" 2>/dev/null | while read trial; do
        echo "  - $trial"
    done
    echo ""
    echo "Launching TensorBoard for all trials"
else
    LOGDIR="${LOG_DIR}/${TRIAL_NAME}"
    if [ ! -d "$LOGDIR" ]; then
        echo "Error: Trial directory not found: $LOGDIR"
        echo ""
        echo "Available trials:"
        ls -1 "$LOG_DIR" 2>/dev/null | while read trial; do
            echo "  - $trial"
        done
        exit 1
    fi
    echo "Launching TensorBoard for trial: $TRIAL_NAME"
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate geolexels

echo "Log directory: $LOGDIR"
echo "TensorBoard will be available at: http://localhost:${PORT}"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all
