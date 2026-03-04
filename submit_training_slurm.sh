#!/bin/bash
#
# Example SLURM job submission script for training SuiT with GeoLexels
# Adjust parameters based on your cluster configuration
#
# Usage:
#   sbatch submit_training_slurm.sh
#

#SBATCH --job-name=suit_geolexels
#SBATCH --partition=gpu              # Use GPU partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4            # Request 4 GPUs
#SBATCH --cpus-per-task=16           # CPUs per GPU
#SBATCH --mem=256G                   # Memory per node
#SBATCH --time=24:00:00              # Maximum 24 hours
#SBATCH --output=logs/train_%j.log   # Log filename
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

# Set up environment
module load cuda/11.8
module load cudnn/8.6
module load pytorch

# Activate Python environment
source /path/to/venv/bin/activate  # Adjust path to your Python environment

# Set working directory
cd /path/to/GeoLexels/transformers/SuiT  # Adjust path

# Set PyTorch to use all available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting SuiT training with GeoLexels..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Training parameters
MODEL="suit_tiny_224"
BATCH_SIZE=32         # Total batch size across all GPUs
EPOCHS=100
LR=0.001
N_SPIX=196
DOWNSAMPLE=2

# Run training with distributed data parallel
python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --data-path /path/to/datasets/SUNRGBD \
    --data-set SUNRGBD \
    --geolexels-cache-dir /path/to/datasets/SUNRGBD/.geolexels_cache \
    --n-spix-segments "$N_SPIX" \
    --downsample "$DOWNSAMPLE" \
    --num-workers 4 \
    --seed 0 \
    --input-size 224 \
    --output-dir ./outputs/suit_${SLURM_JOB_ID}

echo "Training complete!"
echo "Date: $(date)"
