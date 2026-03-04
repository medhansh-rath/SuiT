#!/bin/bash
#
# Example SLURM job submission script for preprocessing GeoLexels on a cluster
# Adjust parameters based on your cluster configuration
#
# Usage:
#   sbatch submit_preprocessing_slurm.sh
#

#SBATCH --job-name=geolexels_prep
#SBATCH --partition=cpu              # Use CPU partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16           # Adjust based on available cores
#SBATCH --mem=32G                    # Memory per node
#SBATCH --time=72:00:00              # Maximum 3 days (adjust as needed)
#SBATCH --output=logs/slurm_%j.log   # Log filename
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

# Set up environment
module load cuda/11.8              # Load CUDA if needed for fast_cloud
module load cmake
module load gcc

# Activate Python environment
source /path/to/venv/bin/activate  # Adjust path to your Python environment

# Set working directory
cd /path/to/GeoLexels/transformers/SuiT  # Adjust path

# Run preprocessing
echo "Starting GeoLexels preprocessing..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Run with maximum images or without limit
python3 precompute_geolexels.py \
    --dataset-root /path/to/datasets/SUNRGBD \
    --fast-cloud-exe /path/to/pointcloud/build/fast_cloud \
    --skip-existing \
    --temp-dir /scratch/$SLURM_JOB_ID  # Use scratch space if available

echo "Preprocessing complete!"
echo "Date: $(date)"
