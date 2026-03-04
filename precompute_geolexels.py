#!/usr/bin/env python3
"""
Batch process SUNRGBD dataset to precompute GeoLexels superpixels.
This script finds all RGB/Depth pairs and runs fast_cloud to generate GeoLexels.

Usage:
    python precompute_geolexels.py --dataset-root /path/to/SUNRGBD --output-dir /path/to/cache --batch-size 8
    
    Or use the provided bash script:
    bash run_geolexels_preprocessing.sh
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_rgb_depth_pairs(dataset_root):
    """
    Find all RGB and Depth image pairs in SUNRGBD dataset.
    Returns list of tuples: (rgb_path, depth_path, output_path)
    """
    pairs = []
    dataset_path = Path(dataset_root)
    
    # Search for all jpg files (RGB images)
    for rgb_path in dataset_path.rglob('image/*.jpg'):
        # Find corresponding depth image
        scene_dir = rgb_path.parent.parent
        frame_name = rgb_path.stem
        depth_path = scene_dir / 'depth' / f'{frame_name}.png'
        
        if depth_path.exists():
            # Create output directory structure
            relative_path = scene_dir.relative_to(dataset_path)
            output_dir = Path(dataset_root) / '.geolexels_cache' / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f'{frame_name}.npy'
            pairs.append((str(rgb_path), str(depth_path), str(output_file)))
    
    return pairs


def process_geolexels(rgb_path, depth_path, output_path, fast_cloud_exe, temp_dir='/tmp'):
    """
    Process a single RGB/Depth pair using fast_cloud to generate GeoLexels.
    Saves the result as a .npy file.
    """
    try:
        # Create temporary binary file
        temp_bin = os.path.join(temp_dir, f'temp_geolexels_{os.getpid()}.bin')
        
        # Run fast_cloud
        cmd = [
            fast_cloud_exe,
            rgb_path,
            depth_path,
            "-n",  # Compute normals
            "-G",  # True Guided Filter
            "-B",  # Save binary
            "--output", temp_bin
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.warning(f"fast_cloud failed for {rgb_path}: {result.stderr}")
            return False
        
        # Read the binary file
        if not os.path.exists(temp_bin):
            logger.warning(f"Output file not created for {rgb_path}")
            return False
        
        data = np.fromfile(temp_bin, dtype=np.float32)
        
        # Infer dimensions from output message or try common resolutions
        height, width = 480, 640  # Default
        
        # Try to parse dimensions from stdout
        for line in result.stdout.split('\n'):
            if 'Saved' in line and 'x 7 channels' in line:
                try:
                    parts = line.split('(')[1].split('x')
                    width = int(parts[0].strip())
                    height = int(parts[1].strip())
                    break
                except:
                    pass
        
        # Reshape to (H, W, 7)
        total_pixels = width * height
        if data.size == total_pixels * 7:
            data = data.reshape(height, width, 7)
            
            # Save as numpy file
            np.save(output_path, data)
            
            # Clean up temp file
            if os.path.exists(temp_bin):
                os.remove(temp_bin)
            
            return True
        else:
            logger.warning(f"Unexpected data size for {rgb_path}: got {data.size}, expected {total_pixels * 7}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout processing {rgb_path}")
        return False
    except Exception as e:
        logger.error(f"Error processing {rgb_path}: {e}")
        return False


# Global variable to store fast_cloud_exe path for multiprocessing
_FAST_CLOUD_EXE = None
_TEMP_DIR = '/tmp'

def _process_wrapper(args_tuple):
    """Wrapper function for multiprocessing (must be module-level for pickling)"""
    idx, rgb, depth, output = args_tuple
    result = process_geolexels(rgb, depth, output, _FAST_CLOUD_EXE, _TEMP_DIR)
    return (idx, rgb, result)


def main():
    parser = argparse.ArgumentParser(description='Precompute GeoLexels for SUNRGBD dataset')
    
    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    
    parser.add_argument('--dataset-root', type=str, default=str(project_root / 'datasets' / 'SUNRGBD'),
                       help='Root directory of SUNRGBD dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for cache (default: <dataset-root>/.geolexels_cache)')
    parser.add_argument('--fast-cloud-exe', type=str, default=str(project_root / 'pointcloud' / 'build' / 'fast_cloud'),
                       help='Path to fast_cloud executable')
    parser.add_argument('--temp-dir', type=str, default='/tmp',
                       help='Temporary directory for intermediate files')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Start processing from this index (useful for resuming)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Process only first N images (useful for testing)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have cached results')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers for processing (default: 4). Use higher values for better GPU utilization.')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    fast_cloud_exe = os.path.abspath(args.fast_cloud_exe)
    dataset_root = os.path.abspath(args.dataset_root)
    
    if not os.path.exists(fast_cloud_exe):
        logger.error(f"fast_cloud executable not found: {fast_cloud_exe}")
        sys.exit(1)
    
    if not os.path.exists(dataset_root):
        logger.error(f"Dataset root not found: {dataset_root}")
        sys.exit(1)
    
    # Log configuration
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"fast_cloud executable: {fast_cloud_exe}")
    logger.info(f"Temp directory: {args.temp_dir}")
    logger.info(f"Parallel workers: {args.num_workers}")
    
    # Find all RGB/Depth pairs
    logger.info("Finding RGB/Depth pairs...")
    pairs = find_rgb_depth_pairs(dataset_root)
    logger.info(f"Found {len(pairs)} RGB/Depth pairs")
    
    if args.max_images:
        pairs = pairs[:args.max_images]
        logger.info(f"Processing first {len(pairs)} pairs")
    
    # Filter out already processed pairs if requested
    pairs_to_process = []
    skipped = 0
    for i, (rgb_path, depth_path, output_path) in enumerate(pairs):
        if i < args.start_idx:
            continue
        if args.skip_existing and os.path.exists(output_path):
            skipped += 1
            continue
        pairs_to_process.append((i, rgb_path, depth_path, output_path))
    
    # Process pairs in parallel
    processed = 0
    failed = 0
    
    if len(pairs_to_process) > 0:
        # Set global variables for multiprocessing wrapper function
        global _FAST_CLOUD_EXE, _TEMP_DIR
        _FAST_CLOUD_EXE = fast_cloud_exe
        _TEMP_DIR = args.temp_dir
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_wrapper, item): item for item in pairs_to_process}
            
            # Process completed tasks as they finish
            for future in as_completed(futures):
                item = futures[future]
                idx, rgb, depth, output = item
                try:
                    idx_ret, rgb_ret, result = future.result()
                    if result:
                        processed += 1
                    else:
                        failed += 1
                    # Log progress
                    if (processed + failed) % max(1, len(pairs_to_process) // 10) == 0:
                        logger.info(f"Progress: {processed + failed}/{len(pairs_to_process)} processed ({processed} ok, {failed} failed)")
                except Exception as e:
                    failed += 1
                    logger.error(f"Task failed for {rgb}: {e}")
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Skipped:   {skipped}")
    logger.info(f"  Failed:    {failed}")
    logger.info(f"  Total:     {len(pairs)}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
