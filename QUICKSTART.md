# Quick Start: Running Algorithm Comparison

This guide shows how to run the superpixel algorithm comparison on SUNRGBD.

## Pre-requisites

Ensure all dependencies are installed:
```bash
cd /home/medhansh/GeoLexels
pip install scikit-image scipy matplotlib pandas
```

## Running the Comparison

### Option 1: Quick Test (Recommended first run)

Run on first 10 images to verify setup:
```bash
cd /home/medhansh/GeoLexels/transformers/SuiT
bash compare_algorithms.sh
```

**Expected output:**
- CSV results in `datasets/SUNRGBD/.superpixel_comparison/comparison_results.csv`
- Summary JSON in `datasets/SUNRGBD/.superpixel_comparison/comparison_summary.json`
- Per-algorithm results (e.g., `slic_metrics.csv`, `dasp_summary.json`, etc.)

### Option 2: Full Comparison

Run on larger subset (50-100 images):
```bash
bash compare_algorithms.sh --max-images 100
```

### Option 3: Specific Algorithms Only

Compare only SLIC and VCCS:
```bash
bash compare_algorithms.sh --max-images 50 --algorithms slic vccs
```

### Option 4: Manual Python Execution

For more control:
```bash
python3 evaluate_algorithms.py \
    --max-images 100 \
    --algorithms slic dasp vccs geolexels \
    --output-dir ./my_comparison \
    --ue-threshold 0.05 \
    --verbose
```

## Analyzing Results

View summary and generate plots:
```bash
python3 analyze_comparison.py \
    --results-dir datasets/SUNRGBD/.superpixel_comparison \
    --plot \
    --plot-output metrics_comparison.png
```

## Understanding the Output

### Per-Image Results (CSV)
File: `comparison_results.csv`

Columns:
- **algorithm**: Algorithm name (slic, dasp, vccs, geolexels)
- **rgb_path, gt_path**: File paths
- **num_superpixels**: Actual count of superpixels generated
- **nce**: Normal Consistency Error (lower is better)
- **chv**: Color Homogeneity Variance (lower is better)
- **ue**: Under-segmentation Error (lower is better)
- **boundary_recall**: Fraction of GT boundaries detected (higher is better, 0-1)
- **boundary_precision**: Fraction of detected boundaries correct (higher is better, 0-1)
- **f_measure**: F1 score = 2×BR×BP/(BR+BP) (higher is better, 0-1)
- **runtime_seconds**: Wall-clock time for segmentation

### Summary JSON
File: `comparison_summary.json`

Shows:
- Elapsed total time
- Per-algorithm stats:
  - Mean and standard deviation for all metrics
  - Number of successful/failed evaluations

### Per-Algorithm Results
Files: `{algorithm}_metrics.csv`, `{algorithm}_summary.json`, `{algorithm}_failures.json`

Detailed results for each algorithm separately.

## Typical Results Interpretation

For SUNRGBD with ~100 images:

| Metric | Typical Range | Interpretation |
|--------|---------------|-----------------|
| NCE    | 0.05-0.20     | Lower = better superpixel adherence to surface normals |
| CHV    | 0.01-0.08     | Lower = more uniform color within superpixels |
| UE     | 2-8%          | Lower = fewer missed boundaries = better segmentation |
| BR     | 0.45-0.75     | Higher = detects more true boundaries |
| BP     | 0.50-0.80     | Higher = fewer false boundary detections |
| F      | 0.45-0.75     | Higher = overall better boundary detection |
| Runtime | 0.01-1.0s     | Lower = faster, but may sacrifice quality |

## Troubleshooting

### "fast_cloud executable not found"
Build it first:
```bash
cd /home/medhansh/GeoLexels/pointcloud
mkdir -p build && cd build
cmake ..
make
```

### "GeoLexels module not available"
Compile GeoLexels:
```bash
cd /home/medhansh/GeoLexels/GeoLexels
python3 compile_geolexels_lib.py
```

### "DASP failed during build"
Install dependencies:
```bash
sudo apt-get install cmake build-essential libopencv-dev
```

### "VCCS executable (pcl_vccs_segmentation) not found"
Install PCL:
```bash
sudo apt-get install libpcl-dev
```
Then test:
```bash
which pcl_vccs_segmentation
```

### Memory Issues / Out of Memory
Reduce batch size:
```bash
bash compare_algorithms.sh --max-images 5
```

## Next Steps

1. **Compare on full dataset** - Increase `--max-images` to evaluate all images
2. **Tune parameters** - Modify algorithm settings in `create_algorithm()` function in `evaluate_algorithms.py`
3. **Add new algorithms** - Implement `SuperpixelAlgorithm` base class (see `ALGORITHM_COMPARISON.md`)
4. **Analyze per-scene** - Modify `analyze_comparison.py` to group results by scene
5. **Visualize segmentations** - Write a script to render segmentation maps

## References

- **SLIC**: Achanta et al., "SLIC Superpixels Compared to State-of-the-art Superpixel Methods" (TPAMI 2012)
- **DASP**: Weikersdorfer et al., "Depth-Adaptive Superpixels with Parametric Regularization" (3DV 2015)  
- **VCCS**: Papon et al., "Voxel Cloud Connectivity Segmentation: Supervoxels for Point Clouds" (CVPR 2013)
- **SUNRGBD**: Song et al., "SUN RGB-D: A Large Scale RGB-D Indoor Scenes Understanding Benchmark"
