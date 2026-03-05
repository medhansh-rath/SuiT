#!/usr/bin/env python3
"""Run all superpixel algorithms on test images and save segmentation results."""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from algorithms import GeoLexelsAlgorithm

# For GeoLexels, we need fast_cloud
PROJECT_ROOT = Path(__file__).parent.parent.parent
FAST_CLOUD_EXE = PROJECT_ROOT / "pointcloud" / "build" / "fast_cloud"

sys.path.insert(0, str(PROJECT_ROOT / "GeoLexels"))

try:
    from evaluate_sunrgbd_geolexels_metrics import run_fast_cloud, load_fast_cloud_binary
except ImportError:
    print("Warning: Could not import fast_cloud utilities")


def generate_segmentation_image(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Convert label map to RGB image for visualization."""
    h, w = labels.shape
    
    # Generate random colors for each label
    np.random.seed(42)
    colors = np.random.randint(0, 256, (num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Label 0 = black
    
    # Map each label to its color
    rgb = colors[labels]
    
    return rgb


def run_all_algorithms(rgb_path: Path, depth_path: Path, output_dir: Path) -> dict:
    """Run all available algorithms and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading images...")
    print(f"  RGB: {rgb_path}")
    print(f"  Depth: {depth_path}")
    
    # Load images
    rgb_image = np.array(Image.open(rgb_path))
    depth_image = np.array(Image.open(depth_path), dtype=np.float32)
    
    # Normalize depth
    if depth_image.max() > 1.0:
        depth_image = depth_image / 255.0
    
    h, w = rgb_image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    results = {}
    
    # GeoLexels (requires fast_cloud)
    print(f"\n{'='*60}")
    print("Running GeoLexels...")
    print(f"{'='*60}")
    try:
        if not FAST_CLOUD_EXE.exists():
            raise FileNotFoundError(f"fast_cloud not found at {FAST_CLOUD_EXE}")
        
        t0 = time.perf_counter()
        
        # Create temporary binary via fast_cloud
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = tmp.name
        
        # Run fast_cloud
        fast_cloud_result = run_fast_cloud(
            fast_cloud_exe=FAST_CLOUD_EXE,
            rgb_path=rgb_path,
            depth_path=depth_path,
            output_bin=Path(tmp_path),
            timeout_seconds=180,
        )
        
        if fast_cloud_result.returncode != 0:
            raise RuntimeError(f"fast_cloud failed: {fast_cloud_result.stderr}")
        
        # Load binary and run GeoLexels
        binary_data = load_fast_cloud_binary(Path(tmp_path), w, h)
        
        geolexels = GeoLexelsAlgorithm(
            mode=3,
            threshold=0.25,
            weight_depth=0.45,
            weight_normals=0.1,
        )
        result = geolexels.segment_from_binary(binary_data, w, h)
        
        elapsed = time.perf_counter() - t0
        
        print(f"✓ GeoLexels completed in {elapsed:.2f}s")
        print(f"  Superpixels: {result.num_labels}")
        
        # Save label map
        geolexels_labels_path = output_dir / "geolexels_labels.npy"
        np.save(geolexels_labels_path, result.labels)
        
        # Save visualization
        geolexels_vis_path = output_dir / "geolexels_visualization.png"
        geolexels_rgb = generate_segmentation_image(result.labels, result.num_labels)
        Image.fromarray(geolexels_rgb).save(geolexels_vis_path)
        
        results["geolexels"] = {
            "labels_path": str(geolexels_labels_path),
            "visualization_path": str(geolexels_vis_path),
            "num_superpixels": result.num_labels,
            "runtime_seconds": elapsed,
        }
        print(f"  Saved: {geolexels_labels_path}")
        print(f"  Saved: {geolexels_vis_path}")
        
        # Cleanup
        try:
            Path(tmp_path).unlink()
        except:
            pass
    except Exception as e:
        print(f"✗ GeoLexels failed: {e}")
        results["geolexels"] = {"error": str(e)}
    
    # Save input images for reference
    rgb_copy_path = output_dir / "input_rgb.jpg"
    Image.fromarray(rgb_image).save(rgb_copy_path)
    
    depth_uint8 = np.clip(depth_image * 255, 0, 255).astype(np.uint8)
    depth_copy_path = output_dir / "input_depth.png"
    Image.fromarray(depth_uint8).save(depth_copy_path)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Input images saved:")
    print(f"  {rgb_copy_path}")
    print(f"  {depth_copy_path}")
    print(f"\nSegmentation results saved to: {output_dir}")
    
    for algo_name, result in results.items():
        if "error" in result:
            status = f"✗ {result['error']}"
        else:
            status = f"✓ {result['num_superpixels']} superpixels, {result['runtime_seconds']:.2f}s"
        print(f"  {algo_name:12} {status}")
    
    # Save summary as JSON
    import json
    summary_path = output_dir / "results_summary.json"
    
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = {}
    for algo_name, result in results.items():
        if "error" in result:
            results_serializable[algo_name] = result
        else:
            results_serializable[algo_name] = {
                "labels_path": result["labels_path"],
                "visualization_path": result["visualization_path"],
                "num_superpixels": int(result["num_superpixels"]),
                "runtime_seconds": float(result["runtime_seconds"]),
            }
    
    with open(summary_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nSummary: {summary_path}")
    
    return results


if __name__ == "__main__":
    datasets_root = Path("/home/medhansh/GeoLexels/datasets")
    rgb_path = datasets_root / "SUN_color.jpg"
    depth_path = datasets_root / "SUN_depth.png"
    output_dir = datasets_root / "superpixel_results"
    
    if not rgb_path.exists():
        print(f"Error: {rgb_path} not found")
        sys.exit(1)
    
    if not depth_path.exists():
        print(f"Error: {depth_path} not found")
        sys.exit(1)
    
    run_all_algorithms(rgb_path, depth_path, output_dir)
