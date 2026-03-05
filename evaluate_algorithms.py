#!/usr/bin/env python3
"""Evaluate multiple superpixel algorithms on SUNRGBD dataset.

Computes unified metrics for different algorithms:
- SLIC
- DASP  
- VCCS
- GeoLexels
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    from skimage.color import rgb2lab
except Exception:
    rgb2lab = None

from algorithms import (
    GeoLexelsAlgorithm,
    SuperpixelAlgorithm,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GEOLEXELS_DIR = PROJECT_ROOT / "GeoLexels"

if str(GEOLEXELS_DIR) not in sys.path:
    sys.path.insert(0, str(GEOLEXELS_DIR))

# Import fast_cloud runner
try:
    from evaluate_sunrgbd_geolexels_metrics import (
        run_fast_cloud,
        load_fast_cloud_binary,
        load_label_map,
        resize_label_map,
        discover_samples,
        compute_metrics,
        Sample,
    )
except ImportError as e:
    print(f"Error importing from evaluate_sunrgbd_geolexels_metrics: {e}")
    sys.exit(1)

LOGGER = logging.getLogger("superpixel_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare superpixel algorithms on SUNRGBD")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "SUNRGBD",
        help="Root directory of SUNRGBD dataset",
    )
    parser.add_argument(
        "--fast-cloud-exe",
        type=Path,
        default=PROJECT_ROOT / "pointcloud" / "build" / "fast_cloud",
        help="Path to fast_cloud executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <dataset-root>/.superpixel_comparison)",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["slic", "dasp", "vccs", "geolexels"],
        choices=["slic", "dasp", "vccs", "geolexels"],
        help="Algorithms to evaluate",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="Start index into samples")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images")
    parser.add_argument(
        "--ue-threshold",
        type=float,
        default=0.05,
        help="UE threshold (fraction of superpixel size)",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Boundary tolerance in pixels",
    )
    parser.add_argument(
        "--sensor-max-depth",
        type=float,
        default=10.0,
        help="Sensor max depth",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout for fast_cloud",
    )
    parser.add_argument(
        "--keep-binaries",
        action="store_true",
        help="Keep intermediate binary files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def create_algorithm(algo_name: str, sensor_max_depth: float) -> SuperpixelAlgorithm:
    """Create algorithm instance by name."""
    if algo_name == "geolexels":
        return GeoLexelsAlgorithm(
            mode=3,
            threshold=0.25,
            focal_length=1.0,
            weight_depth=0.45,
            weight_normals=0.1,
            sensor_max_depth=sensor_max_depth,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_evaluation(
    algorithms: list[str],
    dataset_root: Path,
    fast_cloud_exe: Path,
    output_dir: Path,
    start_idx: int,
    max_images: int | None,
    ue_threshold: float,
    boundary_tolerance: int,
    sensor_max_depth: float,
    timeout_seconds: int,
    keep_binaries: bool,
) -> int:
    """Run evaluation for all specified algorithms."""
    
    if not dataset_root.exists():
        LOGGER.error("Dataset root not found: %s", dataset_root)
        return 1
    if not fast_cloud_exe.exists():
        LOGGER.error("fast_cloud executable not found: %s", fast_cloud_exe)
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_bin_dir = output_dir / "tmp_fast_cloud"
    temp_bin_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info("Algorithms to evaluate: %s", ", ".join(algorithms))
    
    # Discover and select samples
    samples = discover_samples(dataset_root)
    if not samples:
        LOGGER.error("No valid samples found")
        return 1
    
    selected = samples[start_idx:]
    if max_images is not None:
        selected = selected[:max_images]
    
    LOGGER.info("Evaluating %d samples", len(selected))
    
    # Initialize algorithm instances
    algo_instances = {name: create_algorithm(name, sensor_max_depth) for name in algorithms}
    
    # Results per algorithm
    all_results = {algo: [] for algo in algorithms}
    all_failures = {algo: [] for algo in algorithms}
    
    start_time = time.perf_counter()
    
    for idx, sample in enumerate(selected, start=1):
        sample_id = hashlib.sha1(str(sample.rgb_path).encode("utf-8")).hexdigest()[:12]
        temp_bin = temp_bin_dir / f"{sample.frame_stem}_{sample_id}.bin"
        
        LOGGER.info("[%d/%d] %s", idx, len(selected), sample.rgb_path)
        
        # Run fast_cloud once and reuse for all algorithms
        try:
            with Image.open(sample.rgb_path) as rgb_im:
                rgb_w, rgb_h = rgb_im.size
            
            # Load RGB-D images
            rgb_image = np.array(Image.open(sample.rgb_path))
            depth_image = np.array(Image.open(sample.depth_path), dtype=np.float32) / 255.0
            
            t0 = time.perf_counter()
            fast_cloud_result = run_fast_cloud(
                fast_cloud_exe=fast_cloud_exe,
                rgb_path=sample.rgb_path,
                depth_path=sample.depth_path,
                output_bin=temp_bin,
                timeout_seconds=timeout_seconds,
            )
            fast_cloud_seconds = time.perf_counter() - t0
            
            if fast_cloud_result.returncode != 0:
                raise RuntimeError(f"fast_cloud failed: {fast_cloud_result.stderr}")
            
            if not temp_bin.exists():
                raise FileNotFoundError("fast_cloud did not produce output")
            
            # Load fast_cloud binary
            binary_data = load_fast_cloud_binary(temp_bin, rgb_w, rgb_h)
            
            # Load ground truth
            gt_labels = load_label_map(sample.gt_path)
            gt_labels = resize_label_map(gt_labels, rgb_h, rgb_w)
            
            # Evaluate each algorithm
            for algo_name, algo in algo_instances.items():
                try:
                    if algo_name == "geolexels":
                        # GeoLexels needs binary data
                        result = algo.segment_from_binary(binary_data, rgb_w, rgb_h)
                    else:
                        # Others use RGB and depth directly
                        result = algo.segment(rgb_image, depth_image)
                    
                    # Resize labels if needed
                    labels = result.labels
                    if labels.shape != (rgb_h, rgb_w):
                        from scipy import ndimage
                        labels = ndimage.zoom(labels.astype(float), 
                                             (rgb_h / labels.shape[0], 
                                              rgb_w / labels.shape[1]), 
                                             order=0).astype(np.uint32)
                    
                    # Compute metrics
                    metric_values = compute_metrics(
                        labels=labels,
                        binary_data=binary_data,
                        gt_labels=gt_labels,
                        ue_threshold=ue_threshold,
                        boundary_tolerance=boundary_tolerance,
                    )
                    
                    row = {
                        "index": idx,
                        "algorithm": algo_name,
                        "rgb_path": str(sample.rgb_path),
                        "gt_path": str(sample.gt_path),
                        "scene_rel": str(sample.scene_rel),
                        "frame_stem": sample.frame_stem,
                        "width": rgb_w,
                        "height": rgb_h,
                        "num_superpixels": int(result.num_labels),
                        "nce": metric_values["nce"],
                        "chv": metric_values["chv"],
                        "ue": metric_values["ue"],
                        "boundary_recall": metric_values["boundary_recall"],
                        "boundary_precision": metric_values["boundary_precision"],
                        "f_measure": metric_values["f_measure"],
                        "runtime_seconds": result.runtime_seconds,
                    }
                    all_results[algo_name].append(row)
                    
                    LOGGER.info(
                        "  %s: nce=%.6f chv=%.6f ue=%.6f br=%.6f bp=%.6f f=%.6f (%d superpixels)",
                        algo_name,
                        metric_values["nce"],
                        metric_values["chv"],
                        metric_values["ue"],
                        metric_values["boundary_recall"],
                        metric_values["boundary_precision"],
                        metric_values["f_measure"],
                        int(result.num_labels),
                    )
                
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    all_failures[algo_name].append({"rgb_path": str(sample.rgb_path), "error": message})
                    LOGGER.error("  %s failed: %s", algo_name, message)
        
        finally:
            if temp_bin.exists() and not keep_binaries:
                try:
                    temp_bin.unlink()
                except OSError:
                    pass
    
    elapsed = time.perf_counter() - start_time
    
    # Write results for each algorithm
    write_evaluation_results(
        all_results, all_failures, output_dir, elapsed
    )
    
    return 0


def write_evaluation_results(
    all_results: dict,
    all_failures: dict,
    output_dir: Path,
    elapsed: float,
) -> None:
    """Write evaluation results to CSV and JSON files."""
    
    # Common fieldnames
    fieldnames = [
        "index",
        "algorithm",
        "rgb_path",
        "gt_path",
        "scene_rel",
        "frame_stem",
        "width",
        "height",
        "num_superpixels",
        "nce",
        "chv",
        "ue",
        "boundary_recall",
        "boundary_precision",
        "f_measure",
        "runtime_seconds",
    ]
    
    # Write combined CSV with all algorithms
    combined_csv = output_dir / "comparison_results.csv"
    all_rows = []
    for rows in all_results.values():
        all_rows.extend(rows)
    
    with combined_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    
    LOGGER.info("Wrote combined results CSV: %s", combined_csv)
    
    # Write per-algorithm CSVs and summaries
    for algo_name, rows in all_results.items():
        algo_csv = output_dir / f"{algo_name}_metrics.csv"
        algo_summary = output_dir / f"{algo_name}_summary.json"
        algo_failures = output_dir / f"{algo_name}_failures.json"
        
        # CSV
        with algo_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        # Summary JSON
        def metric_mean(metric_name: str) -> float:
            values = [float(r[metric_name]) for r in rows if not np.isnan(float(r[metric_name]))]
            return float(np.nanmean(values)) if values else float("nan")
        
        def metric_std(metric_name: str) -> float:
            values = [float(r[metric_name]) for r in rows if not np.isnan(float(r[metric_name]))]
            return float(np.nanstd(values)) if values else float("nan")
        
        summary = {
            "algorithm": algo_name,
            "counts": {
                "images_evaluated": len(rows),
                "images_failed": len(all_failures[algo_name]),
            },
            "means": {
                "nce": metric_mean("nce"),
                "chv": metric_mean("chv"),
                "ue": metric_mean("ue"),
                "boundary_recall": metric_mean("boundary_recall"),
                "boundary_precision": metric_mean("boundary_precision"),
                "f_measure": metric_mean("f_measure"),
                "num_superpixels": metric_mean("num_superpixels"),
                "runtime_seconds": metric_mean("runtime_seconds"),
            },
            "stds": {
                "nce": metric_std("nce"),
                "chv": metric_std("chv"),
                "ue": metric_std("ue"),
                "boundary_recall": metric_std("boundary_recall"),
                "boundary_precision": metric_std("boundary_precision"),
                "f_measure": metric_std("f_measure"),
                "num_superpixels": metric_std("num_superpixels"),
                "runtime_seconds": metric_std("runtime_seconds"),
            },
        }
        
        with algo_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        # Failures
        with algo_failures.open("w", encoding="utf-8") as f:
            json.dump(all_failures[algo_name], f, indent=2)
        
        LOGGER.info("Wrote %s results: CSV=%s, Summary=%s", algo_name, algo_csv, algo_summary)
    
    # Write master summary comparing all algorithms
    master_summary = output_dir / "comparison_summary.json"
    master = {
        "elapsed_seconds": elapsed,
        "algorithms": {},
    }
    
    for algo_name, rows in all_results.items():
        def metric_mean(metric_name: str) -> float:
            values = [float(r[metric_name]) for r in rows if not np.isnan(float(r[metric_name]))]
            return float(np.nanmean(values)) if values else float("nan")
        
        master["algorithms"][algo_name] = {
            "num_evaluated": len(rows),
            "num_failed": len(all_failures[algo_name]),
            "metrics": {
                "nce_mean": metric_mean("nce"),
                "chv_mean": metric_mean("chv"),
                "ue_mean": metric_mean("ue"),
                "br_mean": metric_mean("boundary_recall"),
                "bp_mean": metric_mean("boundary_precision"),
                "f_mean": metric_mean("f_measure"),
                "runtime_mean_sec": metric_mean("runtime_seconds"),
                "nsuperpixels_mean": metric_mean("num_superpixels"),
            },
        }
    
    with master_summary.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    
    LOGGER.info("Wrote comparison summary: %s", master_summary)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    
    if args.output_dir is None:
        args.output_dir = args.dataset_root / ".superpixel_comparison"
    
    return run_evaluation(
        algorithms=args.algorithms,
        dataset_root=args.dataset_root,
        fast_cloud_exe=args.fast_cloud_exe,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        max_images=args.max_images,
        ue_threshold=args.ue_threshold,
        boundary_tolerance=args.boundary_tolerance,
        sensor_max_depth=args.sensor_max_depth,
        timeout_seconds=args.timeout_seconds,
        keep_binaries=args.keep_binaries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
