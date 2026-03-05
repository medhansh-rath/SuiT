#!/usr/bin/env python3
"""Evaluate GeoLexels superpixel quality metrics on SUNRGBD.

This script:
1) Finds RGB/depth/ground-truth triplets in SUNRGBD.
2) Runs fast_cloud with flags: -n -G -B.
3) Runs GeoLexels with the requested settings:
   - mode 3
   - threshold 0.25
   - focal length 1.0
   - depth weight 0.45 (implying color weight 0.45 when normals weight is 0.1)
   - normals weight 0.1
   - depth normalization mode sensor_max
   - Laplace distributions for color and depth
   - von Mises-Fisher for normals
   - RGB->CIELAB conversion enabled
4) Computes per-image metrics:
   - Normal Consistency Error (NCE)
   - Color Homogeneity Variance (CHV)
   - Under-segmentation Error (UE)
   - Boundary Recall (BR)
   - Boundary Precision (BP)
   - F-measure
5) Writes per-image CSV and summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GEOLEXELS_DIR = PROJECT_ROOT / "GeoLexels"

if str(GEOLEXELS_DIR) not in sys.path:
    sys.path.insert(0, str(GEOLEXELS_DIR))

try:
    from GeoLexelsDemo import segment
except Exception as import_error:
    segment = None
    SEGMENT_IMPORT_ERROR = import_error
else:
    SEGMENT_IMPORT_ERROR = None


LOGGER = logging.getLogger("sunrgbd_geolexels_eval")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
GT_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy", ".mat"}
GT_DIR_HINTS = (
    "label",
    "labels",
    "seg",
    "segment",
    "mask",
    "groundtruth",
    "ground_truth",
    "annotation",
    "semantic",
    "gt",
)

GT_FILE_HINTS = (
    "seg",
    "label",
    "mask",
    "groundtruth",
    "ground_truth",
    "semantic",
    "gt",
)

FAST_CLOUD_DIM_RE = re.compile(r"\((\d+)\s*x\s*(\d+)\s*x\s*7\s+channels", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    rgb_path: Path
    depth_path: Path
    gt_path: Path
    scene_rel: Path
    frame_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SUNRGBD GeoLexels superpixel metrics")
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
        help="Output directory (default: <dataset-root>/.geolexels_eval)",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="Start index into discovered samples")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to evaluate")
    parser.add_argument(
        "--ue-threshold",
        type=float,
        default=0.05,
        help="UE threshold B, interpreted as fraction of superpixel size |sj|",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Boundary tolerance t in pixels for BR/BP",
    )
    parser.add_argument(
        "--sensor-max-depth",
        type=float,
        default=10.0,
        help="Sensor max depth used with depth normalization mode sensor_max",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout for each fast_cloud invocation",
    )
    parser.add_argument(
        "--keep-binaries",
        action="store_true",
        help="Keep intermediate fast_cloud binary files",
    )
    parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Save GeoLexels label maps as .npy under output_dir/labels",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first failed sample",
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


def find_file_for_stem(directory: Path, stem: str, allowed_exts: set[str]) -> Path | None:
    if not directory.exists() or not directory.is_dir():
        return None

    for ext in allowed_exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    exact = [p for p in files if p.stem == stem]
    if exact:
        return sorted(exact)[0]

    fuzzy = [p for p in files if p.stem.startswith(stem) or stem.startswith(p.stem)]
    if fuzzy:
        return sorted(fuzzy)[0]

    return None


def find_depth_for_frame(scene_dir: Path, frame_stem: str) -> Path | None:
    depth_dirs: list[Path] = []
    canonical = scene_dir / "depth"
    if canonical.exists():
        depth_dirs.append(canonical)

    for child in scene_dir.iterdir():
        if child.is_dir() and "depth" in child.name.lower() and child not in depth_dirs:
            depth_dirs.append(child)

    for depth_dir in depth_dirs:
        depth_path = find_file_for_stem(depth_dir, frame_stem, DEPTH_EXTS)
        if depth_path is not None:
            return depth_path

    return None


def get_gt_dirs(scene_dir: Path) -> list[Path]:
    gt_dirs = []
    for child in scene_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if any(token in name for token in GT_DIR_HINTS):
            gt_dirs.append(child)
    return sorted(gt_dirs)


def find_gt_for_frame(scene_dir: Path, frame_stem: str, gt_dir_cache: dict[Path, list[Path]]) -> Path | None:
    # Common SUNRGBD scene-level GT files (e.g., seg.mat)
    for scene_gt_name in ("seg.mat", "seg.npy", "label.mat", "labels.mat"):
        scene_gt = scene_dir / scene_gt_name
        if scene_gt.exists() and scene_gt.is_file():
            return scene_gt

    # Try frame-stem matching among scene-level files (excluding image/depth files)
    scene_files = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in GT_EXTS]
    for file_path in sorted(scene_files):
        stem_lower = file_path.stem.lower()
        if stem_lower == frame_stem.lower() or stem_lower.startswith(frame_stem.lower()):
            return file_path
        if any(token in stem_lower for token in GT_FILE_HINTS):
            return file_path

    if scene_dir not in gt_dir_cache:
        gt_dir_cache[scene_dir] = get_gt_dirs(scene_dir)

    for gt_dir in gt_dir_cache[scene_dir]:
        gt_path = find_file_for_stem(gt_dir, frame_stem, GT_EXTS)
        if gt_path is not None:
            return gt_path

    return None


def discover_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    gt_dir_cache: dict[Path, list[Path]] = {}
    missing_depth = 0
    missing_gt = 0

    rgb_candidates = sorted(
        p for p in dataset_root.rglob("image/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    for rgb_path in rgb_candidates:
        scene_dir = rgb_path.parent.parent
        frame_stem = rgb_path.stem

        depth_path = find_depth_for_frame(scene_dir, frame_stem)
        if depth_path is None:
            missing_depth += 1
            continue

        gt_path = find_gt_for_frame(scene_dir, frame_stem, gt_dir_cache)
        if gt_path is None:
            missing_gt += 1
            continue

        try:
            scene_rel = scene_dir.relative_to(dataset_root)
        except ValueError:
            scene_rel = scene_dir

        samples.append(
            Sample(
                rgb_path=rgb_path,
                depth_path=depth_path,
                gt_path=gt_path,
                scene_rel=scene_rel,
                frame_stem=frame_stem,
            )
        )

    LOGGER.info("Discovered %d RGB files under image/", len(rgb_candidates))
    LOGGER.info("Matched %d complete RGB/depth/GT samples", len(samples))
    if missing_depth:
        LOGGER.info("Skipped %d samples without matching depth file", missing_depth)
    if missing_gt:
        LOGGER.info("Skipped %d samples without matching GT file", missing_gt)

    return samples


def run_fast_cloud(
    fast_cloud_exe: Path,
    rgb_path: Path,
    depth_path: Path,
    output_bin: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(fast_cloud_exe),
        str(rgb_path),
        str(depth_path),
        "-n",
        "-G",
        "-B",
        "--output",
        str(output_bin),
    ]
    LOGGER.debug("Running fast_cloud: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)


def parse_dims_from_fast_cloud_output(text: str) -> tuple[int, int] | None:
    match = FAST_CLOUD_DIM_RE.search(text)
    if match is None:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    return width, height


def load_fast_cloud_binary(bin_path: Path, width: int, height: int) -> np.ndarray:
    data = np.fromfile(bin_path, dtype=np.float32)
    expected = width * height * 7
    if data.size != expected:
        raise ValueError(f"Unexpected binary size {data.size}, expected {expected} for {width}x{height}x7")
    return data.reshape(height, width, 7)


def load_label_map(label_path: Path) -> np.ndarray:
    suffix = label_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(label_path)
    elif suffix == ".mat":
        if loadmat is None:
            raise ImportError("scipy is required to read .mat ground-truth files")
        mat = loadmat(label_path)
        preferred_keys = ("seglabel", "label", "labels", "groundtruth", "gt", "seg")
        arr = None
        for key in preferred_keys:
            if key in mat and isinstance(mat[key], np.ndarray) and mat[key].ndim >= 2:
                arr = mat[key]
                break
        if arr is None:
            for value in mat.values():
                if isinstance(value, np.ndarray) and value.ndim >= 2 and value.size > 0:
                    arr = value
                    break
        if arr is None:
            raise ValueError(f"No 2D array found in MAT file: {label_path}")
    else:
        with Image.open(label_path) as im:
            arr = np.array(im)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = arr[:, :, :3].astype(np.uint32)
            packed = (arr[:, :, 0] << 16) | (arr[:, :, 1] << 8) | arr[:, :, 2]
            _, inverse = np.unique(packed.reshape(-1), return_inverse=True)
            arr = inverse.reshape(packed.shape)

    if arr.ndim != 2:
        raise ValueError(f"Ground-truth must be 2D labels after decoding, got shape {arr.shape} from {label_path}")

    return arr.astype(np.int32, copy=False)


def resize_label_map(label_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if label_map.shape == (target_h, target_w):
        return label_map
    pil = Image.fromarray(label_map.astype(np.int32), mode="I")
    resized = pil.resize((target_w, target_h), resample=Image.NEAREST)
    return np.array(resized, dtype=np.int32)


def build_boundary_map(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundary


def dilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)

    for dy in range(-radius, radius + 1):
        max_dx = int(math.floor(math.sqrt(radius * radius - dy * dy)))
        for dx in range(-max_dx, max_dx + 1):
            y0 = radius + dy
            x0 = radius + dx
            out |= padded[y0 : y0 + h, x0 : x0 + w]

    return out


def iter_label_groups(flat_labels: np.ndarray) -> Iterable[np.ndarray]:
    order = np.argsort(flat_labels, kind="mergesort")
    sorted_labels = flat_labels[order]
    cuts = np.flatnonzero(np.diff(sorted_labels)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    for start, end in zip(starts, ends):
        yield order[start:end]


def normalize_lab_from_rgb(rgb_0_255: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb_0_255, 0.0, 255.0).astype(np.float32) / 255.0
    if rgb2lab is None:
        LOGGER.warning("scikit-image not available; CHV will be computed in RGB space")
        return rgb

    lab = rgb2lab(rgb).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] / 100.0
    lab[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0
    lab[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0
    return np.clip(lab, 0.0, 1.0)


def compute_nce(labels: np.ndarray, normals: np.ndarray) -> float:
    flat_labels = labels.reshape(-1)
    flat_normals = normals.reshape(-1, 3).astype(np.float64)
    norm_mag = np.linalg.norm(flat_normals, axis=1, keepdims=True)
    valid = norm_mag[:, 0] > 1e-8
    flat_normals[valid] /= norm_mag[valid]
    flat_normals[~valid] = 0.0

    total_error = 0.0
    total_count = 0
    for group_idx in iter_label_groups(flat_labels):
        segment_normals = flat_normals[group_idx]
        segment_valid = np.linalg.norm(segment_normals, axis=1) > 1e-8
        if not np.any(segment_valid):
            continue

        segment_normals = segment_normals[segment_valid]
        mean_normal = segment_normals.mean(axis=0)
        mean_norm = np.linalg.norm(mean_normal)
        if mean_norm < 1e-8:
            continue

        mean_normal = mean_normal / mean_norm
        dots = np.clip(segment_normals @ mean_normal, -1.0, 1.0)
        errors = 1.0 - dots
        total_error += float(errors.sum())
        total_count += int(errors.shape[0])

    if total_count == 0:
        return float("nan")
    return total_error / total_count


def compute_chv(labels: np.ndarray, color_features: np.ndarray) -> float:
    flat_labels = labels.reshape(-1)
    flat_colors = color_features.reshape(-1, 3).astype(np.float64)

    weighted_var_sum = 0.0
    weighted_count = 0
    for group_idx in iter_label_groups(flat_labels):
        segment_colors = flat_colors[group_idx]
        seg_size = int(segment_colors.shape[0])
        if seg_size == 0:
            continue
        segment_var = np.var(segment_colors, axis=0, ddof=0).mean()
        weighted_var_sum += float(segment_var) * seg_size
        weighted_count += seg_size

    if weighted_count == 0:
        return float("nan")
    return weighted_var_sum / weighted_count


def compute_undersegmentation_error(labels: np.ndarray, gt_labels: np.ndarray, threshold_ratio: float) -> float:
    if labels.shape != gt_labels.shape:
        raise ValueError("labels and gt_labels must have the same shape")

    flat_s = labels.reshape(-1)
    flat_g = gt_labels.reshape(-1)
    total_pixels = flat_s.size
    if total_pixels == 0:
        return float("nan")

    s_ids, s_inv, s_sizes = np.unique(flat_s, return_inverse=True, return_counts=True)
    g_ids, g_inv = np.unique(flat_g, return_inverse=True)

    num_s = len(s_ids)
    num_g = len(g_ids)
    joint_index = g_inv * num_s + s_inv
    intersections = np.bincount(joint_index, minlength=num_g * num_s).reshape(num_g, num_s)

    significant = intersections > (threshold_ratio * s_sizes[np.newaxis, :])
    leakage_sum = float((significant * s_sizes[np.newaxis, :]).sum())
    ue = (leakage_sum - total_pixels) / total_pixels
    return max(0.0, ue)


def compute_boundary_metrics(labels: np.ndarray, gt_labels: np.ndarray, tolerance: int) -> tuple[float, float, float]:
    sp_boundary = build_boundary_map(labels)
    gt_boundary = build_boundary_map(gt_labels)

    gt_count = int(gt_boundary.sum())
    sp_count = int(sp_boundary.sum())

    if gt_count == 0 or sp_count == 0:
        return float("nan"), float("nan"), float("nan")

    gt_dilated = dilate_disk(gt_boundary, tolerance)
    sp_dilated = dilate_disk(sp_boundary, tolerance)

    boundary_recall = float((sp_boundary & gt_dilated).sum()) / gt_count
    boundary_precision = float((gt_boundary & sp_dilated).sum()) / sp_count

    denom = boundary_recall + boundary_precision
    if denom <= 1e-12:
        f_measure = 0.0
    else:
        f_measure = 2.0 * boundary_recall * boundary_precision / denom

    return boundary_recall, boundary_precision, f_measure


def compute_metrics(
    labels: np.ndarray,
    binary_data: np.ndarray,
    gt_labels: np.ndarray,
    ue_threshold: float,
    boundary_tolerance: int,
) -> dict[str, float]:
    normals = binary_data[:, :, 4:7]
    color_features = normalize_lab_from_rgb(binary_data[:, :, :3])

    nce = compute_nce(labels, normals)
    chv = compute_chv(labels, color_features)
    ue = compute_undersegmentation_error(labels, gt_labels, ue_threshold)
    br, bp, f_measure = compute_boundary_metrics(labels, gt_labels, boundary_tolerance)

    return {
        "nce": nce,
        "chv": chv,
        "ue": ue,
        "boundary_recall": br,
        "boundary_precision": bp,
        "f_measure": f_measure,
    }


def sanitize_for_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if segment is None:
        LOGGER.error("Failed to import GeoLexelsDemo.segment: %s", SEGMENT_IMPORT_ERROR)
        LOGGER.error("Expected GeoLexels module under: %s", GEOLEXELS_DIR)
        return 1

    dataset_root = args.dataset_root.resolve()
    fast_cloud_exe = args.fast_cloud_exe.resolve()

    if args.output_dir is None:
        output_dir = dataset_root / ".geolexels_eval"
    else:
        output_dir = args.output_dir.resolve()

    if not dataset_root.exists():
        LOGGER.error("Dataset root not found: %s", dataset_root)
        return 1
    if not fast_cloud_exe.exists():
        LOGGER.error("fast_cloud executable not found: %s", fast_cloud_exe)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_bin_dir = output_dir / "tmp_fast_cloud"
    temp_bin_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = output_dir / "labels"
    if args.save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.info("fast_cloud: %s", fast_cloud_exe)
    LOGGER.info("Output dir: %s", output_dir)

    LOGGER.info("GeoLexels settings: mode=3, threshold=0.25, focal=1.0, λz=0.45, λn=0.1")
    LOGGER.info("GeoLexels settings: doRGBtoLAB=True, color_metric=Laplace, depth_metric=Laplace")
    LOGGER.info("GeoLexels settings: normals_metric=vMF, depth_norm=sensor_max, sensor_max_depth=%.4f", args.sensor_max_depth)

    samples = discover_samples(dataset_root)
    if not samples:
        LOGGER.error("No valid RGB/depth/GT samples found under %s", dataset_root)
        return 1

    if args.start_idx < 0:
        LOGGER.error("--start-idx must be >= 0")
        return 1
    if args.start_idx >= len(samples):
        LOGGER.error("--start-idx (%d) is out of range for %d samples", args.start_idx, len(samples))
        return 1

    selected = samples[args.start_idx :]
    if args.max_images is not None:
        if args.max_images <= 0:
            LOGGER.error("--max-images must be > 0")
            return 1
        selected = selected[: args.max_images]

    LOGGER.info("Evaluating %d samples", len(selected))

    rows: list[dict[str, float | int | str]] = []
    failures: list[dict[str, str]] = []

    start_time = time.perf_counter()
    for idx, sample in enumerate(selected, start=1):
        sample_id = hashlib.sha1(str(sample.rgb_path).encode("utf-8")).hexdigest()[:12]
        temp_bin = temp_bin_dir / f"{sample.frame_stem}_{sample_id}.bin"

        LOGGER.info("[%d/%d] %s", idx, len(selected), sample.rgb_path)
        fast_cloud_seconds = float("nan")
        geolexels_seconds = float("nan")

        try:
            with Image.open(sample.rgb_path) as rgb_im:
                rgb_w, rgb_h = rgb_im.size

            t0 = time.perf_counter()
            fast_cloud_result = run_fast_cloud(
                fast_cloud_exe=fast_cloud_exe,
                rgb_path=sample.rgb_path,
                depth_path=sample.depth_path,
                output_bin=temp_bin,
                timeout_seconds=args.timeout_seconds,
            )
            fast_cloud_seconds = time.perf_counter() - t0

            if fast_cloud_result.returncode != 0:
                raise RuntimeError(
                    f"fast_cloud failed (rc={fast_cloud_result.returncode}) stderr={fast_cloud_result.stderr.strip()}"
                )
            if not temp_bin.exists():
                raise FileNotFoundError(f"fast_cloud did not produce output file: {temp_bin}")

            binary_data = None
            try:
                binary_data = load_fast_cloud_binary(temp_bin, rgb_w, rgb_h)
                width, height = rgb_w, rgb_h
            except ValueError:
                parsed_dims = parse_dims_from_fast_cloud_output(fast_cloud_result.stdout + "\n" + fast_cloud_result.stderr)
                if parsed_dims is None:
                    raise
                width, height = parsed_dims
                binary_data = load_fast_cloud_binary(temp_bin, width, height)

            t1 = time.perf_counter()
            labels, numlabels = segment(
                str(temp_bin),
                threshold=0.25,
                doRGBtoLAB=True,
                weight_depth=0.45,
                weight_normals=0.1,
                focal_length=1.0,
                normals_mode=3,
                is_binary=True,
                width=width,
                height=height,
                color_metric=1,
                depth_metric=1,
                normals_metric=2,
                depth_normalization_mode=2,
                sensor_max_depth=float(args.sensor_max_depth),
                trunc_laplace_cutoff=0.5,
            )
            geolexels_seconds = time.perf_counter() - t1

            gt_labels = load_label_map(sample.gt_path)
            gt_labels = resize_label_map(gt_labels, labels.shape[0], labels.shape[1])

            metric_values = compute_metrics(
                labels=labels,
                binary_data=binary_data,
                gt_labels=gt_labels,
                ue_threshold=args.ue_threshold,
                boundary_tolerance=args.boundary_tolerance,
            )

            total_seconds = fast_cloud_seconds + geolexels_seconds

            row = {
                "index": idx,
                "rgb_path": str(sample.rgb_path),
                "depth_path": str(sample.depth_path),
                "gt_path": str(sample.gt_path),
                "scene_rel": str(sample.scene_rel),
                "frame_stem": sample.frame_stem,
                "width": labels.shape[1],
                "height": labels.shape[0],
                "num_superpixels": int(numlabels),
                "nce": metric_values["nce"],
                "chv": metric_values["chv"],
                "ue": metric_values["ue"],
                "boundary_recall": metric_values["boundary_recall"],
                "boundary_precision": metric_values["boundary_precision"],
                "f_measure": metric_values["f_measure"],
                "fast_cloud_seconds": fast_cloud_seconds,
                "geolexels_seconds": geolexels_seconds,
                "total_seconds": total_seconds,
            }
            rows.append(row)

            if args.save_labels:
                label_path = labels_dir / sample.scene_rel / f"{sample.frame_stem}.npy"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(label_path, labels)

            LOGGER.info(
                "[%d/%d] nce=%.6f chv=%.6f ue=%.6f br=%.6f bp=%.6f f=%.6f (%d superpixels)",
                idx,
                len(selected),
                metric_values["nce"],
                metric_values["chv"],
                metric_values["ue"],
                metric_values["boundary_recall"],
                metric_values["boundary_precision"],
                metric_values["f_measure"],
                int(numlabels),
            )

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({"rgb_path": str(sample.rgb_path), "error": message})
            LOGGER.error("[%d/%d] Failed: %s", idx, len(selected), message)
            if args.fail_fast:
                break
        finally:
            if temp_bin.exists() and not args.keep_binaries:
                try:
                    temp_bin.unlink()
                except OSError:
                    pass

    elapsed = time.perf_counter() - start_time

    csv_path = output_dir / "metrics_per_image.csv"
    summary_path = output_dir / "summary.json"
    failures_path = output_dir / "failures.json"

    fieldnames = [
        "index",
        "rgb_path",
        "depth_path",
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
        "fast_cloud_seconds",
        "geolexels_seconds",
        "total_seconds",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def metric_mean(name: str) -> float:
        values = [float(r[name]) for r in rows]
        if not values:
            return float("nan")
        return float(np.nanmean(values))

    summary = {
        "dataset_root": str(dataset_root),
        "fast_cloud_exe": str(fast_cloud_exe),
        "output_dir": str(output_dir),
        "requested_settings": {
            "mode": 3,
            "threshold": 0.25,
            "focal_length": 1.0,
            "weight_color": 0.45,
            "weight_depth": 0.45,
            "weight_normals": 0.1,
            "depth_normalization": "sensor_max",
            "sensor_max_depth": float(args.sensor_max_depth),
            "color_metric": "laplace",
            "depth_metric": "laplace",
            "normals_metric": "von_mises_fisher",
            "convert_to_cielab": True,
            "fast_cloud_flags": ["-n", "-G", "-B"],
        },
        "evaluation_params": {
            "ue_threshold": float(args.ue_threshold),
            "boundary_tolerance": int(args.boundary_tolerance),
        },
        "counts": {
            "images_selected": len(selected),
            "images_succeeded": len(rows),
            "images_failed": len(failures),
        },
        "means": {
            "nce": metric_mean("nce"),
            "chv": metric_mean("chv"),
            "ue": metric_mean("ue"),
            "boundary_recall": metric_mean("boundary_recall"),
            "boundary_precision": metric_mean("boundary_precision"),
            "f_measure": metric_mean("f_measure"),
            "num_superpixels": metric_mean("num_superpixels"),
            "fast_cloud_seconds": metric_mean("fast_cloud_seconds"),
            "geolexels_seconds": metric_mean("geolexels_seconds"),
            "total_seconds": metric_mean("total_seconds"),
        },
        "elapsed_seconds": elapsed,
        "files": {
            "metrics_per_image_csv": str(csv_path),
            "summary_json": str(summary_path),
            "failures_json": str(failures_path),
        },
    }

    summary_safe = sanitize_for_json(summary)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_safe, handle, indent=2)

    with failures_path.open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)

    LOGGER.info("Wrote per-image metrics CSV: %s", csv_path)
    LOGGER.info("Wrote summary JSON: %s", summary_path)
    LOGGER.info("Wrote failures JSON: %s", failures_path)

    if rows:
        LOGGER.info(
            "Mean metrics: NCE=%.6f CHV=%.6f UE=%.6f BR=%.6f BP=%.6f F=%.6f",
            float(summary_safe["means"]["nce"]) if summary_safe["means"]["nce"] is not None else float("nan"),
            float(summary_safe["means"]["chv"]) if summary_safe["means"]["chv"] is not None else float("nan"),
            float(summary_safe["means"]["ue"]) if summary_safe["means"]["ue"] is not None else float("nan"),
            float(summary_safe["means"]["boundary_recall"]) if summary_safe["means"]["boundary_recall"] is not None else float("nan"),
            float(summary_safe["means"]["boundary_precision"]) if summary_safe["means"]["boundary_precision"] is not None else float("nan"),
            float(summary_safe["means"]["f_measure"]) if summary_safe["means"]["f_measure"] is not None else float("nan"),
        )

    if failures:
        LOGGER.warning("Completed with %d failed samples", len(failures))
        return 2 if not rows else 0

    LOGGER.info("Completed successfully for %d samples", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
