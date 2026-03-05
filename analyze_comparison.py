#!/usr/bin/env python3
"""Analyze and visualize superpixel algorithm comparison results."""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None


def load_comparison_data(results_dir: Path) -> dict:
    """Load comparison results from directory."""
    summary_file = results_dir / "comparison_summary.json"
    comparison_csv = results_dir / "comparison_results.csv"
    
    data = {
        "summary": {},
        "per_image": {},
    }
    
    # Load summary
    if summary_file.exists():
        with open(summary_file) as f:
            data["summary"] = json.load(f)
    
    # Load per-image results
    if comparison_csv.exists():
        data["per_image"] = pd.read_csv(comparison_csv)
    
    return data


def print_summary_table(data: dict) -> None:
    """Print summary statistics as table."""
    summary = data["summary"]
    algorithms = summary.get("algorithms", {})
    
    print("\n" + "=" * 100)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 100)
    
    if not algorithms:
        print("No summary data available")
        return
    
    # Determine which metrics to show
    first_algo = list(algorithms.values())[0]
    metric_names = list(first_algo.get("metrics", {}).keys())
    
    # Header
    header = ["Algorithm", "Evaluated", "Failed"]
    header.extend([m.upper() for m in metric_names])
    print(f"{header[0]:15} {header[1]:>10} {header[2]:>8}", end="")
    for m in metric_names:
        print(f" {m:>12}", end="")
    print()
    print("-" * 100)
    
    # Rows
    for algo_name, algo_data in algorithms.items():
        num_eval = algo_data.get("num_evaluated", 0)
        num_failed = algo_data.get("num_failed", 0)
        metrics = algo_data.get("metrics", {})
        
        print(f"{algo_name:15} {num_eval:>10} {num_failed:>8}", end="")
        for metric_name in metric_names:
            value = metrics.get(metric_name, float("nan"))
            if isinstance(value, float):
                if np.isnan(value):
                    print(f" {'N/A':>12}", end="")
                else:
                    print(f" {value:>12.6f}", end="")
            else:
                print(f" {value:>12}", end="")
        print()
    
    print("=" * 100 + "\n")


def print_per_image_stats(data: dict) -> None:
    """Print per-image statistics by algorithm."""
    df = data["per_image"]
    
    if df.empty:
        print("No per-image data available")
        return
    
    print("\n" + "=" * 100)
    print("PER-IMAGE STATISTICS")
    print("=" * 100)
    
    metrics = ["nce", "chv", "ue", "boundary_recall", "boundary_precision", "f_measure", "runtime_seconds"]
    
    grouped = df.groupby("algorithm")
    for algo_name in df["algorithm"].unique():
        algo_df = grouped.get_group(algo_name)
        print(f"\n{algo_name.upper()}:")
        print("-" * 60)
        
        for metric in metrics:
            if metric in algo_df.columns:
                values = pd.to_numeric(algo_df[metric], errors="coerce")
                valid = values.dropna()
                
                if len(valid) > 0:
                    print(
                        f"  {metric:25} - Mean: {valid.mean():>10.6f} "
                        f"Std: {valid.std():>10.6f} "
                        f"Min: {valid.min():>10.6f} Max: {valid.max():>10.6f}"
                    )


def plot_metrics_comparison(data: dict, output_file: Optional[Path] = None) -> None:
    """Create visualization comparing metrics across algorithms."""
    if plt is None:
        print("matplotlib not available, skipping plots")
        return
    
    summary = data["summary"]
    algorithms = summary.get("algorithms", {})
    
    if not algorithms:
        print("No data to plot")
        return
    
    # Extract data for plotting
    algo_names = list(algorithms.keys())
    metrics_dict = {}
    
    for algo_name in algo_names:
        metrics = algorithms[algo_name].get("metrics", {})
        for metric_name, value in metrics.items():
            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = []
            metrics_dict[metric_name].append(value)
    
    # Create subplots for each metric
    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(
        (n_metrics + 2) // 3, 3, figsize=(15, 4 * ((n_metrics + 2) // 3))
    )
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(algo_names)]
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = metrics_dict[metric_name]
        
        bars = ax.bar(algo_names, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f"{metric_name}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9
            )
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze algorithm comparison results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing comparison results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Save plot to file",
    )
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    # Load data
    data = load_comparison_data(args.results_dir)
    
    # Print summaries
    print_summary_table(data)
    print_per_image_stats(data)
    
    # Generate plots
    if args.plot or args.plot_output:
        plot_metrics_comparison(data, args.plot_output)
    
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
