#!/usr/bin/env python3
"""
Plot three METR-LA station curves in one figure.

Each station is shown with 10 points. Two points per station are treated as
missing and rendered as hollow circles; line segments touching a missing point
are dashed. Axes and borders are hidden for a clean figure.

Usage:
    python scripts/plot_metrla_three_stations.py
    python scripts/plot_metrla_three_stations.py --stations 0 25 50
    python scripts/plot_metrla_three_stations.py --output figures/metrla_three.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_metrla_raw


def select_series(values: np.ndarray, station_idx: int, start_idx: int, num_steps: int) -> np.ndarray:
    series = values[start_idx : start_idx + num_steps, station_idx].astype(np.float32)
    series = series.copy()
    series[~np.isfinite(series)] = np.nan
    series[series <= 0] = np.nan
    return series


def find_valid_window(values: np.ndarray, station_idx: int, num_steps: int, missing_positions: np.ndarray) -> int:
    max_start = values.shape[0] - num_steps
    for start_idx in range(max_start + 1):
        series = select_series(values, station_idx, start_idx, num_steps)
        valid_positions = [idx for idx in range(num_steps) if idx not in set(missing_positions)]
        if np.all(np.isfinite(series[valid_positions])):
            return start_idx
    raise ValueError(
        f"Could not find a {num_steps}-step window for station {station_idx} "
        f"with only the specified missing positions {missing_positions.tolist()}."
    )


def draw_station(axis, x: np.ndarray, y: np.ndarray, missing_mask: np.ndarray, color: str) -> None:
    line_y = y.copy()
    valid_mask = ~missing_mask & np.isfinite(y)
    if valid_mask.sum() >= 2:
        line_y = np.interp(x, x[valid_mask], y[valid_mask])

    for idx in range(len(x) - 1):
        linestyle = "--" if (missing_mask[idx] or missing_mask[idx + 1]) else "-"
        axis.plot(x[idx : idx + 2], line_y[idx : idx + 2], color=color, linewidth=4, linestyle=linestyle)

    axis.plot(x[valid_mask], y[valid_mask], linestyle="None", marker="o", markersize=16,
              markerfacecolor=color, markeredgecolor=color)
    axis.plot(x[missing_mask], line_y[missing_mask], linestyle="None", marker="o", markersize=17,
              markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.8)


def add_axis_arrows(axis) -> None:
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()

    axis.annotate(
        "",
        xy=(x_max * 1.02, y_min),
        xytext=(x_min, y_min),
        arrowprops=dict(arrowstyle="->", lw=3.2, color="black", mutation_scale=24),
        annotation_clip=False,
    )
    axis.annotate(
        "",
        xy=(x_min, y_max * 1.03),
        xytext=(x_min, y_min),
        arrowprops=dict(arrowstyle="->", lw=3.2, color="black", mutation_scale=24),
        annotation_clip=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="datasets")
    parser.add_argument("--stations", type=int, nargs=3, default=[0, 50, 100])
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="figures/metrla_three_stations.png")
    args = parser.parse_args()

    values, _ = load_metrla_raw("METR-LA", data_dir=args.dataset_dir)
    num_stations = values.shape[1]

    station_ids = []
    for station in args.stations:
        if station < 0 or station >= num_stations:
            raise ValueError(f"station index {station} out of range [0, {num_stations - 1}]")
        station_ids.append(station)

    fig, ax = plt.subplots(figsize=(10, 4))
    time_axis = np.arange(args.num_steps)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    missing_positions_by_station = [np.array([3, 6]), np.array([2, 7]), np.array([5])]

    for color, station_idx, missing_positions in zip(colors, station_ids, missing_positions_by_station):
        start_idx = find_valid_window(values, station_idx, args.num_steps, missing_positions)
        series = select_series(values, station_idx, start_idx, args.num_steps)
        missing_mask = np.zeros_like(series, dtype=bool)
        for position in missing_positions:
            if 0 <= position < len(missing_mask):
                missing_mask[position] = True
        draw_station(ax, time_axis, series, missing_mask, color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xticks(time_axis)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xlabel("")
    ax.margins(x=0.02, y=0.15)
    add_axis_arrows(ax)
    fig.tight_layout(pad=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
