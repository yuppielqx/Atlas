#!/usr/bin/env python3
"""
2×5 spatial scatter of source-client subsets + aligned data table below.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRACTIONS = ["05", "25", "50", "75", "90"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_IDS_DIR = os.path.join(REPO, "client_ids")


# ── Load coordinates ────────────────────────────────────────────────────

def load_coords_metrla():
    import csv
    path = os.path.join(REPO, "datasets", "METR-LA", "graph_sensor_locations.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((int(row["index"]), float(row["longitude"]), float(row["latitude"])))
    rows.sort(key=lambda x: x[0])
    return np.array([(r[1], r[2]) for r in rows])


def load_coords_pemsbay():
    path = os.path.expanduser(
        "~/baselines/KDD2024_FedGTP/data/PEMS_BAY/graph_sensor_locations_bay.csv")
    rows = []
    with open(path) as f:
        for line in f:
            _, lat, lon = line.strip().split(",")
            rows.append((float(lon), float(lat)))
    return np.array(rows)


def load_source_ids(ds, frac):
    import glob as gmod
    pattern = os.path.join(CLIENT_IDS_DIR, ds, f"frac{frac}_n*.txt")
    for m in sorted(gmod.glob(pattern)):
        with open(m) as f:
            return [int(line.strip()) for line in f if line.strip()]
    raise FileNotFoundError(pattern)


# ── Plot ────────────────────────────────────────────────────────────────

def main():
    coords = {"metr-la": load_coords_metrla(), "pems-bay": load_coords_pemsbay()}
    ds_labels = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
    ds_short  = {"metr-la": "metrla",   "pems-bay": "pemsbay"}

    # ── Load summary metrics ──
    sdf = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "source_client_mae_summary.csv"))
    metrics = {}
    for _, r in sdf.iterrows():
        metrics[(r["dataset"], f"{r['frac']:02d}")] = {
            "mae": r["mean_source_mae"], "bar_mae": r["mean_allclients_mae"],
            "rmse": r["mean_source_rmse"], "bar_rmse": r["mean_allclients_rmse"],
        }

    # ── Figure: 4 rows × 5 cols ──
    # row 0 → METR-LA maps,  row 1 → METR-LA data
    # row 2 → PEMS-BAY maps,  row 3 → PEMS-BAY data
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})
    fig, axes = plt.subplots(4, 5, figsize=(10.0, 6.0), facecolor="white",
                             gridspec_kw={"height_ratios": [3.0, 1.6, 3.0, 1.6]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.02,
                        wspace=0.03, hspace=0.1)

    color_bg  = "#B0B0B0"
    color_src = "#D62728"

    datasets = ["metr-la", "pems-bay"]
    ds_map_row = [0, 2]   # map rows
    ds_data_row = [1, 3]  # data rows

    for idx, ds in enumerate(datasets):
        map_row = ds_map_row[idx]
        data_row = ds_data_row[idx]
        all_xy = coords[ds]
        ds_s = ds_short[ds]

        for col, frac in enumerate(FRACTIONS):
            ax = axes[map_row, col]
            src_ids = load_source_ids(ds, frac)

            ax.scatter(all_xy[:, 0], all_xy[:, 1], c=color_bg, s=2.5,
                       edgecolors="none", alpha=0.5, zorder=1)
            src_xy = all_xy[src_ids]
            ax.scatter(src_xy[:, 0], src_xy[:, 1], c=color_src, s=6.0,
                       edgecolors="none", alpha=0.9, zorder=2)

            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#999999")

            if idx == 0:
                ax.set_title(f"{int(frac)}%", fontsize=14, fontweight="bold", pad=15)

            # ── Data table below this map ──
            m = metrics.get((ds_s, frac), {})
            cell_text = [
                ["MAE",  f"{m['mae']:.2f}",     f"{m['bar_mae']:.2f}"],
                ["RMSE", f"{m['rmse']:.2f}",    f"{m['bar_rmse']:.2f}"],
            ] if m else [["", "", ""], ["", "", ""]]
            col_labels = ["Metric", "Few", "All"]

            ax_data = axes[data_row, col]
            tbl = ax_data.table(cellText=cell_text, colLabels=col_labels,
                                cellLoc="center", loc="center",
                                bbox=[0, 0.0, 1, 1.0])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.3)
            for key, cell in tbl.get_celld().items():
                cell.set_linewidth(0.3)
                cell.set_edgecolor("#CCCCCC")
                r, c = key
                if r == 0:  # header row
                    cell.set_facecolor("#EEEEEE")
                    cell.set_text_props(fontsize=12)
                    cell.set_height(cell.get_height() * 0.8)
                elif c == 0:  # MAE / RMSE label
                    cell.set_facecolor("#EEEEEE")
                    cell.set_text_props(fontsize=12)
                else:  # numeric values
                    cell.set_facecolor("white")
                    cell.set_text_props(fontsize=11)
            ax_data.axis("off")
            for spine in ax_data.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#999999")

        # Row label centered across map + data rows
        bbox_map = axes[map_row, 0].get_position()
        bbox_data = axes[data_row, 0].get_position()
        y_center = (bbox_map.y1 + bbox_data.y0) / 2  # midpoint of combined block
        x_pos = bbox_map.x0 - 0.04  # left of the maps
        fig.text(x_pos, y_center, ds_labels[ds],
                 fontsize=16, fontweight="bold", rotation=90,
                 ha="center", va="center")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fraction_source_map.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved {out}  ({out.replace('.pdf', '.png')})")


if __name__ == "__main__":
    main()
