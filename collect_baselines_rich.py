"""
Collect fraction-transfer baselines into a clean CSV.
Only includes: dataset, frac, n_source, n_rest,
and for each baseline: test_mae, test_rmse.

Usage:
    python collect_baselines_rich.py
"""

import os, sys, json, csv, re
import numpy as np

DATASETS = {"metr-la": {"tag": "metrla", "total_nodes": 207}, "pems-bay": {"tag": "pemsbay", "total_nodes": 325}}
FRACTIONS = ["05", "25", "50", "75", "90"]
SRC_COUNTS = {"metr-la": {"05": 10, "25": 52, "50": 104, "75": 155, "90": 186},
              "pems-bay": {"05": 16, "25": 81, "50": 162, "75": 244, "90": 292}}

EXPERIMENTS = [
    ("xfer",                 "xfer_{t}_frac{f}_to_rest_k10_r200"),
    ("randbase",             "randbase_{t}_frac{f}_to_rest_k10_r200"),
    ("fedrest",              "fedrest_{t}_frac{f}_k10_r200"),
    ("randbase_trainable",   "randbase_trainable_{t}_frac{f}_to_rest_k10_r200"),
]


def get_metrics(run_dir):
    path = os.path.join("output", run_dir, "results.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as fp:
        d = json.load(fp)
    mae = d.get("test_mae") or d.get("metrics", {}).get("MAE")
    rmse = d.get("test_rmse") or d.get("metrics", {}).get("RMSE")
    return mae, rmse


def load_bar():
    path = "output/allclients_bar_rest_subsets.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    bar = load_bar()
    rows = []

    for ds, info in DATASETS.items():
        t = info["tag"]
        for frac in FRACTIONS:
            n_src = SRC_COUNTS[ds][frac]
            n_rest = info["total_nodes"] - n_src
            row = {"dataset": ds, "frac": frac, "n_source": n_src, "n_rest": n_rest}

            if bar and ds in bar.get("results", {}):
                row["allclients_bar_mae"] = bar["results"][ds]["fractions"].get(frac, {}).get("avg_mae")
                row["allclients_bar_rmse"] = bar["results"][ds]["fractions"].get(frac, {}).get("avg_rmse")

            for key, tmpl in EXPERIMENTS:
                run_dir = tmpl.replace("{t}", t).replace("{f}", frac)
                mae, rmse = get_metrics(run_dir)
                row[f"{key}_mae"] = mae
                row[f"{key}_rmse"] = rmse

            rows.append(row)

    # ── Print ──
    print("=" * 130)
    print("Fraction Transfer Baselines")
    print("=" * 130)
    header = (f"{'Dataset':10} {'Frac':>5} {'N_src':>6} {'N_rest':>7}  "
              f"{'All-bar':>12}  "
              f"{'①Xfer':>12}  {'②RandFrz':>12}  {'③FedRest':>12}  {'④RandTrn':>12}")
    print(header)
    print(header.replace('MAE', '───').replace('│', '─'))
    # Sub-header for MAE / RMSE
    sub = f"{'':>10} {'':>5} {'':>6} {'':>7}  {'MAE':>6} {'RMSE':>6}  {'MAE':>6} {'RMSE':>6}  {'MAE':>6} {'RMSE':>6}  {'MAE':>6} {'RMSE':>6}  {'MAE':>6} {'RMSE':>6}"
    print(sub)
    print("-" * 130)
    for ds_key in ["metr-la", "pems-bay"]:
        label = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}[ds_key]
        first = True
        for row in rows:
            if row["dataset"] != ds_key:
                continue
            name = label if first else ""
            first = False
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "N/A"
            print(f"{name:10} {row['frac']:>5} {row['n_source']:>6} {row['n_rest']:>7}  "
                  f"{fmt(row.get('allclients_bar_mae')):>6} {fmt(row.get('allclients_bar_rmse')):>6}  "
                  f"{fmt(row.get('xfer_mae')):>6} {fmt(row.get('xfer_rmse')):>6}  "
                  f"{fmt(row.get('randbase_mae')):>6} {fmt(row.get('randbase_rmse')):>6}  "
                  f"{fmt(row.get('fedrest_mae')):>6} {fmt(row.get('fedrest_rmse')):>6}  "
                  f"{fmt(row.get('randbase_trainable_mae')):>6} {fmt(row.get('randbase_trainable_rmse')):>6}")
        print()

    # ── CSV ──
    cols = ["dataset", "frac", "n_source", "n_rest",
            "allclients_bar_mae", "allclients_bar_rmse",
            "xfer_mae", "xfer_rmse", "randbase_mae", "randbase_rmse",
            "fedrest_mae", "fedrest_rmse", "randbase_trainable_mae", "randbase_trainable_rmse"]
    with open("output/fraction_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"CSV → output/fraction_baselines.csv  ({len(cols)} cols)")

    # ── Cross-dataset ──
    print("=" * 80)
    print("Cross-Dataset Transfer (all clients, frozen codebook, local training)")
    print("=" * 80)

    CROSS = {
        "metr-la":  {"cross": "xfer_cross_bay2la_zinv_k10_r200", "self": "xfer_zinv_frac100_to_all_k10_r200",
                     "random": "randbase_all_metrla_k10_r200", "fed": "zinv_ksweep_all_k10_r200"},
        "pems-bay": {"cross": "xfer_cross_la2bay_zinv_k10_r200", "self": "xfer_self_bay_zinv_k10_r200",
                     "random": "randbase_all_pemsbay_k10_r200", "fed": "pemsbay_stage1_k10_r200"},
    }

    print(f"{'Target':10} {'Cross':>10} {'Self':>10} {'Random':>10} {'Fed':>10}")
    print("-" * 50)
    cross_rows = []
    for target, runs in CROSS.items():
        row = {"target": target}
        for key, rd in runs.items():
            mae, rmse = get_metrics(rd)
            row[f"{key}_mae"] = mae
            row[f"{key}_rmse"] = rmse
        cross_rows.append(row)
        label = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}[target]
        def f(v): return f"{v:.4f}" if v else "N/A"
        print(f"{label:10} {f(row['cross_mae']):>10} {f(row['self_mae']):>10} {f(row['random_mae']):>10} {f(row['fed_mae']):>10}")

    cross_cols = ["target", "cross_mae", "cross_rmse", "self_mae", "self_rmse",
                  "random_mae", "random_rmse", "fed_mae", "fed_rmse"]
    with open("output/fraction_baselines_cross_dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cross_cols)
        w.writeheader()
        for row in cross_rows:
            w.writerow(row)
    print(f"\nCSV → output/fraction_baselines_cross_dataset.csv  ({len(cross_cols)} cols)")


if __name__ == "__main__":
    main()
