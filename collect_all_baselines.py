"""
Collect all fraction-transfer baselines into a single CSV.

Reads results.json from:
  ① xfer:          xfer_{ds}_frac{f}_to_rest_k10_r200
  ② randbase:      randbase_{ds}_frac{f}_to_rest_k10_r200
  ③ fedrest:       fedrest_{ds}_frac{f}_k10_r200
  ④ randbase_trainable: randbase_trainable_{ds}_frac{f}_to_rest_k10_r200
  all-clients bar: output/allclients_bar_rest_subsets.json

Usage:
    python collect_all_baselines.py
    python collect_all_baselines.py --csv results.csv
"""

import os
import sys
import json
import csv
import argparse
import numpy as np

# ── Config ──
DATASETS = {
    "metr-la":  {"tag": "metrla",  "total_nodes": 207},
    "pems-bay": {"tag": "pemsbay", "total_nodes": 325},
}
FRACTIONS = ["05", "25", "50", "75", "90"]

SRC_COUNTS = {
    "metr-la":  {"05": 10, "25": 52, "50": 104, "75": 155, "90": 186},
    "pems-bay": {"05": 16, "25": 81, "50": 162, "75": 244, "90": 292},
}

EXPERIMENTS = [
    {
        "key": "xfer",
        "label": "Transfer (source Q, frozen, local)",
        "tmpl": "xfer_{ds_tag}_frac{frac}_to_rest_k10_r200",
    },
    {
        "key": "randbase",
        "label": "Random Q, frozen, local",
        "tmpl": "randbase_{ds_tag}_frac{frac}_to_rest_k10_r200",
    },
    {
        "key": "fedrest",
        "label": "Random Q, trainable, federated",
        "tmpl": "fedrest_{ds_tag}_frac{frac}_k10_r200",
    },
    {
        "key": "randbase_trainable",
        "label": "Random Q, trainable, local",
        "tmpl": "randbase_trainable_{ds_tag}_frac{frac}_to_rest_k10_r200",
    },
]


def get_mae(run_dir):
    """Extract test_mae from results.json."""
    path = os.path.join("output", run_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    # results.json can have 'test_mae' at top level or nested
    if "test_mae" in d:
        return d["test_mae"]
    if "metrics" in d and isinstance(d["metrics"], dict) and "MAE" in d["metrics"]:
        return d["metrics"]["MAE"]
    return None


def load_allclients_bar():
    """Load pre-computed all-clients bar."""
    path = "output/allclients_bar_rest_subsets.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="output/fraction_baselines.csv")
    args = ap.parse_args()

    bar_data = load_allclients_bar()
    rows = []

    for ds_key, ds_info in DATASETS.items():
        ds_tag = ds_info["tag"]
        for frac in FRACTIONS:
            n_src = SRC_COUNTS[ds_key][frac]
            n_rest = ds_info["total_nodes"] - n_src

            row = {
                "dataset": ds_key,
                "frac": frac,
                "n_source": n_src,
                "n_rest": n_rest,
            }

            # All-clients bar
            if bar_data and ds_key in bar_data["results"]:
                frac_info = bar_data["results"][ds_key]["fractions"].get(frac, {})
                row["allclients_bar"] = frac_info.get("avg_mae", None)

            # Each experiment
            for exp in EXPERIMENTS:
                run_dir = exp["tmpl"].format(ds_tag=ds_tag, frac=frac)
                mae = get_mae(run_dir)
                row[exp["key"]] = mae

            rows.append(row)

    # ── Print summary ──
    print("=" * 110)
    print("Fraction Transfer Baselines — Full Results")
    print("=" * 110)

    header = ["dataset", "frac", "n_src", "n_rest",
              "allclients_bar", "xfer", "randbase", "fedrest", "randbase_trainable"]
    col_names = {
        "dataset": "Dataset", "frac": "Frac", "n_src": "N_src", "n_rest": "N_rest",
        "allclients_bar": "All-clients bar",
        "xfer": "① Transfer",
        "randbase": "② Rand+freeze+local",
        "fedrest": "③ Rand+fed",
        "randbase_trainable": "④ Rand+train+local",
    }

    # Print header
    print(f"{'Dataset':10} {'Frac':>5} {'N_src':>6} {'N_rest':>7}  "
          f"{'All-bar':>8}  {'①Xfer':>8}  {'②RandFrz':>9}  {'③FedRest':>9}  {'④RandTrn':>9}")
    print("-" * 100)

    for ds_key in ["metr-la", "pems-bay"]:
        ds_name = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}[ds_key]
        first = True
        for row in rows:
            if row["dataset"] != ds_key:
                continue
            label = ds_name if first else ""
            first = False

            def fmt(v):
                return f"{v:.4f}" if v is not None and np.isfinite(v) else "N/A"

            print(f"{label:10} {row['frac']:>5} {row['n_source']:>6} {row['n_rest']:>7}  "
                  f"{fmt(row.get('allclients_bar')):>8}  "
                  f"{fmt(row.get('xfer')):>8}  "
                  f"{fmt(row.get('randbase')):>9}  "
                  f"{fmt(row.get('fedrest')):>9}  "
                  f"{fmt(row.get('randbase_trainable')):>9}")
        print()

    # ── Write CSV ──
    csv_cols = ["dataset", "frac", "n_source", "n_rest",
                "allclients_bar", "xfer", "randbase", "fedrest", "randbase_trainable"]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writerow({c: col_names.get(c, c) for c in csv_cols})
        for row in rows:
            writer.writerow({c: row.get(c) for c in csv_cols})

    print(f"CSV saved to {args.csv}")

    # ── Count completeness ──
    for exp in EXPERIMENTS:
        done = sum(1 for r in rows if r.get(exp["key"]) is not None)
        print(f"  {exp['key']}: {done}/{len(rows)} available")

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: Cross-dataset transfer (all clients)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Cross-Dataset Transfer (all clients, frozen codebook, local training)")
    print("=" * 80)

    CROSS_RUNS = {
        "metr-la": {
            "cross":   "xfer_cross_bay2la_zinv_k10_r200",    # PEMS-BAY Q → METR-LA
            "self":    "xfer_zinv_frac100_to_all_k10_r200",   # METR-LA Q → METR-LA
            "random":  "randbase_all_metrla_k10_r200",        # random Q → METR-LA
            "fed":     "zinv_ksweep_all_k10_r200",            # METR-LA federated
        },
        "pems-bay": {
            "cross":   "xfer_cross_la2bay_zinv_k10_r200",    # METR-LA Q → PEMS-BAY
            "self":    "xfer_self_bay_zinv_k10_r200",         # PEMS-BAY Q → PEMS-BAY
            "random":  "randbase_all_pemsbay_k10_r200",       # random Q → PEMS-BAY
            "fed":     "pemsbay_stage1_k10_r200",             # PEMS-BAY federated
        },
    }

    cross_data = {}
    for target, runs in CROSS_RUNS.items():
        cross_data[target] = {}
        for key, run_dir in runs.items():
            mae = get_mae(run_dir)
            cross_data[target][key] = mae

    print(f"{'Target':10} {'Cross (other Q)':>17} {'Self (own Q)':>17} "
          f"{'Random (frozen)':>17} {'Fed (ceiling)':>17}")
    print("-" * 80)
    for target in ["metr-la", "pems-bay"]:
        cd = cross_data[target]
        def fmt(v):
            return f"{v:.4f}" if v is not None and np.isfinite(v) else "N/A"
        label = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}[target]
        print(f"{label:10} {fmt(cd['cross']):>17} {fmt(cd['self']):>17} "
              f"{fmt(cd['random']):>17} {fmt(cd['fed']):>17}")

    # Write cross-dataset CSV
    cross_csv = args.csv.replace(".csv", "_cross_dataset.csv")
    with open(cross_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "cross_transfer", "self_transfer", "random_frozen", "federated_ceiling"])
        for target in ["metr-la", "pems-bay"]:
            cd = cross_data[target]
            writer.writerow([target, cd["cross"], cd["self"], cd["random"], cd["fed"]])
    print(f"\nCross-dataset CSV saved to {cross_csv}")

    for target in ["metr-la", "pems-bay"]:
        cd = cross_data[target]
        done = sum(1 for v in cd.values() if v is not None)
        print(f"  {target}: {done}/4 available")
