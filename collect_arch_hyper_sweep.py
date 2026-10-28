#!/usr/bin/env python3
"""Collect architecture hyperparameter sweep results into a CSV."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

OUTPUT_DIR = Path("output")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Define all experiments
DATASETS = ["metr-la", "pems-bay"]

CONFIGS = [
    # (param_name, values, run_name_template)
    ("hid_dim", [32, 128, 256], "arch_hid{val}_{ds}_r200"),
    ("tau", [0.01, 0.05, 0.2, 0.5, 1.0], "arch_tau{val_tag}_{ds}_r200"),
    ("stid_num_layers", [1, 2, 4, 5], "arch_nlayer{val}_{ds}_r200"),
]


def load_results(run_name: str) -> dict | None:
    path = OUTPUT_DIR / run_name / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main():
    rows = []
    missing = []

    for param, values, template in CONFIGS:
        for ds in DATASETS:
            for val in values:
                if param == "tau":
                    val_tag = str(val).replace(".", "").replace("00", "0").rstrip("0")
                    # Ensure two digits: 0.01->001, 0.05->005, 0.2->02, 0.5->05, 1.0->10
                    if val == 0.01:
                        val_tag = "001"
                    elif val == 0.05:
                        val_tag = "005"
                    elif val == 0.2:
                        val_tag = "02"
                    elif val == 0.5:
                        val_tag = "05"
                    elif val == 1.0:
                        val_tag = "10"
                    run_name = template.format(val_tag=val_tag, ds=ds)
                else:
                    run_name = template.format(val=val, ds=ds)

                res = load_results(run_name)
                if res is None:
                    missing.append(run_name)
                    rows.append({
                        "param": param,
                        "value": val,
                        "dataset": ds,
                        "run_name": run_name,
                        "MAE": None,
                        "RMSE": None,
                        "MAPE": None,
                        "best_val_loss": None,
                        "best_round": None,
                    })
                    continue

                metrics = res.get("metrics", {})
                rows.append({
                    "param": param,
                    "value": val,
                    "dataset": ds,
                    "run_name": run_name,
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "MAPE": metrics.get("MAPE"),
                    "best_val_loss": res.get("best_val_loss"),
                    "best_round": res.get("best_round"),
                })

    # Write CSV
    out_path = RESULTS_DIR / "arch_hyper_sweep.csv"
    fieldnames = ["param", "value", "dataset", "run_name", "MAE", "RMSE", "MAPE", "best_val_loss", "best_round"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print summary to stdout
    print(f"Total experiments: {len(rows)} ({len(rows) - len(missing)} complete, {len(missing)} missing)")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m}")

    # Print per-param summary
    for param in sorted(set(r["param"] for r in rows)):
        complete = [r for r in rows if r["param"] == param and r["MAE"] is not None]
        print(f"\n{param}: {len(complete)}/{sum(1 for r in rows if r['param'] == param)} complete")
        if complete:
            for ds in DATASETS:
                ds_rows = [r for r in complete if r["dataset"] == ds]
                ds_rows.sort(key=lambda r: r["value"])
                for r in ds_rows:
                    print(f"  {ds}  {param}={r['value']}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}" if r["MAE"] else f"  {ds}  {param}={r['value']}  MISSING")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
