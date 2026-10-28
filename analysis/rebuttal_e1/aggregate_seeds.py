#!/usr/bin/env python3
"""Aggregate E1 independent checkpoint analyses without matching slot labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", type=Path, help="one E1 summary.json per checkpoint")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = [json.loads(path.read_text()) for path in args.summaries]
    if len({d["checkpoint_sha256"] for d in data}) != len(data):
        raise ValueError("duplicate checkpoint supplied")
    for key in ("split_bounds_half_open", "weather_thresholds_train_only",
                "n_clients", "probe_train_stride"):
        if any(d[key] != data[0][key] for d in data[1:]):
            raise ValueError(f"seed analyses use different {key}")
    records = []
    for path, d in zip(args.summaries, data):
        base, aug = d["linear_control"], d["linear_control_plus_prototypes"]
        row = {
            "checkpoint": d["checkpoint"],
            "seed": d["checkpoint_config"].get("seed"),
            "n_train_probe_windows": d["n_train_probe_windows"],
            "n_test_windows": d["n_test_windows"],
            "test_high_precip_rate": d["high_precip_test_rate"],
            "linear_auc_control": base["auc"],
            "linear_auc_plus_slots": aug["auc"],
            "linear_auc_gain": aug["auc"] - base["auc"],
            "linear_brier_control": base["brier"],
            "linear_brier_plus_slots": aug["brier"],
            "linear_brier_gain": base["brier"] - aug["brier"],
            "effective_slots_exp_entropy": d["effective_slots_exp_entropy"],
        }
        assoc = d.get("matched_conditional_slot_association")
        if assoc:
            row["max_abs_matched_slot_weight_difference"] = assoc["max_abs_slot_effect"]
            row["max_slot_exploratory_fwer_p"] = assoc["max_slot_fwer_p_exploratory"]
            row["n_mixed_weather_strata"] = assoc["n_mixed_weather_strata"]
        proto_csv = path.parent / "per_prototype.csv"
        if proto_csv.exists():
            proto = pd.read_csv(proto_csv)
            row["max_hard_precip_enrichment"] = float(proto["high_precip_hard_enrichment"].max())
        n = d.get("nonlinear_control_comparison")
        if n:
            row["nonlinear_auc_control"] = n["control"]["auc"]
            row["nonlinear_auc_plus_slots"] = n["control_plus_prototypes"]["auc"]
            row["nonlinear_auc_gain"] = row["nonlinear_auc_plus_slots"] - row["nonlinear_auc_control"]
            row["nonlinear_brier_gain"] = n["control"]["brier"] - n["control_plus_prototypes"]["brier"]
        records.append(row)
    frame = pd.DataFrame(records).sort_values("seed")
    frame.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary = {
        "n_independent_checkpoints": len(records),
        "thresholds_train_only": data[0]["weather_thresholds_train_only"],
        "note": "Slot labels are permutation-invariant across independently trained seeds; per-slot rows are not aligned or averaged. Max-slot enrichment is selected post hoc and descriptive.",
        "metrics": {},
    }
    for col in frame.select_dtypes(include="number").columns:
        if col in {"seed", "n_train_probe_windows", "n_test_windows",
                   "n_mixed_weather_strata"}:
            continue
        v = frame[col].to_numpy(dtype=float)
        summary["metrics"][col] = {
            "mean": float(v.mean()),
            "sample_sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "min": float(v.min()), "max": float(v.max()),
            "values_by_seed": dict(zip(frame["seed"].astype(str), v.tolist())),
        }
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print(frame.to_string(index=False))
    print(f"Saved {args.output_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
