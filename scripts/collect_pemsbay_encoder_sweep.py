#!/usr/bin/env python3
"""Summarise the PEMS-BAY K=10 two-stage encoder sweep results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ENCODERS = ["stid", "dlinear", "patchtst", "nlinear", "tsmixer"]
# stid reuses the main two-stage baseline runs.
RUNS = {
    "stid": ("pemsbay_stage1_k10_r200", "pemsbay_stage2_k10_r200"),
}


def run_names(enc: str) -> tuple[str, str]:
    if enc in RUNS:
        return RUNS[enc]
    return (
        f"pemsbay_encsweep_stage1_{enc}_k10_r200",
        f"pemsbay_encsweep_stage2_{enc}_k10_r200",
    )


def load_metrics(run_name: str) -> dict | None:
    path = REPO / "output" / run_name / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f).get("metrics", {})


def fmt(metrics: dict | None, key: str):
    if metrics is None:
        return ""
    value = metrics.get(key)
    return f"{value:.4f}" if value is not None else ""


def main() -> None:
    out = REPO / "results" / "encoder_sweep_pemsbay_k10.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for enc in ENCODERS:
        s1_name, s2_name = run_names(enc)
        s1, s2 = load_metrics(s1_name), load_metrics(s2_name)
        rows.append({
            "encoder": enc,
            "stage1_run": s1_name,
            "stage2_run": s2_name,
            "stage1_MAE": fmt(s1, "MAE"),
            "stage2_MAE": fmt(s2, "MAE"),
            "stage1_RMSE": fmt(s1, "RMSE"),
            "stage2_RMSE": fmt(s2, "RMSE"),
            "stage1_MAPE": fmt(s1, "MAPE"),
            "stage2_MAPE": fmt(s2, "MAPE"),
        })

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        print(f"{r['encoder']:>10}  stage1={r['stage1_MAE']:>8}  stage2={r['stage2_MAE']:>8}")


if __name__ == "__main__":
    main()
