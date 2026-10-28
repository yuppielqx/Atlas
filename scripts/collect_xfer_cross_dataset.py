#!/usr/bin/env python3
"""Summarise the cross-dataset codebook-transfer runs (PEMS-BAY <-> METR-LA)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# target dataset -> list of (row label, zinv run, stage2 run)
TABLE = {
    "metr-la": [
        ("cross: Q from PEMS-BAY", "xfer_cross_bay2la_zinv_k10_r200", "xfer_cross_bay2la_stage2_k10_r200"),
        ("self anchor: frozen own Q", "xfer_zinv_frac100_to_all_k10_r200", "xfer_stage2_frac100_to_all_k10_r200"),
        ("native (no transfer)", "zinv_ksweep_all_k10_r200", "stage2_k10_r200"),
        ("local_proto (per-client Q)", "stage1_localproto_enc_dec_k10_r200", None),
    ],
    "pems-bay": [
        ("cross: Q from METR-LA", "xfer_cross_la2bay_zinv_k10_r200", "xfer_cross_la2bay_stage2_k10_r200"),
        ("self anchor: frozen own Q", "xfer_self_bay_zinv_k10_r200", "xfer_self_bay_stage2_k10_r200"),
        ("native (no transfer)", "pemsbay_stage1_k10_r200", "pemsbay_stage2_k10_r200"),
        ("local_proto (per-client Q)", "pemsbay_stage1_localproto_enc_dec_k10_r200", None),
    ],
}


def load_metrics(run_name: str | None) -> dict | None:
    if run_name is None:
        return None
    path = REPO / "output" / run_name / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f).get("metrics", {})


def fmt(metrics: dict | None, key: str = "MAE"):
    if metrics is None:
        return ""
    value = metrics.get(key)
    return f"{value:.4f}" if value is not None else ""


def main() -> None:
    out = REPO / "results" / "xfer_cross_dataset.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for target, entries in TABLE.items():
        for label, zinv_run, s2_run in entries:
            zinv, s2 = load_metrics(zinv_run), load_metrics(s2_run)
            rows.append({
                "target_dataset": target,
                "row": label,
                "zinv_MAE": fmt(zinv),
                "stage2_MAE": fmt(s2),
                "zinv_RMSE": fmt(zinv, "RMSE"),
                "stage2_RMSE": fmt(s2, "RMSE"),
            })

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        print(f"{r['target_dataset']:>9} | {r['row']:<28} zinv={r['zinv_MAE']:>8}  stage2={r['stage2_MAE']:>8}")


if __name__ == "__main__":
    main()
