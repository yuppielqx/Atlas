#!/usr/bin/env python3
"""Summarise the METR-LA K=10 two-stage encoder sweep results."""
from __future__ import annotations

import json
import math
from pathlib import Path

ENCODERS = [
    "stid",
    "dlinear",
    "patchtst",
    "nlinear",
    "tsmixer",
    "timesnet",
    "autoformer",
    "informer",
]
RUNS = {
    "stid": ("zinv_ksweep_all_k10_r200", "stage2_k10_r200"),
}


def result_path(run_name: str) -> Path:
    return Path("output") / run_name / "results.json"


def load(run_name: str) -> dict | None:
    path = result_path(run_name)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def metric(result: dict | None, name: str) -> float | None:
    if result is None:
        return None
    value = result.get("metrics", {}).get(name)
    return float(value) if value is not None else None


def scalar(result: dict | None, name: str) -> float | int | None:
    if result is None:
        return None
    value = result.get(name)
    if value is None:
        return None
    return value


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4f}"
    return str(value)


def runs_for(encoder: str) -> tuple[str, str]:
    return RUNS.get(
        encoder,
        (f"encsweep_stage1_{encoder}_k10_r200", f"encsweep_stage2_{encoder}_k10_r200"),
    )


def main() -> None:
    rows = []
    for encoder in ENCODERS:
        stage1_run, stage2_run = runs_for(encoder)
        s1 = load(stage1_run)
        s2 = load(stage2_run)
        rows.append(
            {
                "encoder": encoder,
                "stage1_MAE": metric(s1, "MAE"),
                "stage1_RMSE": metric(s1, "RMSE"),
                "stage1_MAPE": metric(s1, "MAPE"),
                "stage2_MAE": metric(s2, "MAE"),
                "stage2_RMSE": metric(s2, "RMSE"),
                "stage2_MAPE": metric(s2, "MAPE"),
                "stage2_best_val_loss": scalar(s2, "best_val_loss"),
                "stage2_best_round": scalar(s2, "best_round"),
                "stage2_nonfinite": len(s2.get("nonfinite_clients", [])) if s2 else None,
                "stage1_run": stage1_run,
                "stage2_run": stage2_run,
            }
        )

    rows.sort(key=lambda r: (r["stage2_MAE"] is None, r["stage2_MAE"] or float("inf")))
    headers = [
        "encoder",
        "stage1_MAE",
        "stage1_RMSE",
        "stage1_MAPE",
        "stage2_MAE",
        "stage2_RMSE",
        "stage2_MAPE",
        "stage2_best_val_loss",
        "stage2_best_round",
        "stage2_nonfinite",
        "stage1_run",
        "stage2_run",
    ]
    widths = {h: max(len(h), *(len(fmt(r[h])) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(fmt(row[h]).ljust(widths[h]) for h in headers))


if __name__ == "__main__":
    main()
