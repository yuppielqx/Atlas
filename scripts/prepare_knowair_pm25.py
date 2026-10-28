#!/usr/bin/env python3
"""Create a BasicTS/METR-LA-style PM2.5-only dataset from KnowAir.npy.zip."""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def build_time_features(num_steps: int, start: datetime, freq_hours: int) -> np.ndarray:
    """Return [time_of_day, day_of_week] features matching the repo format."""
    out = np.empty((num_steps, 2), dtype=np.float32)
    for i in range(num_steps):
        ts = start + timedelta(hours=i * freq_hours)
        seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
        out[i, 0] = seconds / 86400.0
        out[i, 1] = ts.weekday() / 7.0  # Monday=0, same convention as BasicTS data here.
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/KnowAir.npy.zip")
    parser.add_argument("--out-dir", default="datasets/KnowAir-PM25")
    parser.add_argument("--start", default="2015-01-01T00:00:00+00:00")
    parser.add_argument("--freq-hours", type=int, default=3)
    parser.add_argument("--target-index", type=int, default=17)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source) as zf:
        with zf.open("KnowAir.npy") as f:
            raw = np.load(f)

    if raw.ndim != 3:
        raise ValueError(f"expected KnowAir array [T, N, F], got shape={raw.shape}")
    if not 0 <= args.target_index < raw.shape[-1]:
        raise ValueError(f"target index {args.target_index} out of range for shape={raw.shape}")

    pm25 = raw[:, :, args.target_index].astype(np.float32, copy=False)
    if not np.isfinite(pm25).all():
        raise ValueError("PM2.5 channel contains NaN/Inf; clean it before export")

    start = datetime.fromisoformat(args.start)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    time_features = build_time_features(pm25.shape[0], start, args.freq_hours)

    # METR-LA/BasicTS-style data.dat: [time, node, (target, time_of_day, day_of_week)].
    data = np.empty((pm25.shape[0], pm25.shape[1], 3), dtype=np.float32)
    data[:, :, 0] = pm25
    data[:, :, 1] = time_features[:, 0:1]
    data[:, :, 2] = time_features[:, 1:2]

    dat_path = out_dir / "data.dat"
    mem = np.memmap(dat_path, dtype=np.float32, mode="w+", shape=data.shape)
    mem[:] = data[:]
    mem.flush()
    del mem

    desc = {
        "name": "KnowAir-PM25",
        "domain": "air quality PM2.5",
        "shape": list(data.shape),
        "num_time_steps": int(data.shape[0]),
        "num_nodes": int(data.shape[1]),
        "num_features": int(data.shape[2]),
        "feature_description": ["PM2.5", "time of day", "day of week"],
        "source": str(source),
        "source_shape": list(raw.shape),
        "source_target_index": int(args.target_index),
        "source_target_description": "PM2.5 concentration",
        "start_time_utc": start.isoformat(),
        "frequency (minutes)": int(args.freq_hours * 60),
        "has_graph": False,
        "regular_settings": {
            "INPUT_LEN": 12,
            "OUTPUT_LEN": 12,
            "TRAIN_VAL_TEST_RATIO": [0.7, 0.1, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0,
        },
    }
    with (out_dir / "desc.json").open("w") as f:
        json.dump(desc, f, indent=4)

    print(f"Wrote {dat_path} with shape {data.shape} dtype=float32")
    print(f"Wrote {out_dir / 'desc.json'}")
    print(
        "PM2.5 stats: "
        f"mean={pm25.mean():.4f} std={pm25.std():.4f} "
        f"min={pm25.min():.4f} max={pm25.max():.4f}"
    )


if __name__ == "__main__":
    main()
