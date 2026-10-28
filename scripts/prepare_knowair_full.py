#!/usr/bin/env python3
"""Create a BasicTS-style 18-feature KnowAir dataset from KnowAir.npy.zip.

Output layout (KnowAir-Full/):
  data.dat       [T, N, 18]  float32  — all 17 meteo channels + PM2.5
  desc.json      metadata with feature names and statistics

The 18 channels (verified against PM2.5-GNN paper + ERA5 single-level fields):

  0  z_t2m           z-scored 2m temperature
  1  z_blh           z-scored boundary layer height
  2  t2m             2m temperature (K)
  3  skt             skin temperature (K)
  4  blh             boundary layer height (m)
  5  kx              K-index (stability)
  6  r               relative humidity at surface (%)
  7  r_950           relative humidity at 950 hPa (%)
  8  tp              total precipitation (m)
  9  sp              surface pressure (Pa)
 10  d2m             2m dewpoint temperature (K)
 11  t_950           temperature at 950 hPa (K)
 12  sf              snowfall (m water equiv.)
 13  z_sp            z-scored surface pressure
 14  z_u10           z-scored 10m u-wind
 15  u10             10m u-component of wind (m/s)
 16  v10             10m v-component of wind (m/s)
 17  pm25            PM2.5 concentration (ug/m3)  ← target
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "z_t2m",
    "z_blh",
    "t2m",
    "skt",
    "blh",
    "kx",
    "r",
    "r_950",
    "tp",
    "sp",
    "d2m",
    "t_950",
    "sf",
    "z_sp",
    "z_u10",
    "u10",
    "v10",
    "pm25",
]

FEATURE_DESCRIPTIONS = [
    "z-scored 2m temperature",
    "z-scored boundary layer height",
    "2m temperature (K)",
    "skin temperature (K)",
    "boundary layer height (m)",
    "K-index (atmospheric stability)",
    "relative humidity at surface (%)",
    "relative humidity at 950 hPa (%)",
    "total precipitation (m)",
    "surface pressure (Pa)",
    "2m dewpoint temperature (K)",
    "temperature at 950 hPa (K)",
    "snowfall (m water equivalent)",
    "z-scored surface pressure",
    "z-scored 10m u-wind component",
    "10m u-component of wind (m/s)",
    "10m v-component of wind (m/s)",
    "PM2.5 concentration (ug/m3)",
]

TARGET_INDEX = 17  # PM2.5


def build_time_features(
    num_steps: int, start: datetime, freq_hours: int
) -> np.ndarray:
    """Return [time_of_day, day_of_week] matching the repo format."""
    out = np.empty((num_steps, 2), dtype=np.float32)
    for i in range(num_steps):
        ts = start + timedelta(hours=i * freq_hours)
        seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
        out[i, 0] = seconds / 86400.0
        out[i, 1] = ts.weekday() / 7.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/KnowAir.npy.zip")
    parser.add_argument("--out-dir", default="datasets/KnowAir-Full")
    parser.add_argument("--start", default="2015-01-01T00:00:00+00:00")
    parser.add_argument("--freq-hours", type=int, default=3)
    parser.add_argument("--target-index", type=int, default=TARGET_INDEX)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source) as zf:
        with zf.open("KnowAir.npy") as f:
            raw = np.load(f, allow_pickle=True)

    if raw.ndim != 3:
        raise ValueError(f"expected [T, N, F], got shape={raw.shape}")
    if raw.shape[-1] != 18:
        raise ValueError(f"expected 18 features, got {raw.shape[-1]}")

    T, N, F = raw.shape
    data = raw.astype(np.float32, copy=False)  # [T, N, 18]

    # BasicTS-style desc.json
    desc = {
        "name": "KnowAir-Full",
        "domain": "air quality PM2.5 with full meteorology",
        "shape": list(data.shape),
        "num_time_steps": int(T),
        "num_nodes": int(N),
        "num_features": int(F),
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "target_index": int(args.target_index),
        "target_name": "pm25",
        "target_description": "PM2.5 concentration (ug/m3)",
        "source": str(source),
        "source_target_index": int(args.target_index),
        "start_time_utc": args.start,
        "frequency_minutes": int(args.freq_hours * 60),
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

    # Write data.dat (memmap)
    dat_path = out_dir / "data.dat"
    mem = np.memmap(dat_path, dtype=np.float32, mode="w+", shape=data.shape)
    mem[:] = data[:]
    mem.flush()
    del mem

    # Write desc.json
    with (out_dir / "desc.json").open("w") as f:
        json.dump(desc, f, indent=2)

    # Feature statistics
    print(f"Wrote {dat_path}  shape={data.shape}  dtype=float32")
    print(f"Wrote {out_dir / 'desc.json'}")
    print(f"\nFeature statistics (aggregated over all stations):")
    print(f"{'Idx':>3s} {'name':>8s}  {'mean':>12s}  {'std':>10s}  {'min':>12s}  {'max':>12s}")
    print("-" * 65)
    for i in range(F):
        ch = data[:, :, i]
        print(
            f"{i:3d} {FEATURE_NAMES[i]:>8s}  "
            f"{np.mean(ch):12.4f}  {np.std(ch):10.4f}  "
            f"{np.min(ch):12.4f}  {np.max(ch):12.4f}"
        )

    # PM2.5 per-station summary
    pm25 = data[:, :, TARGET_INDEX]
    station_means = np.mean(pm25, axis=0)
    print(f"\nPM2.5 station means: min={np.min(station_means):.1f}  "
          f"max={np.max(station_means):.1f}  mean={np.mean(station_means):.1f}")
    print(f"Top-5 highest PM2.5 stations (0-indexed): "
          f"{np.argsort(station_means)[-5:][::-1].tolist()}")
    print(f"Top-5 lowest  PM2.5 stations (0-indexed): "
          f"{np.argsort(station_means)[:5].tolist()}")


if __name__ == "__main__":
    main()
