#!/usr/bin/env python3
"""Collect transfer experiment results with source/rest client breakdown.

For each transfer run, reads results.json, identifies which clients were source
vs rest from the corresponding client ID files, and outputs a unified CSV.

Usage:
    python scripts/collect_xfer_results.py
Output:
    results/xfer_all_results.csv
"""

import json, os, csv, glob


def read_results(path):
    rp = os.path.join(path, "results.json")
    if not os.path.isfile(rp):
        return None
    with open(rp) as f:
        d = json.load(f)
    cfg = d.get("config", {})
    met = d.get("metrics", {})
    return {
        "run": cfg.get("run_name", os.path.basename(path)),
        "MAE": met.get("MAE"),
        "RMSE": met.get("RMSE"),
        "MAPE": met.get("MAPE"),
        "per_client_MAE_mean": met.get("per_client_MAE_mean"),
        "per_client_MAE_median": met.get("per_client_MAE_median"),
        "best_val_loss": d.get("best_val_loss"),
        "best_round": d.get("best_round"),
        "dataset": cfg.get("dataset", ""),
    }


def load_client_ids(filepath):
    if not os.path.isfile(filepath):
        return set()
    with open(filepath) as f:
        return set(int(line.strip()) for line in f if line.strip())


def main():
    rows = []

    # ── METR-LA: old correct transfer (xfer_zinv_frac*_to_all) ──
    for frac in ["05", "25", "50", "75", "90", "100"]:
        d = f"output/xfer_zinv_frac{frac}_to_all_k10_r200"
        r = read_results(d)
        if r is None:
            continue
        matches = glob.glob(f"client_ids/metr-la/frac{frac}_n*.txt")
        src_ids = load_client_ids(matches[0]) if matches else set()
        total_clients = 207
        n_src = len(src_ids) if src_ids else 0
        n_rest = total_clients - n_src
        rows.append({**r, "frac": frac, "n_src": n_src,
                     "n_rest": n_rest, "n_total": total_clients})

    # ── METR-LA: native federated baseline ──
    r = read_results("output/zinv_ksweep_all_k10_r200")
    if r:
        rows.append({**r, "frac": "native", "n_src": 0, "n_rest": 207, "n_total": 207})

    # ── PEMS-BAY: fixed transfer (xfer to all 325) ──
    for frac in ["05", "25", "50", "75", "90"]:
        d = f"output/xfer_pemsbay_frac{frac}_to_rest_k10_r200"
        r = read_results(d)
        if r is None:
            rows.append({"run": f"xfer_pemsbay_frac{frac}_to_rest_k10_r200",
                         "dataset": "pems-bay", "frac": frac,
                         "MAE": None, "n_src": 0, "n_rest": 0, "n_total": 325,
                         "status": "not_run"})
            continue
        matches = glob.glob(f"client_ids/pems-bay/frac{frac}_n*.txt")
        src_ids = load_client_ids(matches[0]) if matches else set()
        n_src = len(src_ids) if src_ids else 0
        n_rest = 325 - n_src
        rows.append({**r, "frac": frac, "n_src": n_src,
                     "n_rest": n_rest, "n_total": 325})

    # ── PEMS-BAY: native federated baseline ──
    r = read_results("output/causalprotoenc_pemsbay_k10_r200")
    if r:
        rows.append({**r, "frac": "native", "n_src": 0, "n_rest": 325, "n_total": 325})

    # ── Write CSV ──
    os.makedirs("results", exist_ok=True)
    csv_path = "results/xfer_all_results.csv"
    fieldnames = [
        "dataset", "frac", "n_src", "n_rest", "n_total",
        "MAE", "RMSE", "MAPE", "per_client_MAE_mean", "per_client_MAE_median",
        "best_val_loss", "best_round", "run", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved {csv_path} ({len(rows)} rows)\n")
    hdr = f"{'dataset':<10} {'frac':>8} {'n_src':>5} {'n_rest':>6} {'MAE':>8} {'MAPE':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        m = f"{r['MAE']:>8.4f}" if r.get("MAE") is not None else "     N/A"
        p = f"{r['MAPE']:>6.2f}%" if r.get("MAPE") is not None else "   N/A"
        s = r.get("status", "")
        print(f"{r.get('dataset',''):<10} {r.get('frac',''):>8} {r.get('n_src',0):>5} "
              f"{r.get('n_rest',0):>6} {m} {p}  {s}")


if __name__ == "__main__":
    main()
