#!/usr/bin/env python3
"""
Compare per-client MAE of source-fraction clients between:
  1. Source-only checkpoint (trained with only k% clients)
  2. All-clients checkpoint (full federation, as a bar)

Usage:
    python analysis/compare_source_client_mae.py
    python analysis/compare_source_client_mae.py --device cuda:1
"""

import os, sys, json, argparse, glob as gmod
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import generate_metrla_clients
from federated.trainer import build_clients_and_server
from models.encoders import build_encoder_kwargs
from main import DATASET_DIRS, SAFE_SIGMA

# ── Config ──────────────────────────────────────────────────────────────
ALL_CKPTS = {
    "metr-la":  "output/zinv_ksweep_all_k10_r200",
    "pems-bay": "output/pemsbay_stage1_k10_r200",
}

SOURCE_RUNS = {
    "metr-la": {
        "05": "output/zinv_src_frac05_n10_k10_r200",
        "25": "output/zinv_src_frac25_n52_k10_r200",
        "50": "output/zinv_src_frac50_n104_k10_r200",
        "75": "output/zinv_src_frac75_n155_k10_r200",
        "90": "output/zinv_src_frac90_n186_k10_r200",
    },
    "pems-bay": {
        "05": "output/pemsbay_src_nores_frac05_n16_k10_r200",
        "25": "output/pemsbay_src_nores_frac25_n81_k10_r200",
        "50": "output/pemsbay_src_nores_frac50_n162_k10_r200",
        "75": "output/pemsbay_src_nores_frac75_n244_k10_r200",
        "90": "output/pemsbay_src_nores_frac90_n292_k10_r200",
    },
}

FRACTIONS = ["05", "25", "50", "75", "90"]
CLIENT_IDS_DIR = "client_ids"


# ── Helpers ─────────────────────────────────────────────────────────────

def _read_client_ids(filename):
    """Read client IDs from file (one per line or comma/whitespace separated)."""
    with open(filename) as f:
        text = f.read().strip()
    ids = []
    for token in text.replace(",", " ").split():
        token = token.strip()
        if token:
            ids.append(int(token))
    return ids


def _parse_n_src(run_dir):
    """Extract n_src from run dir name like '..._n104_k10_r200'."""
    import re
    m = re.search(r'_n(\d+)_', os.path.basename(run_dir))
    return int(m.group(1)) if m else None


# ── Core evaluation ─────────────────────────────────────────────────────

def evaluate_checkpoint(run_dir, device, client_ids=None):
    """Load checkpoint, rebuild model+data, evaluate per-client test MAE.

    Args:
        run_dir: path to output directory containing checkpoint.pt
        device: torch device
        client_ids: if provided, only load+eval these clients; if None, load all

    Returns:
        dict[int client_id -> float MAE (raw units)]
    """
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint.pt in {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dev = torch.device(device)

    # ── Rebuild client data ──
    clients_data = generate_metrla_clients(
        n_clients=None if client_ids is None else None,
        client_ids=client_ids,
        window_size=cfg["window_size"],
        horizon=cfg["horizon"],
        use_time_features=False,
        dataset_name=DATASET_DIRS[cfg["dataset"]],
        data_dir=cfg["data_dir"],
        seed=cfg["seed"],
    )

    # ── Rebuild model ──
    encoder_kwargs = build_encoder_kwargs(cfg["encoder_type"], cfg.get("encoder_opt", []))
    use_prototype = cfg["fl_mode"] not in {"local", "fedavg", "fedavg_enc", "ditto"}
    use_residual = cfg.get("use_residual", False)

    model_kwargs = dict(
        input_dim=1,
        hid_dim=cfg["hid_dim"],
        seq_len=cfg["window_size"],
        horizon=cfg["horizon"],
        K=cfg["K"],
        tau=cfg["tau"],
        stid_num_layers=cfg.get("stid_num_layers", 3),
        dropout=cfg.get("dropout", 0.1),
        use_residual=use_residual,
        residual_mode=cfg.get("residual_mode", "none") if use_residual else "none",
        residual_scale=cfg.get("residual_scale", 1.0),
        residual_gate_init=cfg.get("residual_gate_init", -2.0),
        encoder_type=cfg["encoder_type"],
        encoder_kwargs=encoder_kwargs,
        use_prototype=use_prototype,
    )

    clients, server = build_clients_and_server(
        clients_data=clients_data,
        model_kwargs=model_kwargs,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        local_epochs=cfg["local_epochs"],
        device=dev,
        weight_decay=cfg.get("weight_decay", 0.0),
        align_mode=cfg.get("proto_align", "none"),
        align_eps=cfg.get("proto_align_eps", 0.05),
        align_iters=cfg.get("proto_align_iters", 10),
        mi_weight=cfg.get("mi_weight", 0.0),
    )

    # ── Load saved weights ──
    assert len(clients) == len(ckpt["client_models"]), \
        f"client count mismatch: {len(clients)} vs {len(ckpt['client_models'])}"
    for client, sd in zip(clients, ckpt["client_models"]):
        client.model.load_state_dict(sd)
        client.model.to(dev)
    global_protos = ckpt["server_prototypes"].to(dev)

    # ── Build test loaders ──
    for i, client in enumerate(clients):
        test_tensors = [
            torch.from_numpy(clients_data[i]["X_test"]),
            torch.from_numpy(clients_data[i]["Y_test"]),
        ]
        if "T_test" in clients_data[i]:
            test_tensors.append(torch.from_numpy(clients_data[i]["T_test"]))
        client.test_loader = DataLoader(
            TensorDataset(*test_tensors), batch_size=cfg["batch_size"], shuffle=False
        )

    # ── Evaluate per-client ──
    per_client_mae = {}
    per_client_rmse = {}
    for i, client in enumerate(clients):
        cid = int(clients_data[i]["client_id"])
        client.set_prototypes(global_protos)
        client.model.eval()
        target_mean = clients_data[i]["target_mean"]
        target_std = clients_data[i]["target_std"]

        yt_all, yp_all = [], []
        with torch.no_grad():
            for batch in client.test_loader:
                bx, by = batch[:2]
                bt = batch[2] if len(batch) > 2 else None
                bx, by = bx.to(dev), by.to(dev)
                if bt is not None:
                    bt = bt.to(dev)
                yh, _, _ = client.model(bx, temporal_id=bt)
                yt_all.append(by[..., :1].cpu().numpy())
                yp_all.append(yh.cpu().numpy())

        yt = np.concatenate(yt_all, axis=0)
        yp = np.concatenate(yp_all, axis=0)
        yp = np.clip(yp, -SAFE_SIGMA, SAFE_SIGMA)
        yt_raw = yt * target_std + target_mean
        yp_raw = yp * target_std + target_mean
        mask = np.isfinite(yt_raw) & np.isfinite(yp_raw) & (yt_raw > 0.1)
        if mask.sum() > 0:
            err = yt_raw[mask] - yp_raw[mask]
            per_client_mae[cid] = float(np.abs(err).mean())
            per_client_rmse[cid] = float(np.sqrt((err ** 2).mean()))

    return per_client_mae, per_client_rmse, cfg


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    all_rows = []

    for ds_key in ["metr-la", "pems-bay"]:
        ds_short = {"metr-la": "metrla", "pems-bay": "pemsbay"}[ds_key]
        all_run = ALL_CKPTS[ds_key]

        # ── Step 1: Evaluate all-clients checkpoint once ──
        print(f"{'='*70}")
        print(f"  {ds_key} ({ds_short}) — All-clients checkpoint")
        print(f"{'='*70}")
        print(f"  Loading: {all_run}")
        all_mae, all_rmse, all_cfg = evaluate_checkpoint(all_run, device, client_ids=None)
        all_mean = np.mean(list(all_mae.values()))
        print(f"  {len(all_mae)} clients evaluated, mean MAE = {all_mean:.4f}\n")

        # Log architecture info for transparency
        all_use_res = all_cfg.get("use_residual", False)
        print(f"  All-clients use_residual = {all_use_res}")

        # ── Step 2: Evaluate each source fraction ──
        for frac in FRACTIONS:
            src_run = SOURCE_RUNS[ds_key][frac]

            # Find source client ID file
            src_id_files = sorted(gmod.glob(
                os.path.join(CLIENT_IDS_DIR, ds_key, f"frac{frac}_n*.txt")))
            if not src_id_files:
                print(f"  frac {frac}: no source client ID file found, skip")
                continue
            src_ids = _read_client_ids(src_id_files[0])
            n_src = _parse_n_src(src_run)

            # Evaluate source checkpoint
            print(f"\n  frac {frac} (n_src={len(src_ids)}): {src_run}")
            src_mae, src_rmse, src_cfg = evaluate_checkpoint(src_run, device, client_ids=src_ids)
            src_use_res = src_cfg.get("use_residual", False)

            if src_use_res != all_use_res:
                print(f"  ⚠  Architecture differs: source use_residual={src_use_res}, "
                      f"all-clients use_residual={all_use_res}")

            # Build comparison
            src_mean_mae = np.mean(list(src_mae.values()))
            src_mean_rmse = np.mean(list(src_rmse.values()))
            all_subset_mae = [all_mae[cid] for cid in src_ids if cid in all_mae]
            all_subset_rmse = [all_rmse[cid] for cid in src_ids if cid in all_rmse]
            all_subset_mean_mae = np.mean(all_subset_mae) if all_subset_mae else float("nan")
            all_subset_mean_rmse = np.mean(all_subset_rmse) if all_subset_rmse else float("nan")
            missing = set(src_ids) - set(all_mae.keys())
            if missing:
                print(f"  ⚠  {len(missing)} client IDs missing from all-clients eval")

            delta_mae = src_mean_mae - all_subset_mean_mae
            n_better = sum(1 for cid in src_ids
                           if cid in all_mae and all_mae[cid] < src_mae.get(cid, float("inf")))
            n_eval = len([cid for cid in src_ids if cid in all_mae])

            print(f"  Source-only MAE:      mean = {src_mean_mae:.4f}")
            print(f"  All-clients MAE (bar): mean = {all_subset_mean_mae:.4f}  "
                  f"(Δ = {delta_mae:+.4f})")
            print(f"  Source-only RMSE:      mean = {src_mean_rmse:.4f}")
            print(f"  All-clients RMSE (bar): mean = {all_subset_mean_rmse:.4f}  "
                  f"(Δ = {src_mean_rmse - all_subset_mean_rmse:+.4f})")
            print(f"  Federation improved:  {n_better}/{n_eval} clients "
                  f"({100*n_better/max(1,n_eval):.0f}%)")

            # Collect per-client rows
            for cid in sorted(src_ids):
                s_mae = src_mae.get(cid, float("nan"))
                a_mae = all_mae.get(cid, float("nan"))
                s_rmse = src_rmse.get(cid, float("nan"))
                a_rmse = all_rmse.get(cid, float("nan"))
                all_rows.append({
                    "dataset": ds_short,
                    "frac": frac,
                    "n_src": n_src or len(src_ids),
                    "client_id": cid,
                    "source_mae": round(s_mae, 6),
                    "allclients_mae": round(a_mae, 6),
                    "delta_mae": round(s_mae - a_mae, 6) if not (np.isnan(s_mae) or np.isnan(a_mae)) else float("nan"),
                    "source_rmse": round(s_rmse, 6),
                    "allclients_rmse": round(a_rmse, 6),
                    "delta_rmse": round(s_rmse - a_rmse, 6) if not (np.isnan(s_rmse) or np.isnan(a_rmse)) else float("nan"),
                })

        print()

    # ── Save CSV ──
    df = pd.DataFrame(all_rows)
    out_path = "analysis/source_client_mae_comparison.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved per-client CSV: {out_path}  ({len(df)} rows)")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    summary = df.groupby(["dataset", "frac", "n_src"]).agg(
        mean_source_mae=("source_mae", "mean"),
        mean_allclients_mae=("allclients_mae", "mean"),
        mean_delta_mae=("delta_mae", "mean"),
        median_source_mae=("source_mae", "median"),
        median_allclients_mae=("allclients_mae", "median"),
        mean_source_rmse=("source_rmse", "mean"),
        mean_allclients_rmse=("allclients_rmse", "mean"),
        mean_delta_rmse=("delta_rmse", "mean"),
        median_source_rmse=("source_rmse", "median"),
        median_allclients_rmse=("allclients_rmse", "median"),
        n_clients=("client_id", "count"),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_path = "analysis/source_client_mae_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
