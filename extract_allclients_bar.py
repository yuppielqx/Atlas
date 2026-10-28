"""
Extract per-client test MAE from all-clients federated checkpoints,
averaged over the rest-client subsets for each fraction.

This gives the "all-clients bar" — the best these clients could achieve
if they participated in the full federation.

Usage:
    python extract_allclients_bar.py
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import generate_metrla_clients
from federated.trainer import build_clients_and_server
from models.encoders import build_encoder_kwargs
from main import DATASET_DIRS, SAFE_SIGMA

# ── All-clients checkpoints ──
ALL_CKPTS = {
    "metr-la":   "output/zinv_ksweep_all_k10_r200",
    "pems-bay":  "output/pemsbay_stage1_k10_r200",
}

FRACTIONS = ["05", "25", "50", "75", "90"]

CLIENT_IDS_DIR = "client_ids"


def evaluate_checkpoint(run_dir, device="cuda:0"):
    """Load checkpoint, evaluate per-client test MAE, return dict[client_id] = MAE."""
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint.pt in {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dev = torch.device(device)

    dataset_name = DATASET_DIRS[cfg["dataset"]]

    # Regenerate deterministic client data
    clients_data = generate_metrla_clients(
        n_clients=None,
        window_size=cfg["window_size"],
        horizon=cfg["horizon"],
        use_time_features=False,
        dataset_name=dataset_name,
        data_dir=cfg["data_dir"],
        seed=cfg["seed"],
    )

    # Rebuild model
    encoder_kwargs = build_encoder_kwargs(cfg["encoder_type"], cfg.get("encoder_opt", []))
    use_prototype = cfg["fl_mode"] not in {"local", "fedavg", "fedavg_enc", "ditto"}
    use_residual = cfg["use_residual"]
    model_kwargs = dict(
        input_dim=1,
        hid_dim=cfg["hid_dim"],
        seq_len=cfg["window_size"],
        horizon=cfg["horizon"],
        K=cfg["K"],
        tau=cfg["tau"],
        stid_num_layers=cfg["stid_num_layers"],
        dropout=cfg["dropout"],
        use_residual=use_residual,
        residual_mode=cfg["residual_mode"] if use_residual else "none",
        residual_scale=cfg["residual_scale"],
        residual_gate_init=cfg["residual_gate_init"],
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
        weight_decay=cfg["weight_decay"],
        align_mode=cfg["proto_align"],
        align_eps=cfg["proto_align_eps"],
        align_iters=cfg["proto_align_iters"],
        mi_weight=cfg["mi_weight"],
    )

    # Load saved weights
    assert len(clients) == len(ckpt["client_models"]), \
        f"client count mismatch: {len(clients)} vs {len(ckpt['client_models'])}"
    for client, sd in zip(clients, ckpt["client_models"]):
        client.model.load_state_dict(sd)
        client.model.to(dev)
    global_protos = ckpt["server_prototypes"].to(dev)

    # Build test loaders
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

    # Evaluate per-client
    per_client_mae = {}
    per_client_rmse = {}
    for i, client in enumerate(clients):
        cid = clients_data[i]["client_id"]
        client.set_prototypes(global_protos)
        client.model.eval()
        target_mean = clients_data[i]["target_mean"]
        target_std = clients_data[i]["target_std"]

        yt_all, yp_all = [], []
        with torch.no_grad():
            for batch in client.test_loader:
                bx, by = batch[:2]
                bt = batch[2] if len(batch) > 2 else None
                bx = bx.to(dev)
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
            per_client_mae[int(cid)] = float(np.abs(err).mean())
            per_client_rmse[int(cid)] = float(np.sqrt((err ** 2).mean()))

    return per_client_mae, per_client_rmse, cfg


def load_client_ids(filename):
    """Load client IDs from file (one per line)."""
    with open(filename) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    all_results = {}

    for ds_key in ["metr-la", "pems-bay"]:
        run_dir = ALL_CKPTS[ds_key]
        ds_short = {"metr-la": "metrla", "pems-bay": "pemsbay"}[ds_key]

        print(f"=== {ds_key} ({ds_short}) ===")
        print(f"  Loading checkpoint: {run_dir}")
        per_client_mae, per_client_rmse, cfg = evaluate_checkpoint(run_dir, device=device)

        all_ids = sorted(per_client_mae.keys())
        all_mae = np.mean([per_client_mae[cid] for cid in all_ids])
        all_rmse = np.mean([per_client_rmse[cid] for cid in all_ids])
        print(f"  All {len(all_ids)} clients: avg MAE = {all_mae:.4f}, RMSE = {all_rmse:.4f}")
        print()

        all_results[ds_key] = {"all_avg_mae": all_mae, "all_avg_rmse": all_rmse, "fractions": {}}

        for frac in FRACTIONS:
            rest_file = os.path.join(CLIENT_IDS_DIR, ds_key, f"rest_frac{frac}_n*.txt")
            import glob as gmod
            matches = sorted(gmod.glob(rest_file))
            if not matches:
                print(f"  frac {frac}: no rest file found ({rest_file})")
                continue
            rest_file = matches[0]

            rest_ids = load_client_ids(rest_file)
            maes, rmses, missing = [], [], []
            for cid in rest_ids:
                if cid in per_client_mae:
                    maes.append(per_client_mae[cid])
                    rmses.append(per_client_rmse[cid])
                else:
                    missing.append(cid)
            if missing:
                print(f"  frac {frac}: WARNING {len(missing)} client IDs not found: {missing[:5]}...")

            avg_mae = np.mean(maes) if maes else float('nan')
            avg_rmse = np.mean(rmses) if rmses else float('nan')
            tag = f"randbase_{ds_short}_frac{frac}_to_rest_k10_r200"
            print(f"  frac {frac}: {len(rest_ids)} rest clients, avg MAE = {avg_mae:.4f}, RMSE = {avg_rmse:.4f}  (→ bar for {tag})")
            all_results[ds_key]["fractions"][frac] = {
                "n_rest": len(rest_ids),
                "avg_mae": avg_mae,
                "avg_rmse": avg_rmse,
            }

        print()

    # ── Summary table ──
    print("=" * 80)
    print("SUMMARY: All-clients bar (MAE/RMSE from full federation, rest-client subset only)")
    print("=" * 90)
    print(f"{'Dataset':12} {'Frac':>6} {'N_rest':>7} {'MAE':>10} {'RMSE':>10}")
    print("-" * 50)
    for ds_key in ["metr-la", "pems-bay"]:
        ds_short = {"metr-la": "metrla", "pems-bay": "pemsbay"}[ds_key]
        all_mae = all_results[ds_key]["all_avg_mae"]
        all_rmse = all_results[ds_key]["all_avg_rmse"]
        print(f"{ds_short:12} {'ALL':>6} {'-':>7} {all_mae:10.4f} {all_rmse:10.4f}")
        for frac in FRACTIONS:
            if frac in all_results[ds_key]["fractions"]:
                info = all_results[ds_key]["fractions"][frac]
                print(f"{'':12} {frac:>6} {info['n_rest']:>7} {info['avg_mae']:10.4f} {info['avg_rmse']:10.4f}")

    # Save JSON
    out = {
        "description": "Per-client test MAE from all-clients federated checkpoint, averaged over rest-client subsets",
        "checkpoints": ALL_CKPTS,
        "results": all_results,
    }
    out_path = "output/allclients_bar_rest_subsets.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
