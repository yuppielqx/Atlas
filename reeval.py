"""
Re-evaluate finished runs from their saved checkpoints, recomputing the
raw-scale metrics with the corrected safety clip (normalized-space ±SAFE_SIGMA
instead of the old hardcoded raw ±100 that truncated PEMS03/PEMS04 flow).

No retraining: rebuilds each run's model from config, loads the saved
per-client state_dicts + shared prototype, regenerates the (deterministic) test
loaders, and recomputes MAE/RMSE/MAPE. Writes results back into each run dir as
results_fixed.json and prints a before/after table.

    python reeval.py output/pems04__stid__causal_proto [more dirs...]
    python reeval.py --glob 'output/pems0[34]__*'        # all pems03/04 runs
"""

import os
import sys
import json
import glob as globmod
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import generate_metrla_clients
from federated.trainer import build_clients_and_server
from utils.metrics import compute_all_metrics
from models.encoders import build_encoder_kwargs

from main import DATASET_DIRS, SAFE_SIGMA


def reeval_run(run_dir, device="cuda:0"):
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint.pt in {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dev = torch.device(device)

    # ── Regenerate the same (deterministic) client data / test loaders ──
    n_clients = cfg["n_clients"] if cfg["n_clients"] and cfg["n_clients"] > 0 else None
    clients_data = generate_metrla_clients(
        n_clients=n_clients,
        window_size=cfg["window_size"],
        horizon=cfg["horizon"],
        use_time_features=False,
        dataset_name=DATASET_DIRS[cfg["dataset"]],
        data_dir=cfg["data_dir"],
        seed=cfg["seed"],
    )

    # ── Rebuild the model exactly as main.py did ──
    encoder_kwargs = build_encoder_kwargs(cfg["encoder_type"], cfg.get("encoder_opt", []))
    use_prototype = cfg["fl_mode"] not in {"local", "fedavg", "fedavg_enc"}
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

    # Load saved weights + shared prototype.
    assert len(clients) == len(ckpt["client_models"]), "client count mismatch"
    for client, sd in zip(clients, ckpt["client_models"]):
        client.model.load_state_dict(sd)
        client.model.to(dev)
    global_protos = ckpt["server_prototypes"].to(dev)

    from torch.utils.data import DataLoader, TensorDataset
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

    # ── Evaluation with the corrected normalized-space safety clip ──
    all_y_true, all_y_pred = [], []
    for i, client in enumerate(clients):
        client.set_prototypes(global_protos)
        client.model.eval()
        target_mean = clients_data[i]["target_mean"]
        target_std = clients_data[i]["target_std"]

        cyt, cyp = [], []
        with torch.no_grad():
            for batch in client.test_loader:
                bx, by = batch[:2]
                bt = batch[2] if len(batch) > 2 else None
                bx = bx.to(dev)
                if bt is not None:
                    bt = bt.to(dev)
                yh, _, _ = client.model(bx, temporal_id=bt)
                cyt.append(by[..., :1].cpu().numpy())
                cyp.append(yh.cpu().numpy())

        yt = np.concatenate(cyt, axis=0)
        yp = np.concatenate(cyp, axis=0)
        yp = np.clip(yp, -SAFE_SIGMA, SAFE_SIGMA)        # <-- the fix
        all_y_true.append(yt * target_std + target_mean)
        all_y_pred.append(yp * target_std + target_mean)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.1)
    metrics = compute_all_metrics(y_true, y_pred, mask=mask)

    per_client_mae = []
    for yt_raw, yp_raw in zip(all_y_true, all_y_pred):
        m = np.isfinite(yt_raw) & np.isfinite(yp_raw) & (yt_raw > 0.1)
        if m.sum() > 0:
            per_client_mae.append(float(np.abs(yt_raw[m] - yp_raw[m]).mean()))
    if per_client_mae:
        pca = np.array(per_client_mae)
        metrics["per_client_MAE_mean"] = float(pca.mean())
        metrics["per_client_MAE_median"] = float(np.median(pca))

    metrics = {k: float(v) for k, v in metrics.items()}
    out = {
        "config": cfg,
        "metrics": metrics,
        "metrics_old": ckpt.get("metrics", {}),
        "reeval": True,
        "safe_sigma": SAFE_SIGMA,
    }
    with open(os.path.join(run_dir, "results_fixed.json"), "w") as f:
        json.dump(out, f, indent=2)
    return ckpt.get("metrics", {}), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="run directories under output/")
    ap.add_argument("--glob", default=None, help="glob pattern for run dirs")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dirs = list(args.run_dirs)
    if args.glob:
        dirs += [d for d in globmod.glob(args.glob) if os.path.isdir(d)]
    dirs = sorted(set(d.rstrip("/") for d in dirs))
    if not dirs:
        print("no run dirs given"); return

    print(f"{'run':45} {'MAE_old':>9} {'MAE_new':>9} {'RMSE_new':>9} {'MAPE_new':>9}")
    for d in dirs:
        try:
            old, new = reeval_run(d, device=args.device)
        except Exception as e:
            print(f"{os.path.basename(d):45} ERROR {type(e).__name__}: {e}")
            continue
        print(f"{os.path.basename(d):45} {old.get('MAE', float('nan')):9.3f} "
              f"{new['MAE']:9.3f} {new['RMSE']:9.3f} {new['MAPE']:9.3f}")


if __name__ == "__main__":
    main()
