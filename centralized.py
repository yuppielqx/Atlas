"""
Centralized (pooled-data) baseline for CausalFedST.

Instead of federated training (one Client per node + aggregated prototypes),
pool ALL nodes' windows and train ONE STID + prototype + predictor model under a
single global scaler, then evaluate it per node (same masked metrics as the
federated runs). This is the centralized upper-bound baseline.

    python centralized.py --dataset metr-la --encoder_type stid --n_clients -1

Federated training (main.py) is untouched.
"""

import os
import sys
import json
import copy
import argparse
from datetime import datetime


import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_matplotlib_{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import generate_centralized_data
from models.causal_fedst import CausalFedST
from models.encoders import build_encoder_kwargs
from utils.metrics import compute_all_metrics
from main import DATASET_DIRS, str2bool   # reuse dataset table + bool parser


def parse_args():
    p = argparse.ArgumentParser(description="CausalFedST centralized baseline")
    p.add_argument("--dataset", type=str, default="metr-la", choices=sorted(DATASET_DIRS))
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--n_clients", type=int, default=-1,
                   help="number of nodes to pool; <=0 = ALL nodes")
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--hid_dim", type=int, default=64)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--encoder_type", type=str, default="stid")
    p.add_argument("--encoder_opt", action="append", default=[], metavar="KEY=VAL",
                   help="encoder-specific hyperparameter override (repeatable)")
    p.add_argument("--stid_num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_residual", type=str2bool, default=True)
    p.add_argument("--residual_mode", type=str, default="fixed",
                   choices=["none", "fixed", "learnable", "detach", "detach_fixed"])
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--central_epochs", type=int, default=30,
                   help="epochs over the pooled dataset (much fewer than FL rounds)")
    p.add_argument("--lr_scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--patience", type=int, default=8, help="early-stop patience (<=0 disables)")
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def masked_mae_norm(y_hat, y, g_mean, g_std, thr=0.1):
    """Masked MAE in normalized space; mask drops raw targets <= thr (NULL_VAL=0)."""
    y_raw = y * g_std + g_mean
    mask = (y_raw > thr).float()
    err = torch.abs(y_hat - y) * mask
    return err.sum() / mask.sum().clamp(min=1.0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or (
        f"central_{args.dataset}_{args.encoder_type}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    n_clients = args.n_clients if args.n_clients and args.n_clients > 0 else None
    dataset_dir = DATASET_DIRS[args.dataset]

    print("=" * 60)
    print("CausalFedST — CENTRALIZED baseline")
    print(f"  Dataset: {args.dataset} | nodes: {'ALL' if n_clients is None else n_clients}")
    print(f"  Encoder: {args.encoder_type} | hid_dim {args.hid_dim} | K {args.K} | prototype: True")
    print(f"  Epochs: {args.central_epochs} | batch {args.batch_size} | lr {args.lr}")
    print(f"  Output: {run_dir}")
    print("=" * 60)

    print(f"\n[1/3] Loading + pooling {dataset_dir} (global scaler)...")
    data = generate_centralized_data(
        n_clients=n_clients, window_size=args.window_size, horizon=args.horizon,
        dataset_name=dataset_dir, data_dir=args.data_dir, seed=args.seed,
    )
    g_mean, g_std = data["target_mean"], data["target_std"]
    n_nodes = len(data["test_nodes"])
    print(f"  Nodes pooled: {n_nodes} | global mean {g_mean:.4f} std {g_std:.4f}")
    print(f"  Pooled train windows: {data['X_train'].shape[0]:,} | "
          f"val: {data['X_val'].shape[0]:,}")

    def make_loader(X, Y, T, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(T)),
            batch_size=args.batch_size, shuffle=shuffle,
        )

    train_loader = make_loader(data["X_train"], data["Y_train"], data["T_train"], True)
    val_loader = make_loader(data["X_val"], data["Y_val"], data["T_val"], False)

    # ── Build one model ──
    print("\n[2/3] Building model and training...")
    use_residual = args.use_residual
    model_kwargs = dict(
        input_dim=1, hid_dim=args.hid_dim, seq_len=args.window_size, horizon=args.horizon,
        K=args.K, tau=args.tau, stid_num_layers=args.stid_num_layers, dropout=args.dropout,
        use_residual=use_residual, residual_mode=(args.residual_mode if use_residual else "none"),
        residual_scale=args.residual_scale, encoder_type=args.encoder_type,
        encoder_kwargs=build_encoder_kwargs(args.encoder_type, args.encoder_opt),
        use_prototype=True,
    )
    model = CausalFedST(**model_kwargs).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-6)
             if args.lr_scheduler == "plateau" else None)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_round, best_state, no_imp = float("inf"), -1, None, 0
    for epoch in range(args.central_epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y, t in train_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            opt.zero_grad(set_to_none=True)
            y_hat, _, _ = model(x, temporal_id=t)
            loss = masked_mae_norm(y_hat, y[..., :1], g_mean, g_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            model.normalize_prototypes_()
            tot += loss.item(); nb += 1
        train_loss = tot / max(nb, 1)

        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for x, y, t in val_loader:
                x, y, t = x.to(device), y.to(device), t.to(device)
                y_hat, _, _ = model(x, temporal_id=t)
                vtot += masked_mae_norm(y_hat, y[..., :1], g_mean, g_std).item(); vnb += 1
        val_loss = vtot / max(vnb, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step(val_loss)

        if val_loss < best_val - args.min_delta:
            best_val, best_round, no_imp = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        print(f"  [E {epoch}] train {train_loss:.4f} | val {val_loss:.4f} | "
              f"best {best_val:.4f}@{best_round} (no_imp {no_imp})")
        if args.patience and args.patience > 0 and no_imp >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_round"] = best_round
    history["best_val_loss"] = best_val

    # ── Per-node evaluation (single shared model) ──
    print("\n[3/3] Per-node evaluation...")
    model.eval()
    all_y_true, all_y_pred, per_node_mae, per_node_records = [], [], [], []
    for node in data["test_nodes"]:
        loader = make_loader(node["X_test"], node["Y_test"], node["T_test"], False)
        yts, yps = [], []
        with torch.no_grad():
            for x, y, t in loader:
                x, t = x.to(device), t.to(device)
                y_hat, _, _ = model(x, temporal_id=t)
                yts.append(y[..., :1].numpy())
                yps.append(y_hat.cpu().numpy())
        yt = np.concatenate(yts, axis=0)
        yp = np.clip(np.concatenate(yps, axis=0), -100.0, 100.0)
        yt_raw = yt * g_std + g_mean
        yp_raw = yp * g_std + g_mean
        all_y_true.append(yt_raw); all_y_pred.append(yp_raw)
        m = np.isfinite(yt_raw) & np.isfinite(yp_raw) & (yt_raw > 0.1)
        if m.sum() > 0:
            mae_i = float(np.abs(yt_raw[m] - yp_raw[m]).mean())
            per_node_mae.append(mae_i)
            per_node_records.append({"client_id": node["client_id"], "raw_mae": mae_i})

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.1)
    metrics = compute_all_metrics(y_true, y_pred, mask=mask)

    print("\n" + "=" * 60)
    print("RESULTS (centralized)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    if per_node_mae:
        pca = np.array(per_node_mae)
        print(f"  Per-node MAE (macro): mean {pca.mean():.4f}, median {np.median(pca):.4f}, "
              f"min {pca.min():.4f}, max {pca.max():.4f}")
        metrics["per_client_MAE_mean"] = float(pca.mean())
        metrics["per_client_MAE_median"] = float(np.median(pca))
    print(f"  Best epoch: {best_round} (val {best_val:.4f})")

    # ── Save ──
    results = {
        "config": vars(args),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "best_round": int(best_round),
        "best_val_loss": float(best_val),
        "global_mean": g_mean,
        "global_std": g_std,
        "n_nodes": n_nodes,
        "worst_clients": sorted(per_node_records, key=lambda r: r["raw_mae"], reverse=True)[:10],
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save({
        "model_state": model.state_dict(),
        "model_kwargs": model_kwargs,
        "global_mean": g_mean, "global_std": g_std,
        "history": history, "config": vars(args),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }, os.path.join(run_dir, "checkpoint.pt"))

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = list(range(len(history["train_loss"])))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MAE (normalized)")
    ax.set_title(f"Centralized {args.dataset}/{args.encoder_type} (K={args.K})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curve.png"), dpi=120)
    plt.close(fig)
    print(f"\n  Saved to {run_dir}/")


if __name__ == "__main__":
    main()
