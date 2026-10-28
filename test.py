"""
Standalone evaluation script — loads a checkpoint and computes test metrics.

Usage:
    python test.py --checkpoint output/checkpoint.pt
    python test.py --checkpoint output/checkpoint.pt --device cuda:0
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import generate_metrla_clients
from models.causal_fedst import CausalFedST
from utils.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CausalFedST checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to checkpoint.pt")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load checkpoint ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    metrics_saved = ckpt["metrics"]
    print(f"  Saved metrics in checkpoint: MAE={metrics_saved['MAE']:.4f}, "
          f"RMSE={metrics_saved['RMSE']:.4f}, MAPE={metrics_saved['MAPE']:.4f}")

    # ── Rebuild dataset (same as training) ──
    if config.get("dataset", "metr-la") != "metr-la":
        raise ValueError("Only METR-LA checkpoints are supported")

    input_dim = 1

    print(f"Rebuilding dataset: {config['dataset']}, {config['n_clients']} clients")
    clients_data = generate_metrla_clients(
        n_clients=config["n_clients"],
        window_size=config["window_size"],
        horizon=config["horizon"],
        use_time_features=False,
        dataset_name="METR-LA",
        data_dir=config.get("data_dir", "./datasets"),
        seed=config["seed"],
    )

    # ── Rebuild models ──
    model_kwargs = dict(
        input_dim=input_dim,
        hid_dim=config["hid_dim"],
        seq_len=config["window_size"],
        horizon=config["horizon"],
        K=config["K"],
        tau=config["tau"],
        stid_num_layers=config.get("stid_num_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_temporal_identity=config.get("use_temporal_identity", True)
        and not config.get("no_temporal_identity", False),
        use_residual=config.get("use_residual", True)
        and not config.get("no_residual", False),
        residual_mode=config.get(
            "residual_mode",
            "learnable" if config.get("use_residual_gate", False) else "none",
        ),
        residual_scale=config.get("residual_scale", 0.25),
        residual_gate_init=config.get("residual_gate_init", -2.0),
    )

    device = torch.device(args.device)

    # ── Load client models with saved weights & global prototypes ──
    server_prototypes = ckpt["server_prototypes"].to(device)
    client_weights = ckpt["client_models"]

    assert len(client_weights) == len(clients_data), \
        f"Mismatch: {len(client_weights)} models vs {len(clients_data)} clients in data"

    # ── Evaluate ──
    print(f"Evaluating {len(client_weights)} clients...")
    all_y_true, all_y_pred, all_mask = [], [], []

    for i, data in enumerate(clients_data):
        model = CausalFedST(**model_kwargs).to(device)
        model.load_state_dict(client_weights[i])
        model.set_prototypes(server_prototypes)
        model.eval()

        target_mean = data["target_mean"]
        target_std = data["target_std"]

        test_tensors = [torch.from_numpy(data["X_test"]), torch.from_numpy(data["Y_test"])]
        if "T_test" in data:
            test_tensors.append(torch.from_numpy(data["T_test"]))
        loader = DataLoader(
            TensorDataset(*test_tensors),
            batch_size=config.get("batch_size", 32),
            shuffle=False,
        )

        client_y_true, client_y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y = batch[:2]
                batch_t = batch[2] if len(batch) > 2 else None
                batch_x = batch_x.to(device)
                if batch_t is not None:
                    batch_t = batch_t.to(device)
                y_hat, _, _ = model(batch_x, temporal_id=batch_t)
                client_y_true.append(batch_y[..., :1].cpu().numpy())
                client_y_pred.append(y_hat.cpu().numpy())

        yt = np.concatenate(client_y_true, axis=0)
        yp = np.concatenate(client_y_pred, axis=0)

        # Match training/main.py evaluation: METR-LA missing values are encoded
        # as 0 in raw scale and excluded with the same threshold.
        yt_raw = yt * target_std + target_mean
        yp_raw = yp * target_std + target_mean
        yp_raw = np.clip(yp_raw, -100.0, 100.0)
        client_mask = yt_raw > 0.1

        all_y_true.append(yt_raw)
        all_y_pred.append(yp_raw)
        all_mask.append(client_mask)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    metrics = compute_all_metrics(y_true, y_pred, mask=mask)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        diff = ""
        if k in metrics_saved:
            d = v - metrics_saved[k]
            diff = f"  (diff: {d:+.4f})"
        print(f"  {k}: {v:.4f}{diff}")
    print("=" * 60)


if __name__ == "__main__":
    main()
