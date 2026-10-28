"""
Export prototype assignment data to CSV.

Reads the same data used for prototype_weights_boxplot_val.png and exports:
  1. prototype_weights_matrix.csv  — (N_clients × K) average softmax weight matrix
  2. client_prototype_stats.csv    — per-client summary with all metrics

Usage:
    python scripts/export_prototype_weights_csv.py \
        --checkpoint output/zinv_ksweep_all_k10_r200/checkpoint.pt \
        --data_split val
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import generate_metrla_clients
from models.causal_fedst import CausalFedST
from models.encoders import build_encoder_kwargs, ENCODER_REGISTRY


DATASET_DIRS = {
    "metr-la": "METR-LA",
    "pems-bay": "PEMS-BAY",
    "pems03": "PEMS03",
    "pems04": "PEMS04",
    "knowair-pm25": "KnowAir-PM25",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export prototype weights to CSV")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_split", type=str, default="val",
                        choices=["val", "test", "train"])
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def build_model_from_config(config, device):
    encoder_type = config.get("encoder_type", "stid")
    encoder_kwargs = build_encoder_kwargs(encoder_type, config.get("encoder_opt", []))
    use_prototype = config.get("fl_mode", "causal_proto") not in {
        "local", "fedavg", "fedavg_enc", "ditto"}
    use_residual = config.get("use_residual", True)
    residual_mode = config.get("residual_mode", "fixed") if use_residual else "none"

    model = CausalFedST(
        input_dim=config.get("input_dim", 1)
                  if not config.get("graph_context", False) else 2,
        hid_dim=config.get("hid_dim", 64),
        seq_len=config.get("window_size", 12),
        horizon=config.get("horizon", 12),
        K=config.get("K", 10),
        tau=config.get("tau", 0.1),
        stid_num_layers=config.get("stid_num_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_residual=use_residual,
        residual_mode=residual_mode,
        residual_scale=config.get("residual_scale", 1.0),
        residual_gate_init=config.get("residual_gate_init", -2.0),
        encoder_type=encoder_type,
        encoder_kwargs=encoder_kwargs,
        use_prototype=use_prototype,
        use_aux_inv=(config.get("aux_inv_weight", 0.0) > 0),
        separate_residual=config.get("separate_residual", False),
    )
    return model


def collect_weights(checkpoint_path, args):
    """Reuse the same logic as visualize_prototype_weights.py."""
    device = torch.device(args.device)
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt["config"]
    client_models = ckpt["client_models"]
    global_prototypes = ckpt["server_prototypes"].float().to(device)
    K = config.get("K", 10)
    tau = config.get("tau", 0.1)

    print(f"K={K}, tau={tau}, clients={len(client_models)}, "
          f"prototypes shape={tuple(global_prototypes.shape)}")

    dataset_name = config.get("dataset", "metr-la")
    dataset_dir = DATASET_DIRS[dataset_name]
    window_size = config.get("window_size", 12)
    horizon = config.get("horizon", 12)
    graph_context = config.get("graph_context", False)
    client_ids_file = config.get("client_ids_file", None)

    explicit_ids = None
    if client_ids_file:
        import re
        with open(client_ids_file) as f:
            text = f.read()
        toks = [t for t in re.split(r"[\s,]+", text.strip()) if t]
        explicit_ids = [int(t) for t in toks]

    n_clients = len(client_models)
    if explicit_ids is not None:
        n_clients = len(explicit_ids)

    clients_data = generate_metrla_clients(
        n_clients=n_clients if explicit_ids is None else None,
        client_ids=explicit_ids,
        window_size=window_size,
        horizon=horizon,
        use_time_features=False,
        dataset_name=dataset_dir,
        data_dir=args.data_dir,
        seed=config.get("seed", 42),
        graph_context=graph_context,
    )

    model = build_model_from_config(config, device)
    model.set_prototypes(global_prototypes)
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset

    all_rows = []        # one dict per client
    n = min(len(clients_data), len(client_models))

    for i in range(n):
        sd = client_models[i]
        model.load_state_dict(sd, strict=False)
        model.to(device)

        data = clients_data[i]
        if args.data_split == "train":
            X = data["X_train"]
            T = data.get("T_train")
        elif args.data_split == "test":
            X = data["X_test"]
            T = data.get("T_test")
        else:
            X = data["X_val"] if data.get("X_val") is not None else data["X_train"]
            T = data.get("T_val") if data.get("T_val") is not None else data.get("T_train")

        X_t = torch.from_numpy(X).float()
        T_t = torch.from_numpy(T).float() if T is not None else None
        if T_t is not None:
            loader = DataLoader(TensorDataset(X_t, T_t), batch_size=args.batch_size, shuffle=False)
        else:
            loader = DataLoader(TensorDataset(X_t), batch_size=args.batch_size, shuffle=False)

        client_id = int(data.get("client_id", i))
        sum_weights = np.zeros(K)
        sum_entropy = 0.0
        sum_z_norm = 0.0
        sum_max_sim = 0.0
        n_samples = 0

        # Per-sample weights for variability stats (optional)
        all_sample_weights = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(device)
                batch_t = batch[1].to(device) if len(batch) > 1 else None

                z = model.encoder(batch_x, temporal_id=batch_t)
                B_, T_, D_ = z.shape
                z_flat = z.reshape(B_ * T_, D_)
                z_norm = F.normalize(z_flat, dim=-1)
                p_norm = F.normalize(global_prototypes, dim=-1)
                sim = z_norm @ p_norm.T
                w = F.softmax(sim / tau, dim=-1)

                w_np = w.cpu().numpy()
                sum_weights += w_np.sum(axis=0)
                n_samples += w_np.shape[0]

                eps = 1e-10
                ent = -np.sum(w_np * np.log(w_np + eps), axis=1)
                sum_entropy += ent.sum()
                sum_z_norm += z_flat.norm(dim=-1).cpu().numpy().sum()
                sum_max_sim += sim.max(dim=-1).values.cpu().numpy().sum()

                all_sample_weights.append(w_np)

        avg_weights = sum_weights / max(n_samples, 1)
        avg_entropy = sum_entropy / max(n_samples, 1)
        avg_z_norm = sum_z_norm / max(n_samples, 1)
        avg_max_sim = sum_max_sim / max(n_samples, 1)

        # Weight-of-variation across samples (std per prototype, then mean)
        sample_w_all = np.concatenate(all_sample_weights, axis=0)  # (N_samples, K)
        w_std_per_proto = sample_w_all.std(axis=0)  # (K,)
        primary_prototype = int(np.argmax(avg_weights))

        row = {
            "client_id": client_id,
            "entropy": avg_entropy,
            "max_weight": float(avg_weights.max()),
            "primary_prototype": primary_prototype,
            "z_norm": avg_z_norm,
            "max_cos_sim": avg_max_sim,
        }
        for k in range(K):
            row[f"weight_p{k}"] = float(avg_weights[k])
            row[f"weight_std_p{k}"] = float(w_std_per_proto[k])

        all_rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} clients...")

    print(f"  Done. {n} clients processed.")
    return all_rows, K, config


def main():
    args = parse_args()
    rows, K, config = collect_weights(args.checkpoint, args)
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)

    # --- CSV 1: Prototype weight matrix (N_clients × K) ---
    matrix_path = os.path.join(out_dir, f"prototype_weights_matrix_{args.data_split}.csv")
    with open(matrix_path, "w") as f:
        header = "client_id," + ",".join([f"weight_p{k}" for k in range(K)])
        f.write(header + "\n")
        for row in rows:
            line = str(row["client_id"])
            for k in range(K):
                line += f",{row[f'weight_p{k}']:.6f}"
            f.write(line + "\n")
    print(f"Saved weight matrix: {matrix_path}")

    # --- CSV 2: Full per-client stats ---
    stats_path = os.path.join(out_dir, f"client_prototype_stats_{args.data_split}.csv")
    weight_cols = [f"weight_p{k}" for k in range(K)]
    std_cols = [f"weight_std_p{k}" for k in range(K)]
    columns = ["client_id", "entropy", "max_weight", "primary_prototype",
               "z_norm", "max_cos_sim"] + weight_cols + std_cols

    with open(stats_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            vals = [str(row.get(c, "")) for c in columns]
            f.write(",".join(vals) + "\n")
    print(f"Saved client stats: {stats_path}")

    # --- Print summary ---
    entropies = np.array([r["entropy"] for r in rows])
    max_weights = np.array([r["max_weight"] for r in rows])
    max_ent = np.log(K)
    print(f"\nSummary ({len(rows)} clients, K={K}, split={args.data_split}):")
    print(f"  Entropy: mean={entropies.mean():.4f}, max possible={max_ent:.4f}")
    print(f"  Max weight: mean={max_weights.mean():.4f}")
    primary_counts = np.bincount([r["primary_prototype"] for r in rows], minlength=K)
    print(f"  Primary prototype distribution: {primary_counts.tolist()}")


if __name__ == "__main__":
    main()
