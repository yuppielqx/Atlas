"""
Visualize per-client soft assignment weights: z -> prototypes.

For each client, compute the average softmax weight distribution over its
validation (or test) samples, showing which prototypes each client relies on.

Usage:
    python scripts/visualize_prototype_weights.py \
        --checkpoint output/zinv_ksweep_all_k10_r200/checkpoint.pt \
        --data_split val
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
    parser = argparse.ArgumentParser(description="Visualize prototype soft weights")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to checkpoint.pt")
    parser.add_argument("--data_split", type=str, default="val",
                        choices=["val", "test", "train"],
                        help="which data split to use for computing weights")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory for figures (default: checkpoint dir)")
    parser.add_argument("--max_clients", type=int, default=None,
                        help="max clients to show in heatmap (default: all)")
    return parser.parse_args()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def build_model_from_config(config, device):
    """Reconstruct a CausalFedST model from the config dict stored in the ckpt."""
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


def collect_weights(checkpoint, args):
    """For each client, collect the average softmax weight over all samples."""
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    client_models = ckpt["client_models"]
    global_prototypes = ckpt["server_prototypes"].float().to(device)
    K = config.get("K", 10)
    tau = config.get("tau", 0.1)

    print(f"Checkpoint config: dataset={config.get('dataset')}, K={K}, "
          f"tau={tau}, fl_mode={config.get('fl_mode')}")
    print(f"Number of clients: {len(client_models)}")
    print(f"Global prototypes shape: {tuple(global_prototypes.shape)}")

    # Reconstruct data
    dataset_name = config.get("dataset", "metr-la")
    dataset_dir = DATASET_DIRS[dataset_name]
    window_size = config.get("window_size", 12)
    horizon = config.get("horizon", 12)
    graph_context = config.get("graph_context", False)
    client_ids_file = config.get("client_ids_file", None)

    # Read explicit client ids if they were used
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

    print(f"Loading dataset {dataset_dir} with {n_clients} clients...")
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
    print(f"  Loaded {len(clients_data)} clients")

    if len(clients_data) != len(client_models):
        print(f"  WARNING: {len(clients_data)} data clients vs {len(client_models)} "
              f"model clients; using min")

    # Build model template
    model = build_model_from_config(config, device)
    model.set_prototypes(global_prototypes)
    model.eval()

    # Collect per-client average weights
    all_client_weights = []  # list of (client_id, avg_weight_vector of shape (K,))
    all_client_entropy = []  # per-client assignment entropy
    all_client_z_norms = []  # average ||z|| per client
    all_client_max_sim = []  # average max cosine similarity

    from torch.utils.data import DataLoader, TensorDataset

    n = min(len(clients_data), len(client_models))
    for i in range(n):
        # Load this client's model weights
        sd = client_models[i]
        model.load_state_dict(sd, strict=False)
        model.to(device)

        # Select data split
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

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(device)
                batch_t = batch[1].to(device) if len(batch) > 1 else None

                # Manually run encoder + prototype matching to get weights
                z = model.encoder(batch_x, temporal_id=batch_t)  # (B, T, D)
                B_, T_, D_ = z.shape
                z_flat = z.reshape(B_ * T_, D_)
                z_norm = F.normalize(z_flat, dim=-1)
                p_norm = F.normalize(global_prototypes, dim=-1)
                sim = z_norm @ p_norm.T  # (B*T, K)
                w = F.softmax(sim / tau, dim=-1)  # (B*T, K)

                w_np = w.cpu().numpy()
                sum_weights += w_np.sum(axis=0)  # sum over (B*T) samples
                n_samples += w_np.shape[0]

                # Entropy per sample, then average
                eps = 1e-10
                ent = -np.sum(w_np * np.log(w_np + eps), axis=1)
                sum_entropy += ent.sum()

                # Average ||z||
                sum_z_norm += z_flat.norm(dim=-1).cpu().numpy().sum()

                # Average max cosine similarity
                sum_max_sim += sim.max(dim=-1).values.cpu().numpy().sum()

        avg_weights = sum_weights / max(n_samples, 1)
        avg_entropy = sum_entropy / max(n_samples, 1)
        avg_z_norm = sum_z_norm / max(n_samples, 1)
        avg_max_sim = sum_max_sim / max(n_samples, 1)

        all_client_weights.append((client_id, avg_weights))
        all_client_entropy.append((client_id, avg_entropy))
        all_client_z_norms.append((client_id, avg_z_norm))
        all_client_max_sim.append((client_id, avg_max_sim))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} clients...")

    print(f"  Done. Processed {n} clients.")

    # Sort by client_id for consistent display
    all_client_weights.sort(key=lambda x: x[0])
    all_client_entropy.sort(key=lambda x: x[0])
    all_client_z_norms.sort(key=lambda x: x[0])
    all_client_max_sim.sort(key=lambda x: x[0])

    return {
        "weights": all_client_weights,
        "entropy": all_client_entropy,
        "z_norms": all_client_z_norms,
        "max_sim": all_client_max_sim,
        "K": K,
        "tau": tau,
        "prototypes": global_prototypes.cpu().numpy(),
        "config": config,
    }


def plot_heatmap(results, args):
    """Main figure: heatmap of per-client average prototype weights."""
    weights = results["weights"]
    K = results["K"]
    client_ids = [w[0] for w in weights]
    weight_matrix = np.stack([w[1] for w in weights])  # (N_clients, K)

    if args.max_clients and args.max_clients < len(client_ids):
        # Subsample evenly
        idx = np.linspace(0, len(client_ids) - 1, args.max_clients, dtype=int)
        client_ids = [client_ids[i] for i in idx]
        weight_matrix = weight_matrix[idx]

    N = len(client_ids)
    print(f"  Plotting heatmap: {N} clients x {K} prototypes")

    # ── Figure 1: Heatmap ──
    fig, ax = plt.subplots(figsize=(max(8, K * 0.7), max(6, N * 0.15)))

    im = ax.imshow(weight_matrix, aspect="auto", cmap="YlOrRd",
                   norm=Normalize(vmin=0, vmax=max(0.3, weight_matrix.max())))

    ax.set_xlabel("Prototype index", fontsize=12)
    ax.set_ylabel("Client (node) ID", fontsize=12)
    ax.set_title(f"Per-Client Average Prototype Assignment Weights\n"
                 f"(K={K}, τ={results['tau']}, {args.data_split} split)",
                 fontsize=13)

    # Ticks: show every prototype; thin out client labels if too many
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(k) for k in range(K)], fontsize=8)

    if N <= 50:
        ax.set_yticks(range(N))
        ax.set_yticklabels([str(c) for c in client_ids], fontsize=6)
    else:
        step = max(1, N // 30)
        tick_pos = list(range(0, N, step))
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([str(client_ids[p]) for p in tick_pos], fontsize=6)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Avg softmax weight", fontsize=11)

    fig.tight_layout()
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    path1 = os.path.join(out_dir, f"prototype_weights_heatmap_{args.data_split}.png")
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    print(f"  Saved heatmap: {path1}")
    plt.close(fig)

    # ── Figure 2: Weight distribution per prototype (boxplot / violin) ──
    fig, ax = plt.subplots(figsize=(max(8, K * 0.6), 5))
    box_data = [weight_matrix[:, k] for k in range(K)]
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.5)

    ax.set_xlabel("Prototype index", fontsize=12)
    ax.set_ylabel("Avg softmax weight across clients", fontsize=12)
    ax.set_title(f"Distribution of Prototype Weights Across Clients\n"
                 f"(K={K}, {N} clients, {args.data_split} split)", fontsize=13)
    ax.set_xticklabels([str(k) for k in range(K)], fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path2 = os.path.join(out_dir, f"prototype_weights_boxplot_{args.data_split}.png")
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    print(f"  Saved boxplot: {path2}")
    plt.close(fig)

    # ── Figure 3: Entropy histogram ──
    entropies = np.array([e[1] for e in results["entropy"]])
    max_ent = np.log(K)
    entropies_clipped = entropies[client_ids] if len(entropies) > N else entropies
    # align with displayed clients
    ent_display = np.array([results["entropy"][
        [j for j, e in enumerate(results["entropy"]) if e[0] == cid][0]
    ][1] for cid in client_ids])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(ent_display, bins=30, color="C2", edgecolor="white", alpha=0.8)
    ax.axvline(max_ent, color="red", linestyle="--", linewidth=1.5,
               label=f"Max entropy (uniform) = {max_ent:.2f}")
    ax.set_xlabel("Assignment entropy (nats)", fontsize=12)
    ax.set_ylabel("Number of clients", fontsize=12)
    ax.set_title(f"Per-Client Prototype Assignment Entropy\n"
                 f"(K={K}, lower = more concentrated)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(ent_display, weight_matrix.max(axis=1), alpha=0.5, s=10, color="C3")
    ax.set_xlabel("Assignment entropy (nats)", fontsize=12)
    ax.set_ylabel("Max prototype weight", fontsize=12)
    ax.set_title("Max Weight vs Entropy per Client", fontsize=12)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path3 = os.path.join(out_dir, f"prototype_entropy_{args.data_split}.png")
    fig.savefig(path3, dpi=200, bbox_inches="tight")
    print(f"  Saved entropy plots: {path3}")
    plt.close(fig)

    # ── Figure 4: Top-3 prototype diversity ──
    top3_idx = np.argsort(-weight_matrix, axis=1)[:, :3]  # (N, 3)
    proto_usage = np.zeros(K)
    for k in range(K):
        proto_usage[k] = (top3_idx == k).sum()
    proto_usage /= top3_idx.shape[0]  # fraction of (client, rank) slots

    # Also compute "primary prototype" per client
    primary = np.argmax(weight_matrix, axis=1)
    primary_counts = np.bincount(primary, minlength=K)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    ax.bar(range(K), primary_counts, color=colors, edgecolor="white")
    ax.set_xlabel("Prototype index", fontsize=12)
    ax.set_ylabel("Number of clients (primary prototype)", fontsize=12)
    ax.set_title(f"Primary Prototype Assignment per Client\n"
                 f"(each client assigned to its max-weight prototype)", fontsize=12)
    ax.set_xticks(range(K))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    width = 0.25
    x = np.arange(K)
    ax.bar(x - width, proto_usage, width, label="Top-1", color="C0", alpha=0.8)
    # top-2 and top-3 combined appearance
    top2_frac = np.zeros(K)
    for k in range(K):
        top2_frac[k] = (top3_idx[:, :2] == k).sum() / top3_idx.shape[0]
    ax.bar(x, top2_frac - proto_usage, width, label="Top-2", color="C1", alpha=0.8,
           bottom=proto_usage)
    top3_frac = np.zeros(K)
    for k in range(K):
        top3_frac[k] = (top3_idx[:, :3] == k).sum() / top3_idx.shape[0]
    ax.bar(x + width, top3_frac - top2_frac, width, label="Top-3", color="C2", alpha=0.8,
           bottom=top2_frac)
    ax.set_xlabel("Prototype index", fontsize=12)
    ax.set_ylabel("Fraction of (client, rank) slots", fontsize=12)
    ax.set_title("Prototype Usage in Top-K Rankings", fontsize=12)
    ax.set_xticks(range(K))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path4 = os.path.join(out_dir, f"prototype_usage_{args.data_split}.png")
    fig.savefig(path4, dpi=200, bbox_inches="tight")
    print(f"  Saved usage plots: {path4}")
    plt.close(fig)

    # ── Figure 5: Prototype-prototype cosine similarity matrix ──
    proto = results["prototypes"]  # (K, D)
    proto_norm = proto / (np.linalg.norm(proto, axis=-1, keepdims=True) + 1e-10)
    proto_sim = proto_norm @ proto_norm.T  # (K, K)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(proto_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xlabel("Prototype index", fontsize=12)
    ax.set_ylabel("Prototype index", fontsize=12)
    ax.set_title(f"Prototype-Prototype Cosine Similarity (K={K})", fontsize=13)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{proto_sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(proto_sim[i, j]) < 0.6 else "white")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Cosine similarity", fontsize=11)

    fig.tight_layout()
    path5 = os.path.join(out_dir, f"prototype_similarity_matrix.png")
    fig.savefig(path5, dpi=200, bbox_inches="tight")
    print(f"  Saved prototype similarity: {path5}")
    plt.close(fig)

    # Print summary
    print(f"\n─── Summary ───")
    print(f"  Mean entropy: {ent_display.mean():.3f} / {max_ent:.3f} (max)")
    print(f"  Mean max weight: {weight_matrix.max(axis=1).mean():.3f}")
    print(f"  Prototype usage (primary): {primary_counts}")
    print(f"  Prototype usage (top-3 fraction):")
    for k in range(K):
        print(f"    proto {k}: top1={primary_counts[k]}/{N} "
              f"({100*primary_counts[k]/max(N,1):.1f}%), "
              f"top3_slots={top3_frac[k]:.3f}")


def main():
    args = parse_args()
    results = collect_weights(args.checkpoint, args)
    plot_heatmap(results, args)


if __name__ == "__main__":
    main()
