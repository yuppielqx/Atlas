"""
Multi-panel prototype analysis figure for supplementary material.

Layout (2 rows, top row spans full width):
  ┌──────────────────────────────────────────────────┐
  │ (top) t-SNE of prototype trajectories across      │
  │       training rounds (colored by round)          │
  ├────────────────────────┬─────────────────────────┤
  │ (a) Prototype cosine   │ (b) Boxplot: per-       │
  │     similarity matrix  │     prototype weight    │
  │     (KxK heatmap)      │     across clients      │
  └────────────────────────┴─────────────────────────┘

All panels use the TEST split for (b).
The t-SNE panel uses proto_history from the checkpoint (saved every 10 rounds).

Usage:
    python scripts/fig_prototype_supp.py \
        --checkpoint output/zinv_ksweep_all_k10_r200/checkpoint.pt \
        --device cuda:0
"""

import os
import re
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
    p = argparse.ArgumentParser(description="Multi-panel prototype analysis figure")
    p.add_argument("--checkpoint", type=str,
                   default="output/zinv_ksweep_all_k10_r200/checkpoint.pt")
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=None,
                   help="output directory (default: checkpoint dir)")
    p.add_argument("--tsne_perplexity", type=float, default=30)
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Model reconstruction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Per-client weight collection (test split)
# ---------------------------------------------------------------------------

def collect_weights(checkpoint_path, data_split, device, data_dir, batch_size):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    client_models = ckpt["client_models"]
    global_prototypes = ckpt["server_prototypes"].float().to(device)
    K = config.get("K", 10)
    tau = config.get("tau", 0.1)

    print(f"Checkpoint: dataset={config.get('dataset')}, K={K}, tau={tau}, "
          f"clients={len(client_models)}")

    dataset_name = config.get("dataset", "metr-la")
    dataset_dir = DATASET_DIRS[dataset_name]
    window_size = config.get("window_size", 12)
    horizon = config.get("horizon", 12)
    graph_context = config.get("graph_context", False)
    client_ids_file = config.get("client_ids_file", None)

    explicit_ids = None
    if client_ids_file:
        with open(client_ids_file) as f:
            text = f.read()
        toks = [t for t in re.split(r"[\s,]+", text.strip()) if t]
        explicit_ids = [int(t) for t in toks]

    n_clients = len(client_models)
    if explicit_ids is not None:
        n_clients = len(explicit_ids)

    print(f"Loading dataset {dataset_dir} ({n_clients} clients, split={data_split})...")
    clients_data = generate_metrla_clients(
        n_clients=n_clients if explicit_ids is None else None,
        client_ids=explicit_ids,
        window_size=window_size, horizon=horizon,
        use_time_features=False, dataset_name=dataset_dir,
        data_dir=data_dir, seed=config.get("seed", 42),
        graph_context=graph_context,
    )
    print(f"  Loaded {len(clients_data)} data clients")

    model = build_model_from_config(config, device)
    model.set_prototypes(global_prototypes)
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset

    all_weight_rows = []
    n = min(len(clients_data), len(client_models))

    for i in range(n):
        sd = client_models[i]
        model.load_state_dict(sd, strict=False)
        model.to(device)

        data = clients_data[i]
        if data_split == "train":
            X = data["X_train"]; T = data.get("T_train")
        elif data_split == "test":
            X = data["X_test"];  T = data.get("T_test")
        else:
            X = data["X_val"] if data.get("X_val") is not None else data["X_train"]
            T = data.get("T_val") if data.get("T_val") is not None else data.get("T_train")

        X_t = torch.from_numpy(X).float()
        T_t = torch.from_numpy(T).float() if T is not None else None
        loader = (DataLoader(TensorDataset(X_t, T_t), batch_size=batch_size, shuffle=False)
                  if T_t is not None
                  else DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False))

        client_id = int(data.get("client_id", i))
        sum_weights = np.zeros(K)
        n_samples = 0

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

        avg_weights = sum_weights / max(n_samples, 1)
        all_weight_rows.append((client_id, avg_weights))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} clients...")

    print(f"  Done. {n} clients processed.")
    all_weight_rows.sort(key=lambda x: x[0])
    weight_matrix = np.stack([r[1] for r in all_weight_rows])
    all_client_ids = [r[0] for r in all_weight_rows]

    return {
        "weight_matrix": weight_matrix,
        "client_ids": all_client_ids,
        "prototypes": global_prototypes.cpu().numpy(),
        "K": K,
        "tau": tau,
        "config": config,
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_figure(results, proto_history, proto_rounds, args):
    K = results["K"]
    tau = results["tau"]
    weight_matrix = results["weight_matrix"]          # (N, K)
    all_client_ids = results["client_ids"]            # length N
    prototypes = results["prototypes"]                 # (K, D)
    N = len(all_client_ids)

    has_proto_history = proto_history is not None and len(proto_history) > 1

    # ---- Create figure ----
    if has_proto_history:
        fig = plt.figure(figsize=(16, 11))
        gs = GridSpec(2, 2, figure=fig,
                      width_ratios=[1, 1.1],
                      height_ratios=[0.7, 1],
                      hspace=0.35, wspace=0.28)
        ax_top = fig.add_subplot(gs[0, :])   # top row: proto t-SNE across rounds
        ax_a = fig.add_subplot(gs[1, 0])     # bottom-left: cosine sim
        ax_b = fig.add_subplot(gs[1, 1])     # bottom-right: boxplot
    else:
        fig = plt.figure(figsize=(16, 6.5))
        gs = GridSpec(1, 2, figure=fig,
                      width_ratios=[1, 1.1],
                      hspace=0.30, wspace=0.28)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_top = None

    LABEL_FONTSIZE = 12
    SUBLABEL_FONTSIZE = 13
    TITLE_FONTSIZE = 10.5

    # =====================================================================
    # (top) Prototype t-SNE trajectory across rounds
    # =====================================================================
    if has_proto_history:
        from sklearn.manifold import TSNE

        # proto_history: list of (K, D) numpy arrays at saved rounds
        n_snapshots = len(proto_history)
        print(f"\nProto history: {n_snapshots} snapshots at rounds {proto_rounds}")

        # Stack all prototypes: (n_snapshots * K, D)
        all_protos = np.concatenate(proto_history, axis=0)
        # Normalize before t-SNE (prototypes are on unit sphere but accumulate
        # tiny numerical drift; normalise for a clean cosine geometry).
        all_protos = all_protos / (np.linalg.norm(all_protos, axis=-1, keepdims=True) + 1e-10)

        perp = min(args.tsne_perplexity, all_protos.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    n_jobs=-1, verbose=1)
        X_2d = tsne.fit_transform(all_protos)  # (n_snapshots * K, 2)

        # Plot: each prototype as a trajectory across rounds
        # Color by round index (early = cool, late = warm)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cmap = plt.cm.viridis
        norm = Normalize(vmin=min(proto_rounds), vmax=max(proto_rounds))

        for k in range(K):
            px = X_2d[k::K, 0]
            py = X_2d[k::K, 1]
            # Draw trajectory as a thin line connecting consecutive rounds
            ax_top.plot(px, py, color="gray", alpha=0.25, linewidth=0.6, zorder=1)
            # Plot markers colored by round
            for s in range(n_snapshots):
                ax_top.scatter(px[s], py[s], c=[cmap(norm(proto_rounds[s]))],
                               s=35, edgecolors="white", linewidths=0.3, zorder=2)
            # Label the final position
            ax_top.annotate(f"P{k}", (px[-1], py[-1]),
                            textcoords="offset points", xytext=(3, 3),
                            fontsize=9, fontweight="bold", zorder=5)
            # Small circle at the start position
            ax_top.scatter(px[0], py[0], facecolors="none",
                           edgecolors=["red"], linewidths=0.8, s=60, zorder=3)

        # Colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax_top, shrink=0.9)
        cbar.set_label("Round", fontsize=9)
        cbar.ax.tick_params(labelsize=7)

        # Decorative: mark start/end
        ax_top.scatter([], [], c="red", marker="o", s=40, label="Start (round 0)")
        ax_top.scatter([], [], c="black", marker="s", s=30, label="End")
        ax_top.legend(fontsize=8, loc="upper right")

        ax_top.set_xlabel("t-SNE 1", fontsize=LABEL_FONTSIZE)
        ax_top.set_ylabel("t-SNE 2", fontsize=LABEL_FONTSIZE)
        ax_top.set_title(
            f"Prototype t-SNE trajectory across rounds (K={K}, "
            f"{n_snapshots} snapshots, rounds {proto_rounds[0]}–{proto_rounds[-1]})",
            fontsize=TITLE_FONTSIZE, fontweight="bold")

    # =====================================================================
    # (a) Cosine similarity matrix
    # =====================================================================
    proto = prototypes
    proto_norm = proto / (np.linalg.norm(proto, axis=-1, keepdims=True) + 1e-10)
    proto_sim = proto_norm @ proto_norm.T

    im = ax_a.imshow(proto_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(K):
        for j in range(K):
            ax_a.text(j, i, f"{proto_sim[i, j]:.2f}", ha="center", va="center",
                      fontsize=7.5,
                      color="black" if abs(proto_sim[i, j]) < 0.6 else "white")
    ax_a.set_xticks(range(K))
    ax_a.set_yticks(range(K))
    ax_a.set_xticklabels([f"P{k}" for k in range(K)], fontsize=8)
    ax_a.set_yticklabels([f"P{k}" for k in range(K)], fontsize=8)
    ax_a.set_xlabel("Prototype index", fontsize=LABEL_FONTSIZE)
    ax_a.set_ylabel("Prototype index", fontsize=LABEL_FONTSIZE)
    ax_a.set_title("Prototype cosine similarity matrix (test)", fontsize=TITLE_FONTSIZE,
                   fontweight="bold")
    cbar_a = plt.colorbar(im, ax=ax_a, shrink=0.85)
    cbar_a.set_label("Cosine similarity", fontsize=9)
    cbar_a.ax.tick_params(labelsize=7)

    mask = ~np.eye(K, dtype=bool)
    off_diag = proto_sim[mask]
    ax_a.text(0.5, -0.18,
              f"off-diag: mean={off_diag.mean():.3f}, "
              f"max={off_diag.max():.3f}, min={off_diag.min():.3f}",
              transform=ax_a.transAxes, ha="center", fontsize=8, style="italic")

    # =====================================================================
    # (b) Boxplot of prototype weights
    # =====================================================================
    box_data = [weight_matrix[:, k] for k in range(K)]
    bp = ax_b.boxplot(box_data, patch_artist=True, showfliers=True,
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.5)
    ax_b.axhline(y=1.0 / K, color="red", linestyle="--", linewidth=1.0,
                 label=f"uniform 1/K = {1.0/K:.2f}")
    ax_b.set_xticklabels([f"P{k}" for k in range(K)], fontsize=9)
    ax_b.set_xlabel("Prototype", fontsize=LABEL_FONTSIZE)
    ax_b.set_ylabel("Avg softmax weight across clients", fontsize=LABEL_FONTSIZE)
    ax_b.set_title("Per-prototype weight distribution (test)", fontsize=TITLE_FONTSIZE,
                   fontweight="bold")
    ax_b.legend(fontsize=9, loc="upper right")
    ax_b.grid(axis="y", alpha=0.3)

    # =====================================================================
    # Letter labels
    # =====================================================================
    panels = [(ax_a, "a"), (ax_b, "b")]
    if ax_top is not None:
        panels.insert(0, (ax_top, ""))  # top panel has no letter label
    for ax, letter in panels:
        if letter:
            ax.text(-0.06, 1.04, letter, transform=ax.transAxes,
                    fontsize=SUBLABEL_FONTSIZE, fontweight="bold",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.85))

    # =====================================================================
    # Caption
    # =====================================================================
    metrics_str = ""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "metrics" in ckpt:
        m = ckpt.get("metrics", {})
        metrics_str = (f"test MAE={m.get('MAE', 'N/A'):.2f}, "
                       f"RMSE={m.get('RMSE', 'N/A'):.2f}, "
                       f"MAPE={m.get('MAPE', 'N/A'):.2f}%")
    ds_name = results['config'].get('dataset', 'metr-la').upper()
    if has_proto_history:
        caption = (f"K={K} prototypes, {N} {ds_name} clients (test split); "
                   f"{metrics_str}. "
                   f"Top: {n_snapshots} prototype snapshots over rounds "
                   f"{proto_rounds[0]}–{proto_rounds[-1]} (t-SNE).")
    else:
        caption = (f"K={K} prototypes, {N} {ds_name} clients (test split); "
                   f"{metrics_str}. "
                   f"Proto history not available — re-run training with the "
                   f"updated trainer.py to capture per-round snapshots.")
    fig.text(0.5, 0.003, caption, ha="center", va="bottom",
             fontsize=8.5, style="italic")

    # =====================================================================
    # Save
    # =====================================================================
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    svg_path = os.path.join(out_dir, "prototype_supp_figure.svg")
    png_path = os.path.join(out_dir, "prototype_supp_figure.png")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", format="svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {svg_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)

    # =====================================================================
    # Summary
    # =====================================================================
    entropies = []
    for i in range(N):
        w = weight_matrix[i]
        eps = 1e-10
        entropies.append(-np.sum(w * np.log(w + eps)))
    entropies = np.array(entropies)
    max_ent = np.log(K)
    max_weights = weight_matrix.max(axis=1)
    off_mean = off_diag.mean()
    off_abs_max = np.abs(off_diag).max()
    primary = np.argmax(weight_matrix, axis=1)
    primary_counts = np.bincount(primary, minlength=K)

    print(f"\n─── Summary (test split, {N} clients) ───")
    print(f"  Entropy: mean={entropies.mean():.4f}, max possible={max_ent:.4f}")
    print(f"  Max weight: mean={max_weights.mean():.4f}, "
          f"min={max_weights.min():.4f}, max={max_weights.max():.4f}")
    print(f"  Prototype off-diag sim: mean={off_mean:.4f}, max_by_abs={off_abs_max:.4f}")
    print(f"  Primary prototype counts: {primary_counts.tolist()}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Collect per-client prototype weights (test split)
    results = collect_weights(
        args.checkpoint, data_split="test",
        device=args.device, data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # 2. Load proto_history from checkpoint (if available)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    proto_history = ckpt.get("proto_history", None)
    proto_rounds = ckpt.get("proto_history_rounds", None)

    if proto_history is not None and len(proto_history) > 0:
        proto_history = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                         for p in proto_history]
        if proto_rounds is None:
            proto_rounds = list(range(0, 10 * len(proto_history), 10))

        # Also include the final server_prototypes as the last snapshot
        final = ckpt["server_prototypes"].cpu().numpy()
        proto_history.append(final)
        proto_rounds = list(proto_rounds) + [proto_rounds[-1] + 10 if proto_rounds else 200]

        print(f"Proto history: {len(proto_history)} snapshots, "
              f"rounds {proto_rounds}")
    else:
        print("No proto_history in checkpoint. "
              "Top panel will be skipped.")
        proto_history = None
        proto_rounds = None

    # 3. Plot
    plot_figure(results, proto_history, proto_rounds, args)


if __name__ == "__main__":
    main()
