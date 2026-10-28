#!/usr/bin/env python3
"""
t-SNE Prototype Case Study — 3-panel figure comparing encoder z-space,
per-client prototype usage, and prototype similarity.

Panel A (top, full-width): t-SNE of encoder z-vectors from 3–5 representative
    clients, with global prototype vectors overlaid as labelled markers.
    Colored by client ID to show whether different clients' encoders map
    similar inputs to similar regions of the representation space.

Panel B (bottom-left): Boxplot of per-prototype average softmax weights
    across all clients (test split).  A uniform 1/K reference line marks
    the "equal usage" baseline; prototypes with high cross-client variance
    carry client-specific information.

Panel C (bottom-right): Pairwise cosine similarity matrix of the global
    prototype vectors.  Off-diagonal entries near +1 signal prototype
    collapse (redundancy); entries near 0 indicate a well-spread codebook.

Usage:
    python analysis/tsne_prototype_case_study.py \
        --checkpoint output/knowair-pm25_K10_nall_horizon12_20260714_091638/checkpoint.pt \
        --device cuda:0
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FixedFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from models.causal_fedst import CausalFedST
from data.dataset import generate_metrla_clients

# ── Constants ──────────────────────────────────────────────────────────
P10 = np.array([
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
    "#B07AA1", "#EDC948", "#B6992D", "#FF9DA7", "#9C755F",
])
# Per-client scatter colours (distinct from P10 for contrast)
CLIENT_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
REGIME_NAMES = {0: "Clean", 1: "Moderate", 2: "Polluted", 3: "Rising", 4: "Falling"}
REGIME_COLORS = {0: "#2ECC71", 1: "#3498DB", 2: "#E74C3C", 3: "#F39C12", 4: "#9B59B6"}

DATASET_DIRS = {
    "metr-la": "METR-LA", "pems-bay": "PEMS-BAY",
    "pems03": "PEMS03", "pems04": "PEMS04",
    "knowair-pm25": "KnowAir-PM25",
}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="t-SNE Prototype Case Study")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (default: analysis/case_study)")
    p.add_argument("--n_select_clients", type=int, default=5,
                   help="Number of representative clients for Panel A")
    p.add_argument("--n_samples_per_client", type=int, default=1000,
                   help="Max z-vectors per client for t-SNE (subsampled)")
    p.add_argument("--tsne_perplexity", type=float, default=30,
                   help="t-SNE perplexity")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint loading + model reconstruction
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint_and_model(ckpt_path, device):
    """Load checkpoint and return (ckpt, model_kwargs, proto_tensor, K)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    proto = ckpt["server_prototypes"].to(device)
    K = proto.shape[0]

    # Detect input_dim from encoder embedding layer
    sd0 = ckpt["client_models"][0]
    ts = sd0.get("encoder.time_series_emb_layer.weight")
    if ts is not None:
        in_channels = ts.shape[1]
        seq_len = cfg.get("window_size", 12)
        input_dim = in_channels // seq_len
    else:
        input_dim = cfg.get("input_dim", 1)

    # Detect multivariate from prototypes or encoder
    encoder_type = cfg.get("encoder_type", "stid")
    multivariate = input_dim > 1
    if encoder_type == "cnfgnn":
        multivariate = cfg.get("multivariate", input_dim > 1)

    use_residual = cfg.get("use_residual", False)
    residual_mode = cfg.get("residual_mode", "none") if use_residual else "none"

    model_kw = dict(
        input_dim=input_dim,
        hid_dim=cfg.get("hid_dim", 64),
        seq_len=cfg.get("window_size", 12),
        horizon=cfg.get("horizon", 12),
        K=K,
        tau=cfg.get("tau", 0.1),
        stid_num_layers=cfg.get("stid_num_layers", 3),
        dropout=cfg.get("dropout", 0.1),
        use_residual=use_residual,
        residual_mode=residual_mode,
        residual_scale=cfg.get("residual_scale", 1.0),
        residual_gate_init=cfg.get("residual_gate_init", -2.0),
        encoder_type=encoder_type,
        encoder_kwargs=cfg.get("encoder_kwargs"),
        use_prototype=True,
        use_aux_inv=(cfg.get("aux_inv_weight", 0.0) > 0),
        separate_residual=cfg.get("separate_residual", False),
        normalize_codebook=cfg.get("codebook_norm", True),
        prototype_init=cfg.get("prototype_init", "randn"),
        multivariate=multivariate,
    )
    return ckpt, model_kw, proto, K, cfg


# ═══════════════════════════════════════════════════════════════════════════
# PM2.5 regime classification (reused from knowair_prototype_case_study.py)
# ═══════════════════════════════════════════════════════════════════════════

def classify_pm25_regime(pm25, window=8):
    """Classify each time step into Clean/Moderate/Polluted/Rising/Falling."""
    T = len(pm25)
    labels = np.full(T, -1, dtype=int)
    lo, hi = np.percentile(pm25, 30), np.percentile(pm25, 70)
    sigma = pm25.std()
    for t in range(T):
        labels[t] = 0 if pm25[t] < lo else (2 if pm25[t] > hi else 1)
    for t in range(window, T):
        d = pm25[t] - pm25[t - window]
        if d > sigma:
            labels[t] = 3
        elif d < -sigma:
            labels[t] = 4
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Client selection
# ═══════════════════════════════════════════════════════════════════════════

def select_representative_clients(clients_data, n_select=5):
    """Select clients by PM2.5 variance: highest, lowest, and evenly spaced.

    Returns list of (client_index, label) tuples.
    """
    n = len(clients_data)
    # Compute PM2.5 std per client (from test data, target channel)
    stds = []
    for i, cdata in enumerate(clients_data):
        t_mean = cdata.get("target_mean", 0)
        t_std = cdata.get("target_std", 1)
        # X_test has shape (N, 12, channels); channel 0 is PM2.5 (normalised)
        pm25_norm = cdata["X_test"][:, -1, 0]
        pm25_raw = pm25_norm * t_std + t_mean
        stds.append((i, pm25_raw.std()))

    stds.sort(key=lambda x: x[1], reverse=True)
    n_available = len(stds)

    if n_select >= n_available:
        selected = [(idx, f"c{idx}") for idx, s in stds]
    else:
        selected = []
        # Highest variance
        selected.append((stds[0][0], f"c{stds[0][0]}"))
        # Lowest variance
        selected.append((stds[-1][0], f"c{stds[-1][0]}"))
        # Evenly spaced from the rest
        mid = stds[1:-1] if n_select > 2 else []
        if mid and n_select > 2:
            step = max(1, len(mid) // (n_select - 2))
            for j in range(0, len(mid), step):
                idx, s = mid[j]
                selected.append((idx, f"c{idx}"))
                if len(selected) >= n_select:
                    break
        selected = selected[:n_select]

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_z_vectors(clients_data, client_models, selected_clients,
                       model_kw, global_proto, device, n_max_per_client):
    """Collect encoder z-vectors + PM2.5 values for selected clients.

    Returns:
        z_list: list of (n_samples, hid_dim) numpy arrays, one per selected client
        pm25_list: list of (n_samples,) numpy arrays (raw PM2.5)
        client_labels: list of str labels for plotting
    """
    z_list, pm25_list, client_labels = [], [], []
    K = global_proto.shape[0]

    for ci, label in selected_clients:
        sd = client_models[ci]
        cdata = clients_data[ci]
        model = CausalFedST(**model_kw).to(device)
        model.load_state_dict(sd, strict=False)
        model.set_prototypes(global_proto)
        model.eval()

        t_mean = cdata.get("target_mean", 0)
        t_std = cdata.get("target_std", 1)

        X = torch.from_numpy(cdata["X_test"]).float()
        T_id = torch.from_numpy(cdata["T_test"]).float() if cdata.get("T_test") is not None else None

        n_total = X.shape[0]
        n_use = min(n_total, n_max_per_client * 2)  # oversample then subsample

        # Process in batches
        all_z = []
        all_pm25 = []
        for start in range(0, n_use, 512):
            end = min(start + 512, n_use)
            x_batch = X[start:end, :, :1].to(device)  # only PM2.5 channel
            t_batch = T_id[start:end].to(device) if T_id is not None else None
            pm25_norm = cdata["X_test"][start:end, -1, 0]
            pm25_raw = pm25_norm * t_std + t_mean

            z = model.encoder(x_batch, temporal_id=t_batch)  # (B, 1, hid_dim)
            all_z.append(z.squeeze(1).cpu().numpy())          # (B, hid_dim)
            all_pm25.append(pm25_raw)

        z_all = np.concatenate(all_z, axis=0)
        pm25_all = np.concatenate(all_pm25, axis=0)

        # Subsample uniformly
        if len(z_all) > n_max_per_client:
            rng = np.random.default_rng(42)
            idxs = rng.choice(len(z_all), n_max_per_client, replace=False)
            z_all = z_all[idxs]
            pm25_all = pm25_all[idxs]

        z_list.append(z_all)
        pm25_list.append(pm25_all)
        client_labels.append(label)
        print(f"  Client {ci}: {len(z_all)} z-vectors, "
              f"PM2.5 ∈ [{pm25_all.min():.0f}, {pm25_all.max():.0f}]")

    return z_list, pm25_list, client_labels


@torch.no_grad()
def collect_all_client_weights(clients_data, client_models, model_kw,
                                global_proto, device, batch_size):
    """Compute per-client average prototype softmax weights (test split).

    Returns:
        weight_matrix: (N_clients, K) float32 array
        client_ids: list of int client IDs
        per_client_mae: (N_clients,) float32 array of per-client test MAE (normalised)
    """
    K = global_proto.shape[0]
    tau = model_kw.get("tau", 0.1)
    n = min(len(clients_data), len(client_models))

    weight_rows, client_ids, mae_rows = [], [], []

    for i in range(n):
        sd = client_models[i]
        cdata = clients_data[i]
        model = CausalFedST(**model_kw).to(device)
        model.load_state_dict(sd, strict=False)
        model.set_prototypes(global_proto)
        model.eval()

        X = torch.from_numpy(cdata["X_test"]).float()
        Y = torch.from_numpy(cdata["Y_test"]).float()
        T_id = torch.from_numpy(cdata["T_test"]).float() if cdata.get("T_test") is not None else None

        sum_weights = np.zeros(K)
        n_samples = 0
        se_sum = 0.0

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            x_batch = X[start:end, :, :1].to(device)
            y_batch = Y[start:end].to(device)
            t_batch = T_id[start:end].to(device) if T_id is not None else None

            z = model.encoder(x_batch, temporal_id=t_batch)
            z_flat = z.reshape(-1, z.shape[-1])
            z_norm = F.normalize(z_flat, dim=-1)
            p_norm = F.normalize(global_proto, dim=-1)
            sim = z_norm @ p_norm.T
            w = F.softmax(sim / tau, dim=-1)

            sum_weights += w.cpu().numpy().sum(axis=0)
            n_samples += w.shape[0]

            # Prediction for MAE
            z_inv = w.reshape(z.shape[0], z.shape[1], K) @ global_proto.unsqueeze(0)
            y_hat = model.predictor(z_inv, None)
            se_sum += F.l1_loss(y_hat, y_batch[..., :1], reduction="sum").item()

        avg_weights = sum_weights / max(n_samples, 1)
        weight_rows.append(avg_weights)
        client_ids.append(int(cdata.get("client_id", i)))
        mae_rows.append(se_sum / max(n_samples, 1))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} clients...")

    weight_matrix = np.stack(weight_rows)
    mae_array = np.array(mae_rows)
    print(f"  Done. {n} clients processed.")
    return weight_matrix, client_ids, mae_array


# ═══════════════════════════════════════════════════════════════════════════
# Panel A: z-space t-SNE
# ═══════════════════════════════════════════════════════════════════════════

def panel_z_space_tsne(ax, z_list, pm25_list, client_labels, global_proto, args):
    """Draw t-SNE of encoder z-vectors with prototype overlay."""
    from sklearn.manifold import TSNE

    K = global_proto.shape[0]
    D = global_proto.shape[1]

    # Concatenate all z vectors + prototypes for joint t-SNE
    all_z = np.concatenate(z_list, axis=0)  # (total_samples, D)
    proto_np = global_proto.cpu().numpy()    # (K, D)

    # Normalize to unit sphere (like prototypes)
    all_z_norm = all_z / (np.linalg.norm(all_z, axis=-1, keepdims=True) + 1e-10)
    proto_norm = proto_np / (np.linalg.norm(proto_np, axis=-1, keepdims=True) + 1e-10)

    joint = np.concatenate([all_z_norm, proto_norm], axis=0)
    n_total = joint.shape[0]
    perp = min(args.tsne_perplexity, max(5, n_total - 1))
    print(f"\n  t-SNE: {n_total} points (perplexity={perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed,
                n_jobs=-1, verbose=1)
    X_2d = tsne.fit_transform(joint)

    z_2d = X_2d[:-K]      # z vectors
    proto_2d = X_2d[-K:]   # prototypes

    # Split z back per client
    sizes = [len(z) for z in z_list]
    offsets = np.cumsum([0] + sizes)

    # Draw z-vector scatter (small, semi-transparent)
    for j in range(len(z_list)):
        sl = slice(offsets[j], offsets[j + 1])
        ax.scatter(z_2d[sl, 0], z_2d[sl, 1],
                   c=[CLIENT_COLORS[j % len(CLIENT_COLORS)]],
                   s=8, alpha=0.35, edgecolors="none",
                   label=client_labels[j], zorder=1)

    # Draw prototype markers (star, medium-small)
    for k in range(K):
        ax.scatter(proto_2d[k, 0], proto_2d[k, 1],
                   c=[P10[k]], s=180, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(f"P{k}", (proto_2d[k, 0], proto_2d[k, 1]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=16, fontweight="bold", color='black', zorder=6)

    ax.legend(fontsize=16, loc="upper right", markerscale=1.5,
              framealpha=0.85, edgecolor="gray", ncol=5,
              labelspacing=0.02, handletextpad=0.05, columnspacing=0.3,
              borderpad=0.3, handlelength=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    y_ticks = [tick for tick in ax.get_yticks() if abs(tick) > 1e-12]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in y_ticks]))
    ax.text(0.5, -0.14,
            f"(a) t-SNE of prototypes and z-space",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=20, fontweight="bold")
    ax.grid(alpha=0.15, linewidth=0.3)

    return z_2d, proto_2d, sizes


# ═══════════════════════════════════════════════════════════════════════════
# Panel B: Per-prototype weight boxplot
# ═══════════════════════════════════════════════════════════════════════════

def panel_weight_boxplot(ax, weight_matrix, K, client_ids, mae_array):
    """Boxplot of per-prototype weight distribution across clients."""
    box_data = [weight_matrix[:, k] for k in range(K)]

    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for ki, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(P10[ki])
        patch.set_alpha(0.55)

    # Annotate variance per prototype
    variances = weight_matrix.var(axis=0)
    # for ki in range(K):
    #     v = variances[ki]
    #     ax.annotate(f"σ²={v:.4f}", (ki + 1, weight_matrix[:, ki].max()),
    #                 ha="center", fontsize=7, color="black", alpha=0.7,
    #                 xytext=(0, 4), textcoords="offset points")

    ax.set_xticklabels([f"P{k}" for k in range(K)], fontsize=16)
    ax.set_xlabel("")
    # ax.set_ylabel("Avg softmax weight across clients", fontsize=16)
    ax.set_title("")
    y_ticks = [tick for tick in ax.get_yticks() if abs(tick) > 1e-12]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in y_ticks]))
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.14,
            f"(b) Per-prototype weight distribution",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=20, fontweight="bold")

    # Print top-2 most variable prototypes
    top_var = np.argsort(variances)[::-1][:2]
    for k in top_var:
        print(f"  Proto {k}: mean_w={weight_matrix[:, k].mean():.3f}, "
              f"std={weight_matrix[:, k].std():.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Panel C: Prototype cosine similarity heatmap
# ═══════════════════════════════════════════════════════════════════════════

def panel_similarity_heatmap(ax, global_proto, K):
    """Cosine similarity matrix of global prototypes."""
    proto_np = global_proto.cpu().numpy()
    proto_norm = proto_np / (np.linalg.norm(proto_np, axis=-1, keepdims=True) + 1e-10)
    proto_sim = proto_norm @ proto_norm.T

    im = ax.imshow(proto_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{proto_sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(proto_sim[i, j]) > 0.5 else "black")

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"P{k}" for k in range(K)], fontsize=16)
    ax.set_yticklabels([f"P{k}" for k in range(K)], fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.tick_params(labelsize=14)

    # Off-diagonal stats (kept for return value)
    mask = ~np.eye(K, dtype=bool)
    off_diag = proto_sim[mask]
    ax.text(0.5, -0.14,
            f"(c) Prototype similarity",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=20, fontweight="bold")

    return off_diag


# ═══════════════════════════════════════════════════════════════════════════
# Figure assembly
# ═══════════════════════════════════════════════════════════════════════════

def make_figure(z_list, pm25_list, client_labels, weight_matrix, client_ids,
                mae_array, global_proto, K, ckpt, args):
    """Assemble the 3-panel figure and save."""

    # ── matplotlib setup ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1, 1],
                  height_ratios=[0.85, 1],
                  hspace=0.25, wspace=0.06)

    ax_a = fig.add_subplot(gs[0, :])   # Panel A: full-width t-SNE
    ax_b = fig.add_subplot(gs[1, 0])   # Panel B: boxplot
    ax_c = fig.add_subplot(gs[1, 1])   # Panel C: similarity heatmap

    # ── Panel A: z-space t-SNE ──
    panel_z_space_tsne(ax_a, z_list, pm25_list, client_labels, global_proto, args)

    # ── Panel B: weight boxplot ──
    panel_weight_boxplot(ax_b, weight_matrix, K, client_ids, mae_array)

    # ── Panel C: cosine similarity ──
    off_diag = panel_similarity_heatmap(ax_c, global_proto, K)

    # ── Save ──
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analysis", "case_study")
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, "tsne_prototype_case_study.pdf")
    png_path = os.path.join(out_dir, "tsne_prototype_case_study.png")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(weight_matrix, K, off_diag, client_ids, mae_array):
    """Print diagnostic summary to stdout."""
    N = weight_matrix.shape[0]
    entropies = -np.sum(weight_matrix * np.log(weight_matrix + 1e-10), axis=1)
    max_ent = np.log(K)
    max_weights = weight_matrix.max(axis=1)
    primary = np.argmax(weight_matrix, axis=1)
    primary_counts = np.bincount(primary, minlength=K)
    variances = weight_matrix.var(axis=0)

    print(f"\n{'='*60}")
    print(f"Summary ({N} clients, K={K} prototypes)")
    print(f"{'='*60}")
    print(f"  Prototype usage entropy: {entropies.mean():.3f} ± {entropies.std():.3f} "
          f"(max possible = {max_ent:.3f})")
    print(f"  Max weight per client:   mean={max_weights.mean():.3f}, "
          f"min={max_weights.min():.3f}, max={max_weights.max():.3f}")
    print(f"  Prototype off-diag cos:  mean={off_diag.mean():.3f}, "
          f"max={off_diag.max():.3f}, min={off_diag.min():.3f}")
    print(f"  Cross-client weight variance per prototype: "
          f"{'  '.join(f'P{k}={variances[k]:.4f}' for k in range(K))}")
    print(f"  Primary prototype counts: "
          f"{'  '.join(f'P{k}={primary_counts[k]}' for k in range(K))}")
    print(f"  Per-client MAE (normalised): "
          f"mean={mae_array.mean():.4f}, "
          f"min={mae_array.min():.4f} (c{client_ids[int(np.argmin(mae_array))]}), "
          f"max={mae_array.max():.4f} (c{client_ids[int(np.argmax(mae_array))]})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("t-SNE Prototype Case Study")
    print("=" * 60)

    # 1. Load checkpoint
    print("\n[1/5] Loading checkpoint & building model...")
    ckpt, model_kw, global_proto, K, cfg = load_checkpoint_and_model(
        args.checkpoint, device)
    proto_history = ckpt.get("proto_history", None)
    has_proto_hist = proto_history is not None and len(proto_history) > 1
    print(f"  K={K}  hid_dim={model_kw['hid_dim']}  "
          f"dataset={cfg.get('dataset', '?')}  "
          f"proto_history={'yes' if has_proto_hist else 'no'}  "
          f"MAE={ckpt.get('metrics', {}).get('MAE', '?'):.2f}")

    client_models = ckpt["client_models"]
    n_clients_total = len(client_models)

    # 2. Load dataset
    print(f"\n[2/5] Loading dataset ({n_clients_total} clients)...")
    dataset_name = cfg.get("dataset", "metr-la")
    dataset_dir = DATASET_DIRS.get(dataset_name, dataset_name.upper())
    window_size = cfg.get("window_size", 12)
    horizon = cfg.get("horizon", 12)
    clients_data = generate_metrla_clients(
        n_clients=None, window_size=window_size, horizon=horizon,
        use_time_features=False, dataset_name=dataset_dir,
        data_dir=args.data_dir, seed=cfg.get("seed", 42),
    )
    # Trim to checkpoint client count
    n_data = min(len(clients_data), n_clients_total)
    clients_data = clients_data[:n_data]
    client_models = client_models[:n_data]
    print(f"  Loaded {len(clients_data)} data clients, {len(client_models)} models")

    # 3. Select representative clients + collect z-vectors
    print(f"\n[3/5] Selecting {args.n_select_clients} representative clients "
          f"& collecting z-vectors...")
    selected = select_representative_clients(clients_data, args.n_select_clients)
    for _, label in selected:
        print(f"  {label}")
    z_list, pm25_list, client_labels = collect_z_vectors(
        clients_data, client_models, selected, model_kw, global_proto, device,
        args.n_samples_per_client)

    # 4. Collect per-client prototype weights
    print(f"\n[4/5] Collecting per-client prototype weights...")
    weight_matrix, client_ids, mae_array = collect_all_client_weights(
        clients_data, client_models, model_kw, global_proto, device,
        args.batch_size)

    # 5. Make figure
    print(f"\n[5/5] Creating figure...")
    off_diag = np.array([])
    proto_np = global_proto.cpu().numpy()
    proto_norm = proto_np / (np.linalg.norm(proto_np, axis=-1, keepdims=True) + 1e-10)
    proto_sim = proto_norm @ proto_norm.T
    mask = ~np.eye(K, dtype=bool)
    off_diag = proto_sim[mask]

    make_figure(z_list, pm25_list, client_labels, weight_matrix, client_ids,
                mae_array, global_proto, K, ckpt, args)

    print_summary(weight_matrix, K, off_diag, client_ids, mae_array)
    print("\nDone.")


if __name__ == "__main__":
    main()
