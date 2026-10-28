#!/usr/bin/env python3
"""
Causal representation analysis for trained FedTS models on KnowAir-Full.

Analyzes:
  1. Prototype assignment patterns — which prototypes activate for which weather regimes?
  2. Feature contribution to z_inv — which weather variables drive the invariant representation?
  3. Station-level causal profiles — do learned prototypes align with statistical causality?
  4. Prototype geography — do nearby stations share similar prototype usage?

Usage:
  python3 analysis/analyze_fedts_causal.py --checkpoint <path/to/checkpoint.pt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.dataset import generate_multifeature_clients
from models.causal_fedst import CausalFedST

FEATURE_NAMES = [
    "z_t2m", "z_blh", "t2m", "skt", "blh", "kx",
    "r", "r_950", "tp", "sp", "d2m", "t_950",
    "sf", "z_sp", "z_u10", "u10", "v10", "pm25",
]

WEATHER_LABELS = {
    "t2m": "2m Temp", "skt": "Skin Temp", "blh": "BLH",
    "kx": "K-index", "r": "RH", "r_950": "RH@950",
    "tp": "Precip", "sp": "Pressure", "d2m": "Dewpoint",
    "t_950": "T@950", "sf": "Snow", "u10": "U-wind", "v10": "V-wind",
    "z_t2m": "z(T2m)", "z_blh": "z(BLH)", "z_sp": "z(SP)", "z_u10": "z(U10)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to checkpoint.pt")
    p.add_argument("--out_dir", default="analysis/causal_analysis")
    p.add_argument("--n_stations", type=int, default=20,
                   help="number of stations to analyze in detail")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_model_and_data(checkpoint_path, device):
    """Load the trained model and re-create client data."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    global_protos = ckpt["server_prototypes"].to(device)

    # Detect input_dim from checkpoint: time_series_emb_layer.weight shape
    # is [out_ch, seq_len * series_input_dim, 1, 1]
    cms = ckpt["client_models"]
    sample_sd = cms[0]
    ts_emb = sample_sd.get("encoder.time_series_emb_layer.weight")
    if ts_emb is not None:
        in_channels = ts_emb.shape[1]
        seq_len = cfg.get("window_size", 12)
        series_input_dim = in_channels // seq_len
        if series_input_dim == 1:
            # Legacy: series_input_dim=1 when input_dim>=3 (METR-LA format)
            # or when input_dim=1 with the old convention
            input_dim = 1
            multivariate = False
        else:
            input_dim = series_input_dim
            multivariate = True
    else:
        input_dim = cfg.get("input_dim", 1)  # fallback
        multivariate = cfg.get("multivariate", False)

    model_kwargs = dict(
        input_dim=input_dim,
        hid_dim=cfg.get("hid_dim", 64),
        seq_len=cfg.get("window_size", 12),
        horizon=cfg.get("horizon", 12),
        K=cfg.get("K", 10),
        tau=cfg.get("tau", 0.1),
        stid_num_layers=cfg.get("stid_num_layers", 3),
        dropout=cfg.get("dropout", 0.1),
        use_residual=cfg.get("use_residual", True),
        residual_mode=cfg.get("residual_mode", "fixed"),
        residual_scale=cfg.get("residual_scale", 1.0),
        encoder_type=cfg.get("encoder_type", "stid"),
        use_prototype=True,
        separate_residual=cfg.get("separate_residual", False),
        normalize_codebook=cfg.get("codebook_norm", True),
        prototype_init=cfg.get("prototype_init", "randn"),
        multivariate=multivariate,
    )

    # Load data (same config as training)
    clients_data = generate_multifeature_clients(
        n_clients=None,
        window_size=cfg.get("window_size", 12),
        horizon=cfg.get("horizon", 12),
        dataset_name="KnowAir-Full",
        data_dir=cfg.get("data_dir", "./datasets"),
        seed=cfg.get("seed", 42),
        include_target_in_input=True,  # match training setup
    )

    # If the model was trained with fewer input channels than the data provides
    # (e.g., STID encoder only reads channel 0 when input_dim>=3 and not multivariate),
    # subset the data to match the effective input dimension.
    actual_data_dim = clients_data[0]["X_train"].shape[-1]
    if input_dim < actual_data_dim:
        print(f"  Note: Model expects input_dim={input_dim}, data has {actual_data_dim}. "
              f"Subsetting to first {input_dim} channel(s).")
        for c in clients_data:
            for key in ["X_train", "X_val", "X_test"]:
                c[key] = c[key][:, :, :input_dim]

    return ckpt, model_kwargs, clients_data, global_protos, input_dim


def analyze_prototype_assignments(ckpt, clients_data, global_protos, model_kwargs,
                                   device, out_dir):
    """Analyze which prototypes activate for which weather conditions."""
    print("\n=== Prototype Assignment Analysis ===")

    K = global_protos.shape[0]
    n_clients = len(clients_data)

    # Compute per-client prototype usage
    client_proto_usage = np.zeros((n_clients, K))
    client_worst_mae = []

    for i, cdata in enumerate(clients_data):
        # Quick forward pass to get prototype weights
        model = CausalFedST(**model_kwargs).to(device)
        model.load_state_dict(ckpt["client_models"][i], strict=False)
        model.set_prototypes(global_protos)
        model.eval()

        # Sample a subset of test data
        X_test = torch.from_numpy(cdata["X_test"][:500]).to(device)
        T_test = torch.from_numpy(cdata["T_test"][:500]).to(device)

        with torch.no_grad():
            _, _, weights = model(X_test, temporal_id=T_test)
        # weights: [B, T, K]
        avg_weights = weights.mean(dim=(0, 1)).cpu().numpy()
        client_proto_usage[i] = avg_weights

        # Also record the client id
        client_worst_mae.append(cdata["client_id"])

    # Plot prototype usage heatmap
    fig, ax = plt.subplots(figsize=(14, 20))
    # Sort clients by dominant prototype
    dominant = np.argmax(client_proto_usage, axis=1)
    sort_idx = np.lexsort([dominant, client_proto_usage.argmax(axis=1)])

    im = ax.imshow(client_proto_usage[sort_idx], aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Prototype", fontsize=12)
    ax.set_ylabel("Station (sorted)", fontsize=12)
    ax.set_title(f"Prototype Assignment Weights Across {n_clients} Stations\n"
                 f"(KnowAir-Full, K={K})", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Softmax Weight", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "prototype_usage_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Per-prototype statistics
    proto_total = client_proto_usage.sum(axis=0)
    proto_norm = proto_total / proto_total.sum()
    print("\n  Prototype usage distribution:")
    for k in range(K):
        top_clients = np.argsort(client_proto_usage[:, k])[-5:][::-1]
        print(f"    Proto {k}: {proto_norm[k]:.3f}  "
              f"top stations: {top_clients.tolist()}")

    # Entropy of prototype distribution per client
    entropies = -np.sum(client_proto_usage * np.log(client_proto_usage + 1e-8), axis=1)
    max_entropy = np.log(K)
    normalized_entropy = entropies / max_entropy
    print(f"\n  Prototype entropy: mean={normalized_entropy.mean():.3f}  "
          f"std={normalized_entropy.std():.3f}  "
          f"(1.0 = uniform, 0.0 = one-hot)")

    return client_proto_usage


def analyze_feature_importance(ckpt, clients_data, global_protos, model_kwargs,
                                 device, out_dir, input_dim):
    """Gradient-based feature importance: which input features drive z_inv?"""
    print("\n=== Feature Importance Analysis ===")

    # Pick a few representative stations
    sample_stations = [0, 5, 10, 50, 100, 150]

    all_importances = []

    for s_idx in sample_stations:
        cdata = clients_data[s_idx]
        model = CausalFedST(**model_kwargs).to(device)
        model.load_state_dict(ckpt["client_models"][s_idx], strict=False)
        model.set_prototypes(global_protos)
        model.train()  # enable gradients

        X = torch.from_numpy(cdata["X_test"][:100]).to(device)
        T = torch.from_numpy(cdata["T_test"][:100]).to(device)
        X.requires_grad_(True)

        z = model.encoder(X, temporal_id=T)
        z_inv, _ = model.prototype(z)

        # Gradient of ||z_inv|| w.r.t. input
        z_norm = z_inv.norm()
        z_norm.backward()

        grad = X.grad.abs().mean(dim=(0, 1))  # average over batch and time
        all_importances.append(grad.cpu().numpy())

    avg_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    # Map to feature names
    # input_dim = 18 (1 PM2.5 AR + 17 weather)
    input_names = ["PM2.5 (AR)"] + [WEATHER_LABELS.get(n, n) for n in FEATURE_NAMES[:17]]

    # Sort by importance
    sorted_idx = np.argsort(avg_importance)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_idx))
    bars = ax.bar(x, avg_importance[sorted_idx], yerr=std_importance[sorted_idx],
                  color=["#E74C3C" if i == 0 else "#3498DB" for i in sorted_idx],
                  edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([input_names[i] for i in sorted_idx],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |∂z_inv/∂feature|", fontsize=11)
    ax.set_title("Feature Importance for Invariant Representation (z_inv)\n"
                 "(Gradient-based, averaged over 6 stations)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "feature_importance_zinv.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    print("\n  Top-5 features driving z_inv:")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"    {rank}. {input_names[idx]}: {avg_importance[idx]:.4f} ± {std_importance[idx]:.4f}")

    return avg_importance, input_names


def analyze_causal_correlation(ckpt, clients_data, global_protos, model_kwargs,
                                device, out_dir, input_dim):
    """Correlate prototype assignments with weather conditions."""
    print("\n=== Causal Correlation: Prototypes vs Weather ===")

    K = global_protos.shape[0]
    n_weather = input_dim - 1  # excluding AR PM2.5

    # For each station, compute: prototype_usage[k] vs weather_variable_mean[j]
    all_proto_weather_corr = np.zeros((K, n_weather))

    for station_idx in range(min(50, len(clients_data))):
        cdata = clients_data[station_idx]
        model = CausalFedST(**model_kwargs).to(device)
        model.load_state_dict(ckpt["client_models"][station_idx], strict=False)
        model.set_prototypes(global_protos)
        model.eval()

        X = torch.from_numpy(cdata["X_test"]).to(device)
        T = torch.from_numpy(cdata["T_test"]).to(device)

        with torch.no_grad():
            _, _, weights = model(X, temporal_id=T)

        w_np = weights.mean(dim=1).cpu().numpy()  # [N_samples, K]
        x_np = cdata["X_test"][:, -1, :]  # last input step, all features

        # Correlate each prototype weight with each weather feature
        for k in range(K):
            for j in range(1, n_weather + 1):  # skip AR channel (index 0)
                if x_np[:, j].std() > 0:
                    corr = np.corrcoef(w_np[:, k], x_np[:, j])[0, 1]
                    all_proto_weather_corr[k, j - 1] += corr

    all_proto_weather_corr /= 50  # average over stations

    # Plot
    weather_labels = [WEATHER_LABELS.get(n, n) for n in FEATURE_NAMES[:n_weather]]
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(all_proto_weather_corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                    aspect="auto")

    ax.set_xticks(range(n_weather))
    ax.set_xticklabels(weather_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"Proto {k}" for k in range(K)], fontsize=10)
    ax.set_xlabel("Weather Variable", fontsize=11)
    ax.set_ylabel("Prototype", fontsize=11)
    ax.set_title("Prototype–Weather Correlation\n"
                 "(Which prototypes capture which weather regimes?)",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Correlation", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "prototype_weather_correlation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Strongest correlations
    print("\n  Strongest prototype–weather correlations:")
    for k in range(K):
        top_j = np.argsort(np.abs(all_proto_weather_corr[k]))[-3:][::-1]
        descs = [f"{weather_labels[j]}({all_proto_weather_corr[k,j]:+.3f})"
                 for j in top_j]
        print(f"    Proto {k}: {', '.join(descs)}")

    return all_proto_weather_corr, weather_labels


def compare_statistical_vs_learned(all_proto_weather_corr, weather_labels, out_dir):
    """Compare the FedTS-learned causal structure with statistical analysis."""
    print("\n=== Comparison: Statistical vs Learned Causal Structure ===")

    # Load statistical results
    stat_path = REPO_ROOT / "analysis" / "knowair_causal_results.json"
    if not stat_path.exists():
        print("  [warn] Statistical results not found; skipping comparison")
        return

    with open(stat_path) as f:
        stat_results = json.load(f)

    # Aggregate learned: average |correlation| across prototypes
    learned_importance = np.abs(all_proto_weather_corr).mean(axis=0)

    # Statistical: partial correlation with PM2.5
    stat_pcorr = stat_results["partial_corr_with_pm25"]
    stat_importances = np.array([
        abs(stat_pcorr.get(name, 0))
        for name in FEATURE_NAMES[:17]
    ])

    # Normalize to [0, 1] for comparison
    learned_norm = learned_importance / (learned_importance.sum() + 1e-8)
    stat_norm = stat_importances / (stat_importances.sum() + 1e-8)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(weather_labels))
    width = 0.35

    ax.bar(x - width / 2, stat_norm, width, label="Statistical (Partial Corr)",
           color="#3498DB", edgecolor="white")
    ax.bar(x + width / 2, learned_norm, width, label="FedTS Learned (Proto–Weather Corr)",
           color="#E74C3C", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(weather_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalized Importance", fontsize=11)
    ax.set_title("Causal Structure Comparison:\n"
                 "Statistical Analysis vs FedTS Learned Representation",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "statistical_vs_learned.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(learned_importance, stat_importances)
    print(f"\n  Spearman rank correlation: ρ={rho:.4f} (p={pval:.4f})")
    if rho > 0.3:
        print("  ✓ FedTS prototypes capture similar causal structure to statistical analysis!")
    else:
        print("  ✗ Learned structure differs from statistical analysis (different mechanisms?)")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("FedTS Causal Representation Analysis — KnowAir-Full")
    print("=" * 60)

    # Load
    ckpt, model_kwargs, clients_data, global_protos, input_dim = \
        load_model_and_data(args.checkpoint, device)
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"  Dataset: {ckpt['config'].get('dataset', '?')}")
    print(f"  K={model_kwargs['K']}  hid_dim={model_kwargs['hid_dim']}  "
          f"input_dim={input_dim}")
    print(f"  Best round: {ckpt.get('history', {}).get('best_round', '?')}")
    print(f"  MAE: {ckpt.get('metrics', {}).get('MAE', '?'):.4f}")

    # 1. Prototype assignments
    client_proto_usage = analyze_prototype_assignments(
        ckpt, clients_data, global_protos, model_kwargs, device, args.out_dir)

    # 2. Feature importance
    avg_importance, input_names = analyze_feature_importance(
        ckpt, clients_data, global_protos, model_kwargs, device,
        args.out_dir, input_dim)

    # 3. Causal correlation
    all_proto_weather_corr, weather_labels = analyze_causal_correlation(
        ckpt, clients_data, global_protos, model_kwargs, device,
        args.out_dir, input_dim)

    # 4. Compare with statistical analysis
    compare_statistical_vs_learned(all_proto_weather_corr, weather_labels,
                                   args.out_dir)

    print("\n" + "=" * 60)
    print("Analysis complete. Figures saved to", args.out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
