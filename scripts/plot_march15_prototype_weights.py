"""
Single-day prototype weight visualization for a specific sensor and date.

Shows:
  (top)    Traffic speed: input window + ground truth + model prediction
  (bottom) Prototype soft-weight heatmap — how the codebook decomposes each time step

Usage:
    python scripts/plot_march15_prototype_weights.py \
        --checkpoint output/zinv_ksweep_all_k10_r200/checkpoint.pt \
        --sensor_id 718379 \
        --date 2012-03-15
"""

import os, sys, argparse, json, csv
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_mpl_{os.getuid()}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta

from models.causal_fedst import CausalFedST
from data.dataset import load_metrla_raw, sliding_window


# ── Palette ────────────────────────────────────────────────────────────
P10 = np.array([
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
    "#B07AA1", "#EDC948", "#B6992D", "#FF9DA7", "#9C755F",
])


def parse_args():
    p = argparse.ArgumentParser(description="March 15 prototype weight visualization")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sensor_id", type=int, default=718379)
    p.add_argument("--date", type=str, default="2012-03-15")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg", "all"])
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--dataset", type=str, default="metr-la")
    return p.parse_args()


# ── Data ────────────────────────────────────────────────────────────────

def load_sensor_data(sensor_id, dataset_name, data_dir, window_size=12, horizon=12):
    """Load windows for a specific sensor."""
    ds_map = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
    ds_dir = os.path.join(data_dir, ds_map[dataset_name])

    values, timestamps = load_metrla_raw(ds_map[dataset_name], data_dir)

    # Find sensor column index
    with open(os.path.join(ds_dir, "graph_sensor_locations.csv")) as f:
        sid2i = {int(r["sensor_id"]): int(r["index"]) for r in csv.DictReader(f)}

    col = sid2i.get(int(sensor_id))
    if col is None:
        raise ValueError(f"Sensor {sensor_id} not found in graph_sensor_locations.csv")

    speed = values[:, col]  # (L,)
    temporal = timestamps  # (L, 2) - these are the raw temporal features

    X, Y = sliding_window(speed, window_size, horizon)
    # Temporal IDs: use the last input step for each window
    n_windows = X.shape[0]
    temporal_ids = np.array([
        temporal[i + window_size - 1] for i in range(n_windows)
    ])

    return X, Y, temporal_ids, speed, temporal


def date_to_windows(date_str, n_windows, window_size=12, horizon=12, interval_min=5):
    """Map a date string to window indices.

    METR-LA starts 2012-03-01 00:00, 5-min intervals.
    """
    start = datetime(2012, 3, 1, 0, 0)
    target = datetime.strptime(date_str, "%Y-%m-%d")

    if target < start:
        raise ValueError(f"Date {date_str} is before METR-LA start ({start.date()})")

    delta_days = (target - start).days
    points_per_day = 24 * 60 // interval_min  # 288
    start_idx = delta_days * points_per_day
    end_idx = start_idx + points_per_day

    # Clamp to valid window range (need window_size + horizon steps ahead)
    max_idx = n_windows
    start_idx = max(0, min(start_idx, max_idx - 1))
    end_idx = max(start_idx + 1, min(end_idx, max_idx))

    return start_idx, end_idx


# ── Model ────────────────────────────────────────────────────────────────

def build_model_from_checkpoint(ckpt_path, device="cpu"):
    """Reconstruct a CausalFedST model from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = CausalFedST(
        input_dim=1,
        hid_dim=config.get("hid_dim", 64),
        seq_len=config.get("window_size", 12),
        horizon=config.get("horizon", 12),
        K=config.get("K", 10),
        tau=config.get("tau", 0.1),
        stid_num_layers=config.get("stid_num_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_residual=config.get("use_residual", False),
        residual_mode=config.get("residual_mode", "fixed"),
        residual_scale=config.get("residual_scale", 1.0),
        use_residual_gate=config.get("residual_gate_init", None),
        residual_gate_init=config.get("residual_gate_init", -2.0),
        encoder_type=config.get("encoder_type", "stid"),
        use_aux_inv=config.get("aux_inv_weight", 0.0) > 0,
    ).to(device)

    # Load weights from the first client model (or server_prototypes if available)
    # All client models should have the same final weights for a converged federated model
    cm0 = ckpt["client_models"][0]
    # Load state dict
    model.load_state_dict(cm0, strict=False)
    model.eval()
    return model, ckpt


# ── Plot ─────────────────────────────────────────────────────────────────

def plot_march15(X_day, Y_day, tids_day, speed_day, w_day, y_hat_day,
                 sensor_id, date_str, K, out_paths, window_size=12, horizon=12):
    """Main plot: speed curve + prototype weight heatmap."""

    n_windows = X_day.shape[0]  # ~288 for one day
    # Convert temporal IDs to hours for x-axis
    hours = tids_day[:, 0] * 24.0  # time_of_day → hours [0, 24)

    fig = plt.figure(figsize=(16, 8))

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1.0],
                           hspace=0.08, left=0.08, right=0.97,
                           top=0.94, bottom=0.08)

    # ═════════════════════════════════════════════════════════════════════
    # Top panel: Speed curves
    # ═════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])

    # Input speed: average over the 12-step input window
    input_speed = X_day.mean(axis=1)  # (N,)
    target_speed = Y_day.mean(axis=1)  # (N,)
    pred_speed = y_hat_day.mean(axis=1)  # (N,)

    ax1.plot(hours, input_speed, color="0.4", lw=1.0, alpha=0.7, label="Input (avg 1h)")
    ax1.plot(hours, target_speed, color="0.05", lw=1.6, label="Ground truth (avg 1h)")
    ax1.plot(hours, pred_speed, color="#E15759", lw=1.6, alpha=0.85,
             label="Prediction (z_inv only)")

    # Shade the forecast horizon region conceptually
    ax1.axhline(y=0, color="0.85", lw=0.5)

    ax1.set_ylabel("Speed (mph)", fontsize=11, labelpad=6)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xticklabels([])
    ax1.tick_params(labelsize=8.5)
    ax1.grid(axis="y", alpha=0.12, lw=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Time-of-day shading
    for h_start, h_end, label, alpha in [
        (6, 10, "Morning\npeak", 0.06),
        (16, 20, "Evening\npeak", 0.06),
    ]:
        ax1.axvspan(h_start, h_end, color="#F28E2B", alpha=alpha, lw=0)
    ax1.text(8, ax1.get_ylim()[1] * 0.95, "AM peak", fontsize=7.5,
             color="#D07020", ha="center", va="top")
    ax1.text(18, ax1.get_ylim()[1] * 0.95, "PM peak", fontsize=7.5,
             color="#D07020", ha="center", va="top")

    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.7,
               ncol=1, handlelength=1.2, borderpad=0.3,
               labelspacing=0.2)

    ax1.set_title(f"Sensor {sensor_id}  |  {date_str}  |  METR-LA  |  K={K}  |  z_inv-only",
                  fontsize=12, fontweight="bold", loc="left", pad=8)

    # ═════════════════════════════════════════════════════════════════════
    # Bottom panel: Prototype soft-weight heatmap
    # ═════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])

    # w_day: (N, K) — softmax weights
    wT = w_day.T  # (K, N) for imshow
    im = ax2.pcolormesh(hours, np.arange(K) + 0.5, wT,
                        cmap="YlOrRd", vmin=0, vmax=1.0,
                        shading="auto", rasterized=True)

    # Mark dominant prototype transitions
    dominant = w_day.argmax(axis=1)
    transitions = []
    for t in range(1, len(dominant)):
        if dominant[t] != dominant[t - 1]:
            transitions.append((hours[t], dominant[t - 1], dominant[t]))
    for tx, _, _ in transitions:
        ax2.axvline(x=tx, color="0.2", lw=0.4, alpha=0.5, ls="--")

    ax2.set_yticks(range(K))
    ax2.set_yticklabels([f"Proto {k}" for k in range(K)], fontsize=8.5)
    ax2.set_xlabel("Hour of day", fontsize=11, labelpad=4)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.85, pad=0.015)
    cbar.set_label("P(proto | x)  soft weight", fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=7.5)

    # Annotate transition count
    n_trans = len(transitions)
    ax2.text(0.99, 0.92, f"{n_trans} regime transitions",
             transform=ax2.transAxes, fontsize=8, ha="right",
             color="0.3", style="italic")

    fig.text(0.5, 0.01,
             "Each column = one 12-step sliding window. Input = mean speed over the 12-step window. "
             "Prototype weights = softmax(z_enc · Q_k / τ). Transitions (dashed) = argmax changes.",
             ha="center", fontsize=7, color="0.45", style="italic")

    for fmt, path in out_paths:
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Build model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = build_model_from_checkpoint(args.checkpoint, device)
    config = ckpt["config"]
    K = config.get("K", 10)
    window_size = config.get("window_size", 12)
    horizon = config.get("horizon", 12)
    print(f"  K={K}, window_size={window_size}, horizon={horizon}")

    # Load the server (federated) prototypes — these are the shared codebook
    server_protos = ckpt["server_prototypes"].to(device)
    model.set_prototypes(server_protos)
    print(f"  Loaded server prototypes: {server_protos.shape}")

    # 2. Load data for the target sensor
    print(f"Loading data for sensor {args.sensor_id}...")
    X, Y, tids, raw_speed, raw_temporal = load_sensor_data(
        args.sensor_id, args.dataset, args.data_dir, window_size, horizon)
    n_windows, _, _ = X.shape
    print(f"  {n_windows} total windows")

    # 3. Extract windows for the target date
    w_start, w_end = date_to_windows(args.date, n_windows, window_size, horizon)
    print(f"  Target date {args.date}: windows [{w_start}, {w_end}) = {w_end - w_start} windows")

    X_day = X[w_start:w_end]
    Y_day = Y[w_start:w_end]
    tids_day = tids[w_start:w_end]

    if X_day.shape[0] == 0:
        print("ERROR: No windows for target date!")
        return

    # 4. Run inference and collect prototype weights
    print("Running inference...")
    model.eval()
    all_weights = []
    all_preds = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, X_day.shape[0], batch_size):
            x_batch = torch.tensor(X_day[i:i + batch_size], dtype=torch.float32, device=device)
            t_batch = torch.tensor(tids_day[i:i + batch_size], dtype=torch.float32, device=device)

            y_hat, z_inv, weights = model(x_batch, temporal_id=t_batch)
            all_preds.append(y_hat.squeeze(-1).cpu().numpy())
            all_weights.append(weights.mean(dim=1).cpu().numpy())  # avg over T dimension

    preds = np.concatenate(all_preds, axis=0)  # (N, horizon)
    weights = np.concatenate(all_weights, axis=0)  # (N, K)

    print(f"  Predictions: {preds.shape}, weights: {weights.shape}")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Avg entropy: {float(-np.sum(weights * np.log(weights + 1e-10), axis=1).mean()):.3f}")

    # Dominant prototype distribution
    dominant = weights.argmax(axis=1)
    proto_counts = np.bincount(dominant, minlength=K)
    print(f"  Dominant proto distribution: {proto_counts}")
    print(f"  Unique protos used: {np.sum(proto_counts > 0)}")

    # 5. Plot
    fmts = ["pdf", "png", "svg"] if args.format == "all" else [args.format]
    out_paths = []
    for fmt in fmts:
        path = os.path.join(out_dir, f"fig_march15_proto_weights_s{args.sensor_id}.{fmt}")
        out_paths.append((fmt, path))

    plot_march15(X_day, Y_day, tids_day, raw_speed, weights, preds,
                 args.sensor_id, args.date, K, out_paths,
                 window_size=window_size, horizon=horizon)
    print("Done.")


if __name__ == "__main__":
    main()
