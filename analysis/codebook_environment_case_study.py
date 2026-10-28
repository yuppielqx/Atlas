#!/usr/bin/env python3
"""
Codebook Environment Case Study — Focused Two-Window Comparison

Logic (correct causal narrative):
  1. Find two input windows where PM2.5 distribution clearly differs
     (e.g., a "clean" window vs a "polluted" window at the same station)
  2. Observe how the codebook's soft weight distribution shifts
     between these two windows
  3. Show that other meteorological variables (temperature, wind, humidity...)
     also differ between these windows
  4. → The codebook weights encode the full environmental regime,
     not just PM2.5; each prototype combination corresponds to a
     physically meaningful atmospheric state.

Usage:
  python3 analysis/codebook_environment_case_study.py \
    --checkpoint output/knowair-pm25_K10_nall_horizon12_20260714_091638/checkpoint.pt \
    --out_dir analysis/case_study
"""

from __future__ import annotations
import argparse, os, sys, zipfile
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from models.causal_fedst import CausalFedST
from data.dataset import generate_metrla_clients

# ── Metadata ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "z_t2m","z_blh","t2m","skt","blh","kx","r","r_950","tp","sp",
    "d2m","t_950","sf","z_sp","z_u10","u10","v10","pm25",
]
FEATURE_LABELS = {
    "t2m": "2m Temp (K)",    "d2m": "2m Dewpoint (K)",
    "sp": "Surface Pressure", "blh": "Boundary Layer Height",
    "u10": "U-wind (m/s)",   "v10": "V-wind (m/s)",
    "tp": "Total Precip",    "r": "Rel Humidity",
    "sf": "Snowfall",        "kx": "K Index",
    "pm25": "PM2.5 (μg/m³)",
}
PROTO_COLORS = plt.cm.tab10.colors

# KnowAir dataset: starts 2015-01-01, 3-hourly
DATA_START = datetime(2015, 1, 1, 0, 0, 0)
DATA_FREQ_H = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", default="analysis/case_study")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--station", type=int, default=None,
                   help="Station index; auto-picked by PM2.5 variance if unset")
    p.add_argument("--window_len", type=int, default=48,
                   help="Length of each comparison window (time steps)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_raw_knowair():
    with zipfile.ZipFile(REPO_ROOT / "datasets" / "KnowAir.npy.zip") as zf:
        with zf.open("KnowAir.npy") as f:
            return np.load(f, allow_pickle=True).astype(np.float64)


def load_checkpoint_and_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    proto = ckpt["server_prototypes"].to(device)
    K = proto.shape[0]
    sd0 = ckpt["client_models"][0]
    ts = sd0.get("encoder.time_series_emb_layer.weight")
    in_dim = (ts.shape[1] // cfg.get("window_size", 12)) if ts is not None else 1

    model_kw = dict(
        input_dim=in_dim, hid_dim=cfg.get("hid_dim", 64),
        seq_len=cfg.get("window_size", 12), horizon=cfg.get("horizon", 12),
        K=K, tau=cfg.get("tau", 0.1),
        stid_num_layers=cfg.get("stid_num_layers", 3),
        dropout=cfg.get("dropout", 0.1),
        use_residual=cfg.get("use_residual", False),
        residual_mode=cfg.get("residual_mode", "none"),
        residual_scale=cfg.get("residual_scale", 1.0),
        encoder_type=cfg.get("encoder_type", "stid"),
        use_prototype=True,
        normalize_codebook=cfg.get("codebook_norm", True),
        prototype_init=cfg.get("prototype_init", "randn"),
        multivariate=in_dim > 1,
    )
    return ckpt, model_kw, proto, K


# ═══════════════════════════════════════════════════════════════════════
# Window selection
# ═══════════════════════════════════════════════════════════════════════

def find_contrasting_windows(pm25_series, window_len=48, gap=24):
    """
    Find two windows of length `window_len` where PM2.5 distribution
    differs maximally: one with low mean PM2.5 (clean), one with high
    mean PM2.5 (polluted). Windows must be separated by at least `gap`
    steps to avoid temporal leakage.
    """
    T = len(pm25_series)
    n_windows = T - window_len + 1
    window_means = np.array([
        pm25_series[i:i + window_len].mean() for i in range(n_windows)
    ])

    # Clean window: lowest mean PM2.5 (but not all zeros)
    clean_order = np.argsort(window_means)
    polluted_order = np.argsort(window_means)[::-1]

    clean_idx = None
    for idx in clean_order:
        if pm25_series[idx:idx + window_len].std() > 1.0:  # non-trivial
            clean_idx = idx
            break

    polluted_idx = None
    for idx in polluted_order:
        if abs(idx - clean_idx) >= gap + window_len:
            if pm25_series[idx:idx + window_len].std() > 1.0:
                polluted_idx = idx
                break

    return clean_idx, polluted_idx


# ═══════════════════════════════════════════════════════════════════════
# Model forward pass for a window
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_codebook_weights_for_window(model, X_windows, T_windows, device):
    """
    Run the model on a batch of input windows.

    Args:
        X_windows: (N, 12, 1) — N consecutive 12-step input windows
        T_windows: (N, 2)    — temporal IDs for each window

    Returns:
        weights: (N, K) soft assignment weights
        (Encoder outputs (B, 1, hid_dim), so prototype gives (B, 1, K) → squeeze to (N, K))
    """
    model.eval()
    X = torch.from_numpy(X_windows).float().to(device)   # (N, 12, 1)
    T_id = torch.from_numpy(T_windows).to(device)         # (N, 2)
    z = model.encoder(X, temporal_id=T_id)                # (N, 1, hid_dim)
    _, weights = model.prototype(z)                       # (N, 1, K)
    return weights.squeeze(1).cpu().numpy()               # (N, K)


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def make_figure(
    pm25_clean, pm25_polluted,
    weights_clean, weights_polluted,
    weather_clean, weather_polluted,
    K, out_path, station_idx,
    date_clean, date_polluted,
):
    """2x2 publication-quality figure.

    (a) top-left:     PM2.5 - clean window
    (b) top-right:    PM2.5 - polluted window
    (c) bottom-left:  Codebook soft-weight distribution (clean vs polluted)
    (d) bottom-right: Weather context shift (dumbbell chart with actual values)
    """
    window_len = len(pm25_clean)
    t_axis = np.arange(window_len)
    wx_vars = [f for f in FEATURE_NAMES[:17] if f in FEATURE_LABELS]

    # Pick date ticks: one per day (8 steps = 24h at 3h intervals)
    def _date_ticks(dates, step=8):
        idxs = list(range(0, len(dates), step))
        if idxs[-1] != len(dates) - 1:
            idxs.append(len(dates) - 1)
        return idxs, [dates[i] for i in idxs]

    ticks_clean, labels_clean = _date_ticks(date_clean)
    ticks_poll, labels_poll = _date_ticks(date_polluted)

    # ── Colour palette ──
    C_CLEAN    = "#2166AC"   # deep blue
    C_CLEAN_BG = "#81C6F4"   # light blue fill
    C_POLL     = "#B2182B"   # deep red
    C_POLL_BG  = "#F69C92"   # light red fill
    C_NEUTRAL  = "#525252"   # dark grey for text / grid

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
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.35,
                           height_ratios=[1, 0.7])
    ax_c1 = fig.add_subplot(gs[0, 0])  # clean stackplot
    ax_c2 = fig.add_subplot(gs[0, 1])  # polluted stackplot
    ax_wx_c = fig.add_subplot(gs[1, 0])  # weather at clean rep point
    ax_wx_p = fig.add_subplot(gs[1, 1])  # weather at polluted rep point

    # ── PM2.5 summary inset (stat boxes only, no separate PM2.5 panels) ──
    # (handled inside the stackplot titles)

    # ─────────────────────────────────────────────────────────────
    # (a, b) Codebook weight stackplots — one per period
    # ─────────────────────────────────────────────────────────────
    w_clean_mean = weights_clean.mean(axis=0)  # (K,)
    w_polluted_mean = weights_polluted.mean(axis=0)
    delta_w = w_polluted_mean - w_clean_mean
    cos_sim = np.dot(w_clean_mean, w_polluted_mean) / (
        np.linalg.norm(w_clean_mean) * np.linalg.norm(w_polluted_mean) + 1e-8)

    # Keep top-4 most variable prototypes + group the rest as "others"
    proto_rank = np.argsort(np.abs(delta_w))[::-1]
    top4 = proto_rank[:4]
    other_mask = np.ones(K, dtype=bool)
    other_mask[top4] = False
    proto_order = list(top4)  # top 4 in decreasing importance

    proto_colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#F6BCB5",
        "#937860", "#D5E8F5", "#8C8C8C", "#FBDDAD", "#ACA4B4",
    ][:K]

    def _draw_stackplot(ax, weights, pm25, dates, ticks, labels, title,
                         period_color, panel_label, marker_t=None, marker_label="",
                         marker_y=0.50, legend_loc=(0.0, 0.50),
                         marker_offset=(0, 0)):
        """Draw a prototype weight stackplot with PM2.5 overlay."""
        w_top4 = weights[:, proto_order]
        w_other = weights[:, other_mask].sum(axis=1, keepdims=True)
        w_stacked = np.concatenate([w_top4, w_other], axis=1)

        stack_colors = [proto_colors[k] for k in proto_order] + ["#BDBDBD"]
        stack_labels = [f"P{k}" for k in proto_order] + ["others"]

        ax.stackplot(np.arange(len(weights)), w_stacked.T,
                     colors=stack_colors, labels=stack_labels, alpha=0.82,
                     edgecolor="white", linewidth=0.4)

        ax2 = ax.twinx()
        ax2.plot(np.arange(len(pm25)), pm25, color="black", linewidth=2.0,
                 alpha=0.7, zorder=5)
        ax2.set_ylabel("PM2.5", fontsize=18, color="black")
        ax2.tick_params(axis="y", labelcolor="black", labelsize=16)

        # Mark the representative time point
        if marker_t is not None:
            ax.axvline(x=marker_t, color=period_color, linewidth=2.0,
                       linestyle="--", alpha=0.9, zorder=6)
            ax.annotate(marker_label, xy=(marker_t, marker_y),
                        xycoords=("data", "axes fraction"), fontsize=16,
                        color=period_color, fontweight="bold",
                        xytext=marker_offset, textcoords="offset points",
                        ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  alpha=0.9, edgecolor=period_color, linewidth=0.8))

        ax.set_xlim(0, len(weights) - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, ha="center", fontsize=16)
        ax.set_ylabel("Prototype weight", fontsize=18)
        ax.set_ylim(0, 1.0)
        # Hide "0.0" y-tick label to avoid overlap with x-axis date ticks
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '' if abs(v) < 1e-6 else f'{v:.1f}'))
        ax.grid(alpha=0.15, linewidth=0.3)
        # Combine prototype legend + PM2.5 entry
        h1, l1 = ax.get_legend_handles_labels()
        pm25_line = plt.Line2D([0], [0], color="black", linewidth=2.0, alpha=0.7)
        ax.legend(handles=list(h1) + [pm25_line],
                  labels=list(l1) + ["PM2.5"],
                  loc="center left", bbox_to_anchor=legend_loc,
                  fontsize=14, ncol=2, framealpha=0.3, edgecolor="black"
                  , labelspacing=0.05, handletextpad=0.1, columnspacing=0.4, borderpad=0.3)
        # Title below x-axis with panel label
        ax.text(0.5, -0.1, f"({panel_label})  {title}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold")

    # Identify representative time points
    # Clean window: point where P6 weight is highest (most "clean-like")
    p6_clean = proto_order[0] if proto_order[0] in [0,1,2,3,4,5,6,7,8,9] else proto_order[0]
    # Actually find the prototype that dominates in clean but drops in polluted
    p_clean_dom = int(np.argmin(delta_w))  # most negative delta = drops most in polluted
    p_poll_dom  = int(np.argmax(delta_w))  # most positive delta = rises most in polluted
    t_clean_mark = int(np.argmax(weights_clean[:, p_clean_dom]))
    t_poll_mark  = int(np.argmax(weights_polluted[:, p_poll_dom]))

    _draw_stackplot(ax_c1, weights_clean, pm25_clean, date_clean,
                    ticks_clean, labels_clean,
                    "Clean period",
                    C_CLEAN, panel_label="a", marker_t=t_clean_mark,
                    marker_label=f"P{p_clean_dom} peak \n (PM2.5={pm25_clean[t_clean_mark]:.0f})",
                    marker_y=0.65, legend_loc=(0.0, 0.65),
                    marker_offset=(60, 0))

    _draw_stackplot(ax_c2, weights_polluted, pm25_polluted, date_polluted,
                    ticks_poll, labels_poll,
                    "Polluted period",
                    C_POLL, panel_label="b", marker_t=t_poll_mark,
                    marker_label=f"P{p_poll_dom} peak \n (PM2.5={pm25_polluted[t_poll_mark]:.0f})",
                    marker_y=0.28, legend_loc=(0.0, 0.45),
                    marker_offset=(-70, -30))

    # ─────────────────────────────────────────────────────────────
    # (d) Weather context shift — dumbbell chart with actual values
    # ─────────────────────────────────────────────────────────────
    WX_UNITS = {
        "t2m": "K", "d2m": "K", "skt": "K", "t_950": "K",
        "sp": "hPa", "tp": "mm", "sf": "mm",
        "blh": "m", "r": "%", "r_950": "%",
        "u10": "m/s", "v10": "m/s",
        "kx": "°C",
        # z-scored variables
        "z_t2m": "σ", "z_blh": "σ", "z_sp": "σ", "z_u10": "σ",
    }
    WX_CONV = {
        "sp": 0.01,   # Pa → hPa
        "tp": 1000.0,  # m → mm
        "sf": 1000.0,  # m → mm
        "v10": None,   # drop — all zeros in this dataset
    }

    # Variables to drop entirely (e.g. all zeros)
    WX_DROP = {"v10", "sf"}  # v10 all zeros; snowfall negligible in this dataset

    # Use the two representative time points (not window averages)
    wx_info = []
    for j, name in enumerate(FEATURE_NAMES[:17]):
        if name not in FEATURE_LABELS:
            continue
        if name in WX_DROP:
            continue
        conv = WX_CONV.get(name, 1.0)
        c_val = weather_clean[t_clean_mark, j] * conv
        p_val = weather_polluted[t_poll_mark, j] * conv
        es = abs(p_val - c_val)  # raw difference for ranking
        unit = WX_UNITS.get(name, "")
        label = FEATURE_LABELS.get(name, name)
        wx_info.append((name, label, c_val, p_val, unit, es, j))

    # Keep top-8 most changed variables
    wx_info.sort(key=lambda x: x[5], reverse=True)
    wx_info = wx_info[:8]

    # Shorter labels for the weather panels
    WX_SHORT = {
        "2m Temp (K)": "2m Temp", "2m Dewpoint (K)": "2m Dewpoint",
        "Surface Pressure": "Surf. Pressure", "Boundary Layer Height": "BL Height",
        "U-wind (m/s)": "U-wind", "V-wind (m/s)": "V-wind",
        "Total Precip": "Total Precip", "Rel Humidity": "Rel Humidity",
        "Snowfall": "Snowfall", "K Index": "K Index",
    }

    def _draw_wx_bar(ax, values, vmin_vals, vmax_vals, labels, units, color, title, panel_label):
        """Horizontal bar chart — position normalised per variable with 5% floor."""
        n = len(values)
        positions = np.zeros(n)
        for i in range(n):
            raw_span = vmax_vals[i] - vmin_vals[i] + 1e-8
            # Pad range by 5% each side so bars never vanish
            pad = raw_span * 0.05
            span = raw_span + 2 * pad
            positions[i] = (values[i] - vmin_vals[i] + pad) / span
            positions[i] = max(0.02, min(0.98, positions[i]))  # clamp

        y_pos = np.arange(n)
        ax.barh(y_pos, positions, 0.55, color=color, edgecolor="white", linewidth=0.3,
                alpha=0.88)

        for i in range(n):
            v = values[i]
            if abs(v) >= 100: s = f"{v:.0f}{units[i]}"
            elif abs(v) >= 10: s = f"{v:.1f}{units[i]}"
            elif abs(v) >= 1: s = f"{v:.2f}{units[i]}"
            else: s = f"{v:.3f}{units[i]}"
            ax.text(positions[i] + 0.02, i, s, ha="left", va="center",
                    fontsize=14, color='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=16)
        ax.set_xlim(-0.02, 1.25)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.15, linewidth=0.4)
        # Title below x-axis with panel label
        ax.text(0.5, -0.1, f"({panel_label})  {title}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=24, fontweight="bold")

    # Compute normalisation ranges from full window data
    wx_labels = [WX_SHORT.get(info[1], info[1]) for info in wx_info]
    wx_units  = [info[4] for info in wx_info]
    wx_clean_vals = np.array([info[2] for info in wx_info])
    wx_poll_vals  = np.array([info[3] for info in wx_info])
    wx_vmin = np.zeros(len(wx_info))
    wx_vmax = np.zeros(len(wx_info))
    for i, (name, label, c_val, p_val, unit, es, orig_j) in enumerate(wx_info):
        conv = WX_CONV.get(name, 1.0)
        all_vals = np.concatenate([weather_clean[:, orig_j] * conv,
                                    weather_polluted[:, orig_j] * conv])
        wx_vmin[i] = all_vals.min()
        wx_vmax[i] = all_vals.max()

    _draw_wx_bar(ax_wx_c, wx_clean_vals, wx_vmin, wx_vmax, wx_labels, wx_units,
                 C_CLEAN_BG,
                 f"Clean snapshot",
                 "c")
    _draw_wx_bar(ax_wx_p, wx_poll_vals, wx_vmin, wx_vmax, wx_labels, wx_units,
                 C_POLL_BG,
                 f"Polluted snapshot",
                 "d")

    # ─────────────────────────────────────────────────────────────
    # Finishing
    # ─────────────────────────────────────────────────────────────
    # fig.suptitle(
    #     f"Station {station_idx} — Codebook encodes environmental regime, not just PM$_{{2.5}}$",
    #     fontsize=13, fontweight="bold", y=1.01)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.91, bottom=0.14,
                         wspace=0.35)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Summary: Station {station_idx}")
    print(f"{'='*60}")
    print(f"  Clean window:    PM2.5 = {pm25_clean.mean():.1f} ± {pm25_clean.std():.1f}")
    print(f"  Polluted window: PM2.5 = {pm25_polluted.mean():.1f} ± {pm25_polluted.std():.1f}")
    print(f"\n  Top-3 shifting prototypes:")
    top3 = np.argsort(np.abs(delta_w))[-3:][::-1]
    for k in top3:
        print(f"    P{k}: {w_clean_mean[k]:.3f} → {w_polluted_mean[k]:.3f}  (Δ={delta_w[k]:+.3f})")
    print(f"\n  Representative-point weather (P{p_clean_dom} peak → P{p_poll_dom} peak):")
    for name, label, c_val, p_val, unit, es, orig_j in wx_info[-5:]:
        print(f"    {label}: {c_val:.2f}{unit} → {p_val:.2f}{unit}")
    print(f"\n  L1 codebook shift: {np.abs(delta_w).sum():.4f}")
    print(f"  Cos-sim(clean_w, polluted_w): {cos_sim:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Codebook Environment Case Study — Two-Window Comparison")
    print("=" * 60)

    # 1. Load checkpoint & model
    print("\n[1/4] Loading checkpoint & building model...")
    ckpt, model_kw, proto, K = load_checkpoint_and_model(args.checkpoint, device)
    print(f"  K={K}  MAE={ckpt.get('metrics', {}).get('MAE', '?'):.2f}")

    # 2. Load raw data (for weather context) & client data (for model input)
    print("\n[2/4] Loading data & selecting station...")
    raw = load_raw_knowair()  # (11688, 184, 18)

    cfg = ckpt["config"]
    cd = generate_metrla_clients(
        n_clients=None, window_size=cfg.get("window_size", 12),
        horizon=cfg.get("horizon", 12), dataset_name="KnowAir-PM25",
        data_dir=cfg.get("data_dir", "./datasets"), seed=cfg.get("seed", 42))

    # Pick station
    if args.station is not None:
        station_idx = args.station
    else:
        pm25_var = np.array([raw[:, s, 17].std() for s in range(raw.shape[1])])
        station_idx = int(np.argmax(pm25_var))
    print(f"  Selected station {station_idx}")

    # 3. Build model for this station & load weights
    print("\n[3/4] Building station model & computing codebook weights...")
    model = CausalFedST(**model_kw).to(device)
    model.load_state_dict(ckpt["client_models"][station_idx], strict=False)
    model.set_prototypes(proto)
    model.eval()

    # Use dataset test windows directly for model forward pass
    X_test = cd[station_idx]["X_test"]  # (N, 12, 3) — [PM2.5_norm, tod, dow]
    T_test = cd[station_idx]["T_test"]  # (N, 2)
    n_test = len(X_test)

    # PM2.5 at last step of each window, denormalized
    t_mean = cd[station_idx].get("target_mean", 0)
    t_std = cd[station_idx].get("target_std", 1)
    pm25_test_raw = X_test[:, -1, 0] * t_std + t_mean

    print(f"  Test samples: {n_test}, PM2.5 range: [{pm25_test_raw.min():.0f}, {pm25_test_raw.max():.0f}]")

    # Find contrasting windows in the test set
    clean_idx, polluted_idx = find_contrasting_windows(
        pm25_test_raw, window_len=args.window_len)

    wlen = args.window_len
    print(f"  Clean window:   idx [{clean_idx}:{clean_idx + wlen}], "
          f"PM2.5 mean={pm25_test_raw[clean_idx:clean_idx + wlen].mean():.1f}")
    print(f"  Polluted window: idx [{polluted_idx}:{polluted_idx + wlen}], "
          f"PM2.5 mean={pm25_test_raw[polluted_idx:polluted_idx + wlen].mean():.1f}")

    # Extract consecutive test windows for model forward pass
    # Model input uses only PM2.5 channel (channel 0); temporal IDs from T_test
    X_clean = X_test[clean_idx:clean_idx + wlen, :, :1]        # (wlen, 12, 1)
    T_clean = T_test[clean_idx:clean_idx + wlen]                # (wlen, 2)
    X_polluted = X_test[polluted_idx:polluted_idx + wlen, :, :1]
    T_polluted = T_test[polluted_idx:polluted_idx + wlen]

    # Get codebook weights via model forward pass
    weights_clean = get_codebook_weights_for_window(
        model, X_clean, T_clean, device)       # (wlen, K)
    weights_polluted = get_codebook_weights_for_window(
        model, X_polluted, T_polluted, device)  # (wlen, K)

    # Align raw data for PM2.5 and weather display
    # Dataset windows start at raw index: test_start + window_idx
    T_raw = raw.shape[0]
    test_start = int(T_raw * 0.8)
    raw_test = raw[test_start:, station_idx, :]  # aligned to test split start

    # Each dataset window i covers raw steps [i, i+12); we show the
    # PM2.5 at the last step of each window as the "current" value.
    pm25_clean = np.array([
        raw_test[clean_idx + i + 11, 17] for i in range(wlen)
    ])
    pm25_polluted = np.array([
        raw_test[polluted_idx + i + 11, 17] for i in range(wlen)
    ])

    # Weather: use the same time points (last step of each window)
    weather_clean = np.array([
        raw_test[clean_idx + i + 11, :17] for i in range(wlen)
    ])
    weather_polluted = np.array([
        raw_test[polluted_idx + i + 11, :17] for i in range(wlen)
    ])

    # 4. Make figure
    print("\n[4/4] Creating figure...")

    # Compute actual dates for x-axis ticks
    test_start = int(raw.shape[0] * 0.8)  # raw index where test split begins
    def _window_dates(dataset_start_idx, wlen):
        """Return 'Mon DD' strings for each step in a window."""
        dates = []
        for i in range(wlen):
            raw_idx = test_start + dataset_start_idx + i + 11  # last step of each 12-step input
            dt = DATA_START + timedelta(hours=int(raw_idx) * DATA_FREQ_H)
            dates.append(dt.strftime("%b %d"))
        return dates

    date_clean = _window_dates(clean_idx, wlen)
    date_polluted = _window_dates(polluted_idx, wlen)

    out_path = os.path.join(args.out_dir, "codebook_interpretation.pdf")
    make_figure(
        pm25_clean, pm25_polluted,
        weights_clean, weights_polluted,
        weather_clean, weather_polluted,
        K, out_path, station_idx,
        date_clean, date_polluted,
    )

    print(f"\nDone. Figure saved to {out_path}")


if __name__ == "__main__":
    main()
