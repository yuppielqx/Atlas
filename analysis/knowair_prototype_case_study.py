#!/usr/bin/env python3
"""
KnowAir-PM25 Prototype Case Study — Event-Driven Attribution

Logic: PM2.5 regime change → prototype weight distribution shifts
       → weather context changes → prototype codebook captures environments.

Uses the FULL softmax weight distribution (not argmax), because individual
prototypes form a distributed basis.

Panels:
  A. PM2.5-regime prototype response: per-regime mean weight vector ± shift
  B. Single-station event strips: PM2.5 + prototype weight stackplot + weather
  C. Prototype–weather association via weight-weather correlation per station
  D. Cross-station consistency: same prototype responds to same weather

Usage:
  python3 analysis/knowair_prototype_case_study.py \
    --checkpoint output/knowair-pm25_K10_nall_horizon12_20260714_091638/checkpoint.pt \
    --out_dir analysis/case_study
"""

from __future__ import annotations
import argparse, json, os, sys, zipfile
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from models.causal_fedst import CausalFedST

FEATURE_NAMES = [
    "z_t2m","z_blh","t2m","skt","blh","kx","r","r_950","tp","sp",
    "d2m","t_950","sf","z_sp","z_u10","u10","v10","pm25",
]
TARGET_IDX = 17
PROTO_CMAP = plt.cm.tab10

REGIME_NAMES = {0:"Clean",1:"Moderate",2:"Polluted",3:"Rising",4:"Falling"}
REGIME_COLORS = {0:"#2ECC71",1:"#3498DB",2:"#E74C3C",3:"#F39C12",4:"#9B59B6"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", default="analysis/case_study")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", type=int, default=500)
    return p.parse_args()


def load_raw_knowair():
    with zipfile.ZipFile(REPO_ROOT / "datasets" / "KnowAir.npy.zip") as zf:
        with zf.open("KnowAir.npy") as f:
            return np.load(f, allow_pickle=True).astype(np.float64)


def load_checkpoint_and_data(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    proto = ckpt["server_prototypes"].to(device)
    K = proto.shape[0]
    sd0 = ckpt["client_models"][0]
    ts = sd0.get("encoder.time_series_emb_layer.weight")
    in_dim = (ts.shape[1] // cfg.get("window_size",12)) if ts is not None else 1

    # Load raw data for weather context
    raw = load_raw_knowair()

    model_kw = dict(input_dim=in_dim, hid_dim=cfg.get("hid_dim",64),
        seq_len=cfg.get("window_size",12), horizon=cfg.get("horizon",12),
        K=K, tau=cfg.get("tau",0.1), stid_num_layers=cfg.get("stid_num_layers",3),
        dropout=cfg.get("dropout",0.1),
        use_residual=cfg.get("use_residual",False),
        residual_mode=cfg.get("residual_mode","none"),
        residual_scale=cfg.get("residual_scale",1.0),
        encoder_type=cfg.get("encoder_type","stid"),
        use_prototype=True, normalize_codebook=cfg.get("codebook_norm",True),
        prototype_init=cfg.get("prototype_init","randn"),
        multivariate=in_dim>1)
    return ckpt, model_kw, raw, proto, K, in_dim


def classify_pm25_regime(pm25, window=8):
    T = len(pm25)
    labels = np.full(T, -1, dtype=int)
    lo, hi = np.percentile(pm25, 30), np.percentile(pm25, 70)
    sigma = pm25.std()
    for t in range(T):
        labels[t] = 0 if pm25[t] < lo else (2 if pm25[t] > hi else 1)
    for t in range(window, T):
        d = pm25[t] - pm25[t-window]
        if d > sigma:   labels[t] = 3
        elif d < -sigma: labels[t] = 4
    return labels


@torch.no_grad()
def collect_all(ckpt, model_kw, proto, device, n_samples):
    """Run forward pass on all 184 stations to collect prototype weights and PM2.5."""
    K = proto.shape[0]
    from data.dataset import generate_metrla_clients
    cfg = ckpt["config"]
    cd = generate_metrla_clients(
        n_clients=None, window_size=cfg.get("window_size",12),
        horizon=cfg.get("horizon",12), dataset_name="KnowAir-PM25",
        data_dir=cfg.get("data_dir","./datasets"), seed=cfg.get("seed",42))

    all_pm25 = {}; all_w = {}; all_regime = {}
    for i in range(len(cd)):
        cdata = cd[i]
        m = CausalFedST(**model_kw).to(device)
        m.load_state_dict(ckpt["client_models"][i], strict=False)
        m.set_prototypes(proto)
        m.eval()
        n = min(n_samples, len(cdata["X_test"]))
        # X has 3 channels: [PM2.5, tod, dow]. Model expects input_dim=1 (only PM2.5).
        X = torch.from_numpy(cdata["X_test"][:n, :, :1]).float().to(device)
        T_id = torch.from_numpy(cdata["T_test"][:n]).to(device)
        t_mean, t_std = cdata.get("target_mean",0), cdata.get("target_std",1)
        # Raw PM2.5 from channel 0
        pm25_z = cdata["X_test"][:n, -1, 0]
        pm25_raw = pm25_z * t_std + t_mean
        all_pm25[i] = pm25_raw
        regime = classify_pm25_regime(pm25_raw)
        all_regime[i] = regime
        # Forward
        with torch.no_grad():
            z = m.encoder(X, temporal_id=T_id)
            _, w = m.prototype(z)
        all_w[i] = w.mean(dim=1).cpu().numpy()  # (T, K)
        if i % 50 == 0: print(f"  Station {i}/{len(cd)}...")
    return all_pm25, all_w, all_regime, cd


# ═══════════════════════ Panel A: Regime-conditioned prototype weights ═══════════════════════

def panel_regime_response(all_w, all_regime, K, out_path):
    """For each regime, compute mean weight per prototype. Show as heatmap
    of Δweight from baseline (uniform=0.1), highlighting regime-specific shifts."""
    r_sum = {r: np.zeros(K) for r in range(5)}
    r_cnt = {r: 0 for r in range(5)}
    for sidx in all_w:
        w = all_w[sidx]; r = all_regime[sidx]
        for t in range(len(w)):
            if r[t] >= 0:
                r_sum[r[t]] += w[t]; r_cnt[r[t]] += 1
    r_avg = {r: r_sum[r] / max(r_cnt[r],1) for r in range(5)}
    # stack
    W = np.stack([r_avg[r] for r in range(5)])  # (5, K)
    baseline = W.mean(axis=0, keepdims=True)
    delta = W - baseline  # deviation from average

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
        gridspec_kw={"width_ratios":[2.5,1]})

    # Left: Δ heatmap
    ax = axes[0]
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.03, vmax=0.03, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"P{k}" for k in range(K)], fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels([REGIME_NAMES[r] for r in range(5)], fontsize=10)
    for r in range(5):
        for k in range(K):
            v = delta[r,k]
            if abs(v) > 0.005:
                ax.text(k, r, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if abs(v)>0.015 else "black")
    ax.set_title("Panel A: Prototype Weight Deviation by PM2.5 Regime\n"
                 "Δ = regime_mean − cross-regime_mean (red = above avg)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Prototype", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82); cbar.set_label("Δ weight", fontsize=10)

    # Right: which prototypes shift most?
    ax = axes[1]
    w_range = W.max(axis=0) - W.min(axis=0)
    order = np.argsort(w_range)[::-1]
    colors = [PROTO_CMAP(k/K) for k in order]
    ax.barh(range(K), w_range[order], color=colors, edgecolor="white")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"P{k}" for k in order], fontsize=10)
    ax.set_xlabel("Weight range across regimes", fontsize=11)
    ax.set_title("Prototype Responsiveness", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for i, k in enumerate(order):
        best_r = np.argmax(np.abs(delta[:,k]))
        ax.text(w_range[k]+0.0001, i, REGIME_NAMES[best_r], va="center", fontsize=8)

    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path}")

    # Print shifts
    for k in order[:3]:
        shifts = {REGIME_NAMES[r]: delta[r,k] for r in range(5)}
        top2 = sorted(shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        print(f"    Proto {k}: range={w_range[k]:.4f}  "
              f"{top2[0][0]}({top2[0][1]:+.3f}), {top2[1][0]}({top2[1][1]:+.3f})")
    return delta, order


# ═══════════════════════ Panel B: Event strips ═══════════════════════

def find_events(pm25, regime, n_events=3):
    """Find PM2.5 events: rising, polluted, falling blocks."""
    evts = []
    T = len(pm25)
    i = 0
    while i < T and len(evts) < 10:
        if regime[i] in (2,3,4):
            s = max(0, i-48); e = min(T, i+96)
            if all(abs(s - ev["start"]) > 48 for ev in evts):
                evts.append({"start":s,"end":e,"peak_idx":i,
                    "regime":regime[i],"pm25_peak":pm25[i]})
            i = e
        else:
            i += 1
    picked = []
    for target in [3,2,4]:
        for ev in evts:
            if ev["regime"] == target:
                picked.append(ev); break
    return picked[:n_events]


def panel_event_strips(all_pm25, all_w, all_regime, raw, K, out_path):
    """Pick station with highest PM2.5 variability, show 3 events."""
    # Pick station
    pm25_var = [all_pm25[s].std() for s in range(len(all_pm25))]
    sidx = int(np.argmax(pm25_var))
    pm25 = all_pm25[sidx]; w = all_w[sidx]; regime = all_regime[sidx]
    events = find_events(pm25, regime)
    print(f"  Station {sidx}: PM2.5 [{pm25.min():.0f}, {pm25.max():.0f}]")

    # Weather: raw data, 17 weather vars, station sidx
    wx_raw = raw[:, sidx, :17]  # (T_total, 17)
    T_total = len(wx_raw)
    test_start = int(T_total * 0.8)
    wx_test = wx_raw[test_start:test_start+len(pm25)]

    n_events = len(events)
    fig, axes = plt.subplots(n_events, 1, figsize=(18, 4*n_events), sharex=False)
    if n_events == 1: axes = [axes]
    colors = [PROTO_CMAP(k/K) for k in range(K)]

    # Top varying weather
    wx_var = wx_test.std(axis=0)
    top3 = np.argsort(wx_var)[-3:][::-1]

    for ei, ev in enumerate(events):
        ax = axes[ei]; s, e = ev["start"], ev["end"]
        t_ax = np.arange(s, e)

        # Regime background
        for t in range(s, e):
            r = regime[t]
            if r >= 0:
                ax.axvspan(t-0.5, t+0.5, alpha=0.10, color=REGIME_COLORS.get(r,"gray"))

        # Stacked prototype weights
        ax.stackplot(t_ax, w[s:e].T, colors=colors,
                     labels=[f"P{k}" for k in range(K)], alpha=0.8)
        ax.set_ylabel("Prototype\nWeight", fontsize=9); ax.set_ylim(0, 1.0)

        # PM2.5 twin axis
        ax2 = ax.twinx()
        ax2.plot(t_ax, pm25[s:e], "k", linewidth=2.5, zorder=5)
        ax2.set_ylabel("PM2.5 (μg/m³)", fontsize=9, color="black")

        # Weather
        ax3 = ax.twinx(); ax3.spines["right"].set_position(("axes", 1.12))
        wx_c = ["#E74C3C","#2980B9","#27AE60"]
        wnames = [FEATURE_NAMES[j] for j in top3]
        for ri, wi in enumerate(top3):
            if s < len(wx_test) and e <= len(wx_test):
                wx_z = (wx_test[s:e,wi] - wx_test[s:e,wi].mean()) / (wx_test[s:e,wi].std()+1e-8)
                ax3.plot(t_ax, wx_z, color=wx_c[ri], linewidth=1.0, alpha=0.5, linestyle="--")
        ax3.set_ylabel("Weather (z)", fontsize=8); ax3.set_yticks([])

        rname = REGIME_NAMES.get(ev["regime"],"?")
        ax.set_title(f"Event {ei+1}: {rname}  |  Peak PM2.5={ev['pm25_peak']:.0f} μg/m³  |  "
                     f"Weather: {', '.join(wnames)}", fontsize=11, fontweight="bold")
        if ei == 0: ax.legend(loc="upper left", fontsize=7, ncol=K, framealpha=0.5)

    fig.suptitle(f"Panel B: Station {sidx} — Prototype Weight Shifts During PM2.5 Events\n"
                 f"(color bands = PM2.5 regime; stackplot = prototype weights; black = PM2.5)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path}")
    return sidx


# ═══════════════════════ Panel C: Prototype–Weather association ═══════════════════════

def panel_proto_weather(all_w, all_pm25, raw, K, out_path):
    """For each prototype k, compute Pearson r(weight_k, weather_j) per station.
    Show boxplot of these correlations across 184 stations."""
    N = len(all_w)
    n_wx = 17
    proto_wx_corr = np.zeros((K, N, n_wx))  # (K, N_stations, n_weather)

    for sidx in range(N):
        w = all_w[sidx]; pm25 = all_pm25[sidx]
        T_s = len(w)
        # Weather from raw data (aligned to test split)
        T_total = raw.shape[0]
        test_start = int(T_total * 0.8)
        wx_s = raw[test_start:test_start+T_s, sidx, :17]
        for k in range(K):
            for j in range(n_wx):
                if wx_s[:,j].std() > 1e-8:
                    proto_wx_corr[k, sidx, j] = np.corrcoef(w[:,k], wx_s[:,j])[0,1]

    # For each prototype, which weather variables have strongest |correlation|?
    mean_abs = np.abs(proto_wx_corr).mean(axis=1)  # (K, n_wx)
    wx_names_short = [n.upper() for n in FEATURE_NAMES[:17]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ki in range(min(6, K)):
        ax = axes[ki//3, ki%3]
        order = np.argsort(mean_abs[ki])[::-1][:10]
        bp = ax.boxplot([proto_wx_corr[ki,:,j] for j in order],
                        positions=range(10), widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=PROTO_CMAP(ki/K), alpha=0.5),
                        medianprops=dict(color="black", linewidth=1.5))
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(10))
        ax.set_xticklabels([wx_names_short[j] for j in order],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_title(f"Proto {ki}", fontsize=12, fontweight="bold",
                     color=PROTO_CMAP(ki/K))
        ax.grid(axis="y", alpha=0.3)
        # Mark significant
        from scipy.stats import ttest_1samp
        for jj, j in enumerate(order):
            vals = proto_wx_corr[ki,:,j]
            t, p = ttest_1samp(vals[~np.isnan(vals)], 0)
            if p < 0.001: ax.annotate("***", (jj, vals.max()+.005),
                ha="center", fontsize=10, color="#E74C3C")

    fig.suptitle("Panel C: Per-Prototype Weather Correlation Across 184 Stations\n"
                 "(boxplot = distribution of Pearson r per station; *** = p<0.001)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path}")

    # Print strongest associations
    for ki in range(K):
        top3 = np.argsort(mean_abs[ki])[-3:][::-1]
        desc = ", ".join(f"{wx_names_short[j]}(r̄={mean_abs[ki,j]:+.3f})" for j in top3)
        print(f"  Proto {ki}: {desc}")

    return proto_wx_corr, mean_abs


# ═══════════════════════ Panel D: Cross-station prototype consistency ═══════════════════════

def panel_cross_station(all_w, all_pm25, raw, delta, proto_wx_corr, K, out_path):
    """Find the most regime-responsive prototype. Show that its weight correlates
    with the SAME weather variable direction across stations."""
    # Most responsive prototype
    w_range = np.abs(delta).max(axis=0) - np.abs(delta).min(axis=0)
    top_k = int(np.argmax(w_range))

    # Which weather variable has strongest correlation with this prototype?
    mean_abs = np.abs(proto_wx_corr[top_k]).mean(axis=0)
    top_wx = int(np.argmax(mean_abs))

    # Correlation per station
    stat_corr = proto_wx_corr[top_k, :, top_wx]  # (N,)
    stat_pm25 = np.array([all_pm25[s].mean() for s in range(len(all_pm25))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter: proto-weight vs weather correlation per station
    ax = axes[0]
    ax.scatter(stat_pm25, stat_corr, c=stat_corr, cmap="RdBu_r",
               s=25, alpha=0.7, edgecolors="white", linewidth=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean PM2.5 (μg/m³)", fontsize=11)
    ax.set_ylabel(f"r(weight_P{top_k}, {FEATURE_NAMES[top_wx]})", fontsize=11)
    ax.set_title(f"Proto {top_k} vs {FEATURE_NAMES[top_wx].upper()}\n"
                 f"({(stat_corr>0).mean():.0%} stations positive, "
                 f"mean r={stat_corr.mean():+.3f})",
                 fontsize=11, fontweight="bold")
    from scipy.stats import pearsonr
    rho, pval = pearsonr(stat_pm25[~np.isnan(stat_corr)],
                         stat_corr[~np.isnan(stat_corr)])
    ax.text(0.95, 0.95, f"ρ={rho:+.3f}\np={pval:.1e}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3)

    # Right: for top prototype, show its weather association profile
    ax = axes[1]
    wx_order = np.argsort(mean_abs)[::-1]
    top10 = wx_order[:10]
    bp = ax.boxplot([proto_wx_corr[top_k,:,j] for j in top10],
                    positions=range(10), widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=PROTO_CMAP(top_k/K), alpha=0.5),
                    medianprops=dict(color="black", linewidth=1.5))
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(10))
    ax.set_xticklabels([FEATURE_NAMES[j].upper() for j in top10],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r across 184 stations", fontsize=11)
    ax.set_title(f"Proto {top_k} Weather Profile (most regime-responsive)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Panel D: Cross-Station Consistency", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out_path}")

    # Which stations show strongest proto-weather coupling?
    top_stations = np.argsort(np.abs(stat_corr))[-5:][::-1]
    print(f"\n  Proto {top_k} most coupled with {FEATURE_NAMES[top_wx]}:")
    for s in top_stations:
        print(f"    Station {s}: r={stat_corr[s]:+.3f}, mean PM2.5={stat_pm25[s]:.0f}")


# ═══════════════════════ Main ═══════════════════════

def main():
    args = parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("KnowAir-PM25 Prototype Case Study — Event-Driven")
    print("="*60)

    print("\n[0/4] Loading...")
    ckpt, model_kw, raw, proto, K, in_dim = load_checkpoint_and_data(args.checkpoint, device)
    print(f"  K={K} input_dim={in_dim} MAE={ckpt.get('metrics',{}).get('MAE','?'):.2f}")

    print(f"\n[1/4] Collecting prototype weights ({args.n_samples} samples × 184 stations)...")
    all_pm25, all_w, all_regime, cd = collect_all(ckpt, model_kw, proto, device, args.n_samples)

    print("\n[2/4] Panel A: Regime-conditioned prototype response...")
    delta, order = panel_regime_response(
        all_w, all_regime, K,
        os.path.join(args.out_dir, "panel_A_regime_response.png"))

    print("\n[3/4] Panel B: Event strips...")
    panel_event_strips(all_pm25, all_w, all_regime, raw, K,
                       os.path.join(args.out_dir, "panel_B_event_strips.png"))

    print("\n[4/4] Panel C+D: Prototype–weather + cross-station...")
    proto_wx_corr, mean_abs = panel_proto_weather(
        all_w, all_pm25, raw, K,
        os.path.join(args.out_dir, "panel_C_proto_weather.png"))
    panel_cross_station(all_w, all_pm25, raw, delta, proto_wx_corr, K,
                        os.path.join(args.out_dir, "panel_D_cross_station.png"))

    print(f"\nDone. Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
