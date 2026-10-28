"""
3-panel prototype clustering figure for CausalFedST.

Shows that prototypes partition sensors into spatial/functional groups,
NOT temporal regimes. The codebook learns a sensor-level clustering where
each prototype anchors a distinctive daily traffic pattern.

Panel (a): Spatial map — 207 METR-LA sensors colored by dominant prototype.
Panel (b): Per-prototype mean daily speed profile (±1 std band).
Panel (c): MAE distribution across prototypes — de-confounding verification.

Usage:
    # Step 1: collect data (run the data collection script first)
    python scripts/plot_prototype_clustering_figure.py \\
        --data_pkl /tmp/prototype_sensor_data.pkl \\
        --output_dir output/zinv_ksweep_all_k10_r200
"""

import os, sys, argparse, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/causal_fedts_mpl_{os.getuid()}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

P10 = np.array([
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
    "#B07AA1", "#EDC948", "#B6992D", "#FF9DA7", "#9C755F",
])


def parse_args():
    p = argparse.ArgumentParser(description="3-panel prototype clustering figure")
    p.add_argument("--data_pkl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--format", type=str, default="pdf",
                   choices=["pdf", "png", "svg", "all"])
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Panel (a): Spatial map of prototype membership
# ═══════════════════════════════════════════════════════════════════════

def panel_a(ax, results, K):
    lats = np.array([r["lat"] for r in results])
    lons = np.array([r["lon"] for r in results])
    protos = np.array([r["dom_proto"] for r in results])

    for k in range(K):
        mask = protos == k
        ax.scatter(lons[mask], lats[mask], c=[P10[k]], s=14, alpha=0.85,
                   edgecolors="white", linewidths=0.3, zorder=3,
                   label=f"Proto {k}  ({mask.sum()})")

    ax.set_xlabel("Longitude", fontsize=8.5, labelpad=2)
    ax.set_ylabel("Latitude", fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=6.5, pad=1.5)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [Patch(facecolor=P10[k], alpha=0.85, label=f"Proto {k}")
               for k in range(K)]
    ax.legend(handles=handles, fontsize=5.5, loc="lower left",
              ncol=2, framealpha=0.6, borderpad=0.2,
              handlelength=0.9, labelspacing=0.12, columnspacing=0.4)

    ax.set_title("(a)  Spatial prototype membership  |  METR-LA 207 sensors  |  K=10",
                 fontsize=10, fontweight="bold", loc="left", pad=10)


# ═══════════════════════════════════════════════════════════════════════
# Panel (b): Prototype daily profiles
# ═══════════════════════════════════════════════════════════════════════

def panel_b(ax, results, K):
    hours = np.arange(24)
    hour_smooth = np.linspace(0, 23, 200)

    for k in range(K):
        members = [r for r in results if r["dom_proto"] == k]
        if len(members) == 0:
            continue
        profiles = np.stack([m["avg_by_hour"] for m in members])  # (n_members, 24)
        mu = profiles.mean(axis=0)
        std = profiles.std(axis=0)

        # Smooth
        try:
            mu_smooth = make_interp_spline(hours, mu, k=3)(hour_smooth)
            # also smooth std for band
            std_smooth = make_interp_spline(hours, mu + std, k=3)(hour_smooth)
            std_lower = make_interp_spline(hours, np.clip(mu - std, 0, None), k=3)(hour_smooth)
        except Exception:
            mu_smooth = np.interp(hour_smooth, hours, mu)
            std_smooth = np.interp(hour_smooth, hours, mu + std)
            std_lower = np.interp(hour_smooth, hours, np.clip(mu - std, 0, None))

        ax.plot(hour_smooth, mu_smooth, color=P10[k], lw=1.8,
                label=f"Proto {k}  (n={len(members)})")
        ax.fill_between(hour_smooth, std_lower, std_smooth,
                        color=P10[k], alpha=0.08, lw=0)

    # Annotate peak hours
    ax.axvspan(6, 10, color="#F28E2B", alpha=0.04, lw=0)
    ax.axvspan(16, 20, color="#F28E2B", alpha=0.04, lw=0)
    ax.text(8, ax.get_ylim()[1] * 0.97, "AM", fontsize=7, ha="center",
            color="#D07020", alpha=0.7)
    ax.text(18, ax.get_ylim()[1] * 0.97, "PM", fontsize=7, ha="center",
            color="#D07020", alpha=0.7)

    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"],
                       fontsize=7.5)
    ax.set_xlabel("Hour of day", fontsize=8.5, labelpad=2)
    ax.set_ylabel("Speed (mph)", fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.grid(axis="y", alpha=0.12, lw=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=5.5, loc="upper left", ncol=2,
              framealpha=0.6, borderpad=0.2,
              handlelength=1.0, labelspacing=0.12, columnspacing=0.4)

    ax.set_title("(b)  Mean daily speed profile per prototype  |  ±1σ band",
                 fontsize=10, fontweight="bold", loc="left", pad=10)


# ═══════════════════════════════════════════════════════════════════════
# Panel (c): MAE by prototype group
# ═══════════════════════════════════════════════════════════════════════

def panel_c(ax, results, K):
    # Box plot of per-sensor MAE within each prototype group
    data_by_proto = []
    labels = []
    colors = []
    for k in range(K):
        members = [r for r in results if r["dom_proto"] == k]
        maes = [m["mae"] for m in members]
        if len(maes) > 0:
            data_by_proto.append(maes)
            labels.append(f"P{k}\n({len(maes)})")
            colors.append(P10[k])

    bp = ax.boxplot(data_by_proto, labels=labels, patch_artist=True,
                    widths=0.55, showfliers=True, flierprops=dict(
                        marker="o", markersize=2.5, markerfacecolor="0.5",
                        markeredgewidth=0.3, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.7)
        whisker.set_color("0.3")
    for cap in bp["caps"]:
        cap.set_linewidth(0.7)
        cap.set_color("0.3")
    for median in bp["medians"]:
        median.set_linewidth(1.2)
        median.set_color("0.15")
    for flier in bp["fliers"]:
        flier.set_markeredgewidth(0.3)

    # Overall mean line
    all_maes = [r["mae"] for r in results]
    overall_mean = np.mean(all_maes)
    ax.axhline(y=overall_mean, color="0.3", lw=0.8, ls="--", alpha=0.6)
    ax.text(len(data_by_proto) - 0.3, overall_mean,
            f"μ={overall_mean:.2f}", fontsize=6.5, va="bottom", ha="right",
            color="0.35")

    ax.set_ylabel("MAE (mph)", fontsize=8.5, labelpad=2)
    ax.set_xlabel("Prototype", fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.12, lw=0.3)

    ax.set_title("(c)  Per-sensor MAE by prototype group  |  z_inv-only  |  test set",
                 fontsize=10, fontweight="bold", loc="left", pad=10)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.data_pkl}...")
    with open(args.data_pkl, "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    K = data["K"]
    print(f"  {len(results)} sensors, K={K}")

    # Summary stats
    for k in range(K):
        members = [r for r in results if r["dom_proto"] == k]
        mean_mae = np.mean([m["mae"] for m in members]) if members else 0
        print(f"  Proto {k}: {len(members)} sensors, mean MAE={mean_mae:.3f}")

    fig = plt.figure(figsize=(14, 11))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.0, 1.0],
                            hspace=0.32, wspace=0.32,
                            left=0.07, right=0.96, top=0.95, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a(ax_a, results, K)
    panel_b(ax_b, results, K)
    panel_c(ax_c, results, K)

    fig.text(0.5, 0.01,
             "CausalFedST prototype clustering  |  METR-LA  |  K=10  |  τ=0.1  |  "
             "z_inv-only federated  |  Prototypes partition sensors into functional groups; "
             "temporal dynamics come from encoder time embeddings",
             ha="center", fontsize=6.5, color="0.45", style="italic")

    fmts = ["pdf", "png", "svg"] if args.format == "all" else [args.format]
    for fmt in fmts:
        path = os.path.join(args.output_dir, f"fig_prototype_clustering.{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved: {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
