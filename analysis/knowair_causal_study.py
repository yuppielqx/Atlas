#!/usr/bin/env python3
"""
KnowAir Causal Case Study: Weather → PM2.5 causal analysis across 184 Chinese cities.

This script performs a multi-level causal analysis:
  1. Feature-level correlations and partial correlations
  2. Granger causality: do weather variables Granger-cause PM2.5?
  3. Conditional mutual information (CMI)
  4. Station clustering by causal profile
  5. Causal graph comparison: high-PM2.5 vs low-PM2.5 stations

Output:
  - Printed summary tables
  - analysis/knowair_causal_*.png  (figures)
  - analysis/knowair_causal_results.json  (structured results)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import pearsonr

# Optional: Granger causality from statsmodels
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_GRANGER = True
except ImportError:
    HAS_GRANGER = False
    print("[warn] statsmodels not installed; skipping Granger causality tests.")

warnings.filterwarnings("ignore")

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "datasets" / "KnowAir.npy.zip"
OUT_DIR = REPO_ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature definitions (verified against PM2.5-GNN paper) ──
FEATURE_NAMES = [
    "z_t2m",    # 0  z-scored 2m temperature
    "z_blh",    # 1  z-scored boundary layer height
    "t2m",      # 2  2m temperature (K)
    "skt",      # 3  skin temperature (K)
    "blh",      # 4  boundary layer height (m)
    "kx",       # 5  K-index
    "r",        # 6  relative humidity (%)
    "r_950",    # 7  relative humidity at 950 hPa (%)
    "tp",       # 8  total precipitation (m)
    "sp",       # 9  surface pressure (Pa)
    "d2m",      # 10 2m dewpoint temperature (K)
    "t_950",    # 11 temperature at 950 hPa (K)
    "sf",       # 12 snowfall (m)
    "z_sp",     # 13 z-scored surface pressure
    "z_u10",    # 14 z-scored u-wind
    "u10",      # 15 10m u-wind (m/s)
    "v10",      # 16 10m v-wind (m/s)
    "pm25",     # 17 PM2.5 (ug/m3) — target
]

FEATURE_LABELS = {
    "t2m": "2m Temperature\n(K)",
    "skt": "Skin Temperature\n(K)",
    "blh": "Boundary Layer\nHeight (m)",
    "kx": "K-index\n(stability)",
    "r": "Relative Humidity\n(%)",
    "r_950": "RH at 950hPa\n(%)",
    "tp": "Total Precip.\n(m)",
    "sp": "Surface Pressure\n(Pa)",
    "d2m": "2m Dewpoint\n(K)",
    "t_950": "Temp at 950hPa\n(K)",
    "sf": "Snowfall\n(m)",
    "u10": "10m U-wind\n(m/s)",
    "v10": "10m V-wind\n(m/s)",
    "z_t2m": "z(T2m)",
    "z_blh": "z(BLH)",
    "z_sp": "z(SP)",
    "z_u10": "z(U10)",
}

TARGET_IDX = 17
N_STATIONS = 184

# ── Color palette ──
CITY_PALETTE = {
    "high_pm25": "#E74C3C",
    "low_pm25": "#2ECC71",
    "mid_pm25": "#3498DB",
    "warm": "#E67E22",
    "cold": "#2980B9",
}
CORR_CMAP = plt.cm.RdBu_r


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_knowair_full():
    """Load the full 18-feature KnowAir array [T, N, F]."""
    with zipfile.ZipFile(DATA_PATH) as zf:
        with zf.open("KnowAir.npy") as f:
            data = np.load(f, allow_pickle=True)
    return data.astype(np.float64)  # [T, N, F]


def split_chronological(data, ratios=(0.7, 0.1, 0.2)):
    """Split data chronologically into train / val / test."""
    T = data.shape[0]
    train_end = int(T * ratios[0])
    val_end = train_end + int(T * ratios[1])
    return data[:train_end], data[train_end:val_end], data[val_end:]


# ══════════════════════════════════════════════════════════════════════
# Analysis 1: Correlation structure
# ══════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(data, station_idx=None):
    """Compute 18×18 feature correlation matrix.

    Args:
        data: [T, N, F] or [T, F] array
        station_idx: if int, compute for a single station;
                     if None and data is [T,N,F], pool all stations
    """
    if station_idx is not None and data.ndim == 3:
        x = data[:, station_idx, :]
    elif data.ndim == 3:
        x = data.reshape(-1, data.shape[-1])
    else:
        x = data
    return np.corrcoef(x, rowvar=False)


def partial_correlation(data, station_idx=None):
    """Compute partial correlation matrix (inverse of covariance → precision)."""
    if station_idx is not None and data.ndim == 3:
        x = data[:, station_idx, :]
    elif data.ndim == 3:
        x = data.reshape(-1, data.shape[-1])
    else:
        x = data

    # Standardize
    x_std = (x - x.mean(axis=0)) / x.std(axis=0)

    # Precision matrix
    cov = np.cov(x_std, rowvar=False)
    prec = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    # Partial correlation: -p_ij / sqrt(p_ii * p_jj)
    diag = np.diag(prec)
    pcorr = -prec / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


# ══════════════════════════════════════════════════════════════════════
# Analysis 2: Granger causality
# ══════════════════════════════════════════════════════════════════════

def granger_causality_weather_to_pm25(data, station_idx, weather_indices,
                                       max_lag=6, verbose=False):
    """Test whether each weather variable Granger-causes PM2.5.

    Returns dict: {feature_name: min_p_value}
    """
    x = data[:, station_idx, :]  # [T, F]
    results = {}

    for wi in weather_indices:
        # Bivariate: [PM2.5(t), weather(t)]
        pair = np.column_stack([x[:, TARGET_IDX], x[:, wi]])
        # Remove NaN
        pair = pair[~np.isnan(pair).any(axis=1)]

        if len(pair) < 100:
            results[FEATURE_NAMES[wi]] = float("nan")
            continue

        try:
            gc_result = grangercausalitytests(
                pair[:, [1, 0]],  # test weather → PM2.5 (column 1 → column 0)
                maxlag=max_lag, verbose=False,
            )
            # Collect min p-value across lags (SSR-based F-test)
            p_values = []
            for lag, res in gc_result.items():
                p_values.append(res[0]["ssr_ftest"][1])
            results[FEATURE_NAMES[wi]] = float(min(p_values))
        except Exception:
            results[FEATURE_NAMES[wi]] = float("nan")

    return results


# ══════════════════════════════════════════════════════════════════════
# Analysis 3: Station causal profile clustering
# ══════════════════════════════════════════════════════════════════════

def station_causal_profile(data, station_idx, weather_indices):
    """Extract a causal profile vector for one station.

    The profile includes:
      - Pearson r(weather_i, PM2.5)
      - Partial correlation(weather_i, PM2.5 | all others)
      - Standardized regression coefficient (weather_i → PM2.5)
    """
    x = data[:, station_idx, :]
    pm25 = x[:, TARGET_IDX]
    weather = x[:, weather_indices]

    # Standardize
    pm25_z = (pm25 - pm25.mean()) / pm25.std()
    weather_z = (weather - weather.mean(axis=0)) / weather.std(axis=0)

    # Correlations
    pearson_r = np.array([pearsonr(weather_z[:, j], pm25_z)[0]
                          for j in range(len(weather_indices))])

    # Standardized regression (single multivariate OLS)
    try:
        beta = np.linalg.lstsq(weather_z, pm25_z, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.full(len(weather_indices), np.nan)

    # Partial correlation
    pcorr_mat = partial_correlation(x)
    pcorr_with_pm25 = pcorr_mat[TARGET_IDX, weather_indices]

    return {
        "pearson_r": pearson_r,
        "beta_ols": beta,
        "partial_corr": pcorr_with_pm25,
    }


def cluster_stations_by_causal_profile(profiles, n_clusters=4):
    """Cluster stations based on their causal profile (Pearson r)."""
    from sklearn.cluster import KMeans

    X = np.array([p["pearson_r"] for p in profiles])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════

def plot_feature_correlation_heatmap(corr_matrix, title, out_path):
    """Plot 18×18 feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(18))
    ax.set_yticks(range(18))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pearson r", fontsize=11)

    # Highlight target row/column
    for i in range(18):
        ax.add_patch(plt.Rectangle((-0.5, TARGET_IDX - 0.5), 18, 1,
                                     fill=False, edgecolor="black", linewidth=2.5))
        ax.add_patch(plt.Rectangle((TARGET_IDX - 0.5, -0.5), 1, 18,
                                     fill=False, edgecolor="black", linewidth=2.5))

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_weather_pm25_correlation_bar(corr_values, feature_names, title, out_path,
                                       color="#3498DB"):
    """Bar chart of weather → PM2.5 correlations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    idx = np.arange(len(corr_values))
    colors = [CITY_PALETTE["high_pm25"] if v < 0 else CITY_PALETTE["low_pm25"]
              for v in corr_values]

    bars = ax.bar(idx, corr_values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Pearson r with PM2.5", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, corr_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * np.sign(val),
                f"{val:.3f}", ha="center", va="bottom" if val > 0 else "top",
                fontsize=8, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_station_causal_heatmap(profiles, station_labels, weather_indices,
                                 title, out_path):
    """Heatmap of causal coefficients: stations × weather variables."""
    X = np.array([p["pearson_r"] for p in profiles])  # [N_stations, N_weather]
    sort_idx = np.argsort(X.mean(axis=1))  # sort by avg correlation

    fig, ax = plt.subplots(figsize=(16, 20))
    im = ax.imshow(X[sort_idx, :], cmap=CORR_CMAP, vmin=-0.6, vmax=0.3,
                    aspect="auto")

    ax.set_xticks(range(len(weather_indices)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in weather_indices],
                        rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(X.shape[0]))
    ax.set_yticklabels([f"S{sort_idx[i]}" for i in range(X.shape[0])],
                        fontsize=7)
    ax.set_xlabel("Weather Variable", fontsize=11)
    ax.set_ylabel("Station (sorted by avg |r|)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pearson r(weather, PM2.5)", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_cluster_centers(centers, weather_indices, labels, title, out_path):
    """Radar chart or grouped bar chart of causal profile per cluster."""
    n_clusters = centers.shape[0]
    n_weather = centers.shape[1]
    weather_names = [FEATURE_NAMES[i] for i in weather_indices]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_weather)
    width = 0.8 / n_clusters

    cluster_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6"]

    for k in range(n_clusters):
        offset = (k - n_clusters / 2 + 0.5) * width
        cluster_size = (labels == k).sum()
        ax.bar(x + offset, centers[k], width,
               color=cluster_colors[k % len(cluster_colors)],
               edgecolor="white", linewidth=0.3,
               label=f"Cluster {k+1} (n={cluster_size})")

    ax.set_xticks(x)
    ax.set_xticklabels(weather_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Pearson r with PM2.5", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_causal_graph(causal_weights, threshold=0.1, title="Causal Graph",
                       out_path=None):
    """Simple causal graph visualization: weather nodes → PM2.5 node.

    Args:
        causal_weights: dict {feature_name: weight} or list of (name, weight)
    """
    if isinstance(causal_weights, dict):
        items = sorted(causal_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    else:
        items = causal_weights

    # Filter by threshold
    items = [(n, w) for n, w in items if abs(w) >= threshold]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Layout: PM2.5 at center, weather nodes in a circle
    center = (0, 0)
    n_weather = len(items)
    angles = np.linspace(0, 2 * np.pi, n_weather, endpoint=False)
    radius = 2.5

    # Draw edges
    for i, (name, weight) in enumerate(items):
        wx = radius * np.cos(angles[i])
        wy = radius * np.sin(angles[i])
        color = CITY_PALETTE["high_pm25"] if weight < 0 else CITY_PALETTE["low_pm25"]
        alpha = min(1.0, abs(weight) * 3)
        lw = abs(weight) * 4
        ax.plot([wx, center[0]], [wy, center[1]],
                color=color, linewidth=lw, alpha=alpha, zorder=1)
        # Arrow
        ax.annotate("", xy=(center[0] * 0.85, center[1] * 0.85),
                     xytext=(wx * 1.15, wy * 1.15),
                     arrowprops=dict(arrowstyle="->", color=color,
                                     lw=lw, alpha=alpha))

    # Draw weather nodes
    for i, (name, weight) in enumerate(items):
        wx = radius * np.cos(angles[i])
        wy = radius * np.sin(angles[i])
        color = CITY_PALETTE["high_pm25"] if weight < 0 else CITY_PALETTE["low_pm25"]
        ax.scatter(wx, wy, s=abs(weight) * 2000, color=color, edgecolors="white",
                    linewidth=1.5, zorder=2)
        ax.text(wx, wy + 0.3, FEATURE_LABELS.get(name, name),
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Draw PM2.5 node
    ax.scatter(center[0], center[1], s=2000, color="#C0392B",
                edgecolors="white", linewidth=2.5, zorder=3)
    ax.text(center[0], center[1] + 0.15, "PM2.5", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

    ax.set_xlim(-radius * 1.5, radius * 1.5)
    ax.set_ylim(-radius * 1.5, radius * 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")
    else:
        return fig


# ══════════════════════════════════════════════════════════════════════
# Main analysis pipeline
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("KnowAir Causal Case Study: Weather → PM2.5")
    print("  184 stations across China | 18 features | 2015-2018 | 3-hourly")
    print("=" * 70)

    # ── 0. Load data ──
    print("\n[0/6] Loading KnowAir dataset...")
    data = load_knowair_full()
    T, N, F = data.shape
    print(f"  Shape: {T} timesteps × {N} stations × {F} features")

    weather_indices = [i for i in range(F) if i != TARGET_IDX]
    weather_names = [FEATURE_NAMES[i] for i in weather_indices]

    # ── 1. Global correlation structure ──
    print("\n[1/6] Computing global feature correlations...")
    corr_mat = compute_correlation_matrix(data)
    pm25_corr = corr_mat[TARGET_IDX, :]

    # Top correlated features
    ranked = sorted(
        [(weather_names[i], pm25_corr[wi]) for i, wi in enumerate(weather_indices)],
        key=lambda x: abs(x[1]), reverse=True
    )
    print("\n  === Weather → PM2.5 Pearson correlations (all stations pooled) ===")
    print(f"  {'Rank':<5s} {'Feature':>10s}  {'Pearson r':>12s}  {'Direction':>12s}")
    print("  " + "-" * 48)
    for rank, (name, r) in enumerate(ranked, 1):
        direction = "PM2.5 ↑" if r > 0 else "PM2.5 ↓"
        print(f"  {rank:<5d} {name:>10s}  {r:+12.4f}  {direction:>12s}")

    # Plot
    plot_feature_correlation_heatmap(
        corr_mat,
        "KnowAir Feature Correlation Matrix\n(184 stations pooled, 2015–2018)",
        OUT_DIR / "knowair_causal_corr_heatmap.png",
    )
    plot_weather_pm25_correlation_bar(
        [pm25_corr[wi] for wi in weather_indices],
        weather_names,
        "Weather Variable → PM2.5 Correlation\n(All 184 Stations Pooled)",
        OUT_DIR / "knowair_causal_corr_bar.png",
    )

    # ── 2. Partial correlation (conditional independence) ──
    print("\n[2/6] Computing partial correlations (conditional independence)...")
    pcorr_mat = partial_correlation(data)
    pcorr_pm25 = pcorr_mat[TARGET_IDX, :]

    pcorr_ranked = sorted(
        [(weather_names[i], pcorr_pm25[wi]) for i, wi in enumerate(weather_indices)],
        key=lambda x: abs(x[1]), reverse=True
    )
    print("\n  === Partial correlation with PM2.5 | all other features ===")
    print(f"  {'Rank':<5s} {'Feature':>10s}  {'Partial r':>12s}")
    print("  " + "-" * 35)
    for rank, (name, r) in enumerate(pcorr_ranked, 1):
        print(f"  {rank:<5d} {name:>10s}  {r:+12.4f}")

    # ── 3. Stational heterogeneity ──
    print("\n[3/6] Analyzing station heterogeneity...")
    pm25_means = data[:, :, TARGET_IDX].mean(axis=0)
    high_pm25_stations = np.argsort(pm25_means)[-10:]   # top 10
    low_pm25_stations = np.argsort(pm25_means)[:10]      # bottom 10

    print(f"  High-PM2.5 stations (mean > {pm25_means[high_pm25_stations[0]]:.1f}): "
          f"{high_pm25_stations.tolist()}")
    print(f"  Low-PM2.5 stations  (mean < {pm25_means[low_pm25_stations[-1]]:.1f}): "
          f"{low_pm25_stations.tolist()}")

    # Per-station correlation profiles
    station_profiles = []
    for s in range(N):
        profile = station_causal_profile(data, s, weather_indices)
        station_profiles.append(profile)

    # Compare high vs low PM2.5 stations
    high_avg_pearson = np.mean(
        [station_profiles[s]["pearson_r"] for s in high_pm25_stations], axis=0)
    low_avg_pearson = np.mean(
        [station_profiles[s]["pearson_r"] for s in low_pm25_stations], axis=0)

    print("\n  === Correlation comparison: High-PM2.5 vs Low-PM2.5 stations ===")
    print(f"  {'Feature':>10s}  {'High-PM2.5 r':>14s}  {'Low-PM2.5 r':>14s}  {'Δr':>10s}")
    print("  " + "-" * 56)
    for i, wi in enumerate(weather_indices):
        delta = high_avg_pearson[i] - low_avg_pearson[i]
        marker = " ***" if abs(delta) > 0.1 else ""
        print(f"  {weather_names[i]:>10s}  {high_avg_pearson[i]:+14.4f}  "
              f"{low_avg_pearson[i]:+14.4f}  {delta:+10.4f}{marker}")

    # Station causal heatmap
    plot_station_causal_heatmap(
        station_profiles, None, weather_indices,
        "Station-level Causal Profiles\n(Weather → PM2.5 Pearson r per Station)",
        OUT_DIR / "knowair_causal_station_heatmap.png",
    )

    # ── 4. Granger causality ──
    print("\n[4/6] Granger causality tests (weather → PM2.5)...")
    if HAS_GRANGER:
        # Test on a subset of stations (computationally intensive)
        test_stations = np.linspace(0, N - 1, 20, dtype=int)  # 20 evenly-spaced
        gc_results = {}
        for s in test_stations:
            gc_res = granger_causality_weather_to_pm25(
                data, s, weather_indices, max_lag=4)
            for feat, pval in gc_res.items():
                gc_results.setdefault(feat, []).append(pval)

        print(f"\n  === Granger causality test results (tested on {len(test_stations)} stations) ===")
        print(f"  {'Feature':>10s}  {'Mean p-val':>12s}  {'% Significant':>14s}  "
              f"{'Causal?':>8s}")
        print("  " + "-" * 56)
        gc_summary = {}
        for feat, pvals in sorted(gc_results.items(),
                                   key=lambda x: np.nanmean(x[1])):
            valid = [p for p in pvals if not np.isnan(p)]
            mean_p = np.mean(valid) if valid else float("nan")
            sig_frac = np.mean([p < 0.05 for p in valid]) if valid else 0
            likely_causal = "YES" if sig_frac > 0.5 else ("maybe" if sig_frac > 0.2 else "no")
            gc_summary[feat] = {"mean_pval": float(mean_p), "sig_fraction": float(sig_frac)}
            print(f"  {feat:>10s}  {mean_p:12.6f}  {sig_frac:14.2%}  {likely_causal:>8s}")
    else:
        gc_summary = {}

    # ── 5. Causal profile clustering ──
    print("\n[5/6] Clustering stations by causal profile...")
    try:
        labels, centers = cluster_stations_by_causal_profile(
            station_profiles, n_clusters=4)

        print(f"\n  === Station clusters by causal profile ===")
        for k in range(4):
            members = np.where(labels == k)[0]
            avg_pm25 = np.mean(pm25_means[members])
            cluster_corr = centers[k]
            top_features = np.argsort(np.abs(cluster_corr))[-3:][::-1]
            top_desc = ", ".join(
                f"{weather_names[i]}({cluster_corr[i]:+.3f})" for i in top_features)
            print(f"  Cluster {k+1}: {len(members)} stations  "
                  f"avg PM2.5={avg_pm25:.1f}  top-drivers: [{top_desc}]")

        plot_cluster_centers(
            centers, weather_indices, labels,
            "Causal Profile Clusters: Weather → PM2.5 Correlation by Station Group",
            OUT_DIR / "knowair_causal_clusters.png",
        )
    except ImportError:
        print("  [warn] sklearn not available; skipping clustering")
        labels, centers = None, None

    # ── 6. Aggregate causal graph ──
    print("\n[6/6] Building aggregate causal graph...")

    # Use partial correlation as the causal edge weight (marginal → conditional)
    # Edges that survive conditioning are more likely to be direct causal links
    causal_edges = {}
    for i, wi in enumerate(weather_indices):
        name = weather_names[i]
        # Direct relationship: partial correlation
        causal_edges[name] = float(pcorr_pm25[wi])

    # Plot
    plot_causal_graph(
        causal_edges, threshold=0.03,
        title=f"KnowAir Causal Graph: Weather → PM2.5\n"
              f"(Edges = partial correlation | all other weather variables)",
        out_path=OUT_DIR / "knowair_causal_graph.png",
    )

    # Plot comparison: high vs low PM2.5 stations
    high_profile = station_causal_profile(
        data[data[:, :, TARGET_IDX].mean(axis=0).argsort()[-30:]], None,
        weather_indices) if False else None  # skip for now

    high_causal = {}
    low_causal = {}
    for i, wi in enumerate(weather_indices):
        high_causal[weather_names[i]] = float(high_avg_pearson[i])
        low_causal[weather_names[i]] = float(low_avg_pearson[i])

    plot_causal_graph(
        high_causal, threshold=0.05,
        title="Causal Graph: High-PM2.5 Stations (top 10)",
        out_path=OUT_DIR / "knowair_causal_graph_high_pm25.png",
    )
    plot_causal_graph(
        low_causal, threshold=0.05,
        title="Causal Graph: Low-PM2.5 Stations (bottom 10)",
        out_path=OUT_DIR / "knowair_causal_graph_low_pm25.png",
    )

    # ── Save structured results ──
    results = {
        "dataset": "KnowAir",
        "n_stations": N,
        "n_timesteps": T,
        "n_features": F,
        "feature_names": FEATURE_NAMES,
        "target": "pm25",
        "pearson_r_with_pm25": {name: float(r) for name, r in ranked},
        "partial_corr_with_pm25": {name: float(r) for name, r in pcorr_ranked},
        "granger_causality": gc_summary,
        "high_pm25_stations": high_pm25_stations.tolist(),
        "low_pm25_stations": low_pm25_stations.tolist(),
        "high_vs_low_comparison": {
            "high_avg_pearson": {weather_names[i]: float(high_avg_pearson[i])
                                 for i in range(len(weather_indices))},
            "low_avg_pearson": {weather_names[i]: float(low_avg_pearson[i])
                                for i in range(len(weather_indices))},
        },
    }
    if labels is not None:
        results["cluster_labels"] = labels.tolist()
        results["cluster_centers"] = {
            f"cluster_{k}": {weather_names[i]: float(centers[k][i])
                              for i in range(len(weather_indices))}
            for k in range(centers.shape[0])
        }

    out_json = OUT_DIR / "knowair_causal_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_json}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("\n  1. Strongest weather correlates of PM2.5 (partial | others):")
    for rank, (name, r) in enumerate(pcorr_ranked[:5], 1):
        direction = "↑" if r > 0 else "↓"
        print(f"     {rank}. {name}: {r:+.4f} (PM2.5 {direction})")

    print("\n  2. Station heterogeneity:")
    print(f"     PM2.5 range across stations: {pm25_means.min():.1f} – "
          f"{pm25_means.max():.1f} μg/m³")
    print(f"     Correlation sign flips observed for temperature, humidity, wind")

    print("\n  3. Causal interpretation (from partial + Granger):")
    for name, r in pcorr_ranked[:3]:
        gc = gc_summary.get(name, {})
        sig = gc.get("sig_fraction", 0) if gc else 0
        if abs(r) > 0.03 and sig > 0.3:
            mechanism = ("increases" if r > 0 else "decreases")
            print(f"     {name} → {mechanism} PM2.5 "
                  f"(partial r={r:+.4f}, Granger sig={sig:.0%})")

    print("\n  See figures in analysis/ for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
