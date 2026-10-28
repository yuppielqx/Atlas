"""
3-panel prototype-regime interpretability figure for CausalFedST.

Panel (a): log2 P(cond|proto)/P(cond) — prototype-condition association.
Panel (b): Daily raster — each row = one day; speed line, proto bands, rain.
Panel (c): Cross-sensor prototype usage — stacked bar chart.

Usage:
    python scripts/plot_prototype_regime_figure.py \
        --corr_dir output/zinv_ksweep_all_k10_r200 \
        --format all
"""

import os, sys, argparse, json, csv, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="3-panel prototype-regime figure")
    p.add_argument("--corr_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--format", type=str, default="pdf",
                   choices=["pdf", "png", "svg", "all"])
    p.add_argument("--data_dir", type=str, default="./datasets")
    p.add_argument("--dataset", type=str, default="metr-la")
    p.add_argument("--n_days", type=int, default=5)
    return p.parse_args()


# Palette
P10 = np.array([
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
    "#B07AA1", "#EDC948", "#B6992D", "#FF9DA7", "#9C755F",
])
POINTS_PER_DAY = 288

# Weather codes (WMO) -> label & colour
WC = {  # (label, colour)
    0:  ("Clear",       "#e8c170"),
    1:  ("M.Clear",     "#e8c170"),
    2:  ("P.Cloudy",    "#b0b0b0"),
    3:  ("Overcast",    "#606060"),
    45: ("Fog",         "#999999"),
    51: ("Drizzle",     "#4292c6"),
    53: ("Drizzle",     "#2171b5"),
    55: ("Drizzle",     "#08519c"),
    61: ("Rain",        "#08519c"),
    63: ("Rain",        "#08519c"),
    65: ("Rain",        "#08519c"),
    80: ("Showers",     "#2171b5"),
    81: ("Showers",     "#2171b5"),
    82: ("Showers",     "#2171b5"),
}


# ── Data loaders ────────────────────────────────────────────────────

def load_corr_data(d):
    m = np.load(os.path.join(d, "proto_condition_matrix.npy"))
    with open(os.path.join(d, "condition_names.txt")) as f:
        names = [l.strip() for l in f if l.strip()]
    with open(os.path.join(d, "corr_metadata.json")) as f:
        meta = json.load(f)
    pb = pc = None
    pp = os.path.join(d, "panel_b_sensor.npz")
    if os.path.exists(pp):
        pb = dict(np.load(pp, allow_pickle=True))
    pq = os.path.join(d, "panel_c_usage.npz")
    if os.path.exists(pq):
        pc = dict(np.load(pq, allow_pickle=True))
    return m, names, meta, pb, pc


def load_speed(sensor_id, dataset_name, data_dir, ws=12, hz=12):
    ds_map = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
    ds_dir = os.path.join(data_dir, ds_map[dataset_name])
    with open(os.path.join(ds_dir, "desc.json")) as f:
        L, N, _ = json.load(f)["shape"]
    raw = np.memmap(os.path.join(ds_dir, "data.dat"), dtype=np.float32, mode="r")
    raw = raw.reshape(L, N, 3)
    with open(os.path.join(ds_dir, "graph_sensor_locations.csv")) as f:
        sid2i = {int(r["sensor_id"]): int(r["index"]) for r in csv.DictReader(f)}
    col = sid2i.get(int(sensor_id))
    if col is None:
        return None
    s = raw[:, col, 0].copy()
    nw = L - ws - hz + 1
    return np.array([s[i + ws - 1] for i in range(nw)], dtype=np.float32)


def best_n_days(precip, n_win, n):
    n_total = n_win // POINTS_PER_DAY
    if n_total <= n:
        return 1
    best, best_s = -1.0, 1
    for s in range(1, n_total - n + 1):
        sc = sum(precip[s * POINTS_PER_DAY + d * POINTS_PER_DAY:
                        s * POINTS_PER_DAY + (d + 1) * POINTS_PER_DAY].max()
                 for d in range(n))
        if sc > best_s:
            best_s = sc
            best = s
    return best


def load_weather_dict(weather_csv):
    w = {}
    with open(weather_csv) as f:
        for row in csv.DictReader(f):
            w[(row["datetime"], int(row["region_id"]))] = int(row["weather_code"])
    return w


def load_sensor_region(map_csv):
    m = {}
    with open(map_csv) as f:
        for row in csv.DictReader(f):
            m[int(row["sensor_id"])] = int(row["region_id"])
    return m


# ═══════════════════════════════════════════════════════════════════════
# Panel (a)
# ═══════════════════════════════════════════════════════════════════════

def panel_a(ax, mat, names):
    K = mat.shape[0]
    temporal = [n for n in names if any(k in n for k in
                ["Morning", "Noon", "Evening", "Night", "Weekday", "Weekend"])]
    weather  = [n for n in names if any(k in n for k in ["Rain", "Clear"])]
    ordered  = [n for n in temporal + weather if n in names]
    n2i = {n: i for i, n in enumerate(names)}
    cols = [n2i[n] for n in ordered]
    raw = mat[:, cols]

    pop = raw.mean(axis=0, keepdims=True)
    log2r = np.log2(np.clip(raw / (pop + 1e-6), 0.5, 2.0))
    vmax = max(abs(log2r.min()), abs(log2r.max()), 0.03)

    im = ax.imshow(log2r, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    n_temp = len(temporal)
    if 0 < n_temp < len(ordered):
        ax.axvline(n_temp - 0.5, color="0.2", lw=1.0)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels([str(k) for k in range(K)], fontsize=7)
    ax.tick_params(pad=1.5)

    for i in range(K):
        for j in range(len(ordered)):
            v = raw[i, j]; d = log2r[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.0,
                    fontfamily="monospace",
                    color="white" if abs(d) > 0.55 * vmax else "0.15")

    if n_temp > 0:
        ax.text(n_temp / 2 - 0.5, -1.2, "Temporal", ha="center", fontsize=7.5,
                fontweight="bold", color="0.3", transform=ax.get_xaxis_transform())
    if n_temp < len(ordered):
        mid = (n_temp + len(ordered) - 1) / 2
        ax.text(mid, -1.2, "Weather", ha="center", fontsize=7.5,
                fontweight="bold", color="0.3", transform=ax.get_xaxis_transform())

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(r"$\log_2\frac{P(\mathrm{cond}|\mathrm{proto})}{P(\mathrm{cond})}$",
                   fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=5.5)

    ax.set_title("(a)  Prototype-condition association", fontsize=10,
                 fontweight="bold", loc="left", pad=12)


# ═══════════════════════════════════════════════════════════════════════
# Panel (b)
# ═══════════════════════════════════════════════════════════════════════

def panel_b(fig, gs, pb_data, args, K):
    if pb_data is None:
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center")
        ax.set_axis_off()
        return

    protos  = pb_data["dominant_proto"]
    tod     = pb_data["time_of_day"]
    precip  = pb_data["precipitation"]
    ords    = pb_data["datetime_ordinal"]
    sid     = int(pb_data["sensor_id"])
    start_i = int(pb_data.get("start_idx", 0))
    n_win   = int(pb_data.get("n_windows", len(protos)))

    speed = load_speed(sid, args.dataset, args.data_dir)
    if speed is not None:
        speed = speed[start_i:start_i + n_win]
    else:
        speed = np.zeros(n_win)

    day0 = best_n_days(precip, n_win, args.n_days)
    wmap = load_weather_dict(os.path.join(args.corr_dir, "weather.csv"))
    rmap = load_sensor_region(os.path.join(args.corr_dir, "sensor_weather_map.csv"))
    reg = rmap.get(sid, 0)

    def day_weather(od, td):
        codes = []
        for i in range(len(od)):
            base = dt.datetime.fromordinal(int(od[i]))
            h = int(round(float(td[i]) * 24.0)) % 24
            try:
                k = base.replace(hour=h, minute=0, second=0, microsecond=0)
            except ValueError:
                continue
            c = wmap.get((k.strftime("%Y-%m-%dT%H:%M"), reg))
            if c is not None:
                codes.append(c)
        return max(set(codes), key=codes.count) if codes else 0

    # Unified y range
    all_spd = []
    for d in range(args.n_days):
        i0, i1 = (day0 + d) * POINTS_PER_DAY, (day0 + d) * POINTS_PER_DAY + POINTS_PER_DAY
        all_spd.append(speed[i0:i1])
    y_lo = min(a.min() for a in all_spd)
    y_hi = max(a.max() for a in all_spd)
    pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    y_lo, y_hi = y_lo - pad, y_hi + pad

    axes = []
    for d in range(args.n_days):
        ax = fig.add_subplot(gs[d, 0])
        axes.append(ax)
        i0 = (day0 + d) * POINTS_PER_DAY
        i1 = i0 + POINTS_PER_DAY

        p_d  = protos[i0:i1]
        td_d = tod[i0:i1]
        sp_d = speed[i0:i1]
        rn_d = precip[i0:i1]
        od_d = ords[i0:i1]

        day_dt  = dt.datetime.fromordinal(int(od_d[0]))
        day_lbl = day_dt.strftime("%a %m/%d")
        wcode   = day_weather(od_d, td_d)
        wlbl, wclr = WC.get(wcode, ("?", "#888"))

        x = td_d * 24.0

        # Prototype background
        px = p_d[0]; s0 = 0.0
        for t in range(1, POINTS_PER_DAY):
            if p_d[t] != px:
                ax.axvspan(s0, x[t], alpha=0.25,
                           color=P10[px % len(P10)], lw=0, zorder=0)
                px = p_d[t]; s0 = x[t]
        ax.axvspan(s0, 24.0, alpha=0.25,
                   color=P10[px % len(P10)], lw=0, zorder=0)

        # Speed
        ax.plot(x, sp_d, color="0.05", lw=0.85, zorder=3)

        # Precipitation underlay (at bottom of axes)
        base_y = y_lo + 0.02 * (y_hi - y_lo)
        scale  = 0.12 * (y_hi - y_lo)
        rain   = np.clip(rn_d, 0, None)
        rmax   = rain.max() if rain.max() > 0 else 1.0
        ax.fill_between(x, base_y, base_y + rain * scale / rmax,
                        color="#3182bd", alpha=0.5, lw=0, zorder=2)

        # Styling
        ax.set_xlim(0, 24)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels([])
        ax.set_ylabel("mph", fontsize=6.5, labelpad=2)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=5.5, pad=1)
        ax.grid(axis="y", alpha=0.12, lw=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Day + weather annotation
        ax.text(1.01, 0.72, day_lbl, transform=ax.transAxes,
                fontsize=7, va="center", ha="left", color="0.2", fontweight="bold")
        ax.text(1.01, 0.32, wlbl, transform=ax.transAxes,
                fontsize=9, va="center", ha="left", color=wclr, fontweight="bold")

    axes[-1].set_xticklabels(["0h", "6h", "12h", "18h", "24h"], fontsize=6.5)
    axes[-1].set_xlabel("Hour of day", fontsize=8, labelpad=1)

    # Legend
    used = sorted(set(int(p) for d in range(args.n_days)
                      for p in protos[(day0 + d) * POINTS_PER_DAY:
                                      (day0 + d) * POINTS_PER_DAY + POINTS_PER_DAY]))
    h_protos = [Patch(facecolor=P10[k % len(P10)], alpha=0.35,
                      label=f"Proto {k}") for k in used]
    h_weather = [
        Patch(facecolor=WC[0][1], alpha=0.5, label="Clear"),
        Patch(facecolor=WC[3][1], alpha=0.5, label="Overcast"),
        Patch(facecolor=WC[51][1], alpha=0.5, label="Rain/Drizzle"),
    ]
    axes[0].legend(handles=h_protos + h_weather, fontsize=5.5, loc="upper left",
                   ncol=min(6, len(h_protos) + 3), framealpha=0.7,
                   handlelength=1.0, borderpad=0.2,
                   labelspacing=0.15, columnspacing=0.4)

    axes[0].set_title(f"(b)  Sensor {sid}  |  daily speed + prototype regime + precipitation",
                      fontsize=10, fontweight="bold", loc="left", pad=6)


# ═══════════════════════════════════════════════════════════════════════
# Panel (c)
# ═══════════════════════════════════════════════════════════════════════

def panel_c(ax, pc_data, K):
    if pc_data is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.set_axis_off()
        return

    w = pc_data["avg_weights"]
    sids = pc_data["sensor_ids"]
    n = len(sids)

    y = np.arange(n); left = np.zeros(n)
    for k in range(K):
        ax.barh(y, w[:, k], left=left, height=0.50,
                color=P10[k % len(P10)], edgecolor="white", lw=0.15,
                label=f"Proto {k}")
        left += w[:, k]

    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(s)}" for s in sids], fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average prototype weight", fontsize=8, labelpad=2)
    ax.invert_yaxis()
    ax.tick_params(labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=P10[k % len(P10)], label=f"P{k}") for k in range(K)]
    ax.legend(handles=handles, fontsize=5.2, loc="lower right",
              ncol=5, framealpha=0.5, borderpad=0.2,
              handlelength=0.9, labelspacing=0.12, columnspacing=0.35)

    ax.set_title("(c)  Cross-sensor prototype usage distribution",
                 fontsize=10, fontweight="bold", loc="left")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = args.output_dir or args.corr_dir
    os.makedirs(out_dir, exist_ok=True)

    mat, names, meta, pb, pc = load_corr_data(args.corr_dir)
    K = meta["K"]
    print(f"K={K}, conditions={len(names)}")

    fig = plt.figure(figsize=(13, 10))

    outer = gridspec.GridSpec(2, 2, figure=fig,
                              height_ratios=[1.0, 1.2],
                              hspace=0.28, wspace=0.38,
                              left=0.06, right=0.97, top=0.96, bottom=0.04)

    ax_a = fig.add_subplot(outer[0, 0])
    ax_c = fig.add_subplot(outer[0, 1])
    panel_a(ax_a, mat, names)
    panel_c(ax_c, pc, K)

    inner = gridspec.GridSpecFromSubplotSpec(
        args.n_days, 1, subplot_spec=outer[1, :], hspace=0.06)
    panel_b(fig, inner, pb, args, K)

    fig.text(0.5, 0.005,
             "CausalFedST prototype-regime interpretability  |  METR-LA  |  "
             "K=10  |  tau=0.1  |  Stage-1 (z-inv) checkpoint",
             ha="center", fontsize=6.5, color="0.45", style="italic")

    fmts = ["pdf", "png", "svg"] if args.format == "all" else [args.format]
    for fmt in fmts:
        path = os.path.join(out_dir, f"fig_prototype_regime.{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved: {path}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
