"""
t-SNE of prototype vectors over training rounds for cast-init runs.
Compares normOFF (CaST-like: free Euclidean prototypes) vs normON (sphere-constrained).
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})


def _avg_abs_cos(p):
    """Average absolute pairwise cosine similarity among prototypes."""
    pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    s = np.abs(pn @ pn.T)
    np.fill_diagonal(s, 0)
    return s.mean()


def load_history(run_name):
    ck = torch.load(
        f"/home/qingxiangliu/causal-FedTS/output/{run_name}/checkpoint.pt",
        map_location="cpu", weights_only=False,
    )
    rounds = ck["proto_history_rounds"]
    protos = [p.float().numpy() for p in ck["proto_history"]]
    return rounds, protos


def plot_one(ax, data, rounds, title, highlight_rounds):
    """
    ax: 2-element array [raw_ax, normed_ax]
    data: list of (K, D) raw prototypes
    """
    K = data[0].shape[0]
    n_snap = len(rounds)

    # Normalised for directional view
    data_norm = [p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8) for p in data]

    for ax_i, (pts_list, label) in enumerate(
        [(data, "Raw prototypes (Euclidean)"), (data_norm, "Normalised (directional)")]
    ):
        all_pts = np.concatenate(pts_list, axis=0)
        perp = min(30, max(5, n_snap * K - 1))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=2000)
        xy = tsne.fit_transform(all_pts)

        colors = plt.cm.coolwarm(np.linspace(0, 1, n_snap))
        for t in range(n_snap):
            pts = xy[t * K : (t + 1) * K]
            is_hl = rounds[t] in highlight_rounds
            ax[ax_i].scatter(
                pts[:, 0], pts[:, 1], color=colors[t],
                alpha=0.9 if is_hl else 0.25,
                edgecolors="k" if is_hl else "none",
                linewidths=1.2 if is_hl else 0,
                s=50 if is_hl else 20,
                label=f"round {rounds[t]}" if is_hl else None,
                zorder=3 if is_hl else 1,
            )

        # Trajectories
        for k in range(K):
            traj = xy[k::K]
            ax[ax_i].plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.2, lw=0.5, zorder=0)

        ax[ax_i].set_title(label, fontsize=11)
        ax[ax_i].legend(fontsize=7, loc="best", markerscale=0.8)
        ax[ax_i].set_xticks([])
        ax[ax_i].set_yticks([])

    ax[0].figure.suptitle(title, fontsize=12, y=1.04)


# ── Load ──
rounds_off, protos_off = load_history("cast_init_normOFF")
rounds_on, protos_on = load_history("cast_init_normON")

highlight = [0, rounds_off[-1] // 2, rounds_off[-1]]

# ── Build titles ──
norms0 = np.linalg.norm(protos_off[0], axis=1)
norms_f = np.linalg.norm(protos_off[-1], axis=1)
title_off = (
    f"CaST init + norm OFF  |  "
    f"Round 0: norm∈[{norms0.min():.2f},{norms0.max():.2f}]  →  "
    f"Round {rounds_off[-1]}: norm∈[{norms_f.min():.2f},{norms_f.max():.2f}]"
)

cos0 = _avg_abs_cos(protos_on[0])
cos_f = _avg_abs_cos(protos_on[-1])
title_on = (
    f"CaST init + norm ON  |  "
    f"Round 0: |cos|={cos0:.3f}  →  Round {rounds_on[-1]}: |cos|={cos_f:.3f}"
)

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plot_one(axes[0], protos_off, rounds_off, title_off, highlight)
plot_one(axes[1], protos_on, rounds_on, title_on, highlight)

plt.tight_layout()
out = "/home/qingxiangliu/causal-FedTS/cast_init_prototype_tsne.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
