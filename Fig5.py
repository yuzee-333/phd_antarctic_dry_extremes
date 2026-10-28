# ============================================================
# SECTION 0: Imports and configuration
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

import sys
from config import *

# ============================================================
# SECTION 1: Paths
# ============================================================

OUT_MAIN = os.path.join(OUT_FOLDER, "04_global_clustering")

K = 10
NPZ_PATH       = os.path.join(OUT_MAIN, "all_basins_rarity_curves.npz")
MEMBERS_PATH   = os.path.join(OUT_MAIN, f"members_global_k{K}.csv")
CENTROIDS_PATH = os.path.join(OUT_MAIN, f"centroids_global_k{K}.csv")
MANUSCRIPT_IMAGES_FOLDER = "/Users/ucnworgu/OneDrive - University of Tasmania/mywork/PhD.things/manuscript_prep/MANU_v5_update/V6_figs/"
os.makedirs(MANUSCRIPT_IMAGES_FOLDER, exist_ok=True)
FIG_OUT_PATH = os.path.join(MANUSCRIPT_IMAGES_FOLDER, f"Fig5_global_kmeans_k{K}.png")

# Panel layout
MAX_COLS   = 5    # k=10 -> 2 rows x 5 cols
PLOT_ROW_H = 3.5  # inches per row
COL_W      = 3.8  # inches per column
FONT_TITLE = 15
FONT_AXIS  = 12

# McKee et al. (1993) drought threshold lines
HLINE_SPECS = [
    (0.50, "#2ca02c", "mild"),
    (0.16, "#1f77b4", "moderate"),
    (0.07, "#ff7f0e", "severe"),
    (0.02, "#d62728", "extreme"),
]

# ============================================================
# SECTION 2: Load the global k-means results for k=10
# (produced by 3_Global_Clustering_and_Diagnostics_Heatmaps.ipynb)
# ============================================================

data        = np.load(NPZ_PATH)
curves      = data["curves"].astype(np.float64)
years       = data["years"]
basin_ids   = data["basin_ids"]
windows_ref = data["windows"]
n_curves    = curves.shape[0]

df_members = pd.read_csv(MEMBERS_PATH)
centroids  = pd.read_csv(CENTROIDS_PATH, index_col=0).values

# Labels aligned to the curve order stored in the NPZ (basin_ids/years)
cluster_lookup = {
    (int(row.basin_id), int(row.year)): int(row.cluster)
    for row in df_members.itertuples()
}
labels = np.array([
    cluster_lookup[(int(b), int(y))] - 1   # back to 0-indexed
    for b, y in zip(basin_ids, years)
])

# ============================================================
# SECTION 3: Panel plot - grey members + bold centroid + threshold lines
# ============================================================

n_rows = math.ceil(K / MAX_COLS)
n_cols = min(K, MAX_COLS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(COL_W * n_cols, PLOT_ROW_H * n_rows + 0.8),
)
axes = np.atleast_2d(axes)

fig.suptitle(f"Global k-means — all 26 basins  |  k={K} cluster",
             fontsize=13, fontweight="bold", y=1.0, family="cambria")

for cl in range(K):
    row_i, col_i = divmod(cl, MAX_COLS)
    ax = axes[row_i, col_i]

    idx  = np.where(labels == cl)[0]
    n_cl = len(idx)

    for i in idx:
        ax.plot(windows_ref, curves[i, :],
                 color="0.75", lw=0.55, alpha=0.7, zorder=2)

    # Bold black centroid
    ax.plot(windows_ref, centroids[cl, :],
             color="black", lw=2.2, zorder=4)

    # Drought threshold lines
    for nep_val, tcol, lbl in HLINE_SPECS:
        ax.axhline(nep_val, color=tcol, lw=0.9,
                   linestyle="--", alpha=0.85, zorder=3)

    ax.set_xlim(windows_ref[0], windows_ref[-1])
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FONT_AXIS, labelfontfamily="cambria")
    # for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    #     lbl.set_fontfamily("cambria")
    ax.set_xlabel("Accumulation period (weeks)", fontsize=FONT_AXIS, family="cambria")
    if col_i == 0:
        ax.set_ylabel("Minimum NEP (rarity)", fontsize=FONT_AXIS, family="cambria")

    ax.set_title(f"C{cl+1}  |  n = {n_cl}",
                 fontsize=FONT_TITLE, fontweight="bold", family="cambria")

plt.tight_layout()
plt.savefig(FIG_OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {FIG_OUT_PATH}")
plt.show()

# ============================================================
# SECTION 4: Global EOF (PCA) on the same rarity curves
# (same curves/windows already loaded above - no need to reload)
# ============================================================

RANDOM_SEED  = 42
N_COMPONENTS = 3   # EOF1 + EOF2 used below; EOF3 computed but unused

pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
scores_all    = pca.fit_transform(curves)
eof_patterns  = pca.components_
var_explained = pca.explained_variance_ratio_ * 100

scores_eof1 = scores_all[:, 0]
scores_eof2 = scores_all[:, 1]
eof1 = eof_patterns[0]
eof2 = eof_patterns[1]

# Sign conventions: EOF1 high score = dry; EOF2 flipped if it rises
# from short to long accumulation windows
if eof1.mean() > 0:
    eof1 = -eof1
    scores_eof1 = -scores_eof1
if eof2[0] < eof2[-1]:
    eof2 = -eof2
    scores_eof2 = -scores_eof2

# ------------------------------------------------------------
# EOF1 & EOF2 loading patterns
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, eof, ve, label in zip(axes,
                               [eof1, eof2],
                               [var_explained[0], var_explained[1]],
                               ["EOF1", "EOF2"]):
    ax.plot(windows_ref, eof, color="black", lw=1.8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Loading", family="cambria")
    ax.set_title(f"{label}  ({ve:.1f}% variance explained)", family="cambria")
    ax.set_xlabel("Accumulation window (weeks)", fontsize=12, family="cambria")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("cambria")

plt.tight_layout()
FIG5B_OUT_PATH = os.path.join(MANUSCRIPT_IMAGES_FOLDER, "Fig5b_eof1_eof2_patterns.png")
plt.savefig(FIG5B_OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {FIG5B_OUT_PATH}")
plt.show()

# ============================================================
# SECTION 5: K-means clusters projected onto EOF1 x EOF2 space,
# with per-cluster centroids (same scatter + centroids)
# ============================================================

CLUSTER_COLORS = {
    # DRY end: stepped reds, dark -> medium
    1:  "#67001f",  # Extreme Dry
    2:  "#d6604d",  # Drying
    5:  "#d6604d",
    8:  "#d6604d",
    4:  "#e8d9a0",  # Median
    7:  "#e8d9a0",
    3:  "#92c5de",  # Wetting
    6:  "#92c5de",
    9:  "#053061",  # Wet
    10: "#053061",
}

fig, ax = plt.subplots(figsize=(10, 8), dpi=400)

for cl in range(1, K + 1):
    idx = np.where(labels == cl - 1)[0]
    if len(idx) == 0:
        continue

    # scatter members
    ax.scatter(
        scores_eof1[idx], scores_eof2[idx],
        color=CLUSTER_COLORS[cl], s=10, alpha=0.45, edgecolors="none",
    )

    # centroid in EOF space
    c1 = scores_eof1[idx].mean()
    c2 = scores_eof2[idx].mean()
    ax.scatter(
        c1, c2, color=CLUSTER_COLORS[cl], s=160,
        edgecolors="black", linewidths=1.2, zorder=5,
    )
    ax.annotate(
        f"C{cl}", xy=(c1, c2), xytext=(4, 4), textcoords="offset points",
        fontsize=15, fontweight="bold", family="cambria", color="black", zorder=6,
    )

ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
ax.axvline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel(f"PC1 score — overall amplitude  ({var_explained[0]:.1f}% variance)", fontsize=15, family="cambria")
ax.set_ylabel(f"PC2 score — temporal shape  ({var_explained[1]:.1f}% variance)", fontsize=15, family="cambria")
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("cambria")

handles = [
    mpatches.Patch(color=CLUSTER_COLORS[int(cl[1:])],
                   label=f"Cluster {cl} (n={int(np.sum(labels == int(cl[1:]) - 1))})")
    for cl in CLUSTER_ORDER
    if np.sum(labels == int(cl[1:]) - 1) > 0
]
leg = ax.legend(handles=handles, fontsize=10, frameon=True, framealpha=0.85,
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, borderaxespad=0.)
for text in leg.get_texts():
    text.set_fontfamily("cambria")

for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily("cambria")

plt.tight_layout()
FIG5C_OUT_PATH = os.path.join(MANUSCRIPT_IMAGES_FOLDER, f"Fig5c_kmeans_eof_projection_k{K}.png")
plt.savefig(FIG5C_OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {FIG5C_OUT_PATH}")
plt.show()

# ============================================================
# SECTION 6: Combined manuscript figure - panels (a)-(l)
# Reuses the arrays already computed in Sections 2-5 above
# (curves, labels, centroids, eof1/eof2, scores_eof1/2, etc.)
# instead of recomputing or reloading anything.
# ============================================================

import string
import matplotlib.gridspec as gridspec

PANEL_LETTERS = list(string.ascii_lowercase[:12])   # a..l
TAG_KW = dict(ha="left", va="top", fontsize=16, fontweight="bold", family="cambria")


def _cambria_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("cambria")

fig = plt.figure(figsize=(20, 16))
outer = gridspec.GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.35,
                            height_ratios=[1, 1, 1.3])

# ---- (a)-(j): the 10 cluster panels, same layout as Section 3 ----
for cl in range(K):
    row_i, col_i = divmod(cl, MAX_COLS)
    ax = fig.add_subplot(outer[row_i, col_i])

    idx = np.where(labels == cl)[0]
    for i in idx:
        ax.plot(windows_ref, curves[i, :], color="0.75", lw=0.55, alpha=0.7, zorder=2)
    ax.plot(windows_ref, centroids[cl, :], color="black", lw=2.2, zorder=4)

    for nep_val, tcol, lbl in HLINE_SPECS:
        ax.axhline(nep_val, color=tcol, lw=0.9, linestyle="--", alpha=0.85, zorder=3)

    ax.set_xlim(windows_ref[0], windows_ref[-1])
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FONT_AXIS)
    _cambria_ticks(ax)
    ax.set_xlabel("Accumulation period (weeks)", fontsize=FONT_AXIS, family="cambria")
    if col_i == 0:
        ax.set_ylabel("Minimum NEP (rarity)", fontsize=FONT_AXIS, family="cambria")
    ax.set_title(f"C{cl+1}  |  n = {len(idx)}", fontsize=FONT_TITLE, fontweight="bold", family="cambria")

    ax.text(0.03, 0.95, f"({PANEL_LETTERS[cl]})", transform=ax.transAxes, **TAG_KW)

# ---- (k): EOF1 + EOF2 stacked, bottom-left (2 of 5 columns) ----
eof_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2, 0:2], hspace=0.35)
eof_axes = [fig.add_subplot(eof_gs[0, 0]), fig.add_subplot(eof_gs[1, 0])]

for ax, eof, ve, label in zip(eof_axes, [eof1, eof2],
                                [var_explained[0], var_explained[1]], ["EOF1", "EOF2"]):
    ax.plot(windows_ref, eof, color="black", lw=1.8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Loading", fontsize=FONT_AXIS, family="cambria")
    ax.set_title(f"{label}  ({ve:.1f}% variance explained)", fontsize=FONT_AXIS, family="cambria")
    _cambria_ticks(ax)
eof_axes[-1].set_xlabel("Accumulation window (weeks)", fontsize=FONT_AXIS, family="cambria")

eof_axes[0].text(0.03, 0.95, f"({PANEL_LETTERS[10]})", transform=eof_axes[0].transAxes, **TAG_KW)

# ---- (l): k-means clusters projected onto EOF1 x EOF2, bottom-right (3 of 5 columns) ----
ax_scatter = fig.add_subplot(outer[2, 2:5])

for cl in range(1, K + 1):
    idx = np.where(labels == cl - 1)[0]
    if len(idx) == 0:
        continue
    ax_scatter.scatter(scores_eof1[idx], scores_eof2[idx],
                        color=CLUSTER_COLORS[cl], s=10, alpha=0.45, edgecolors="none")
    c1, c2 = scores_eof1[idx].mean(), scores_eof2[idx].mean()
    ax_scatter.scatter(c1, c2, color=CLUSTER_COLORS[cl], s=160,
                        edgecolors="black", linewidths=1.2, zorder=5)
    ax_scatter.annotate(f"C{cl}", xy=(c1, c2), xytext=(4, 4), textcoords="offset points",
                         fontsize=13, fontweight="bold", family="cambria", color="black", zorder=6)

ax_scatter.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
ax_scatter.axvline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
ax_scatter.set_xlabel(f"PC1 score — overall amplitude  ({var_explained[0]:.1f}% variance)", fontsize=13, family="cambria")
ax_scatter.set_ylabel(f"PC2 score — temporal shape  ({var_explained[1]:.1f}% variance)", fontsize=13, family="cambria")
_cambria_ticks(ax_scatter)

# ── 5-typology legend (not per-cluster), vertical, right of panel (l) ──
TYPOLOGY_LEGEND_COLORS = {
    "Dry":     "#67001f",
    "Drying":  "#d6604d",
    "Median":  "#e8d9a0",
    "Wetting": "#92c5de",
    "Wet":     "#053061",
}
typ_handles = [mpatches.Patch(color=c, label=l) for l, c in TYPOLOGY_LEGEND_COLORS.items()]
leg_l = ax_scatter.legend(handles=typ_handles, title="Typology",
                            loc="center left", bbox_to_anchor=(1.02, 0.5),
                            ncol=1, fontsize=11, title_fontsize=12,
                            frameon=True, framealpha=0.9)
for text in leg_l.get_texts():
    text.set_fontfamily("cambria")
leg_l.get_title().set_fontfamily("cambria")

ax_scatter.text(0.02, 0.98, f"({PANEL_LETTERS[11]})", transform=ax_scatter.transAxes, **TAG_KW)

FIG5_COMBINED_PATH = os.path.join(MANUSCRIPT_IMAGES_FOLDER, f"Fig5_combined_k{K}.png")
plt.savefig(FIG5_COMBINED_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {FIG5_COMBINED_PATH}")
plt.show()
