# ============================================================
# SECTION 0: Imports and configuration
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import date

import sys
from config import *

# ============================================================
# SECTION 1: Paths
# ============================================================

K = 10
RANDOM_SEED = 42
DATE_PREFIX = date.today().strftime("%Y%m%d")

OUT_MAIN = os.path.join(OUT_FOLDER, "04_global_clustering")
BOOT_DIR = os.path.join(OUT_MAIN, "4.2.0_C1_stability_across_k",
                         "4.2.1_bootstrap_permutation_analysis")

NPZ_PATH             = os.path.join(OUT_MAIN, "all_basins_rarity_curves.npz")
MEMBERS_PATH         = os.path.join(OUT_MAIN, f"members_global_k{K}.csv")
BOOT_FREQ_PATH       = os.path.join(BOOT_DIR, f"c1_inclusion_frequency_k{K}.csv")
MEMBERS_EDITED_PATH  = os.path.join(BOOT_DIR, f"members_global_k{K}_edited_typology.csv")

# Panel / plotting constants (same convention as make_panel() in script 3)
COL_W      = 3.8
PLOT_ROW_H = 3.5
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
# SECTION 2: Load the stacked rarity curves (produced by
# 3_Global_Clustering_and_Diagnostics_Heatmaps.ipynb, Section 1)
# ============================================================

data        = np.load(NPZ_PATH)
curves      = data["curves"].astype(np.float64)
years       = data["years"]
basin_ids   = data["basin_ids"]
windows_ref = data["windows"]

curve_index = {(int(b), int(y)): i for i, (b, y) in enumerate(zip(basin_ids, years))}

# ============================================================
# SECTION 3: Build (or load) the edited 5-typology membership table
#
# C1 (the raw k-means "extreme dry" cluster) is split into:
#   - ExDr : bootstrap-core C1 members (stable across resamples)
#   - Dr / Wtn : non-core C1 members, split by the sign of their
#                EOF2 score (temporal-shape axis)
# Clusters 2-10 map onto the other 4 typologies directly via
# TYPOLOGIES in config.py. This mirrors the "TYPOLOGY EDITED"
# section of script 3.
# ============================================================

if os.path.exists(MEMBERS_EDITED_PATH):
    print(f"Loading existing edited typology table: {MEMBERS_EDITED_PATH}")
    members_edited = pd.read_csv(MEMBERS_EDITED_PATH)

else:
    print("Edited typology table not found - building it ...")

    members_full = pd.read_csv(MEMBERS_PATH)   # columns: cluster, basin_id, year
    members_full["basin_id"] = members_full["basin_id"].astype(int)
    members_full["year"]     = members_full["year"].astype(int)

    if not os.path.exists(BOOT_FREQ_PATH):
        raise FileNotFoundError(
            f"Missing bootstrap C1 inclusion-frequency table: {BOOT_FREQ_PATH}\n"
            "This is produced by the 'Kmeans C1 Stability' bootstrap analysis "
            "in 3_Global_Clustering_and_Diagnostics_Heatmaps.ipynb - run that "
            "section first (it is too expensive to redo here)."
        )
    boot_freq = pd.read_csv(BOOT_FREQ_PATH)
    core_set = set(zip(
        boot_freq.loc[boot_freq["tier"] == "core", "basin_id"].astype(int),
        boot_freq.loc[boot_freq["tier"] == "core", "year"].astype(int),
    ))
    print(f"  Core C1 members (tier=='core'): n = {len(core_set)}")

    # Lightweight PCA for the EOF2 (temporal-shape) score used to
    # split non-core C1 members into Dr / Wtn
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    scores_all = pca.fit_transform(curves)
    eof2        = pca.components_[1]
    scores_eof2 = scores_all[:, 1]
    if eof2[0] < eof2[-1]:
        scores_eof2 = -scores_eof2

    CLUSTER_TO_FIXED_TYPOLOGY = {
        c: t["tag"] for t in TYPOLOGIES for c in t["clusters"] if c != 1
    }

    missing_eof2_keys = []

    def assign_typology(row):
        cl  = int(row["cluster"])
        key = (int(row["basin_id"]), int(row["year"]))

        if cl == 1:
            if key in core_set:
                return TYPOLOGY_TAGS[0]   # "Dry" (ExDr)
            idx = curve_index.get(key)
            if idx is None:
                missing_eof2_keys.append(key)
                return TYPOLOGY_TAGS[1]   # fallback default
            pc2 = scores_eof2[idx]
            return TYPOLOGY_TAGS[3] if pc2 > 0 else TYPOLOGY_TAGS[1]

        return CLUSTER_TO_FIXED_TYPOLOGY.get(cl, "UNASSIGNED")

    members_full["typology"] = members_full.apply(assign_typology, axis=1)

    if missing_eof2_keys:
        print(f"  WARNING: {len(missing_eof2_keys)} non-core C1 basin-years had "
              f"no matching curve index and were defaulted to 'Dr'.")

    os.makedirs(BOOT_DIR, exist_ok=True)
    members_edited = members_full[["cluster", "basin_id", "year", "typology"]]
    members_edited.to_csv(MEMBERS_EDITED_PATH, index=False)
    print(f"  Saved: {MEMBERS_EDITED_PATH}")

typologies_present = [t for t in TYPOLOGY_TAGS if t in members_edited["typology"].unique()]
print(f"Typologies found: {typologies_present}")

# ============================================================
# SECTION 4: Panel plot - 1 row x N typology columns
# (grey member curves + bold centroid + threshold lines)
# ============================================================

n_cols = len(typologies_present)
fig_w  = COL_W * n_cols
fig_h  = PLOT_ROW_H + 1.0

fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
axes = axes[0]

fig.suptitle(f"Five typologies",
             fontsize=13, fontweight="bold", y=1.03, family="cambria")

panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)"]

for col_idx, (typ, t_name) in enumerate(zip(typologies_present, TYPOLOGY_NAMES)):
    ax = axes[col_idx]

    ax.text(0.02, 0.98, panel_tags[col_idx],
            transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight="bold", family="cambria",
            ha="left", va="top", color="black")

    sub = members_edited[members_edited["typology"] == typ]
    member_idx = [
        curve_index[(int(row.basin_id), int(row.year))]
        for row in sub.itertuples()
        if (int(row.basin_id), int(row.year)) in curve_index
    ]

    n_typ = len(member_idx)

    if n_typ > 0:
        member_curves = curves[member_idx, :]
        centroid = member_curves.mean(axis=0)

        for i in member_idx:
            ax.plot(windows_ref, curves[i, :],
                     color="0.75", lw=0.55, alpha=0.7, zorder=2)

        ax.plot(windows_ref, centroid, color="black", lw=2.2, zorder=4)

    for nep_val, tcol, lbl in HLINE_SPECS:
        ax.axhline(nep_val, color=tcol, lw=0.9,
                   linestyle="--", alpha=0.85, zorder=3)

    ax.set_xlim(windows_ref[0], windows_ref[-1])
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FONT_AXIS)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("cambria")

    ax.set_xlabel("Accumulation period (weeks)", fontsize=FONT_AXIS, family="cambria")
    if col_idx == 0:
        ax.set_ylabel("Minimum NEP (rarity)", fontsize=FONT_AXIS, family="cambria")

    t_code = TYPOLOGY_LABELS[typ]
    ax.set_title(f"{t_code} - {t_name}  |  n = {n_typ}",
                  fontsize=FONT_TITLE, fontweight="bold", color="black", family="cambria")

OUT_D = "/Users/ucnworgu/OneDrive - University of Tasmania/mywork/PhD.things/manuscript_prep/MANU_v5_update/V6_figs/"
plt.tight_layout()
out_path = os.path.join(OUT_D, f"Fig6_typologies_{DATE_PREFIX}.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
print(f"\nSaved: {out_path}")
plt.show()

# ============================================================
# SECTION 5: Second version - per-typology panels showing each
# CONSTITUENT CLUSTER's own centroid (colored, inline "C{i}" tag,
# no legend), with the overall typology centroid kept in black.
#
# Constituent clusters are read off members_edited["cluster"]
# directly (not the static TYPOLOGY_MAP), since under the edited
# typology scheme, non-core C1 basin-years get reassigned into the
# Drying/Wetting typologies too - this picks those up correctly.
# ============================================================

SHOW_MEMBER_CURVES = True   # flip to False to hide the grey per-curve background

CLUSTER_LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
axes = axes[0]

fig.suptitle("Five typologies — constituent cluster centroids",
             fontsize=13, fontweight="bold", y=1.03, family="cambria")

for col_idx, (typ, t_name) in enumerate(zip(typologies_present, TYPOLOGY_NAMES)):
    ax = axes[col_idx]

    ax.text(0.02, 0.98, panel_tags[col_idx],
            transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight="bold", family="cambria",
            ha="left", va="top", color="black")

    sub = members_edited[members_edited["typology"] == typ]
    member_idx = [
        curve_index[(int(row.basin_id), int(row.year))]
        for row in sub.itertuples()
        if (int(row.basin_id), int(row.year)) in curve_index
    ]

    n_typ = len(member_idx)

    if n_typ > 0:
        member_curves = curves[member_idx, :]
        centroid = member_curves.mean(axis=0)

        if SHOW_MEMBER_CURVES:
            for i in member_idx:
                ax.plot(windows_ref, curves[i, :],
                         color="0.75", lw=0.55, alpha=0.7, zorder=2)

        # ---- constituent-cluster centroids, colored + inline-tagged ----
        cluster_vals = sorted(sub["cluster"].unique())
        for ci, cl_num in enumerate(cluster_vals):
            cl_rows = sub[sub["cluster"] == cl_num]
            cl_idx = [
                curve_index[(int(r.basin_id), int(r.year))]
                for r in cl_rows.itertuples()
                if (int(r.basin_id), int(r.year)) in curve_index
            ]
            if not cl_idx:
                continue
            cl_centroid = curves[cl_idx, :].mean(axis=0)
            color = CLUSTER_LINE_COLORS[ci % len(CLUSTER_LINE_COLORS)]

            ax.plot(windows_ref, cl_centroid, color=color, lw=1.6, zorder=5)
            ax.annotate(
                f"C{cl_num}", xy=(windows_ref[-1], cl_centroid[-1]),
                xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=FONT_AXIS - 2, fontweight="bold",
                family="cambria", va="center", ha="left", zorder=6,
            )

        # overall typology centroid - stays black, drawn on top
        ax.plot(windows_ref, centroid, color="black", lw=2.2, zorder=7)

    for nep_val, tcol, lbl in HLINE_SPECS:
        ax.axhline(nep_val, color=tcol, lw=0.9,
                   linestyle="--", alpha=0.85, zorder=3)

    ax.set_xlim(windows_ref[0], windows_ref[-1])
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FONT_AXIS)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("cambria")

    ax.set_xlabel("Accumulation period (weeks)", fontsize=FONT_AXIS, family="cambria")
    if col_idx == 0:
        ax.set_ylabel("Minimum NEP (rarity)", fontsize=FONT_AXIS, family="cambria")

    t_code = TYPOLOGY_LABELS[typ]
    ax.set_title(f"{t_code} - {t_name}  |  n = {n_typ}",
                  fontsize=FONT_TITLE, fontweight="bold", color="black", family="cambria")

plt.tight_layout()

if SHOW_MEMBER_CURVES:
    out_path_clusters = os.path.join(OUT_D, f"Fig6_typologies_members_{DATE_PREFIX}.png")
else:
    out_path_clusters = os.path.join(OUT_D, f"Fig6b_typologies_nomembers_{DATE_PREFIX}.png")

fig.savefig(out_path_clusters, dpi=300, bbox_inches="tight", pad_inches=0.3)
print(f"\nSaved: {out_path_clusters}")
plt.show()
