# ============================================================
# SECTION 0: Imports and configuration
# ============================================================

import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from datetime import date

import sys
from config import *

"""
C6
BARPLOTS manuV6
"""

# ============================================================
# SECTION 1: Paths and plot constants
# (mirrors the "2_barplot_by_basin" section of
#  3_Global_Clustering_and_Diagnostics_Heatmaps.ipynb, PLOT_MODE
#  hardcoded to "typology_edited")
# ============================================================

K            = 10
RANDOM_SEED  = 42
DATE_PREFIX  = date.today().strftime("%Y%m%d")
PCT_COLUMN   = "n_years_in_cluster"
PLOT_PREFIX  = "typology_edited"

OUT_MAIN = os.path.join(OUT_FOLDER, "04_global_clustering")
BOOT_DIR = os.path.join(OUT_MAIN, "4.2.0_C1_stability_across_k",
                         "4.2.1_bootstrap_permutation_analysis")
OUT_BARPLOTS = os.path.join(OUT_MAIN, "4.2.2_basin_barplots")
os.makedirs(OUT_BARPLOTS, exist_ok=True)

NPZ_PATH             = os.path.join(OUT_MAIN, "all_basins_rarity_curves.npz")
MEMBERS_PATH         = os.path.join(OUT_MAIN, f"members_global_k{K}.csv")
BOOT_FREQ_PATH       = os.path.join(BOOT_DIR, f"c1_inclusion_frequency_k{K}.csv")
MEMBERS_EDITED_PATH  = os.path.join(BOOT_DIR, f"members_global_k{K}_edited_typology.csv")

# --- Y-axis range (raw year counts) ---
Y_MIN, Y_MAX = 0, 17
Y_PAD        = 0.5
BAR_WIDTH    = 0.65

# --- font sizes ---
FS_ANNOT       = 7     # bar-top count label; set None to suppress
FS_TICK        = 8
FS_BASIN_TITLE = 20    # subplot title + shared legend, in the all-basin panel

ANNOTATE     = True
ANNOT_THRESH = 0

# --- all-basin panel layout (5 cols x 6 rows = 30 slots, 26 used) ---
PANEL_NCOLS  = 5
FIG_SIZE_ALL = (22, 26)

PANEL_LEGEND_LOC    = "lower center"
PANEL_LEGEND_ANCHOR = (0.5, 0.0)
PANEL_LEGEND_NCOL    = 5

X_AXIS_LABEL = "Typology"

# Cluster -> typology colour map ("C1" -> hex) and basin -> region lookup
CLUSTER_COLOR_MAP = {
    f"C{cn}": TYPOLOGY_COLORS[typo]
    for typo, cn_list in TYPOLOGY_MAP.items()
    for cn in cn_list
}
REGION_MEMBERSHIP = {
    bid: region
    for region, basins in REGION_BASIN_MAP.items()
    for bid in basins
}

# ============================================================
# SECTION 2: Build (or load) the per-typology basin occurrence
# tables: basin_stats_edited_k{K}_{typ}.csv, one per typology
# ============================================================

missing_typ_files = [
    typ for typ in TYPOLOGY_LABELS
    if not os.path.exists(os.path.join(BOOT_DIR, f"basin_stats_edited_k{K}_{typ}.csv"))
]

if missing_typ_files:
    print(f"Missing basin_stats_edited files for {missing_typ_files} - "
          f"building from the edited typology table ...")

    # ---- Step A: load or build the edited 5-typology membership table ----
    if os.path.exists(MEMBERS_EDITED_PATH):
        print(f"Loading existing edited typology table: {MEMBERS_EDITED_PATH}")
        members_edited = pd.read_csv(MEMBERS_EDITED_PATH)

    else:
        print("Edited typology table not found either - building it from scratch ...")

        data        = np.load(NPZ_PATH)
        curves      = data["curves"].astype(np.float64)
        years       = data["years"]
        basin_ids   = data["basin_ids"]
        curve_index = {(int(b), int(y)): i for i, (b, y) in enumerate(zip(basin_ids, years))}

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
        scores_all  = pca.fit_transform(curves)
        eof2        = pca.components_[1]
        scores_eof2 = scores_all[:, 1]
        if eof2[0] < eof2[-1]:
            scores_eof2 = -scores_eof2

        CLUSTER_TO_FIXED_TYPOLOGY = {
            c: t["tag"] for t in TYPOLOGIES for c in t["clusters"] if c != 1
        }

        def assign_typology(row):
            cl  = int(row["cluster"])
            key = (int(row["basin_id"]), int(row["year"]))

            if cl == 1:
                if key in core_set:
                    return TYPOLOGY_TAGS[0]   # "Dry" (ExDr)
                idx = curve_index.get(key)
                if idx is None:
                    return TYPOLOGY_TAGS[1]   # fallback default
                pc2 = scores_eof2[idx]
                return TYPOLOGY_TAGS[3] if pc2 > 0 else TYPOLOGY_TAGS[1]

            return CLUSTER_TO_FIXED_TYPOLOGY.get(cl, "UNASSIGNED")

        members_full["typology"] = members_full.apply(assign_typology, axis=1)

        os.makedirs(BOOT_DIR, exist_ok=True)
        members_edited = members_full[["cluster", "basin_id", "year", "typology"]]
        members_edited.to_csv(MEMBERS_EDITED_PATH, index=False)
        print(f"  Saved: {MEMBERS_EDITED_PATH}")

    # ---- Step B: per-typology, per-basin occurrence counts ----
    sorted_basins = sorted(members_edited["basin_id"].unique().tolist())

    for typ in TYPOLOGY_LABELS:
        sub_typ = members_edited[members_edited["typology"] == typ]

        basin_rows = []
        for bid in sorted_basins:
            sub_b = sub_typ[sub_typ["basin_id"] == bid]
            n_b = len(sub_b)
            if n_b == 0:
                continue
            member_years = sorted(sub_b["year"].tolist())
            basin_rows.append({
                "basin_id":           bid,
                "basin_label":        f"Basin {bid:02d}",
                "n_years_in_cluster": n_b,
                "member_years":       ";".join(str(y) for y in member_years),
            })

        df_basin_typ = pd.DataFrame(basin_rows).sort_values(
            "n_years_in_cluster", ascending=False
        ).reset_index(drop=True)

        out_path = os.path.join(BOOT_DIR, f"basin_stats_edited_k{K}_{typ}.csv")
        df_basin_typ.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

# ---- read the (now guaranteed to exist) per-typology tables ----
typ_frames = []
for typ in TYPOLOGY_LABELS:
    fpath = os.path.join(BOOT_DIR, f"basin_stats_edited_k{K}_{typ}.csv")
    df = pd.read_csv(fpath)
    df["typology"] = typ
    typ_frames.append(df)

all_data_typ = pd.concat(typ_frames, ignore_index=True)

label_map = (
    all_data_typ[["basin_id", "basin_label"]]
    .drop_duplicates("basin_id")
    .set_index("basin_id")["basin_label"]
    .to_dict()
)

# ============================================================
# SECTION 3: Pivot table - basin x typology occurrence counts
# ============================================================

pivot = (
    all_data_typ
    .pivot_table(index="basin_id", columns="typology",
                 values=PCT_COLUMN, aggfunc="sum")
    .reindex(index=BASINS_ORDER)
    .reindex(columns=list(TYPOLOGY_LABELS.keys()))
    .fillna(0)
    .astype(int)
)

PLOT_ORDER       = list(TYPOLOGY_LABELS.keys())
PLOT_COLOR_MAP   = {lab: TYPOLOGY_COLORS[lab] for lab in TYPOLOGY_LABELS}
X_AXIS_TICKS     = {lab: lab for lab in TYPOLOGY_LABELS.values()}

_max_val = pivot.values.max()
assert _max_val <= Y_MAX, (
    f"Max value {_max_val} exceeds Y_MAX={Y_MAX}. Raise Y_MAX."
)
print(f"Data loaded. Pivot shape: {pivot.shape}  (max value {_max_val})")

basin_occurrence_name = f"{PLOT_PREFIX}_{DATE_PREFIX}_basin_occurrence_k{K}_pivot.csv"
pivot.to_csv(os.path.join(OUT_BARPLOTS, basin_occurrence_name), index=True)
print(f"Saved pivot table: {os.path.join(OUT_BARPLOTS, basin_occurrence_name)}")

# ============================================================
# SECTION 4: Shared legend handles + per-basin bar-plot helper
# ============================================================

def _make_typology_legend_handles():
    """One patch per unique colour (typology_edited mode: 5 typologies)."""
    seen_colours = set()
    handles = []
    for cat in PLOT_ORDER:
        colour = PLOT_COLOR_MAP[cat]
        if colour in seen_colours:
            continue
        seen_colours.add(colour)
        label = typology_display[cat]
        handles.append(
            mpatches.Patch(facecolor=colour, edgecolor="grey",
                           linewidth=0.5, label=label)
        )
    return handles

LEGEND_HANDLES = _make_typology_legend_handles()


def _get_basin_title(bid):
    raw = label_map.get(bid, str(bid))
    return raw.replace("Basin ", "B")


def _draw_basin_barplot(ax, basin_id, show_xlabel=True, show_ylabel=True):
    """Draw the typology occurrence bar plot for one basin onto `ax`."""
    x_pos  = np.arange(len(PLOT_ORDER))
    counts = pivot.loc[basin_id, PLOT_ORDER].values
    colors = [PLOT_COLOR_MAP[c] for c in PLOT_ORDER]

    bars = ax.bar(
        x_pos, counts,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    if ANNOTATE:
        for bar, val in zip(bars, counts):
            if val > ANNOT_THRESH:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.25,
                    str(int(val)),
                    ha="center", va="bottom",
                    fontsize=FS_ANNOT, color="black",
                )

    ax.set_xlim(-0.6, len(PLOT_ORDER) - 0.4)
    ax.set_ylim(Y_MIN, Y_MAX + Y_PAD)
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1, 5))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([X_AXIS_TICKS[c] for c in X_AXIS_TICKS],
                       fontsize=FS_TICK - 1)

    if show_ylabel:
        ax.set_ylabel("No. of years", fontsize=15, labelpad=4)
    if show_xlabel:
        ax.set_xlabel(X_AXIS_LABEL, fontsize=15, labelpad=4)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FS_TICK - 1)

    # Colour spines by region membership
    region    = REGION_MEMBERSHIP.get(basin_id, "AP")
    spine_col = REGION_COLORS.get(region, "black")
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_col)
        spine.set_linewidth(1.2)

# ============================================================
# SECTION 5: ALL-BASIN PANEL (26 subplots, single shared legend)
# ============================================================

print("\nGenerating all-basin panel ...")

n_basins = len(BASINS_ORDER)
n_cols   = PANEL_NCOLS
n_rows   = int(np.ceil(n_basins / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE_ALL,
                          constrained_layout=False, dpi=400)
axes_flat = axes.flatten()

for idx, bid in enumerate(BASINS_ORDER):
    ax     = axes_flat[idx]
    show_y = (idx % n_cols == 0)

    # letter tag (a, b, c, ...) in the top-left corner
    ax.text(
        0.03, 0.95, string.ascii_lowercase[idx],
        transform=ax.transAxes,
        fontsize=FS_BASIN_TITLE, fontweight="bold",
        ha="left", va="top",
        color="black",
    )

    _draw_basin_barplot(ax, bid, show_xlabel=True, show_ylabel=show_y)

    region = REGION_MEMBERSHIP.get(bid, "AP")
    ax.set_title(
        f"{_get_basin_title(bid)}  [{region}]",
        fontsize=FS_BASIN_TITLE, pad=3,
        color=REGION_COLORS.get(region, "black"),
    )
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    if not show_y:
        ax.set_ylabel("")

for idx in range(n_basins, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(
    f"Cluster occurrence per basin  (k={K})",
    fontsize=20, y=1.002, fontweight="bold",
)

fig.legend(
    handles=LEGEND_HANDLES,
    title=X_AXIS_LABEL,
    loc=PANEL_LEGEND_LOC,
    ncol=PANEL_LEGEND_NCOL,
    fontsize=20,
    title_fontsize=20,
    framealpha=0.88,
    bbox_to_anchor=PANEL_LEGEND_ANCHOR,
)

plt.subplots_adjust(
    hspace=0.55, wspace=0.35,
    left=0.06, right=0.98,
    top=0.97, bottom=0.06,
)

all_panel_path = os.path.join(
    OUT_BARPLOTS, f"{PLOT_PREFIX}_{DATE_PREFIX}_all_basins_occurrence_barplots_k{K}.png"
)
fig.savefig(all_panel_path, dpi=400, bbox_inches="tight")
print(f"Saved: {all_panel_path}")
plt.show()
