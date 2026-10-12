# ============================================================
# SECTION 0: Imports and configuration
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
from datetime import date

import sys
from config import *

# ============================================================
# SECTION 1: Paths and plot constants
# (mirrors the "v5_Extremes hatched" section of
#  4_Chronology_Analyses_Results.ipynb - FULL HEATMAP, all basins)
# ============================================================

K           = 10
RANDOM_SEED = 42
DATE_PREFIX = date.today().strftime("%Y%m%d")

OUT_MAIN = os.path.join(OUT_FOLDER, "04_global_clustering")
YRLY_OUT = os.path.join(OUT_MAIN, "4.3_Chronology")
os.makedirs(YRLY_OUT, exist_ok=True)
BOOT_DIR = os.path.join(OUT_MAIN, "4.2.0_C1_stability_across_k",
                         "4.2.1_bootstrap_permutation_analysis")

NPZ_PATH            = os.path.join(OUT_MAIN, "all_basins_rarity_curves.npz")
MEMBERS_PATH        = os.path.join(OUT_MAIN, f"members_global_k{K}.csv")
BOOT_FREQ_PATH      = os.path.join(BOOT_DIR, f"c1_inclusion_frequency_k{K}.csv")
MEMBERS_EDITED_PATH = os.path.join(BOOT_DIR, f"members_global_k{K}_edited_typology.csv")
CORE_MEMBERS_PATH   = os.path.join(BOOT_DIR, f"c1_core_members_k{K}.csv")

EDITED_TAG = "typology_edited"
V5_TAG     = "typology_edited_v5_ext_persist_hatch"

# Plotting configs (same as the notebook's "PLOTTING CONFIGS" cell)
TICK_LEN          = 10
FIG_SIZE          = (20, 10)
FS_TITLE          = 12
FS_AXIS           = 12
FS_TICK           = 20
REGION_FS         = 12
EVENT_MIXED_COLOR = "orange"
EVENT_LW          = 2.0

# v5 persistent-extreme hatching criterion
EXTREME_WINDOW_MIN        = 4        # weeks
EXTREME_WINDOW_MAX        = 52       # weeks
EXTREME_THRESHOLD         = 0.02     # NEP threshold ("extreme" per McKee et al. 1993)
EXTREME_FRACTION_REQUIRED = 1.00     # fraction of in-window values that must meet the threshold

EXTREME_HATCH       = "/////"
EXTREME_HATCH_COLOR = "black"
EXTREME_HATCH_LW    = 0.9

# ============================================================
# SECTION 2: Load the stacked rarity curves + region membership
# ============================================================

data        = np.load(NPZ_PATH)
curves      = data["curves"].astype(np.float64)
years_arr   = data["years"]
basin_ids   = data["basin_ids"]
windows     = data["windows"]

curve_index = {(int(b), int(y)): i for i, (b, y) in enumerate(zip(basin_ids, years_arr))}

region_membership = {}
start = 0
for region, size in zip(REGION_NAMES, REGION_SIZES):
    for bid in BASINS_ORDER[start:start + size]:
        region_membership[bid] = region
    start += size

# ============================================================
# SECTION 3: Load (or build) the edited 5-typology membership table
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
    core_set_bootstrap = set(zip(
        boot_freq.loc[boot_freq["tier"] == "core", "basin_id"].astype(int),
        boot_freq.loc[boot_freq["tier"] == "core", "year"].astype(int),
    ))

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
            if key in core_set_bootstrap:
                return TYPOLOGY_TAGS[0]   # "Dry" (ExDr)
            idx = curve_index.get(key)
            if idx is None:
                return TYPOLOGY_TAGS[1]
            pc2 = scores_eof2[idx]
            return TYPOLOGY_TAGS[3] if pc2 > 0 else TYPOLOGY_TAGS[1]
        return CLUSTER_TO_FIXED_TYPOLOGY.get(cl, "UNASSIGNED")

    members_full["typology"] = members_full.apply(assign_typology, axis=1)

    os.makedirs(BOOT_DIR, exist_ok=True)
    members_edited = members_full[["cluster", "basin_id", "year", "typology"]]
    members_edited.to_csv(MEMBERS_EDITED_PATH, index=False)
    print(f"  Saved: {MEMBERS_EDITED_PATH}")

n_exdr_edited = int((members_edited["typology"] == TYPOLOGY_TAGS[0]).sum())
n_dr_from_c1  = int(((members_edited["cluster"] == 1) &
                      (members_edited["typology"] == TYPOLOGY_TAGS[1])).sum())
n_wtn_from_c1 = int(((members_edited["cluster"] == 1) &
                      (members_edited["typology"] == TYPOLOGY_TAGS[3])).sum())
print(f"Dry (core): n = {n_exdr_edited}  |  non-core C1 -> Dr: {n_dr_from_c1}  "
      f"|  non-core C1 -> Wtn: {n_wtn_from_c1}")

# ============================================================
# SECTION 4: Labelled pivot (basin x year), event extraction, and
# integer-encoded pivot for imshow
# ============================================================

years = sorted(members_edited["year"].unique())

pivot_labels_edited = (
    members_edited
    .pivot_table(index="basin_id", columns="year",
                 values="typology", aggfunc="first")
    .reindex(index=BASINS_ORDER)
    .reindex(columns=years)
    .fillna("--")
)
pivot_labels_edited.insert(0, "region", [region_membership[b] for b in BASINS_ORDER])
pivot_labels_edited.index.name = "basin_id"
year_cols = [c for c in pivot_labels_edited.columns if c != "region"]

# --- Event extraction: contiguous runs of "ExDr" (strict) ---
DRY_CORE = TYPOLOGY_TAGS[0]

def extract_mixed_events(year_series):
    ELIGIBLE = {DRY_CORE}
    events = []
    labels = list(year_series.items())
    n = len(labels)
    i = 0
    while i < n:
        year_i, label_i = labels[i]
        if label_i not in ELIGIBLE:
            i += 1
            continue
        run = []
        j = i
        while j < n and labels[j][1] in ELIGIBLE:
            run.append(labels[j])
            j += 1
        run_labels = [r[1] for r in run]
        if DRY_CORE in run_labels:
            run_years = [r[0] for r in run]
            n_D  = run_labels.count(DRY_CORE)
            n_Dr = run_labels.count(TYPOLOGY_TAGS[1])   # Dr
            events.append({
                "start_year":   run_years[0],
                "end_year":     run_years[-1],
                "length":       len(run),
                "n_D":          n_D,
                "n_Dr":         n_Dr,
                "n_connective": n_Dr,
                "D_only":       n_Dr == 0,
                "consecutive":  len(run) > 1,
                "sequence":     "-".join(run_labels),
            })
        i = j
    return events

mixed_events_edited = []
for basin_id in BASINS_ORDER:
    row = pivot_labels_edited.loc[basin_id, year_cols]
    region = region_membership[basin_id]
    for ev in extract_mixed_events(row):
        ev["basin_id"] = basin_id
        ev["region"] = region
        mixed_events_edited.append(ev)

col_order = ["basin_id", "region", "start_year", "end_year", "length",
             "n_D", "n_Dr", "n_connective", "D_only", "consecutive", "sequence"]
mixed_df_edited = (
    pd.DataFrame(mixed_events_edited)[col_order]
    if len(mixed_events_edited) else pd.DataFrame(columns=col_order)
)

# --- Integer-encode the typology pivot for imshow ---
label_list   = TYPOLOGY_LABELS
label_to_int = {l: i for i, l in enumerate(label_list)}
cmap_colors  = [TYPOLOGY_COLORS[l] for l in label_list]
cmap = ListedColormap(cmap_colors)
norm = BoundaryNorm(boundaries=np.arange(-0.5, len(label_list) + 0.5, 1),
                     ncolors=len(label_list))

pivot_int_edited = pivot_labels_edited[year_cols].copy()
for col in pivot_int_edited.columns:
    pivot_int_edited[col] = pivot_int_edited[col].map(label_to_int)
pivot_int_edited = pivot_int_edited.fillna(0).astype(int)

year_to_idx  = {yr: i for i, yr in enumerate(year_cols)}
basin_to_row = {bid: i for i, bid in enumerate(BASINS_ORDER)}

# ============================================================
# SECTION 5: Core ExDr members (v4) + v5 persistent-extreme set
# ============================================================

if not os.path.exists(CORE_MEMBERS_PATH):
    raise FileNotFoundError(
        f"Missing core C1 (ExDr) members table: {CORE_MEMBERS_PATH}\n"
        "This is produced by the bootstrap stability analysis in "
        "3_Global_Clustering_and_Diagnostics_Heatmaps.ipynb - run that "
        "section first (it is too expensive to redo here)."
    )
core_members_df = pd.read_csv(CORE_MEMBERS_PATH)
core_set_v4 = set(zip(
    core_members_df["basin_id"].astype(int),
    core_members_df["year"].astype(int),
))
n_core_v5 = len(core_set_v4)
print(f"Core ExDr basin-years (population for v5 criterion): n = {n_core_v5}")

extreme_window_mask = (windows >= EXTREME_WINDOW_MIN) & (windows <= EXTREME_WINDOW_MAX)
print(f"Persistence window: {int(extreme_window_mask.sum())} NEP values between "
      f"{EXTREME_WINDOW_MIN}-{EXTREME_WINDOW_MAX} weeks")

extreme_persistent_set = set()
for key in core_set_v4:
    if key not in curve_index:
        continue
    curve = curves[curve_index[key], :]
    frac_below = np.mean(curve[extreme_window_mask] <= EXTREME_THRESHOLD)
    if frac_below >= EXTREME_FRACTION_REQUIRED:
        extreme_persistent_set.add(key)

n_extreme_persistent = len(extreme_persistent_set)
print(f"Core ExDr basin-years meeting the persistent-extreme criterion: "
      f"n = {n_extreme_persistent} out of {n_core_v5}")

extreme_persistent_df = pd.DataFrame(
    sorted(extreme_persistent_set), columns=["basin_id", "year"]
).sort_values(["basin_id", "year"]).reset_index(drop=True)

# ============================================================
# SECTION 6: FULL HEATMAP (all basins) - v5: event boxes restricted
# to >=2 consecutive core years; hatching = persistent-extreme only
# ============================================================

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=300)
fig.subplots_adjust(bottom=0.12)

im = ax.imshow(pivot_int_edited.values, aspect="auto", cmap=cmap, norm=norm,
                interpolation="nearest")

# multi-year event outlines - only for runs of >=2 consecutive core
# ExDr years; single-year "events" get no box at all
for _, ev in mixed_df_edited.iterrows():
    if not bool(ev["consecutive"]):
        continue

    basin_id = int(ev["basin_id"])
    start_yr = int(ev["start_year"])
    end_yr   = int(ev["end_year"])

    row_idx   = basin_to_row.get(basin_id)
    col_start = year_to_idx.get(start_yr)
    col_end   = year_to_idx.get(end_yr)
    if row_idx is None or col_start is None or col_end is None:
        continue

    rect = mpatches.FancyBboxPatch(
        (col_start - 0.48, row_idx - 0.48),
        (col_end - col_start + 0.96), 0.96,
        boxstyle="round,pad=0.0", linewidth=EVENT_LW,
        edgecolor=EVENT_MIXED_COLOR, facecolor="none", hatch=None,
        zorder=6, clip_on=False,
    )
    ax.add_patch(rect)

# per-cell persistent-extreme hatching - independent of consecutiveness
for basin_id, yr in extreme_persistent_set:
    row_idx = basin_to_row.get(basin_id)
    col_idx = year_to_idx.get(yr)
    if row_idx is None or col_idx is None:
        continue
    cell_rect = mpatches.Rectangle(
        (col_idx - 0.48, row_idx - 0.48), 0.96, 0.96,
        linewidth=EXTREME_HATCH_LW, edgecolor=EXTREME_HATCH_COLOR,
        facecolor="none", hatch=EXTREME_HATCH,
        zorder=5, clip_on=False,
    )
    ax.add_patch(cell_rect)

ax.set_xticks(np.arange(len(years)))
ax.set_xticklabels(years, fontsize=12, rotation=45)
ax.set_xlabel("Year", fontsize=FS_AXIS, labelpad=6)

ax.set_yticks(np.arange(len(BASINS_ORDER)))
ax.set_yticklabels([f"B{b:02d}" for b in BASINS_ORDER], fontsize=FS_TICK)
ax.set_ylabel("Basin", fontsize=15, labelpad=6)

ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(BASINS_ORDER), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.3)
ax.tick_params(which="minor", bottom=False, left=False)

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
region_bounds = np.cumsum(REGION_SIZES)[:-1] - 0.5
starts  = [0] + list(np.cumsum(REGION_SIZES)[:-1])
centers = [(s + s + sz - 1) / 2 for s, sz in zip(starts, REGION_SIZES)]
ax2.set_yticks(region_bounds)
ax2.set_yticklabels([])
ax2.tick_params(axis="y", which="major", direction="inout",
                 length=TICK_LEN, width=1.2, color="red", left=True, right=True)
for spine in ax2.spines.values():
    spine.set_visible(False)
for name, center, colors in zip(REGION_NAMES, centers, REGION_COLORS.values()):
    ax2.text(1.01, center, name, transform=ax2.get_yaxis_transform(),
              va="center", ha="left", fontsize=REGION_FS, color=colors, fontweight="bold")

cell_handles = [mpatches.Patch(facecolor=TYPOLOGY_COLORS[l], edgecolor="grey",
                                label=typology_display[l]) for l in TYPOLOGY_LABELS]
event_handles = [mpatches.Patch(facecolor="none", edgecolor=EVENT_MIXED_COLOR,
                                 hatch=None, linewidth=EVENT_LW,
                                 label="Multi-year event")]
extreme_handle = [mpatches.Patch(facecolor="none", edgecolor=EXTREME_HATCH_COLOR,
                                  hatch=EXTREME_HATCH, linewidth=EXTREME_HATCH_LW,
                                  label="Extreme Dry year")]

ax.legend(handles=cell_handles + event_handles + extreme_handle,
          title="Typology / Event / Persistent-extreme",
          fontsize=12, title_fontsize=12, loc="upper center",
          bbox_to_anchor=(0.5, -0.12),
          ncol=len(cell_handles + event_handles + extreme_handle),
          framealpha=0.85, handlelength=1.5, columnspacing=1.0)

fig.subplots_adjust(right=0.80)
out_path = os.path.join(YRLY_OUT, f"{V5_TAG}_{DATE_PREFIX}_chronology_k{K}.png")
fig.savefig(out_path, dpi=400, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
