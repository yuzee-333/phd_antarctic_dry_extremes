# ============================================================
# SECTION 0: Imports and configuration
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from sklearn.decomposition import PCA
from datetime import date

import sys
from config import *

# ============================================================
# SECTION 1: Paths and constants
# (mirrors the "spatial maps based on Chronology" -> "hatching dry
#  extremes" -> "block basins" cells of
#  4_Chronology_Analyses_Results.ipynb)
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

V6_TAG = "spatial_typology_blocks_dry_wet_hatched"
DRY_TYPOLOGY = "Dry"
WET_TYPOLOGY = "Wet"

# v5 persistent-extreme criterion (same as Fig8.py - needed here to
# flag which basins get the hatch overlay on the spatial map)
EXTREME_WINDOW_MIN        = 4
EXTREME_WINDOW_MAX        = 52
EXTREME_THRESHOLD         = 0.02
EXTREME_FRACTION_REQUIRED = 1.00

# Basins excluded from the "extreme-persistent" hatch flagging
EXCLUDE_BASINS = {3, 19}

EXTREME_HATCH       = "..."
EXTREME_HATCH_COLOR = "black"

CST_PATH = os.path.join(MAR_DATA_DIR, "MARcst-AN35km-176x148.cdf2")

# ============================================================
# SECTION 2: Load the stacked rarity curves
# ============================================================

data        = np.load(NPZ_PATH)
curves      = data["curves"].astype(np.float64)
years_arr   = data["years"]
basin_ids   = data["basin_ids"]
windows     = data["windows"]

curve_index = {(int(b), int(y)): i for i, (b, y) in enumerate(zip(basin_ids, years_arr))}

# ============================================================
# SECTION 3: Load (or build) the edited 5-typology membership table
# ============================================================

if os.path.exists(MEMBERS_EDITED_PATH):
    print(f"Loading existing edited typology table: {MEMBERS_EDITED_PATH}")
    members_edited = pd.read_csv(MEMBERS_EDITED_PATH)

else:
    print("Edited typology table not found - building it ...")

    members_full = pd.read_csv(MEMBERS_PATH)
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
                return TYPOLOGY_TAGS[0]
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

# ============================================================
# SECTION 4: v5 persistent-extreme basin-years (drives the hatch
# overlay), then exclude basins 3 and 19 as in the notebook
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

extreme_window_mask = (windows >= EXTREME_WINDOW_MIN) & (windows <= EXTREME_WINDOW_MAX)

extreme_persistent_set = set()
for key in core_set_v4:
    if key not in curve_index:
        continue
    curve = curves[curve_index[key], :]
    frac_below = np.mean(curve[extreme_window_mask] <= EXTREME_THRESHOLD)
    if frac_below >= EXTREME_FRACTION_REQUIRED:
        extreme_persistent_set.add(key)

extreme_persistent_df = pd.DataFrame(
    sorted(extreme_persistent_set), columns=["basin_id", "year"]
).sort_values(["basin_id", "year"]).reset_index(drop=True)
print(f"Persistent-extreme basin-years (pre-exclusion): {len(extreme_persistent_df)}")

extreme_persistent_df_filtered = extreme_persistent_df[
    ~extreme_persistent_df["basin_id"].isin(EXCLUDE_BASINS)
].reset_index(drop=True)
print(f"After excluding basins {EXCLUDE_BASINS}: "
      f"{len(extreme_persistent_df_filtered)} rows")

# ============================================================
# SECTION 5: Temporal blocks
# ============================================================

BLOCKS = [
    {"label": "1982–1989", "y_start": 1982, "y_end": 1989,
     "narrative_note": "Weddell Sea sector dry extremes"},
    {"label": "1990–1997", "y_start": 1990, "y_end": 1997,
     "narrative_note": "Shift to Amery IS sector"},
    {"label": "1998–2005", "y_start": 1998, "y_end": 2005,
     "narrative_note": "Weddell Sea sector again"},
    {"label": "2006–2013", "y_start": 2006, "y_end": 2013,
     "narrative_note": "Pacific/Ross Sea sector (2007-2010)"},
    {"label": "2014–2022", "y_start": 2014, "y_end": 2022,
     "narrative_note": "Amery IS + Wilkes Land sectors"},
]
for b in BLOCKS:
    b["n_years"] = b["y_end"] - b["y_start"] + 1

# ============================================================
# SECTION 6: Load MAR constants and masks (identical to
# 5_metrics.py's plot_4panel_map setup)
# ============================================================

ds = xr.open_dataset(CST_PATH)
if "y" in ds.dims and "Y" not in ds.dims:
    ds = ds.rename({"y": "Y"})
if "x" in ds.dims and "X" not in ds.dims:
    ds = ds.rename({"x": "X"})

LAT    = ds["LAT"]
LON    = ds["LON"]
AIS    = ds["AIS"]
RIGNOT = ds["RIGNOT"]
AREA   = ds["AREA"]

LON_wrapped = ((LON + 180) % 360) - 180
valid = (AIS > 0.5) & np.isfinite(RIGNOT)

ids = np.unique(RIGNOT.where(valid).values)
ids = ids[np.isfinite(ids)].astype(int)

AREA_vals = AREA.where(valid, 0.0)
AREA_km2 = AREA_vals / 1e6 if float(AREA_vals.sum()) > 1e9 else AREA_vals

ny, nx = AREA_km2.shape
Y_IND = xr.DataArray(
    np.repeat(np.arange(ny)[:, None], nx, axis=1),
    dims=("Y", "X"), coords={"Y": AREA_km2["Y"], "X": AREA_km2["X"]},
)
X_IND = xr.DataArray(
    np.repeat(np.arange(nx)[None, :], ny, axis=0),
    dims=("Y", "X"), coords={"Y": AREA_km2["Y"], "X": AREA_km2["X"]},
)

centroids = []
for bid in ids:
    mask = (RIGNOT == bid) & valid
    if not bool(mask.any()):
        continue
    A = AREA_km2.where(mask, 0.0)
    w = A / (A.sum() + 1e-30)
    y_c = int(np.round((w * Y_IND).sum().item()))
    x_c = int(np.round((w * X_IND).sum().item()))
    y_c = max(0, min(ny - 1, y_c))
    x_c = max(0, min(nx - 1, x_c))
    lat_c = float(LAT.isel(Y=y_c, X=x_c).item())
    lon_c = float(LON_wrapped.isel(Y=y_c, X=x_c).item())
    centroids.append((int(bid), lat_c, lon_c))


def make_field_from_basin_values(basin_val_dict):
    """Continuous field (NaN outside mask, else per-basin value) -
    used for the diverging colour layer."""
    Z = xr.full_like(RIGNOT, np.nan, dtype="float64")
    for bid, val in basin_val_dict.items():
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        Z = xr.where((RIGNOT == int(bid)) & valid, float(val), Z)
    return Z


def make_binary_field(flagged_basin_ids):
    """Binary field (0.0 baseline, 1.0 for flagged basins) - used for
    the contourf hatch overlay."""
    Z = xr.full_like(RIGNOT, 0.0, dtype="float64")
    for bid in flagged_basin_ids:
        Z = xr.where((RIGNOT == int(bid)) & valid, 1.0, Z)
    return Z

# ============================================================
# SECTION 7: Per-block NET score (dry_count - wet_count) per basin,
# and the per-block flagged-basin sets (block basins)
# ============================================================

dry_years = members_edited[members_edited["typology"] == DRY_TYPOLOGY]
wet_years = members_edited[members_edited["typology"] == WET_TYPOLOGY]

all_basin_ids = sorted(members_edited["basin_id"].unique())

block_basin_netscores = []
for b in BLOCKS:
    dry_sub = dry_years[(dry_years["year"] >= b["y_start"]) & (dry_years["year"] <= b["y_end"])]
    wet_sub = wet_years[(wet_years["year"] >= b["y_start"]) & (wet_years["year"] <= b["y_end"])]

    dry_counts = dry_sub.groupby("basin_id").size().to_dict()
    wet_counts = wet_sub.groupby("basin_id").size().to_dict()

    net = {bid: dry_counts.get(bid, 0) - wet_counts.get(bid, 0) for bid in all_basin_ids}
    block_basin_netscores.append(net)

    print(f"  {b['label']}: total Dry = {sum(dry_counts.values())}, "
          f"total Wet = {sum(wet_counts.values())}, "
          f"net range = [{min(net.values())}, {max(net.values())}]")

# symmetric colour scale across all blocks, centred on 0
VMAX_ABS = max(b["n_years"] for b in BLOCKS)
VMIN, VMAX = -VMAX_ABS, VMAX_ABS

# Rebuild the per-block flagged-basin sets from the filtered data
block_flagged_basins = []
for b in BLOCKS:
    extreme_sub = extreme_persistent_df_filtered[
        (extreme_persistent_df_filtered["year"] >= b["y_start"]) &
        (extreme_persistent_df_filtered["year"] <= b["y_end"])
    ]
    flagged = set(extreme_sub["basin_id"].unique().tolist())
    block_flagged_basins.append(flagged)
    print(f"  {b['label']}: basins with >=1 extreme-persistent year "
          f"(excl. {sorted(EXCLUDE_BASINS)}) = {len(flagged)}  ->  {sorted(flagged)}")

# Custom diverging colormap: dark green (wet) -> white -> dark brown (dry)
DRY_WET_CMAP = LinearSegmentedColormap.from_list(
    "dry_wet_diverge",
    ["#1b7837", "#7fbf7b", "#f7f7f7", "#d9b382", "#8c510a"],
    N=256,
)

# ============================================================
# SECTION 8: Plot - N-panel polar map, shared diverging colour
# scale, with extreme-persistent hatching ("block basins")
# ============================================================

n_blocks = len(BLOCKS)
n_cols = 3
n_rows = int(np.ceil(n_blocks / n_cols))

proj = ccrs.SouthPolarStereo(central_longitude=0)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 6.2 * n_rows),
                          subplot_kw={"projection": proj})
fig.subplots_adjust(left=0.03, right=0.88, bottom=0.08, top=0.90,
                     wspace=0.25, hspace=0.25)
axs_flat = np.atleast_1d(axs).flatten()

theta = np.linspace(0, 2 * np.pi, 361)
circle = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(circle * 0.5 + 0.5)

norm = TwoSlopeNorm(vmin=VMIN, vcenter=0, vmax=VMAX)
mappable = None

for i, b in enumerate(BLOCKS):
    ax = axs_flat[i]
    ax.set_boundary(circle_path, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=0)
    ax.coastlines(color="black", linewidth=0.6, zorder=5)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        xlocs=np.arange(-180, 181, 30),
        linewidth=0.5, color="k", alpha=0.5, linestyle="--",
        x_inline=False, y_inline=False, zorder=6
    )
    gl.ylabels_left = gl.ylabels_right = False
    gl.ylabels_top = gl.ylabels_bottom = False
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 8}

    Z = make_field_from_basin_values(block_basin_netscores[i])
    mesh = ax.pcolormesh(
        LON_wrapped, LAT, Z.where(valid),
        transform=ccrs.PlateCarree(),
        cmap=DRY_WET_CMAP, norm=norm, shading="nearest", zorder=4,
    )
    mappable = mesh

    Z_extreme = make_binary_field(block_flagged_basins[i])
    ax.contourf(
        LON_wrapped, LAT, Z_extreme.where(valid),
        levels=[0.5, 1.5], colors="none", hatches=[EXTREME_HATCH],
        transform=ccrs.PlateCarree(), zorder=4,
    )

    ax.contour(
        LON_wrapped, LAT, AIS, levels=[0.5],
        colors="k", linewidths=2.0,
        transform=ccrs.PlateCarree(), zorder=3
    )

    for bid, lat_c, lon_c in centroids:
        dlat = LABEL_OFFSETS.get(bid, {}).get("dlat", 0.0)
        dlon = LABEL_OFFSETS.get(bid, {}).get("dlon", 0.0)
        text_lat = lat_c + dlat
        text_lon = lon_c + dlon

        ax.text(
            text_lon, text_lat, f"{bid}",
            transform=ccrs.PlateCarree(),
            fontsize=LABEL_FONTSIZE,
            ha="center", va="center", color="k",
            bbox=dict(facecolor="white", alpha=1.00, boxstyle="round,pad=0.15", linewidth=0.0),
            zorder=9,
        )

        if bid in ANNOTATION_LINES and (abs(dlat) > 0.1 or abs(dlon) > 0.1):
            style = ANNOTATION_LINES[bid]
            vec_lat = lat_c - text_lat
            vec_lon = lon_c - text_lon
            vec_len = np.sqrt(vec_lat**2 + vec_lon**2)
            if vec_len > 0:
                unit_lat = vec_lat / vec_len
                unit_lon = vec_lon / vec_len
            else:
                unit_lat = unit_lon = 0.0
            tip_offset = style.get("tip_offset", 0.0)
            tip_lat = lat_c - unit_lat * tip_offset
            tip_lon = lon_c - unit_lon * tip_offset

            ax.annotate(
                "",
                xy=(tip_lon, tip_lat),
                xytext=(text_lon, text_lat),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                arrowprops=dict(
                    arrowstyle=style["arrowstyle"], color=style["color"],
                    lw=style["lw"], linestyle=style["ls"],
                    shrinkA=6, shrinkB=3,
                ),
                zorder=7,
            )

    ax.set_title(f"{b['label']}\n{b['narrative_note']}", fontsize=11)

for j in range(n_blocks, n_rows * n_cols):
    axs_flat[j].axis("off")

cax = fig.add_axes([0.91, 0.15, 0.012, 0.68])
cb = plt.colorbar(mappable, cax=cax, orientation="vertical", extend="neither")
cb.set_label("← Wet-dominant        Dry-dominant →\n(net years, within block)",
             fontsize=10)
cb.ax.tick_params(labelsize=10)

hatch_handle = mpatches.Patch(
    facecolor="none", edgecolor=EXTREME_HATCH_COLOR, hatch=EXTREME_HATCH,
    label="≥1 extreme-persistent year in block\n"
          "(NEP≤0.02 for ≥90% of 4–52wk, v5 criterion)"
)
fig.legend(handles=[hatch_handle], loc="lower center", fontsize=10,
           bbox_to_anchor=(0.5, 0.0), ncol=1, framealpha=0.9)

fig.suptitle(f"Spatial Dry vs. Wet balance by temporal block, with extreme-persistent hatching  (k={K})",
             fontsize=13, y=0.97)

out_path = os.path.join(YRLY_OUT, f"{V6_TAG}_{DATE_PREFIX}__k{K}.png")
plt.savefig(out_path, dpi=400, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
