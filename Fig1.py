# ============================================================
# SECTION 0: Imports and configuration
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines   import Line2D

import glob
import math
import xarray as xr
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import colormaps
import matplotlib.dates as mdates


import sys 
from config import *


MAR_FILE      = #"daily-TP-MARv3.12_ERA5-1979-2023.nc" from Zenodo
CST_PATH      = "./MARcst-AN35km-176x148.cdf2"

ds = xr.open_dataset(CST_PATH, engine ='netcdf4')
print(ds.variables)

if "y" in ds.dims and "Y" not in ds.dims:
    ds = ds.rename({"y": "Y"})
if "x" in ds.dims and "X" not in ds.dims:
    ds = ds.rename({"x": "X"})

LAT    = ds["LAT"]
LON    = ds["LON"]
AIS    = ds["AIS"]
RIGNOT = ds["RIGNOT"]
AREA   = ds["AREA"]
GROUND = ds["GROUND"]  # percent (0..100), grounded ice fraction
ICE    = ds["ICE"]      # % ice mask
SH     = ds["SH"]    #surface height
ROCK   = ds["ROCK"]  #rock
SOL    = ds["SOL"]   #surface type


OUT_DIR = "/output path/"

os.makedirs(OUT_DIR, exist_ok=True)

# Output file paths

print(f"Output directory: {OUT_DIR}")

# ============================================================
# SECTION 1: Load MAR constants file and harmonize dims
# ============================================================
# Wrap longitudes to [-180, 180] for plotting
LON_w = ((LON + 180) % 360) - 180

# Global validity mask: inside AIS, finite basin ID
valid_mask = (AIS > 0.5) & np.isfinite(RIGNOT)

# Extract sorted list of basin IDs
basin_ids_all = np.unique(RIGNOT.values)
basin_ids_all = basin_ids_all[np.isfinite(basin_ids_all)].astype(int)
basin_ids_all = np.sort(basin_ids_all)

print(f"\nTotal RIGNOT basin IDs found: {basin_ids_all}")
print(f"Number of basins: {len(basin_ids_all)}")

valid = (AIS > 0.5) & np.isfinite(RIGNOT)

ids = (
    np.unique(RIGNOT.where(valid).values[np.isfinite(RIGNOT.where(valid).values)])
    .astype(int)
)


MIN_AREA_KM2  = 0.0

# -----------------------------
# 2) Centroids in grid-index space
# -----------------------------
AREA_vals = AREA.where(valid, 0.0)
if float(AREA_vals.sum()) > 1e9:
    AREA_km2 = AREA_vals / 1e6
else:
    AREA_km2 = AREA_vals

ny, nx = AREA_km2.shape

Y_IND = xr.DataArray(
    np.repeat(np.arange(ny)[:, None], nx, axis=1),
    dims=("Y", "X"),
    coords={"Y": AREA_km2["Y"], "X": AREA_km2["X"]},
)

X_IND = xr.DataArray(
    np.repeat(np.arange(nx)[None, :], ny, axis=0),
    dims=("Y", "X"),
    coords={"Y": AREA_km2["Y"], "X": AREA_km2["X"]},
)

centroid_rows = {"basin_id": [], "lat_c": [], "lon_c": [], "area_km2": [], "n_cells": []}

for bid in ids:
    mask = (RIGNOT == bid) & valid
    if not bool(mask.any()):
        continue

    A = AREA_km2.where(mask, 0.0)
    w = A / (A.sum() + 1e-30)

    # weighted index centroid
    y_c = int(np.round((w * Y_IND).sum().item()))
    x_c = int(np.round((w * X_IND).sum().item()))
    y_c = max(0, min(ny - 1, y_c))   
    x_c = max(0, min(nx - 1, x_c))

    lat_c = float(LAT.isel(Y=y_c, X=x_c).item())
    lon_c = float(LON_w.isel(Y=y_c, X=x_c).item())

    area_sum = float(A.sum().item())
    n_cells  = int(mask.sum().item())

    centroid_rows["basin_id"].append(bid)
    centroid_rows["lat_c"].append(lat_c)
    centroid_rows["lon_c"].append(lon_c)
    centroid_rows["area_km2"].append(area_sum)
    centroid_rows["n_cells"].append(n_cells)

df = pd.read_csv(os.path.join(OUT_DIR, f"basin_mean_SH_classification.csv"))
df.head()



# ============================================================
# SECTION 5: Map — basin-mean SH coloured continuously,
#            HIGH/LOW classification shown by hatching / edge colour
# ============================================================
LABEL_FONTSIZE = 15
PLOT_BASIN_MEAN_SH = True

if PLOT_BASIN_MEAN_SH:
    SH_masked = SH.where(AIS > 0)
    print(SH_masked.shape)
    FIG_MAP_PATH = os.path.join(OUT_DIR, f"map_basin_with_mean_SH_Line_Tags.png")
else:
    SH_masked = SH.where(AIS > 0)
    SH_masked = SH_masked.where(SH_masked < 230, np.nan)
    FIG_MAP_PATH = os.path.join(OUT_DIR, f"map_basin_with_grey_Line_Tags.png")


proj = ccrs.SouthPolarStereo(central_longitude=0)
pc   = ccrs.PlateCarree()

# Circular boundary helper
theta  = np.linspace(0, 2 * np.pi, 361)
circle = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = 0.5 * circle + 0.5

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": proj}, dpi=400)
ax.set_boundary(mpath.Path(circle), transform=ax.transAxes)
ax.set_extent([-180, 180, -90, -60], crs=pc)
ax.set_facecolor("lightgrey")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)


# gridlines (no labels for a clean polar figure)
gl_lon = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    xlocs=np.arange(-180, 181, 30),
    ylocs=np.arange(-80, -20 + 1, 10),
    linewidth=0.5,
    color="grey",
    alpha=0.6,
    linestyle="--",
    x_inline=False,   # don't write labels along the line
    y_inline=False,
    zorder = 0,
)



# ── Continent fill on top of gridlines ──────────────────────
ax.add_feature(
    cfeature.LAND,
    facecolor="lightgrey",
    edgecolor="none",                  # no edge here — coastline drawn separately
    zorder=2,                          # sits above gridlines
)

ax.coastlines(color="k", linewidth=0.7, zorder=5)



# we only want longitude labels, outside the circle
gl_lon.ylabels_left = False
gl_lon.ylabels_right = False
gl_lon.ylabels_top = False
gl_lon.ylabels_bottom = False   # no automatic latitude labels
gl_lon.xlabels_top = False
gl_lon.xlabels_bottom = True    # longitudes at bottom edge (outside circle)

gl_lon.xformatter = LONGITUDE_FORMATTER
gl_lon.xlabel_style = {"size": 8}



# Continuous SH colour fill (basin-mean SH)
vmin = float(np.nanmin(SH_masked.values))
vmax = 4000  #float(np.nanmax(SH_masked.values))
cmap = plt.cm.gist_earth_r  #RdYlBu_r    # warm = high, cool = low


im = ax.pcolormesh(
    LON_w, LAT, SH_masked,
    cmap=cmap, vmin=vmin, vmax=vmax,
    transform=pc, shading="nearest", zorder=2,
)



# Basin outlines + centroid labels
for b, row in df.iterrows():
    basin_mask_bool = (valid_mask & (RIGNOT == b)).values.astype(float)
    # thin contour at the basin boundary
    try:
        ax.contour(
            LON_w, LAT, basin_mask_bool,
            levels=[0.5], colors="k", linewidths=0.5,
            transform=pc, zorder=4
        )
    except Exception:
        pass

# labels at centroids
for bid, lat_c, lon_c, area_km2, n_cells in zip(
    centroid_rows["basin_id"],
    centroid_rows["lat_c"],
    centroid_rows["lon_c"],
    centroid_rows["area_km2"],
    centroid_rows["n_cells"],
):
    if area_km2 < MIN_AREA_KM2:
        continue

    dlat = LABEL_OFFSETS.get(bid, {}).get("dlat", 0.0)
    dlon = LABEL_OFFSETS.get(bid, {}).get("dlon", 0.0)

    text_lat = lat_c + dlat
    text_lon = lon_c + dlon

    ax.text(
        lon_c + dlon,
        lat_c + dlat,
        f"{bid}",
        fontsize=LABEL_FONTSIZE, fontweight="bold", ha="center", va="center",
        color="white",transform=ccrs.PlateCarree(),zorder=6,
        path_effects=[pe.withStroke(linewidth=1.8, foreground="black")]
    )

    # # Leader line — only for basins in ANNOTATION_LINES with a real offset
    if bid in ANNOTATION_LINES and (abs(dlat) > 0.1 or abs(dlon) > 0.1):
        style = ANNOTATION_LINES[bid]

        # Unit vector pointing FROM label TO centroid
        vec_lat = lat_c - text_lat
        vec_lon = lon_c - text_lon
        vec_len = np.sqrt(vec_lat**2 + vec_lon**2)

        if vec_len > 0:
            unit_lat = vec_lat / vec_len
            unit_lon = vec_lon / vec_len
        else:
            unit_lat = unit_lon = 0.0

        # tip_offset > 0 → tip moves AWAY from centroid (shorter line)
        # tip_offset < 0 → tip moves PAST centroid (longer line)
        # tip_offset in degrees, same units as lat/lon
        tip_offset = style.get("tip_offset", 0.0)
        tip_lat = lat_c - unit_lat * tip_offset
        tip_lon = lon_c - unit_lon * tip_offset

        ax.annotate(
            "",
            xy        = (tip_lon,  tip_lat),
            xytext    = (text_lon, text_lat),
            xycoords  = ccrs.PlateCarree()._as_mpl_transform(ax),
            textcoords= ccrs.PlateCarree()._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle = style["arrowstyle"],
                color      = style["color"],
                lw         = style["lw"],
                linestyle  = style["ls"],
                shrinkA    = 6,
                shrinkB    = 3,
            ),
            zorder=7,
        )

ax.text(
    0.90, 0.20, "60°S",            # (x, y) in axes coords; y<0 puts it outside
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=10,
    rotation=45,
    color="black",
    zorder=10
)


# Colourbar
if PLOT_BASIN_MEAN_SH:
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08,
                    shrink=0.65, aspect=40, extend="both")

    cb_label = "surface height (m)"
    cb.set_label(cb_label, fontsize=11)


plt.tight_layout()
#plt.savefig(FIG_MAP_PATH, dpi=400, bbox_inches="tight")
print(f"Map saved: {FIG_MAP_PATH}")
plt.show()
