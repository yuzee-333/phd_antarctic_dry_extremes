import os
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timedelta
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
from config import *

BASE_OUT_DIR = os.path.join(OUT_FOLDER, '05_drought_metrics')
if not os.path.exists(BASE_OUT_DIR):
    os.makedirs(BASE_OUT_DIR)

TAB_DIR = os.path.join(OUT_FOLDER, '02_APA_NEP_matrices')

OUT_EXTRACT_DIR = os.path.join(BASE_OUT_DIR, "nep_minima_to_apa_extracts_bandmin")
os.makedirs(OUT_EXTRACT_DIR, exist_ok=True)

# "month-equivalent" in weeks (approx)
DURATIONS = {
    "9mo": 39,
    "12mo": 52,
    "24mo": 104,
    "36mo": 156,
}

# ---- band half-width in weeks (tune this) ----
# Example: +/-2 weeks means:
# 3mo band = weeks in [11..15], clipped to available week rows


# ============================================================
# Dry-period climatology (duration-aware) for Antarctica basins
# Extreme drought definition: NEP < 0.02 (McKee et al. SPI-style)
# Durations: 3/6/9/12 months ~ 13/26/39/52 weeks with ±2-week band
#
# For each basin & duration:
#   - drought mask from NEP band: dry(t)=1 if ANY week in band has NEP(t)<0.02
#   - APA_scale(t): mean APA across band weeks (band-mean time series)
#   - compute dry-state stats: mean/variance/quantiles
#   - compute seasonality: monthly dry-day probability, peak month, peak probability
#
# Outputs:
#   - Per-basin duration stats CSV
#   - Combined summary CSV
#   - Maps (4-panel with shared colorbar)
#   - Heatmaps (basin × duration)
# ============================================================

OUT_PARENT = BASE_OUT_DIR
OUT_STATS_DIR = os.path.join(OUT_PARENT, "climatology")
OUT_FIG_DIR   = os.path.join(OUT_STATS_DIR, "figs")
os.makedirs(OUT_STATS_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

DATE_FMT = "%d/%m/%Y"

NEP_THRESH = 0.02
BAND_HALF_WIDTH_WEEKS = 2




# ---------------------
# Paths / config
# ---------------------
OUT_DIR = os.path.join(BASE_OUT_DIR, "drought_frequency")
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_CSV  = os.path.join(OUT_EXTRACT_DIR,
    "ALL_BASINS_extreme_drought_frequency_summary_HY.csv",
)

MAR_DATA_DIR = MAR_DATA_DIR
CST_PATH     = os.path.join(MAR_DATA_DIR, "MARcst-AN35km-176x148.cdf2")


FIG_OUT_DIR  = os.path.join(OUT_DIR, "figs_drought_frequency_HY")
os.makedirs(FIG_OUT_DIR, exist_ok=True)
FIG_PATH     = os.path.join(FIG_OUT_DIR, "PROB_ge1_extreme_drought_event_per_HY_maps.png")

DUR_ORDER = list(DURATIONS.keys())
#["3mo", "6mo", "9mo", "12mo"]
PROB_COL  = "prob_ge1_event_per_hydro_year"  # from your HY summary

# Color scale: probability is naturally [0, 1]
norm = Normalize(vmin=0.0, vmax=1.0)

# Label offsets (keep same offsets you used before)
OLD_TAGS_VERSION = False 
if OLD_TAGS_VERSION:
    LABEL_OFFSETS = {
        5:  {"dlat": 0.5, "dlon": -1.0},
        16: {"dlat": 2.0, "dlon": -1.0},
        17: {"dlat": -0.5, "dlon": 0.5},
        23: {"dlat": 4.0, "dlon":  1.0},
        21: {"dlat": 3.0, "dlon": 1.0},
        25: {"dlat": 1.0, "dlon": 0.0},
        26: {"dlat": -0.5, "dlon": 1.0},
    }

    LABEL_FONTSIZE = 7

else:
    LABEL_OFFSETS = {
        1:  {"dlat": 8.0, "dlon": -1.0},                #✅
        5:  {"dlat": 0.5, "dlon": -1.0},                #✅
        15: {"dlat": 0.0, "dlon": 15.0},                #✅
        16: {"dlat": 4.0, "dlon": -2.0},                #✅
        17: {"dlat": 8.0, "dlon": -15.0},               #✅
        21: {"dlat": 8.5, "dlon":  1.0},                #✅
        22: {"dlat": 8.5, "dlon":  0.0},                #✅
        23: {"dlat": 8.5, "dlon":  1.0},                #✅
        24: {"dlat": 10.0, "dlon":  0.0},               #✅
        25: {"dlat": 8.5, "dlon":  -7.0},               #✅
        26: {"dlat": -0.5, "dlon": 15.0},               #✅
    }

    LABEL_FONTSIZE = 9



LINE_COLOR = "k"

ANNOTATION_LINES = {
    1:  {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": -0.5},
    15: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": -1.5},
    16: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": 0.0},
    17: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": -2.5},
    21: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": 2.5},
    22: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": 0.0},
    23: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": 3.5},
    24: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": 0.5},
    25: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": 0.0},
    26: {"color": LINE_COLOR, "lw": 0.7, "ls": "-", "arrowstyle": "<-", "tip_offset": -0.5},
}
ARROW_TIP_PULLBACK = 0.3   # fraction of the offset vector length


# ---------------------
# Read basin probabilities
# ---------------------
df = pd.read_csv(SUMMARY_CSV)
df["basin_id"] = df["basin"].str.extract(r"basin_(\d+)").astype(int)

# duration -> {basin_id: prob}
dur_to_prob = {}
for dur in DUR_ORDER:
    sub = df[df["duration"] == dur].copy()
    # should be one row per basin/duration, but groupby is safe
    prob_map = sub.groupby("basin_id")[PROB_COL].mean().to_dict()
    dur_to_prob[dur] = prob_map

# ---------------------
# Load MAR constants and masks
# ---------------------
ds = xr.open_dataset(CST_PATH)

# harmonize dims if needed
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

# ---------------------
# Compute area-weighted basin centroids (for labels)
# ---------------------
AREA_vals = AREA.where(valid, 0.0)
AREA_km2 = AREA_vals / 1e6 if float(AREA_vals.sum()) > 1e9 else AREA_vals

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


# =========================
# Helpers
# =========================
def read_table_csv(path, date_fmt=DATE_FMT):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df = df.sort_index()
    df.columns = pd.to_datetime(df.columns, format=date_fmt, errors="raise")
    return df

def select_week_band(weeks, center, half_width):
    weeks = np.asarray(weeks, dtype=int)
    band = weeks[(weeks >= center - half_width) & (weeks <= center + half_width)]
    if band.size == 0:
        # fallback to nearest week if empty (edge cases)
        j = int(np.argmin(np.abs(weeks - int(center))))
        band = np.array([int(weeks[j])], dtype=int)
    return band

def month_name(m):
    return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1]

def season_of_month(m):
    # Standard met seasons by month number
    if m in (12, 1, 2): return "DJF"
    if m in (3, 4, 5):  return "MAM"
    if m in (6, 7, 8):  return "JJA"
    return "SON"

def hydro_year(dt):
    # HY YYYY = Apr YYYY -> Mar YYYY+1
    return dt.year if dt.month >= 4 else dt.year - 1



# =========================
# 1) Compute climatology stats per basin/duration
# =========================
all_rows = []
monthly_rows = []  # optional long table for seasonality curves

for b in range(1, 27):
    basin = f"basin_{b:02d}"
    
    if not os.path.exists(os.path.join(OUT_STATS_DIR, f"{basin}_dry_climatology_extreme_NEPlt002.csv")):
        FILES_PRESENT = False

        f_nep = os.path.join(TAB_DIR, f"NEP_{basin}_rowsWeeks_colsDates.csv")
        f_apa = os.path.join(TAB_DIR, f"APA_{basin}_TRIM_rowsWeeks_colsDates.csv")

        if not os.path.exists(f_nep) or not os.path.exists(f_apa):
            print(f"⚠ Missing NEP/APA for {basin}")
            continue

        NEP = read_table_csv(f_nep)
        APA = read_table_csv(f_apa)

        # align to common weeks/dates
        common_dates = NEP.columns.intersection(APA.columns)
        common_weeks = NEP.index.intersection(APA.index)
        if len(common_dates) == 0 or len(common_weeks) == 0:
            print(f"⚠ No overlap NEP-APA for {basin}")
            continue
        NEP = NEP.loc[common_weeks, common_dates]
        APA = APA.loc[common_weeks, common_dates]

        weeks = NEP.index.values
        dates = NEP.columns
        months = pd.Index(dates.month)
        hydro_years = pd.Index([hydro_year(d) for d in dates])

        for dur_name, center_w in DURATIONS.items():
            band_weeks = select_week_band(weeks, center_w, BAND_HALF_WIDTH_WEEKS)

            nep_band = NEP.loc[band_weeks]  # (band_weeks × dates)
            apa_band = APA.loc[band_weeks]

            # extreme drought day if ANY week in band has NEP < threshold
            dry_mask = (nep_band < NEP_THRESH).any(axis=0).values  # boolean length = n_dates
            n_total_days = len(dry_mask)
            n_dry_days = int(np.sum(dry_mask))

            # band-mean APA time series (consistent with band logic)
            apa_scale = apa_band.mean(axis=0).values  # length n_dates
            apa_dry = apa_scale[dry_mask]

            # stats conditioned on dryness
            if n_dry_days > 0:
                mean_dry = float(np.nanmean(apa_dry))
                var_dry  = float(np.nanvar(apa_dry, ddof=1)) if n_dry_days > 1 else 0.0
                std_dry = float(np.sqrt(var_dry)) if var_dry != 0.0 else 0.0
                stde_dry = float(std_dry / np.sqrt(n_dry_days)) if n_dry_days > 1 else 0.0
                q05, q25, q50, q75, q95 = np.nanpercentile(apa_dry, [5, 25, 50, 75, 95])
            else:
                mean_dry = np.nan
                var_dry = np.nan
                std_dry = np.nan
                stde_dry = np.nan
                q05 = q25 = q50 = q75 = q95 = np.nan

            # frequency (as fraction of days that are dry at that scale)
            frac_dry_days = float(n_dry_days / n_total_days) if n_total_days > 0 else np.nan

            # Seasonality: monthly probability of dry day = (# dry days in month) / (# days in month in record)
            # (calendar-month seasonality; compatible with hydro-year framework)
            monthly_prob = {}
            for m in range(1, 13):
                m_mask = (months.values == m)
                denom = int(np.sum(m_mask))
                num = int(np.sum(dry_mask & m_mask))
                p = (num / denom) if denom > 0 else np.nan
                monthly_prob[m] = p
                monthly_rows.append({
                    "basin": basin,
                    "duration": dur_name,
                    "month": m,
                    "month_name": month_name(m),
                    "p_dry": p,
                    "n_dry_days_month": num,
                    "n_days_month": denom,
                })

            # Peak seasonality diagnostics
            # peak month by highest monthly probability
            valid_months = [m for m in range(1, 13) if np.isfinite(monthly_prob[m])]
            if len(valid_months) > 0:
                peak_m = max(valid_months, key=lambda m: monthly_prob[m])
                peak_p = float(monthly_prob[peak_m])
            else:
                peak_m = np.nan
                peak_p = np.nan

            all_rows.append({
                "basin": basin,
                "basin_id": int(b),
                "duration": dur_name,
                "center_week": int(center_w),
                "band_half_width_weeks": int(BAND_HALF_WIDTH_WEEKS),
                "nep_threshold": float(NEP_THRESH),

                "n_total_days": int(n_total_days),
                "n_dry_days": int(n_dry_days),
                "frac_dry_days": frac_dry_days,

                "mean_APA_dry": mean_dry,
                "var_APA_dry": var_dry,
                "std_dev_dry": std_dry,
                "std_error_dry": stde_dry,
                "q05_APA_dry": float(q05) if np.isfinite(q05) else np.nan,
                "q25_APA_dry": float(q25) if np.isfinite(q25) else np.nan,
                "q50_APA_dry": float(q50) if np.isfinite(q50) else np.nan,
                "q75_APA_dry": float(q75) if np.isfinite(q75) else np.nan,
                "q95_APA_dry": float(q95) if np.isfinite(q95) else np.nan,

                "peak_month": int(peak_m) if np.isfinite(peak_m) else np.nan,
                "peak_month_name": month_name(int(peak_m)) if np.isfinite(peak_m) else np.nan,
                "peak_prob": peak_p,
                "peak_season": season_of_month(int(peak_m)) if np.isfinite(peak_m) else np.nan,

                "n_hydro_years": int(pd.Index(hydro_years).nunique()),
            })

        # per-basin save
        df_basin = pd.DataFrame([r for r in all_rows if r["basin"] == basin])
        df_basin.to_csv(os.path.join(OUT_STATS_DIR, f"{basin}_dry_climatology_extreme_NEPlt002.csv"), index=False)
        print(f"✅ {basin} stats saved")
    
    else:
        FILES_PRESENT = True
        print(f"⚠ Skipping {basin} (dry climatology stats already exist)")

if not FILES_PRESENT:
    # combined tables
    df_stats = pd.DataFrame(all_rows)
    df_month = pd.DataFrame(monthly_rows)

    stats_out = os.path.join(OUT_STATS_DIR, "ALL_BASINS_dry_climatology_extreme_NEPlt002.csv")
    month_out = os.path.join(OUT_STATS_DIR, "ALL_BASINS_monthly_seasonality_extreme_NEPlt002.csv")

    df_stats.to_csv(stats_out, index=False)
    df_month.to_csv(month_out, index=False)

    print("📌 Combined stats:", stats_out)
    print("📌 Monthly seasonality:", month_out)


# =========================
# 2) Load masks + helpers for maps
# =========================
ds = xr.open_dataset(CST_PATH)
if "y" in ds.dims and "Y" not in ds.dims: ds = ds.rename({"y": "Y"})
if "x" in ds.dims and "X" not in ds.dims: ds = ds.rename({"x": "X"})

LAT = ds["LAT"]
LON = ds["LON"]
AIS = ds["AIS"]
RIGNOT = ds["RIGNOT"]
valid = (AIS > 0.5) & np.isfinite(RIGNOT)

LON_wrapped = ((LON + 180) % 360) - 180

df_stats  = pd.read_csv(os.path.join(OUT_STATS_DIR, "ALL_BASINS_dry_climatology_extreme_NEPlt002.csv"))
df_month = pd.read_csv(os.path.join(OUT_STATS_DIR, "ALL_BASINS_monthly_seasonality_extreme_NEPlt002.csv"))

def make_field_from_basin_values(basin_val_dict):
    Z = xr.full_like(RIGNOT, np.nan, dtype="float64")
    for bid, val in basin_val_dict.items():
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        Z = xr.where((RIGNOT == int(bid)) & valid, float(val), Z)
    return Z

def plot_4panel_map(metric_col, title, cbar_label, cmap="viridis", vmin=None, vmax=None, save_file=True, dpi=None, discrete_months=False):
    proj = ccrs.SouthPolarStereo(central_longitude=0)

    fig, axs = plt.subplots(2, 2, figsize=(13, 12), subplot_kw={"projection": proj}, dpi=dpi)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.06, top=0.93, wspace=0.30, hspace=0.18)

    theta = np.linspace(0, 2*np.pi, 361)
    circle = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle_path = mpath.Path(circle * 0.5 + 0.5)

    # prepare limits
    vals = df_stats[metric_col].values
    vals = vals[np.isfinite(vals)]
    if vmin is None: vmin = float(np.nanmin(vals)) if len(vals) else 0.0
    if vmax is None: vmax = float(np.nanmax(vals)) if len(vals) else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    mappable = None

    for i, dur in enumerate(DUR_ORDER):
        ax = axs.flat[i]
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

        sub = df_stats[df_stats["duration"] == dur].copy()
        basin_vals = dict(zip(sub["basin_id"].astype(int), sub[metric_col].astype(float)))

        Z = make_field_from_basin_values(basin_vals)

        mesh = ax.pcolormesh(
            LON_wrapped, LAT, Z.where(valid),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="nearest",
            zorder=4
        )
        mappable = mesh

        ax.contour(
            LON_wrapped, LAT, AIS, levels=[0.5],
            colors="k", linewidths=2.0,
            transform=ccrs.PlateCarree(), zorder=3
        )

        # basin id labels
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
                bbox=dict(facecolor="white", alpha=0.00, boxstyle="round,pad=0.15", linewidth=0.0),
                zorder=7,
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

        ax.set_title(dur, fontsize=12)

    # shared colorbar (right, vertical)
    cax = fig.add_axes([0.95, 0.18, 0.005, 0.64])
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", extend="both")
    cb.set_label(cbar_label, fontsize=11)
    cb.ax.tick_params(labelsize=10)

   # fig.suptitle(title, fontsize=14)
    
    if OLD_TAGS_VERSION:
        out = os.path.join(OUT_FIG_DIR, f"Mean_APA_4panel_OldTAGS_map_{metric_col}.png")
    else:
        out = os.path.join(OUT_FIG_DIR, f"Mean_APA_4panel_map_{metric_col}.png")
        

    if save_file:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print("Saved:", out)
    plt.show()


# =========================
# 3) Maps: mean, variance, seasonality
# =========================

# Mean APA during extreme drought state
plot_4panel_map(
    metric_col="mean_APA_dry",
    title="Mean precipitation accumulation during extreme drought state (NEP < 0.02)",
    cbar_label="Mean APA during dry state (mm w.e.)",
    cmap="BrBG", vmin=0, vmax=1000, save_file=False, dpi=400,
)

#######################################################################
####################   HEAT MAPS   #############
#######################################################################

BASINS_ORDER = [16, 25, 17, 26, 15,                            # AP  
                14, 20, 1, 9, 24, 2, 18, 19,                                               # 1, 24, 2, 9,  14, 20, 18, 19,                   #WA 
                13, 21, 7, 11, 22, 6, 12, 23, 5, 4, 10, 3, 8] 


df_stats = pd.read_csv(os.path.join(OUT_STATS_DIR, "ALL_BASINS_dry_climatology_extreme_NEPlt002.csv"))
df_stats.head()


# =========================
# 4) Heatmaps: basin × duration
# =========================
def plot_heatmap(value_col, title, cbar_label, vmin=None, vmax=None, cmap="viridis"):
    
    pivot = (
        df_stats.pivot(index="basin_id", columns="duration", values=value_col)
        .reindex(index=BASINS_ORDER)
        .reindex(columns=DUR_ORDER)
    )

    fig, ax = plt.subplots(figsize=(4, 9), dpi=300)
    im = ax.imshow(pivot.values, aspect="auto", 
                   vmin=vmin, vmax=vmax, cmap=cmap
                   )

    # Y-ticks for basin IDs
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.values)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.values)

    # ---- Secondary axis for region boundaries + labels ----
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    region_sizes = REGION_SIZES
    region_names = REGION_NAMES
    REGION_COLORS  = {"AP": "#4477AA", "WA": "#EE6677", "EA": "#228833"}
    region_colors = REGION_COLORS

    # boundaries between regions (offset by 0.5 between rows)
    region_bounds = np.cumsum(region_sizes)[:-1] - 0.5

    # centers for text labels
    centers = []
    start = 0
    for size in region_sizes:
        end = start + size
        centers.append((start + end - 1) / 2)  # midpoint of block
        start = end

    # ticks at boundaries only
    ax2.set_yticks(region_bounds)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', direction='inout', length=6, width=1.0, color = "red", left=True, right=True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['left'].set_visible(True)

    # place region labels using data coords
    for name, center, colors in zip(region_names, centers, region_colors.values()):
        ax2.text(
            3.52, center, name,
            transform=ax2.transData,
            va='center', ha='left', fontsize=10, #fontweight='bold',
            color = colors, alpha = 0.9, fontweight="bold",
        )

    ax.set_xlabel("Duration")
    ax.set_ylabel("Basin ID")
   
    out = os.path.join(OUT_FIG_DIR, f"HEATMAP_{value_col}.png") 
    #out = os.path.join(OUT_FIG_DIR, f"03Results_HEATMAP_{value_col}.png") 
    plt.savefig(out, dpi=400, bbox_inches="tight")

    plt.show()


plot_heatmap(
    "mean_APA_dry",
    "Mean APA during extreme drought state (NEP < 0.02)",
    "Mean APA during dry state (mm w.e.)",
    vmin = 0, vmax = 1000, cmap = "BrBG",
)

