import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.path as mpath
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats import norm
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date

from sklearn.neighbors import KernelDensity

import sys
from config import *

DATE_PREFIX = date.today().strftime("%Y%m%d")
# ---------------------
# 0) Paths
# ---------------------


MAR_DATA_DIR = #from zenodo see config.py
CST_PATH      = "./MARcst-AN35km-176x148.cdf2"



APA_TRIM_DIR      = #"timeseries and files generated in '01_build_timeseries_APA_NEP.py', files uploaded to zenodo"

OUT_D = "/output path/"
print(OUT_D)


# ==========================================================
# CONFIG (match your pipeline)
# ==========================================================
DATE_FMT_SAVE     = "%d/%m/%Y"
KDE_BW            = 0.25
MIN_KDE_SAMPLES   = 200

# Choose one region and one accumulation period (week index)
REGION   = "basin_08"   # e.g. "basin_05" or "EA" etc, must match your filenames
WEEK_SEL = 52          # e.g. 13-week accumulation (must exist in df_z.index)

# How many example APA values (Z) to annotate on the CDF plot
N_EXAMPLES = 3


EXAMPLE_MODE = "tails_and_center"

#                MAIN PLOT
#==========================================================
# Pick 3 example dates automatically from left-tail / middle / right-tail
# (percentiles used to define those zones)
P_LEFT   = 5     # left tail ~ 5th percentile
P_MID    = 50    # middle ~ median
P_RIGHT  = 95    # right tail ~ 95th percentile

OUT_FIG = os.path.join(OUT_D, f"Fig2_KDE_{DATE_PREFIX}.png")

# ==========================================================
# LOAD Z TABLE (rows=weeks, cols=dates)
# ==========================================================
f_z = os.path.join(APA_TRIM_DIR, f"Z_deseasonalized_{REGION}_rowsWeeks_colsDates.csv")
if not os.path.exists(f_z):
    raise FileNotFoundError(f"Missing Z file: {f_z}")

df_z = pd.read_csv(f_z, index_col=0)
df_z.index = df_z.index.astype(int)
df_z = df_z.sort_index()

if WEEK_SEL not in df_z.index:
    raise ValueError(f"WEEK_SEL={WEEK_SEL} not found. Example weeks: {df_z.index.values[:10]} ...")

# One row: Z values across all dates for that week
z_row = df_z.loc[WEEK_SEL].astype("float64")

# Keep finite only, but preserve dates
z_row = z_row[np.isfinite(z_row.values)]
if z_row.size == 0:
    raise ValueError("No finite Z values found for this region/week.")

# Series values and matching date strings
z_vals = z_row.values.astype(float)
date_strs = z_row.index.astype(str)  # already in DATE_FMT_SAVE in your files
n = z_vals.size

# ==========================================================
# KDE -> PDF and CDF (same logic as your NEP code)
# ==========================================================
use_kde = (n >= MIN_KDE_SAMPLES)

if use_kde:
    lo, hi = np.quantile(z_vals, [0.001, 0.999])
    pad = 0.1 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, 4001)

    kde = KernelDensity(kernel="gaussian", bandwidth=KDE_BW).fit(z_vals[:, None])
    pdf = np.exp(kde.score_samples(xs[:, None]))
    cdf = np.cumsum(pdf); cdf /= cdf[-1]

    def z_to_nep(vals):
        vals = np.asarray(vals)
        idx = np.clip(np.searchsorted(xs, vals), 0, len(xs) - 1)
        return cdf[idx]

else:
    xs = np.sort(z_vals)
    cdf = np.arange(1, xs.size + 1) / xs.size

    def z_to_nep(vals):
        vals = np.asarray(vals)
        ranks = np.searchsorted(xs, vals, side="right")
        return ranks / xs.size

# ==========================================================
# Select 3 example dates: left tail, middle, right tail
# We pick the *actual observed date* whose Z is closest to the target percentile.
# ==========================================================
targets = np.quantile(z_vals, [P_LEFT/100, P_MID/100, P_RIGHT/100])

# indices of z_vals closest to each target (ensure distinct picks if possible)
picked = []
for t in targets:
    # choose closest not yet used
    order = np.argsort(np.abs(z_vals - t))
    for idx in order:
        if idx not in picked:
            picked.append(idx)
            break

picked = np.array(picked, dtype=int)

example_dates = date_strs[picked]
example_z     = z_vals[picked]
example_nep   = z_to_nep(example_z)

# sort by Z so annotations read left->right
ord2 = np.argsort(example_z)
example_dates = example_dates[ord2]
example_z     = example_z[ord2]
example_nep   = example_nep[ord2]

# Labels for the three picks
zone_labels = ["ex1", "ex2", "ex3"]

# ==========================================================
# PLOT: Histogram+KDE and CDF with NEP mapping
# ==========================================================
fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, dpi=300)

# ---------- Panel A: histogram + KDE (PDF) ----------
bins = 40
ax_pdf.hist(z_vals, bins=bins, density=True, alpha=0.35, edgecolor="none")
ax_pdf.set_xlabel("Deseasonalized APA", fontsize=12, family="cambria")
ax_pdf.set_ylabel("Density", fontsize=12, family="cambria")

if use_kde:
    ax_pdf.plot(xs, pdf, lw=2.0)
    ax_pdf.set_title(f"{REGION} | {WEEK_SEL}-week window\nAPA distribution and KDE", fontsize=12, family="cambria")
    ax_pdf.text(
        0.02, 0.95, f"KDE bandwidth = {KDE_BW}\nN = {n}",
        transform=ax_pdf.transAxes, ha="left", va="top",
        fontsize=12, color="white", bbox=dict(facecolor="white", alpha=0.1, edgecolor="none")
    )
else:
    ax_pdf.set_title(f"{REGION} | {WEEK_SEL}-week window\nAPA distribution and ECDF fallback", fontsize=12, family="cambria")
    ax_pdf.text(
        0.02, 0.95, f"ECDF fallback (N < {MIN_KDE_SAMPLES})\nN = {n}",
        transform=ax_pdf.transAxes, ha="left", va="top",
        fontsize=12, family="cambria", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

# Mark the 3 selected example Z values on the histogram/PDF
for z0, lab, d0 in zip(example_z, zone_labels, example_dates):
    ax_pdf.axvline(z0, lw=1.8, linestyle="--", alpha=0.9)
# small text box listing them
ax_pdf.text(
    0.98, 0.92,
    "\n".join([f"{lab}: APA={z0:.2f}" for lab, z0 in zip(zone_labels, example_z)]),
    transform=ax_pdf.transAxes, ha="right", va="top",
    fontsize=12, family="cambria", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

# ---------- Panel B: CDF + NEP mapping ----------
ax_cdf.plot(xs, cdf, lw=2.2)
ax_cdf.set_xlabel("Deseasonalized APA", fontsize=12, family="cambria")
ax_cdf.set_ylabel("NEP (0–1)", fontsize=12, family="cambria")
ax_cdf.set_title("CDF and NEP mapping: NEP = CDF(APA)", fontsize=12, family="cambria")

# Plot the 3 example points and show how NEP is read off
for z0, nep0, d0, lab in zip(example_z, example_nep, example_dates, zone_labels):
    ax_cdf.plot([z0], [nep0], marker="o", markersize=8)
    ax_cdf.vlines(z0, 0, nep0, linestyles="--", linewidth=1.6, alpha=0.9)
    ax_cdf.hlines(nep0, xs.min(), z0, linestyles="--", linewidth=1.6, alpha=0.9)

    ax_cdf.annotate(
        f"{d0}\nAPA={z0:.2f} \n→ NEP={nep0:.2f}",
        xy=(z0, nep0),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=12, family="cambria",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

# Optional drought guide lines
GUIDE_LINES  = [0.5, 0.16, 0.07, 0.02]
GUIDE_COLORS = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
GUIDE_LABELS = ["mild", "moderate", "severe", "extreme"]


# ---------- Panel tags A and B ----------
ax_pdf.text(0.02, 0.98, "(a)", transform=ax_pdf.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold", family="cambria")
ax_cdf.text(0.02, 0.98, "(b)", transform=ax_cdf.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold", family="cambria")

#fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print(f"✅ Saved illustration: {OUT_FIG}")
plt.show()



