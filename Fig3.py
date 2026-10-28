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
#print(OUT_D)
DATE_PREFIX = date.today().strftime("%Y%m%d")

PLOT_DOUBLE = True
# ---------------------
# 0) Paths
# ---------------------

OUT_DIR = "/Users/ucnworgu/OneDrive - University of Tasmania/mywork/PhD.things/project.main/results/multi.temporal.drought/masks/"
os.makedirs(OUT_DIR, exist_ok=True)

#MAR_DATA_DIR = "/home/nworguu/DATA/auf_data/MAR.data/"
CST_PATH      = os.path.join(MAR_DATA_DIR, "MARcst-AN35km-176x148.cdf2")

OUT_D     = os.path.join(ROOT_DIR, "PhD.things/manuscript_prep/MANU_v5_update/")
print("OUT_D: ", OUT_D) 

APA_TRIM_DIR      = os.path.join(ROOT_DIR, "PhD.things/project.main/results/multi.temporal.drought/rignot.individual.mask.results/3years/yearly_tables_12regions/")


# -----------------------------
# 1) Paths & read NEP matrix
# -----------------------------
ibasin = "05"
st_yr = 2022
text_tag = "(b)"
end_yr = st_yr+1

NEP_PATH = ( ROOT_DIR +
    "PhD.things/project.main/"
    "results/multi.temporal.drought/rignot.individual.mask.results/"
    f"3years/yearly_tables_12regions/NEP_basin_{ibasin}_rowsWeeks_colsDates.csv"
)

# rows = accumulation windows (weeks), columns = dates
df_nep = pd.read_csv(NEP_PATH, index_col=0)

# ensure weeks are integers and sorted
df_nep.index = df_nep.index.astype(int)
df_nep = df_nep.sort_index()
weeks_vec = df_nep.index.values

# parse date columns (stored as "dd/mm/YYYY")
cols_dt = pd.to_datetime(df_nep.columns, format="%d/%m/%Y")

# -----------------------------
# 2) Select hydrological year 2022 (Apr-2022 → Mar-2023)
# -----------------------------
start_year = pd.Timestamp(f"{st_yr}-04-01")
end_year   = pd.Timestamp(f"{end_yr}-03-31")

mask_year = (cols_dt >= start_year) & (cols_dt <= end_year)

df_sel = df_nep.loc[:, mask_year].copy()
nep_all = df_sel.to_numpy(dtype=float)       # shape: (nWeeks, nDays_in_hydro_year)

# lower envelope (rarity curve): min NEP across all days, for each accumulation window
nep_min = np.nanmin(nep_all, axis=1)

# -----------------------------
# 3) Drought guide lines & SPI axis
# -----------------------------
GUIDE_LINES  = [0.5, 0.16, 0.07, 0.02]
GUIDE_COLORS = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
GUIDE_LABELS = [
    "mild (NEP≈0.5, SPI≈-1)",
    "moderate (NEP≈0.16, SPI≈-1.5)",
    "severe (NEP≈0.07, SPI≈-2)",
    "extreme (NEP≈0.02, SPI≈-2.33)",
]

SPI_TICKS = [2.0, 0.0, -1.0, -1.5, -2.0]
NEP_TICKS = [norm.cdf(v) for v in SPI_TICKS]


if not PLOT_DOUBLE:
    # -----------------------------
    # 4) Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    def set_ticklabel_family(ax, family="cambria"):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(family)

    # all daily NEP curves in grey
    for j in range(nep_all.shape[1]):
        ax.plot(weeks_vec, nep_all[:, j],
                color="lightgrey", lw=0.7, alpha=0.7, zorder=1)

    # lower envelope (rarity curve) in red
    ax.plot(weeks_vec, nep_min,
            color="red", lw=1.8, zorder=3, label="Rarity curve (min NEP)")

    # guide lines in NEP space
    for gl, gc, lab in zip(GUIDE_LINES, GUIDE_COLORS, GUIDE_LABELS):
        ax.axhline(gl, color=gc, lw=1.0, ls="--", alpha=0.9, zorder=3)

    ax.set_xlim(weeks_vec.min(), weeks_vec.max())
    ax.set_xticks(np.arange(10, 151, 20))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Accumulation period (antecedent weeks)", fontsize=15, family="cambria")
    ax.set_ylabel("NEP", fontsize=15, family="cambria")
    ax.grid(True, ls="--", alpha=0.3)

    # right-hand y-axis for SPI
    ax2 = ax.twinx()
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks(NEP_TICKS)
    ax2.set_yticklabels([f"{v:.1f}" for v in SPI_TICKS])
    ax2.set_ylabel("SPI", fontsize=15, family="cambria")

    #ax.set_title(f"Basin {ibasin} – rarity curve for hydro-year {st_yr}\n"
    #             f"(April {st_yr} – March {end_yr})")

    # ---------- Panel tags A and B ----------
    ax.text(
        0.02, 0.98, text_tag,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14, fontweight="bold", family="cambria"
    )

    FIGSAVE = os.path.join(OUT_D, f"FIG_3_rarity_curve_{ibasin}_{st_yr}_{DATE_PREFIX}.png")

    plt.tight_layout()
    plt.savefig(FIGSAVE, dpi=300, bbox_inches="tight")
    plt.show()



if PLOT_DOUBLE:
    # ============================================================
    # 5) Side-by-side panels - two predefined hydro-years, same basin
    # ============================================================

    PANELS = [
        {"ibasin": "05", "st_yr": 2020, "text_tag": "(a)"},
        {"ibasin": "05", "st_yr": 2022, "text_tag": "(b)"},
    ]


    def load_rarity_curve(ibasin, st_yr):
        """Load NEP table for one basin and return the rarity curve
        (lower envelope) for the hydro-year starting 1 April st_yr."""
        end_yr = st_yr + 1

        nep_path = (
            ROOT_DIR +
            "PhD.things/project.main/"
            "results/multi.temporal.drought/rignot.individual.mask.results/"
            f"3years/yearly_tables_12regions/NEP_basin_{ibasin}_rowsWeeks_colsDates.csv"
        )

        df_nep = pd.read_csv(nep_path, index_col=0)
        df_nep.index = df_nep.index.astype(int)
        df_nep = df_nep.sort_index()
        weeks = df_nep.index.values

        cols_dt = pd.to_datetime(df_nep.columns, format="%d/%m/%Y")
        start_year = pd.Timestamp(f"{st_yr}-04-01")
        end_year   = pd.Timestamp(f"{end_yr}-03-31")
        mask_year  = (cols_dt >= start_year) & (cols_dt <= end_year)

        nep_block = df_nep.loc[:, mask_year].to_numpy(dtype=float)
        nep_lower = np.nanmin(nep_block, axis=1)

        return weeks, nep_block, nep_lower


    def draw_rarity_panel(ax, weeks, nep_block, nep_lower, text_tag):
        """Draw one rarity-curve panel (grey daily curves + red lower
        envelope + McKee guide lines + panel tag) onto `ax`."""
        for j in range(nep_block.shape[1]):
            ax.plot(weeks, nep_block[:, j],
                    color="lightgrey", lw=0.7, alpha=0.7, zorder=1)

        ax.plot(weeks, nep_lower,
                color="red", lw=1.8, zorder=3, label="Rarity curve (min NEP)")

        for gl, gc in zip(GUIDE_LINES, GUIDE_COLORS):
            ax.axhline(gl, color=gc, lw=1.0, ls="--", alpha=0.9, zorder=3)

        ax.set_xlim(weeks.min(), weeks.max())
        ax.set_xticks(np.arange(10, 151, 20))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Accumulation period (antecedent weeks)", fontsize=15, family="cambria")
        ax.grid(True, ls="--", alpha=0.3)

        ax.text(
            0.02, 0.98, text_tag,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=14, fontweight="bold", family="cambria"
        )

        return ax


    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, panel in zip(axes2, PANELS):
        weeks_p, nep_block_p, nep_lower_p = load_rarity_curve(panel["ibasin"], panel["st_yr"])
        draw_rarity_panel(ax, weeks_p, nep_block_p, nep_lower_p, panel["text_tag"])

    axes2[0].set_ylabel("NEP", fontsize=15, family="cambria")

    # right-hand SPI axis, only on the rightmost panel
    ax2_right = axes2[-1].twinx()
    ax2_right.set_ylim(0.0, 1.0)
    ax2_right.set_yticks(NEP_TICKS)
    ax2_right.set_yticklabels([f"{v:.1f}" for v in SPI_TICKS], family="cambria")
    ax2_right.set_ylabel("SPI", fontsize=15, family="cambria")

    panel_tags = "_".join(f"{p['ibasin']}-{p['st_yr']}" for p in PANELS)
    OUT_D = "/Users/ucnworgu/OneDrive - University of Tasmania/mywork/PhD.things/manuscript_prep/MANU_v5_update/V6_figs/"
    FIGSAVE_PANELS = os.path.join(OUT_D, f"Fig3_rarity_curve_panels_{panel_tags}_{DATE_PREFIX}.png")

    plt.tight_layout()
    plt.savefig(FIGSAVE_PANELS, dpi=300, bbox_inches="tight")
    plt.show()
