import os
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D
import sys
from config import *

RF_PATH = os.path.join(MAR_DATA_DIR, "daily-RF-MARv3.12_ERA5-1979-2023.nc")
SF_PATH = os.path.join(MAR_DATA_DIR, "daily-SF-MARv3.12_ERA5-1979-2023.nc")
TP_PATH = os.path.join(MAR_DATA_DIR, "daily-TP-MARv3.12_ERA5-1979-2023.nc")


# Output directory
OUT_DIR = OUT_FOLDER
print(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

DROP_FEB29 = True
ds_tp = xr.open_dataset(TP_PATH, decode_times=True)

TP = ds_tp["TP"]  # (TIME, Y, X)

if "TIME" in TP.dims:
    TP = TP.rename({"TIME": "time"})

TP = TP.sortby("time")

# Keep the main analysis period (1979-01-01 to 2023-12-31)
TP = TP.sel(time=slice("1979-01-01", "2023-12-31"))

# Optionally drop Feb 29 so all years have 365 days (helps monthly/seasonal grouping consistency)
if DROP_FEB29:
    TP = TP.sel(time=~((TP.time.dt.month == 2) & (TP.time.dt.day == 29)))

print("TP shape:", TP.shape, "| Time span:", str(TP["time"].values[0])[:10], "→", str(TP["time"].values[-1])[:10])

OUT_TMS = "./data/"
os.makedirs(OUT_TMS, exist_ok=True)
print("Output directory for time series:", OUT_TMS)

MAKE_TIMESERIES = False
if MAKE_TIMESERIES:
    time = TP["time"]
    #----------------------------------
    # Define hydrological year (April–March, labelled by the START year)
    #----------------------------------
    hydro_year = xr.where(time.dt.month >= 4, time.dt.year, time.dt.year - 1)

    TP = TP.assign_coords(hydro_year=hydro_year)
    TP = TP.sel(time=slice("1979-04-01", "2023-03-31"))

    print("TP shape:", TP.shape, "| Time span:", str(TP["time"].values[0])[:10], "→", str(TP["time"].values[-1])[:10])

    # Rignot subregions mask
    masks = xr.open_dataset(MASKS_PATH)  # contains: mask(basin,Y,X), weight(basin,Y,X)

    # Get basin IDs from coordinate
    basin_ids = masks["basin"].values
    print(f"Basins found in mask file (Rignot IDs): {basin_ids}")

    # Extract weight array and ensure it matches TP grid
    w_all = masks["weight"].astype("float64")  # (basin, Y, X)

    # Optionally realign coords to TP (defensive; shapes must already match)
    w_all = w_all.assign_coords(Y=TP["Y"], X=TP["X"])

    # ------------------------------
    # COMPUTE DAILY AREA-WEIGHTED MEANS PER BASIN
    # ------------------------------
    tp_regional = {}

    for bid in basin_ids:
        bid_int = int(bid)
        print(f"✓ Computing area-weighted daily precip for basin {bid_int} ...")

        w_b = w_all.sel(basin=bid)                      # (Y, X)
        series = (TP * w_b).sum(("Y", "X")).to_pandas() # 1D time series

        tp_regional[bid_int] = series
        print(f"  → {len(series)} days for basin {bid_int}")

    # ------------------------------
    # SAVE ALL BASIN TIMESERIES
    # ------------------------------
    for bid_int, series in tp_regional.items():
        fpath = os.path.join(OUT_TMS, f"ALL_tms_basin_{bid_int:02d}.csv")
        df = series.to_frame(name="precip_mmWE_per_day")
        df.index.name = "time"
        df.to_csv(fpath)
        print(f"💾 Saved basin {bid_int} → {fpath}")

    # ------------------------------
    # SUMMARY
    # ------------------------------
    if len(tp_regional) > 0:
        dates_all = list(tp_regional.values())[0].index  # assume all same time index
        print("\nDate span:", dates_all[0].date(), "→", dates_all[-1].date(), f"({len(dates_all)} days)")
    print(f"Total basins processed: {len(tp_regional)}")
else:
    print(f"Skipping time series generation. Files already in {OUT_TMS}.")


===========  APA Construction ====================
CSV_DIR = OUT_TMS
DATE_FMT_SAVE = "%d/%m/%Y"

# Accumulation windows in WEEKS (Blanchet use a continuum; start with a practical subset and expand)
ACC_WEEKS = np.arange(4, 157)  # 4..156 weeks (~1 month to 3 years)

APA_DIR, UNTRIM_DIR   = #"refer to timeseries folder in zenodo"

COMPUTE_MAR = False
print(REGION_FILES)
if COMPUTE_MAR:
    
    CSV_DIR = OUT_TMS

    #print(REGION_FILES)
    # ----------------------------------------------------
    # 1) LOAD ALL 26 SUBREGIONAL DAILY SERIES & ALIGN DATES
    # ----------------------------------------------------
    series_dict = {}
    for reg,fname in REGION_FILES.items():
        fpath = os.path.join(CSV_DIR, fname)
        # print(f"Loading {reg} from {fpath} ...")
        df = pd.read_csv(fpath, parse_dates=True, index_col=0)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = pd.read_csv(fpath, parse_dates=["time"]).set_index("time")
            elif "date" in df.columns:
                df = pd.read_csv(fpath, parse_dates=["date"]).set_index("date")
            else:
                raise ValueError(f"{reg}: couldn't find a datetime index.")
        # pick first numeric column
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        s = df[num_cols[0]].copy().sort_index()
        if DROP_FEB29:
            s = s[~((s.index.month == 2) & (s.index.day == 29))]
        series_dict[reg] = s

    # print(f"Loaded {len(series_dict)} regions. Sample:\n{list(series_dict.items())[:1]}")

    # Common date index across all regions
    idx = None
    for s in series_dict.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    for reg in series_dict:
        series_dict[reg] = series_dict[reg].reindex(idx)
    dates_all = series_dict[list(REGION_FILES.keys())[0]].index
    print("Date span:", dates_all.min().date(), "→", dates_all.max().date(), f"({len(dates_all)} days)")

    print(f"Loaded {len(series_dict)} regions. Sample:\n{list(series_dict.items())[:1]}")

    
    # ----------------------------------------------------
    # 2) BUILD APA MATRICES FOR EACH REGION (n_days × n_windows)
    # ----------------------------------------------------
    valid_rows_all = np.ones(len(dates_all), dtype=bool)
    apa_dict = {}
    for reg, s in series_dict.items():
        cols = []
        for j in ACC_WEEKS:
            cols.append(s.rolling(window=7*j, min_periods=7*j).sum().values)
        apa_dict[reg] = np.column_stack(cols)

    # Keep rows (days) where **all regions** have valid sums for **all windows**
    for reg in REGION_FILES:
        A = apa_dict[reg]                        # shape: (n_days, n_windows)
        if A.shape[0] != len(dates_all):
            raise ValueError(f"{reg}: APA rows ({A.shape[0]}) != len(dates_all) ({len(dates_all)}).")
        # day is valid only if ALL windows are finite on that day
        valid_rows_all &= ~np.isnan(A).any(axis=1)

    time_valid = dates_all[valid_rows_all]

    print("APA valid days for all regions:", len(time_valid))


    # ----------------------------------------------------
    # 3) SAVE APA MATRICES (only valid rows) FOR EACH REGION
    # -----------------------------------------------------
    
    for reg, M in apa_dict.items():   # M: numpy array, shape (n_days_used, n_windows)
        nD, nW = M.shape

        # Candidate labels
        idx_full = pd.DatetimeIndex(series_dict[reg].index)  # the series used to build APA for this region

        # Decide which labels match M's rows
        if nD == len(idx_full):
            labels = idx_full
        elif 'time_valid' in locals() and len(time_valid) == nD:
            # APA was trimmed with a common valid mask → use that trimmed date index
            labels = pd.DatetimeIndex(time_valid)
        elif 'valid_rows_all' in locals() and len(valid_rows_all) == len(idx_full) and valid_rows_all.sum() == nD:
            # You trimmed APA rows using valid_rows_all; reuse it to trim the series index
            labels = idx_full[valid_rows_all]
        else:
            raise ValueError(
                f"[{reg}] Cannot match APA rows ({nD}) to any available date index:\n"
                f"- len(series_dict[{reg}].index) = {len(idx_full)}\n"
                f"- len(time_valid) = {len(time_valid) if 'time_valid' in locals() else 'NA'}\n"
                f"- sum(valid_rows_all) = {valid_rows_all.sum() if 'valid_rows_all' in locals() else 'NA'}\n"
                "→ Save right after building APA (untrimmed), or provide the mask used to trim."
            )

        # Build DataFrame with rows=weeks, cols=dates
        col_labels = pd.to_datetime(labels).strftime(DATE_FMT_SAVE)
        df_out = pd.DataFrame(M.T, index=ACC_WEEKS, columns=col_labels)

        out_path = os.path.join(UNTRIM_DIR, f"APA_{reg}_FULL_rowsWeeks_colsDates.csv")
        df_out.to_csv(out_path)
        print(f"💾 Saved APA for {reg}: {out_path} | shape={df_out.shape}")



#==================TRIM APA: REMOVE NANS AND START AT 1ST APRIL=========================================================    
#======================================================================================================================
    
    


    for reg in REGION_FILES.keys():
        f_in = os.path.join(UNTRIM_DIR, f"APA_{reg}_FULL_rowsWeeks_colsDates.csv")
        if not os.path.exists(f_in):
            print(f"⚠️ Missing APA (untrimmed) for {reg}: {f_in}")
            continue

        df = pd.read_csv(f_in, index_col=0)

        # --- Parse date columns ---
        cols_dt = pd.to_datetime(df.columns, format=DATE_FMT_SAVE)
        df.columns = cols_dt

        # --- Step 1: Drop columns with ANY missing value across rows ---
        df = df.dropna(axis=1, how='any')

        # --- Step 2: Determine cutoff start date ---
        T_0 = df.columns.year.min()                          # earliest year remaining
        apr1_T0 = pd.Timestamp(year=T_0, month=4, day=1)

        cols_in_T0 = df.columns[df.columns.year == T_0]
        has_pre_april_in_T0 = (cols_in_T0 <= apr1_T0).any()  # any col on or before Apr 1?

        if has_pre_april_in_T0:
            cutoff = apr1_T0
        else:
            T_1 = T_0 + 1
            cutoff = pd.Timestamp(year=T_1, month=4, day=1)

        # --- Step 3: Trim ---
        df_trim = df.loc[:, df.columns >= cutoff]

        # --- Tidy index ---
        df_trim.index = df_trim.index.astype(int)
        df_trim = df_trim.sort_index()

        # --- Reformat columns back to string for saving ---
        df_trim.columns = df_trim.columns.strftime(DATE_FMT_SAVE)

        f_out = os.path.join(APA_DIR, f"APA_{reg}_TRIM_rowsWeeks_colsDates.csv")
        df_trim.to_csv(f_out)
        print(f"✅ [{reg}] Starts {cutoff.date()} | shape={df_trim.shape}")


====== DESEASONALIZE APA

z_dict = {}
for reg in REGION_FILES.keys():
    f_in = os.path.join(APA_DIR, f"APA_{reg}_TRIM_rowsWeeks_colsDates.csv")
    if not os.path.exists(f_in):
        print(f"⚠️ Missing trimmed APA for {reg}: {f_in}")
        continue

    # Load APA: rows = accumulation weeks, columns = dates (strings)
    df_apa = pd.read_csv(f_in, index_col=0)
    # Ensure week index is integers and sorted
    df_apa.index = df_apa.index.astype(int)
    df_apa = df_apa.sort_index()

    # Parse date columns to DatetimeIndex (these are the labels for time axis)
    cols_dt = pd.to_datetime(df_apa.columns, format=DATE_FMT_SAVE)
    # Transpose → A has shape (n_days × n_windows) to match earlier notation
    A = df_apa.T.values.astype("float64")   # (nD × nW)
    nD, nW = A.shape

    # Month per column (day)
    month_valid = cols_dt.month.to_numpy()

    # Deseasonalize: for each window (column in A), subtract month-wise mean
    Z = A.copy()
    for j in range(nW):
        col = Z[:, j]
        for m in range(1, 13):
            sel = (month_valid == m)
            if np.any(sel):
                mu = np.nanmean(col[sel])
                col[sel] = col[sel] - mu
        Z[:, j] = col
    z_dict[reg] = Z
        

    # Save back with original orientation: rows = weeks, cols = dates
    df_z = pd.DataFrame(Z.T, index=df_apa.index, columns=cols_dt.strftime(DATE_FMT_SAVE))
    f_out = os.path.join(APA_DIR, f"Z_deseasonalized_{reg}_rowsWeeks_colsDates.csv")
    df_z.to_csv(f_out)
    print(f"💾 Saved deseasonalized (Z) for {reg}: {f_out}  | shape={df_z.shape}")


======  COMPUTE NEP 

# ==========================================================
# NEP from deseasonalized (Z) tables saved after trimming 1979
# INPUT : Z_deseasonalized_<reg>_rowsWeeks_colsDates.csv   (rows=weeks, cols=dates)
# OUTPUT: NEP_<reg>_rowsWeeks_colsDates.csv                (rows=weeks, cols=dates)
# ==========================================================

SAVE_TAB_DIR  = APA_DIR  # save NEP next to Z (change if you want)


# ---- config (match your earlier settings) ----
DATE_FMT_SAVE     = "%d/%m/%Y"
KDE_BW            = 0.25
MIN_KDE_SAMPLES   = 200

# If you saved the accumulation-week index separately, load it; otherwise infer from Z files
# weeks_vec = np.loadtxt(os.path.join(OUT_DIR, "accum_windows_weeks.txt"), dtype=int)

for reg in REGION_FILES.keys():
    # 1) Load deseasonalized Z (rows=weeks, cols=dates)
    f_z = os.path.join(APA_DIR, f"Z_deseasonalized_{reg}_rowsWeeks_colsDates.csv")
    if not os.path.exists(f_z):
        print(f"⚠️ Missing Z file for {reg}: {f_z}")
        continue

    df_z = pd.read_csv(f_z, index_col=0)             # index=weeks (as strings), columns=date strings
    # Ensure weeks are ints & sorted
    df_z.index = df_z.index.astype(int)
    df_z = df_z.sort_index()
    weeks_vec = df_z.index.values                     # (nW,)
    # Parse date columns to DatetimeIndex (time axis)
    cols_dt = pd.to_datetime(df_z.columns, format=DATE_FMT_SAVE)
    # 2) Convert to array with shape (n_days × n_windows)
    Z = df_z.T.values.astype("float64")               # transpose → (nD × nW)
    nD, nW = Z.shape

    # 3) KDE/ECDF per window → NEP
    nep = np.full_like(Z, np.nan, dtype="float64")    # (nD × nW)
    for j in range(nW):
        x = Z[:, j]
        x = x[np.isfinite(x)]
        if x.size >= MIN_KDE_SAMPLES:
            # KDE grid over robust quantiles
            lo, hi = np.quantile(x, [0.001, 0.999])
            pad = 0.1 * (hi - lo + 1e-9)
            xs  = np.linspace(lo - pad, hi + pad, 4001)
            kde = KernelDensity(kernel="gaussian", bandwidth=KDE_BW).fit(x[:, None])
            pdf = np.exp(kde.score_samples(xs[:, None]))
            cdf = np.cumsum(pdf); cdf /= cdf[-1]
            # Map all Z[:, j] to CDF via nearest xs index
            vals    = Z[:, j]
            idx_map = np.clip(np.searchsorted(xs, vals), 0, len(xs) - 1)
            nep[:, j] = cdf[idx_map]
        elif x.size > 0:
            # Fallback: empirical CDF
            xs = np.sort(x)
            ranks = np.searchsorted(xs, Z[:, j], side="right")
            nep[:, j] = ranks / xs.size
        # else: leave column as NaN if x.size == 0

    # 4) Save NEP with same layout as input (rows=weeks, cols=dates)
    NEP_df = pd.DataFrame(nep.T, index=weeks_vec, columns=cols_dt.strftime(DATE_FMT_SAVE))
    f_out = os.path.join(SAVE_TAB_DIR, f"NEP_{reg}_rowsWeeks_colsDates.csv")
    NEP_df.to_csv(f_out)
    print(f"💾 Saved NEP for {reg}: {f_out}  | shape={NEP_df.shape}")




































