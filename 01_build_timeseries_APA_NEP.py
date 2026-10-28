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

