import numpy as np
import matplotlib.cm as cm

LINUX_PATH = False

if LINUX_PATH:
    ROOT_DIR     = "/home/nworguu/mywork/AUFRANDE.work/"
    MAR_DATA_DIR = "/home/nworguu/DATA/auf_data/MAR.data/"
    MASKS_PATH = "/home/nworguu/mywork/AUFRANDE.work/PhD.things/project.main/results/multi.temporal.drought/masks/antarctica_rignot_individual_basin_masks.nc"

    
else:
    ROOT_DIR      = "/Users/ucnworgu/OneDrive - University of Tasmania/mywork/"
    MAR_DATA_DIR  = "/Users/ucnworgu/OneDrive - University of Tasmania/DATA/auf_data/MAR.data/"
    MASKS_PATH    = "./antarctica_rignot_individual_basin_masks.nc"


DATE_FMT_SAVE = "%d/%m/%Y"

# Accumulation windows in WEEKS (Blanchet use a continuum; start with a practical subset and expand)
ACC_WEEKS = np.arange(4, 157)  # 4..156 weeks (~1 month to 3 years)

REGION_FILES = {}
bIDs = np.arange(1, 27)
print(bIDs)

for i in bIDs:
    key = f"basin_{i:02d}" #zero-padded
    value = f"ALL_tms_basin_{i:02d}.csv"
    REGION_FILES[key] = value


OUT_FOLDER    = ROOT_DIR+"PhD.things/project.main/results/multi.temporal.drought/rignot.individual.mask.results/3years/"

MANUSCRIPT_IMAGES_FOLDER = (
    ROOT_DIR+"PhD.things/manuscript_prep/002.Results_Outline/"
)
print(OUT_FOLDER)
print(MASKS_PATH)

###########################################################################################
############### Plotting parameters for Antarctic basin Map with Basin tags and  annotation lines
###########################################################################################


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
#
#tip_offset
#Value  |   Effect
# 0.0   |  Tip lands exactly at centroid
# 2.0   |  Tip stops 2° short of centroid — shorter line
# 5.0   |  Tip stops 5° short — visibly shorter 
# -2.0  |  Tip overshoots centroid by 2° — longer line

##################################################################################################################################
##################################################################################################################################
###########    TYPOLOGY, REGIONS AND BASIN ORDER MAPPING  ########################################################################

BASINS_ORDER = [16, 25, 17, 26, 15,                            # AP  
                14, 20, 1, 9, 24, 2, 18, 19,                                             
                13, 21, 7, 11, 8, 22, 6, 12, 23, 5, 4, 10, 3]   #EA

CLUSTER_ORDER = ["C1", "C2", "C5", "C8", "C4", "C7", "C3", "C6", "C9", "C10"] 

REGION_NAMES = ["AP", "WA", "EA"]
REGION_SIZES = [5, 8, 13] 
REGION_COLORS  = {"AP": "#4477AA", "WA": "#EE6677", "EA": "#228833"}


REGION_BASIN_MAP = {
    "AP": [16, 25, 17, 26, 15],
    "WA": [14, 20, 1, 9, 24, 2, 18, 19],
    "EA": [13, 21, 7, 11, 8, 22, 6, 12, 23, 5, 4, 10, 3],  
}


##################################################################################################################################
##################################################################################################################################

# TYPOLOGY MAP AND TYPOLOGY PARAMETERS

# ── SINGLE SOURCE OF TRUTH for drought typologies ───────────────────────────
# Edit only this list. Everything below is derived from it automatically.
TYPOLOGIES = [
    {"tag": "Dry",  "name": "Dry",     "label": "T1", "clusters": [1],       "size": 1},
    {"tag": "Drng", "name": "Drying",  "label": "T2", "clusters": [2, 5, 8], "size": 3},
    {"tag": "Med",  "name": "Median",  "label": "T3", "clusters": [4, 7],    "size": 2},
    {"tag": "Wtn",  "name": "Wetting", "label": "T4", "clusters": [3, 6],    "size": 2},
    {"tag": "Wet",  "name": "Wet",     "label": "T5", "clusters": [9, 10],   "size": 2},
]

# ── everything else, derived — do not hand-edit ─────────────────────────────
typology_display = {t["tag"]: t["name"] for t in TYPOLOGIES}
TYPOLOGY_TAGS     = [t["tag"] for t in TYPOLOGIES]
TYPOLOGY_NAMES    = [t["name"] for t in TYPOLOGIES]
TYPOLOGY_SIZES    = [t["size"] for t in TYPOLOGIES]
TYPOLOGY_ORDER    = TYPOLOGY_NAMES.copy()

TYPOLOGY_LABELS = {t["tag"]: t["label"] for t in TYPOLOGIES}

TYPOLOGY_MAP  = {t["tag"]:  t["clusters"] for t in TYPOLOGIES}   # tag  -> clusters
TYPOLOGY_MAP2 = {t["name"]: t["clusters"] for t in TYPOLOGIES}   # name -> clusters

CLUSTER_TYPOLOGY = {
    f"C{c}": t["tag"]
    for t in TYPOLOGIES
    for c in t["clusters"]
}

# ── COLOUR PALETTE ─────────────────────────────────────────────────────────
# BrBG-sampled, with manual overrides for RecTDr (lighter brown)
# and StTDr (near-white midpoint).
#
# Sampling positions:
#   Dry   0.05  →  dark brown
#   Drng     0.27  →  tan-brown
#   Med    0.50  →  near-white (BrBG midpoint)
#   Wtn    0.78  →  medium teal
#   Wet    0.95  →  dark teal/green
 
_brbg = cm.get_cmap("BrBG") #BrBG #RdYlBu
_positions = [0.05, 0.27, 0.50, 0.78, 0.95]
 
TYPOLOGY_COLORS = {
    lab: "#{:02X}{:02X}{:02X}".format(
        int(_brbg(p)[0] * 255),
        int(_brbg(p)[1] * 255),
        int(_brbg(p)[2] * 255),
    )
    for lab, p in zip(TYPOLOGY_LABELS.keys(), _positions)
}
