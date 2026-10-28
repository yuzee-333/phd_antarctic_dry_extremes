# Antarctic dry extremes: Multi-temporal Rarity Curves

Code for the figures in:

> Nworgu, U. C., Favier, V., Vance, T., Fiddes, S., Blanchet, J., Pohl, B., and Amory, C.: Temporal dynamics and evolution of Antarctic dry extremes using Multi-temporal Rarity Curves, submitted to The Cryosphere.

The scripts apply the Multi-temporal Rarity Curve (MRC) framework (Blanchet et al., 2024) to daily total precipitation from the regional climate model MAR v3.12 forced by ERA5 (1979–2023), aggregated over the Antarctic drainage basins of Rignot et al. They classify each basin-year into a dry-to-wet typology and describe the timing, persistence and spatial organisation of dry extremes.

## Repository contents

| Script | Figure |
|---|---|
| `Fig1.py` | Study area: drainage basins and basin labels |
| `Fig2.py` | Illustration of the deseasonalised kernel density and CDF used to derive non-exceedance probabilities |
| `Fig3.py` | Example Multi-temporal Rarity Curve for one basin and year |
| `Fig4.py` | Probability of at least one extreme dry event per hydrological year, and accumulated precipitation during dry states |
| `Fig5.py` | k-means clustering of rarity curves and EOF projection |
| `Fig6.py` | Rarity-curve panels by typology |
| `Fig7.py` | Basin statistics by typology |
| `Fig8.py` | Basin-by-year typology chronology with multi-year events |
| `Fig9.py` | Spatial typology maps by multi-year block |
| `config.py` | Shared paths and parameters imported by all figure scripts |

## Requirements

Python 3.10 or later with the packages in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ant-dry-extremes
```

## Input data

- **MAR v3.12 daily total precipitation**, ERA5-forced, 1979–2023 (`daily-TP-MARv3.12_ERA5-1979-2023.nc`), together with the MAR grid constants file used for the basin masks. See the Data availability section of the paper for access.
- **Antarctic drainage basin boundaries** (Rignot et al.), rasterised onto the MAR grid.
- **Intermediate tables** produced by the analysis pipeline: non-exceedance probability (NEP) and accumulated precipitation anomaly (APA) tables per basin, cluster memberships and centroids, and bootstrap inclusion frequencies.

## Usage

1. Edit the directory paths in `config.py` to point to your local copies of the input data and to an output folder.
2. Run each figure script from the repository root:

```bash
python Fig1.py
```

Each script writes its figure as a PNG to the output folder defined in `config.py`.

## Citation

If you use this code, please cite the archived release on Zenodo (see `CITATION.cff`) and the accompanying paper.

## Licence

MIT. See `LICENSE`.

## References

Blanchet, J., Ainési, B., Rome, S., and Creutin, J.-D.: Multi-temporal drought rarity curves — A yearly classification of meteorological drought severity in France, Adv. Water Resour., 193, 104829, 2024.
