# DINGO Eccentricity O4a — Data Release

Code to reproduce the figures and tables from the O4a eccentricity paper.

**Data**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19340012.svg)](https://doi.org/10.5281/zenodo.19340012)

## Setup

```bash
uv sync
```

## Data

The data is hosted on Zenodo: https://zenodo.org/records/19340012

Each notebook automatically downloads the required data on first run. You can
also download manually:

```bash
# Option 1: using the built-in download script
uv run python scripts/download_data.py

# Option 2: using zenodo-get
pip install zenodo-get
zenodo_get 19340012 -o zenodo_data
```

Or set the `DINGO_ECC_DATA_ROOT` environment variable to point to wherever you
downloaded the data.

### Expected data layout

```
zenodo_data/
├── summary_stats.h5              # Pre-computed BFs, HDIs, FAR for all 85 events
├── posteriors_ecc.h5             # Full eccentric posteriors (DataFrames) for all events
├── posteriors_qc.h5              # Full quasicircular posteriors for all events
├── posteriors_prec.h5            # Precessing posteriors for all events
├── ecc_pop.pkl                   # Hierarchical inference checkpoint (Fig 3)
├── prior_samples.pkl             # Prior hull data for Fig 4
├── glitch_marginalization/
│   ├── GW190701.h5               # Glitch posteriors + draws
│   ├── GW231114_043211.h5
│   └── GW231223_032836.h5
└── egw_conversions.h5            # e_gw values for 9 events
```

The Zenodo record also includes trained DINGO neural network models
(SEOBNRv5EHM, SEOBNRv5HM, SEOBNRv5PHM). These are not required for
reproducing the plots.

## Reproducing Figures

Run the notebooks in `notebooks/`:

| Notebook | Output |
|----------|--------|
| `figure1_bf_lineplot.ipynb` | Figure 1 — Bayes factor lineplot + histogram |
| `figure2_glitch_marginalization.ipynb` | Figure 2 — Glitch marginalization comparison |
| `figure3_eccentricity_distribution.ipynb` | Figure 3 — Eccentricity distribution + velocity dispersion |
| `figure4_posterior_grid.ipynb` | Figure 4 — 9-event 2D posterior grid (e vs zeta) |
| `tables2_3_event_statistics.ipynb` | Tables 2 and 3 — Event statistics |

All output figures are saved to `figures/`.

## Directory Structure

```
├── pyproject.toml         # Dependencies (use uv sync)
├── README.md
├── notebooks/             # One notebook per figure/table
├── scripts/
│   ├── config.py          # DATA_ROOT + Zenodo record ID
│   ├── download_data.py   # Auto-download from Zenodo
│   ├── common_scripts.py  # Parameter conversion, HDI computation
│   ├── formatting.py      # Matplotlib styling, color palette
│   └── extra.py           # Event list
├── style/
│   └── test.mplstyle      # Publication matplotlib style
└── figures/               # Output directory
```
