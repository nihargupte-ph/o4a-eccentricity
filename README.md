# DINGO Eccentricity O4a — Figure & Data Release

[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19340011.svg)](https://doi.org/10.5281/zenodo.19340011)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21221948.svg)](https://doi.org/10.5281/zenodo.21221948)

Code + data to reproduce **every** figure, table, and number in the O4a
eccentricity paper from the **shipped data products**. The data are archived
on Zenodo ([doi:10.5281/zenodo.19340011](https://doi.org/10.5281/zenodo.19340011));
this code is archived at [doi:10.5281/zenodo.21221948](https://doi.org/10.5281/zenodo.21221948).

## Setup

```bash
uv sync          # env for the figure notebooks
```

`uv sync` installs everything, including `dingo` (the SEOBNRv5EHM branch,
[dingo-gw/dingo#332](https://github.com/dingo-gw/dingo/pull/332)), `lalsuite`, and
`pyseobnr` — pulled in via the `dingo-gw[pyseobnr]` dependency.

## Data

Hosted on Zenodo: [doi:10.5281/zenodo.19340011](https://doi.org/10.5281/zenodo.19340011)
(always resolves to the latest version; the record id is set in
`scripts/config.py`, override with `DINGO_ECC_ZENODO_RECORD_ID`).

### Downloading

`scripts/download_data.py` fetches the release into the foldered layout below.
Run it from `scripts/` (it imports `config.py`):

```bash
cd scripts
python download_data.py                       # -> ../zenodo_data/
python download_data.py --include-networks    # also pull the trained DINGO nets (not needed for figures)
```

It downloads into `zenodo_data/` one level above `code/`, which is where the
notebooks look for it (override the location with `DINGO_ECC_DATA_ROOT`). The
script skips files that already exist, so it is safe to re-run. Set
`DINGO_ECC_ZENODO_RECORD_ID` to pull from a specific Zenodo record.

### Layout

```
zenodo_data/
├── event_data/                              # per-event PE products
│   ├── posteriors_eccentric.h5              # eccentric per-event posteriors (uniform eccentricity prior)
│   ├── posteriors_quasicircular.h5          # quasicircular posteriors
│   ├── posteriors_precessing.h5             # precessing posteriors (for a subset of events)
│   ├── summary_statistics.h5                # per-event BFs, HDIs, FAR (84 events; for GW231114_043211 the
│   │                                        #   BF/eccentricity are glitch-marginalized, see glitch/)
│   ├── egw_conversions.h5                   # e_gw for 9 events
│   ├── e_zeta_prior_hull.h5                 # e–zeta prior hull vertices per total-mass bin
│   ├── posteriors_log_uniform_eccentric.h5  # slimmed per-event posteriors, log-uniform-e reweighted
│   └── posteriors_population_reweighted.h5  # per-event posteriors reweighted by the population-informed
│                                            #   posterior (marginalized over the sigma hyperposterior)
├── glitch/                                  # 3-event glitch-marginalization posteriors
│   ├── GW190701.h5                          # keys: glitch_marginalized, non_glitch_marginalized,
│   │                                        #   HV_only; group glitch_draws/{0..N} (per-realization e)
│   ├── GW231114_043211.h5                    # keys: glitch_marginalized, non_glitch_marginalized;
│   │                                        #   group glitch_draws/{0..N}
│   └── GW231223_032836.h5                    # keys: glitch_marginalized, non_glitch_marginalized;
│                                            #   group glitch_draws/{0..N}
├── selection_function/
│   ├── injection_p_draw.h5                  # p_draw dataframe: draw priors, SNRs, p_det
│   │                                        #   
│   ├── fixed_injection_ecc_sweep.h5         # fixed injection (Mc=35, q=1, chi=0, d_L=2 Gpc),
│   │                                        #   200 e x 100 zeta grid: SNRs + p_det vs e_gw
│   └── survival_function.h5                 # appendix survival MF SNRs: 'eccentric' + 'quasicircular',
│                                            #   each {x, mf_snrs (N x 60), eccentricity}  (60 noise/sample)
├── hierarchical_inference/
│   ├── capture_ecc_table.h5                 # p(log10 e | sigma, M, mu) capture table (v9)
│   ├── gwtc4_hyperposterior.h5              # external LVK mass/spin/z fit (reference / comparison)
│   ├── sigma_posterior.h5                   # hierarchical sigma run result
│   └── branching_fraction_posterior.h5      # GC/NSC branching fraction B (beta=1-B), uniform + rate-informed priors
└── networks/                                # trained DINGO neural nets (not needed for the figures)
```

`injection_p_draw.h5` is the final injection dataframe — the only
selection-function data the inference (`alpha(lambda)` / `VT`) and the
`p_det`-vs-`e_gw` figure need. The upstream generation/search pipeline (waveform
draw, whitening, GPU/CPU template search) and the 267 GB whitened-waveform banks
are **not** released; `p_det` is shipped directly in the dataframe.

## Reproducing the paper

| Output | Notebook | Inputs |
|---|---|---|
| Bayes-factor lineplot + histogram | `notebooks/bayes_factor_lineplot.ipynb` | event_data/summary_statistics |
| Glitch-marginalization comparison | `notebooks/glitch_marginalization.ipynb` | glitch/ |
| Astrophysical eccentricity distribution | `notebooks/eccentricity_distribution.ipynb` | Monte Carlo over the capture model; final cell cross-checks against hierarchical_inference/capture_ecc_table |
| Eccentricity PPD (global-sigma) + typical GC/NSC + per-event log-uniform posteriors | `notebooks/ppd_eccentricity.ipynb` | hierarchical_inference/{capture_ecc_table, sigma_posterior}, event_data/posteriors_log_uniform_eccentric |
| Selection-corrected `sigma` posterior + GC/NSC branching fraction `beta` | `notebooks/velocity_dispersion_posterior.ipynb` | hierarchical_inference/{sigma_posterior, branching_fraction_posterior} |
| e–zeta posterior grid (9 events) | `notebooks/posterior_grid_e_zeta.ipynb` | event_data/{e_zeta_prior_hull, posteriors_eccentric} |
| Event-statistics tables | `notebooks/event_statistics_tables.ipynb` | event_data/summary_statistics |
| SNR / p_det vs e_gw 3-panel (Fig selection_function) | `notebooks/pdet_vs_egw.ipynb` | selection_function/fixed_injection_ecc_sweep |
| Empirical vs analytic survival function (ecc + QC) | `notebooks/survival_function.ipynb` | selection_function/survival_function |
| Population vs GWTC-4: hyperposterior + PPD (density & rate) | `notebooks/gwtc4_population.ipynb` | hierarchical_inference/{gwtc4_hyperposterior, sigma_posterior} |

Output figures land in `figures/`.

Beyond the figure notebooks, `notebooks/analyze_event_with_dingo.ipynb` shows how
to load a trained network from `networks/` and re-analyze an event (requires
`download_data.py --include-networks`), and
`event_data/posteriors_population_reweighted.h5` ships the per-event posteriors
reweighted by the population-informed posterior for downstream re-analysis (no
notebook consumes it).

## Selection-function data

The released selection product is the **p_draw dataframe**
(`zenodo_data/selection_function/injection_p_draw.h5`) and the appendix
survival-function SNRs (`survival_function.h5`), shipped directly. The upstream
generation pipeline (waveform draw, whitening, GPU/CPU template search, and the
matched-filter survival-function computation) lives in the analysis tree and is
**not** part of this figure release; the notebooks above reproduce the figures
from the shipped HDF5 alone.
