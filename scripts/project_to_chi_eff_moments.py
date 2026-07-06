"""Monte-Carlo project the GaussianComponentSpins hyperposterior to the
(mu_chi_eff, sigma_chi_eff) of the implied population-predictive chi_eff
marginal.

For each hyperposterior draw lambda = (alpha_1, ..., lam_0, ..., mu_chi,
sigma_chi, mu_spin, sigma_spin, xi_spin, amax), we:
  1. Draw N_events synthetic events (m1, q) from the population mass model
     with that draw's mass hyperparameters.
  2. Draw a_1, a_2 from TruncNorm(mu_chi, sigma_chi, 0, amax).
  3. Draw cos_tilt_1, cos_tilt_2 from the tilt mixture:
       Bernoulli(xi_spin) -> TruncNorm(mu_spin, sigma_spin, -1, 1)
       else              -> Uniform(-1, 1)
  4. Compute chi_eff = (m1*a_1*cos_tilt_1 + m2*a_2*cos_tilt_2) / (m1 + m2).
  5. Return (mean(chi_eff), std(chi_eff)) for that draw.

The output is an (n_lambda, 2) array used by reweighting/train_gwtc3_flow.py
to assemble the 18-dim training matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import truncnorm as scipy_truncnorm


MASS_PARAMS = [
    "alpha_1", "alpha_2", "beta", "break_mass",
    "lam_0", "lam_1", "mpp_1", "mpp_2", "sigpp_1", "sigpp_2",
    "mlow_1", "mlow_2", "delta_m_1", "delta_m_2", "mmax",
]
SPIN_PARAMS = ["mu_chi", "sigma_chi", "amax", "mu_spin", "sigma_spin", "xi_spin"]


def _sample_truncnorm(mu, sigma, low, high, size, rng):
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    return scipy_truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)


def _sample_mass_from_lvk_model(row, n_events, rng):
    """Sample (m1, m2) from BrokenPowerLawTwoPeaks given a hyperposterior row.

    Grid-samples (m1, q) directly from the model PDF on a fixed
    (m1_grid x q_grid) mesh via a single np.random.choice over flattened
    cell probabilities. Much faster than rejection sampling and entirely
    vectorized -- chi_eff only needs the mass ratio q, so a coarse grid is fine.
    """
    import gwpopulation
    gwpopulation.set_backend("numpy")
    from models.mass import TwoPeakBrokenPowerLawSmoothedMassDistribution

    model = TwoPeakBrokenPowerLawSmoothedMassDistribution(mmin=2.0, mmax=300.0)
    mass_kw = {p: float(row[p]) for p in MASS_PARAMS}

    m1_grid = np.geomspace(2.0, 200.0, 200)
    q_grid = np.linspace(0.05, 1.0, 100)
    M1, Q = np.meshgrid(m1_grid, q_grid, indexing="ij")
    ds = {"mass_1": M1.ravel(), "mass_ratio": Q.ravel()}
    prob = np.asarray(model(ds, **mass_kw))
    prob = np.where(np.isfinite(prob) & (prob > 0), prob, 0.0)
    if prob.sum() <= 0:
        return None, None
    prob = prob / prob.sum()

    idx = rng.choice(prob.size, size=n_events, p=prob)
    m1 = M1.ravel()[idx]
    q = Q.ravel()[idx]
    m2 = m1 * q
    return m1, m2


def project_to_chi_eff_moments(
    hyperposterior_df,
    n_lambda=None,
    n_events_per_lambda=5000,
    seed=0,
    progress_every=50,
):
    """Project a hyperposterior DataFrame to (mu_chi_eff, sigma_chi_eff) per draw.

    Parameters
    ----------
    hyperposterior_df : pd.DataFrame
        Must contain `MASS_PARAMS + SPIN_PARAMS` columns.
    n_lambda : int or None
        Number of rows to project. Defaults to all.
    n_events_per_lambda : int
        Synthetic events per draw.
    seed : int
        RNG seed.
    progress_every : int
        Print one progress line every this many rows.

    Returns
    -------
    np.ndarray of shape (n_lambda, 2): [mu_chi_eff, sigma_chi_eff] per row.
    """
    rng = np.random.default_rng(seed)
    if n_lambda is None or n_lambda > len(hyperposterior_df):
        n_lambda = len(hyperposterior_df)
    idxs = rng.choice(len(hyperposterior_df), size=n_lambda, replace=False)

    out = np.full((n_lambda, 2), np.nan)
    for i_local, k in enumerate(idxs):
        row = hyperposterior_df.iloc[int(k)]

        m1, m2 = _sample_mass_from_lvk_model(row, n_events_per_lambda, rng)
        if m1 is None:
            continue

        mu_chi = float(row["mu_chi"])
        sigma_chi = max(float(row["sigma_chi"]), 1e-3)
        amax = float(row["amax"])
        a1 = _sample_truncnorm(mu_chi, sigma_chi, 0.0, amax, n_events_per_lambda, rng)
        a2 = _sample_truncnorm(mu_chi, sigma_chi, 0.0, amax, n_events_per_lambda, rng)

        mu_spin = float(row["mu_spin"])
        sigma_spin = max(float(row["sigma_spin"]), 1e-3)
        xi_spin = float(row["xi_spin"])
        is_aligned = rng.uniform(0, 1, n_events_per_lambda) < xi_spin
        cos_t1 = np.where(
            is_aligned,
            _sample_truncnorm(mu_spin, sigma_spin, -1, 1, n_events_per_lambda, rng),
            rng.uniform(-1, 1, n_events_per_lambda),
        )
        is_aligned_2 = rng.uniform(0, 1, n_events_per_lambda) < xi_spin
        cos_t2 = np.where(
            is_aligned_2,
            _sample_truncnorm(mu_spin, sigma_spin, -1, 1, n_events_per_lambda, rng),
            rng.uniform(-1, 1, n_events_per_lambda),
        )

        chi_eff = (m1 * a1 * cos_t1 + m2 * a2 * cos_t2) / (m1 + m2)
        out[i_local, 0] = chi_eff.mean()
        out[i_local, 1] = chi_eff.std()

        if (i_local + 1) % progress_every == 0:
            print(f"  projection {i_local+1}/{n_lambda}: "
                  f"mu={out[i_local,0]:.3f}, sigma={out[i_local,1]:.3f}")
    return out
