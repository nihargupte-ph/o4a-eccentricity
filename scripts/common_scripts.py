import arviz as az
import numpy as np
import pandas as pd
from astropy import cosmology, units
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d


def fill_missing_available_parameters(df):
    """
    Fill in missing parameters using available ones in a gravitational wave
    parameter dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing posterior samples with missing parameters.
    """

    def chirp_mass(m1, m2):
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    def chirp_mass_and_mass_ratio_to_component_masses(mc, q):
        m1 = mc * (1 + q) ** (1 / 5) / q ** (3 / 5)
        m2 = q * m1
        return m1, m2

    # Compute missing mass parameters
    if "mass_1" in df:
        df["mass_2"] = df.get("mass_2", df["mass_1"] * df.get("mass_ratio", np.nan))
        df["mass_ratio"] = df.get("mass_ratio", df["mass_2"] / df["mass_1"])
        if "chirp_mass" not in df and "mass_2" in df:
            df["chirp_mass"] = chirp_mass(df["mass_1"], df["mass_2"])
    elif "chirp_mass" in df and "mass_ratio" in df:
        df["mass_1"], df["mass_2"] = chirp_mass_and_mass_ratio_to_component_masses(
            df["chirp_mass"], df["mass_ratio"]
        )

    if "mass_1" in df and "mass_2" in df:
        df["total_mass"] = df.get("total_mass", df["mass_1"] + df["mass_2"])

    # Fill missing chi values
    if "a_1" in df and "a_2" in df:
        for i in [1, 2]:
            df[f"chi_{i}"] = df.get(f"chi_{i}", df.get(f"a_{i}", np.nan))

    # Compute effective spin parameter
    if "chi_eff" not in df and "chi_1" in df and "chi_2" in df:
        df["chi_eff"] = (
            df["chi_1"] * df["mass_1"] + df["chi_2"] * df["mass_2"]
        ) / df["total_mass"]

    # Compute luminosity distance from redshift or vice versa
    luminosity_distances = np.linspace(1, 20000, 1000)
    redshifts = np.array(
        [
            cosmology.z_at_value(Planck18.luminosity_distance, dl * units.Mpc)
            for dl in luminosity_distances
        ]
    )
    z_to_dl = interp1d(redshifts, luminosity_distances, fill_value="extrapolate")
    dl_to_z = interp1d(luminosity_distances, redshifts, fill_value="extrapolate")

    if "redshift" in df and "luminosity_distance" not in df:
        df["luminosity_distance"] = z_to_dl(df["redshift"])
    if "luminosity_distance" in df and "redshift" not in df:
        df["redshift"] = dl_to_z(df["luminosity_distance"])

    # Compute missing source-frame masses
    for mass_type in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
        source_col = f"{mass_type}_src"
        if mass_type in df and source_col not in df and "redshift" in df:
            df[source_col] = df[mass_type] / (1 + df["redshift"])

    # Convert log10 eccentricity to eccentricity
    if "log10_eccentricity" in df:
        df["eccentricity"] = 10 ** df["log10_eccentricity"]

    return df


def get_hdi(samples, level=0.90, multi_modal=True):
    """
    Get the highest density interval for the samples.

    Uses ArviZ implementation:
    https://github.com/arviz-devs/arviz/blob/10f0c5f23a2043b16713969a18d44ae3c589fee8/arviz/stats/stats.py#L449
    """
    if isinstance(samples, pd.Series):
        samples = samples.values
    return az.stats.hdi(samples, hdi_prob=level, multimodal=multi_modal)
