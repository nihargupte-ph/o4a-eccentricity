"""GWTC-4 BrokenPowerLawTwoPeaks gwpopulation mass model.

The only piece of the hierarchical inference retained in this data release: the
figure notebooks (`gwtc4_population.ipynb`) and `scripts/project_to_chi_eff_moments.py`
evaluate this model on the shipped GWTC-4 hyperposterior to draw the population
PPD / density / rate curves. The sampler that PRODUCED the posteriors is not part
of the release (it lives in the local analysis tree, not on Zenodo).

The module-level `backend.set_backend(...)` call is omitted so the importer's
backend choice applies; the notebooks run it under `gwpopulation.set_backend("numpy")`.
"""
import inspect
from functools import partial

from gwpopulation.utils import xp, to_numpy
from gwpopulation.models.mass import (
    double_power_law_primary_mass,
    truncnorm,
    BaseSmoothedMassDistribution,
)


def four_component_double_power_law_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_mass, lam_0, lam_1, lam_2,
    mpp_1, sigpp_1, mpp_2, sigpp_2, mpp_3, sigpp_3, gaussian_mass_maximum=100,
):
    lam_3 = 1 - lam_2 - lam_1 - lam_0
    break_fraction = (break_mass - mmin) / (mmax - mmin)
    p_pow = double_power_law_primary_mass(
        mass, alpha_1=alpha_1, alpha_2=alpha_2, mmin=mmin, mmax=mmax,
        break_fraction=break_fraction,
    )
    p_norm1 = truncnorm(mass, mu=mpp_1, sigma=sigpp_1, high=gaussian_mass_maximum, low=mmin)
    p_norm2 = truncnorm(mass, mu=mpp_2, sigma=sigpp_2, high=gaussian_mass_maximum, low=mmin)
    p_norm3 = truncnorm(mass, mu=mpp_3, sigma=sigpp_3, high=gaussian_mass_maximum, low=mmin)
    return lam_0 * p_pow + lam_1 * p_norm1 + lam_2 * p_norm2 + lam_3 * p_norm3


def three_component_double_power_law_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_mass, lam_0, lam_1,
    mpp_1, sigpp_1, mpp_2, sigpp_2, gaussian_mass_maximum=100,
):
    lam_2 = 1 - lam_1 - lam_0
    return four_component_double_power_law_primary_mass(
        mass, alpha_1=alpha_1, alpha_2=alpha_2, mmin=mmin, mmax=mmax,
        break_mass=break_mass, lam_0=lam_0, lam_1=lam_1, lam_2=lam_2,
        mpp_1=mpp_1, sigpp_1=sigpp_1, mpp_2=mpp_2, sigpp_2=sigpp_2,
        mpp_3=0, sigpp_3=1, gaussian_mass_maximum=gaussian_mass_maximum,
    )


def two_component_double_power_law_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_mass, lam_0,
    mpp_1, sigpp_1, gaussian_mass_maximum=100,
):
    lam_1 = 1 - lam_0
    return four_component_double_power_law_primary_mass(
        mass, alpha_1=alpha_1, alpha_2=alpha_2, mmin=mmin, mmax=mmax,
        break_mass=break_mass, lam_0=lam_0, lam_1=lam_1, lam_2=0,
        mpp_1=mpp_1, sigpp_1=sigpp_1, mpp_2=0, sigpp_2=1, mpp_3=0, sigpp_3=1,
        gaussian_mass_maximum=gaussian_mass_maximum,
    )


class BrokenPowerLawPlusPeaksSmoothedMassDistribution(BaseSmoothedMassDistribution):
    primary_model = None

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)

    def __init__(self, mmin=2, mmax=200, normalization_shape=(1000, 500), cache=True, spacing="log"):
        self.mmin = mmin
        self.mmax = mmax
        if spacing == "log":
            self.m1s = xp.logspace(xp.log10(mmin), xp.log10(mmax), normalization_shape[0])
        elif spacing == "linear":
            self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache
        self.spacing = spacing

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin_1 = kwargs.pop("mlow_1", self.mmin)
        mmin_2 = kwargs.pop("mlow_2", self.mmin)
        delta_m_1 = kwargs.pop("delta_m_1", 0)
        delta_m_2 = kwargs.pop("delta_m_2", 0)
        mmax = kwargs.get("mmax", self.mmax)
        if "jax" not in xp.__name__:
            if mmin_1 < self.mmin or mmin_2 < self.mmin:
                raise ValueError(f"{self.__class__}: mlow < self.mmin ({self.mmin})")
            if mmax > self.mmax:
                raise ValueError(f"{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})")
        p_m1 = self.p_m1(dataset, mmin=mmin_1, delta_m=delta_m_1, **kwargs, **self.kwargs)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin_2, delta_m=delta_m_2)
        return p_m1 * p_q

    def p_q(self, dataset, beta, mmin, delta_m):
        """JAX-friendly p_q that uses xp.interp instead of the side-effecting
        `_cache_q_norms`/`_q_interpolant` path. The norm-vs-m1 grid is
        evaluated on self.m1s and linearly interpolated to dataset["mass_1"].
        """
        from gwpopulation.utils import powerlaw

        m1_samples = dataset["mass_1"]
        q_samples = dataset["mass_ratio"]

        p_q = powerlaw(q_samples, beta, 1, mmin / m1_samples)
        p_q *= self.smoothing(
            m1_samples * q_samples,
            mmin=mmin,
            mmax=m1_samples,
            delta_m=delta_m,
        )

        # Compute norm(m1) on the static m1s grid then xp.interp to samples
        p_q_grid = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q_grid *= self.smoothing(
            self.m1s_grid * self.qs_grid,
            mmin=mmin,
            mmax=self.m1s_grid,
            delta_m=delta_m,
        )
        norms_grid = xp.where(
            xp.array(delta_m) > 0,
            xp.nan_to_num(xp.trapezoid(p_q_grid, self.qs, axis=0)),
            xp.ones(self.m1s.shape),
        )
        norm_at_samples = xp.interp(m1_samples, self.m1s, norms_grid)
        return xp.nan_to_num(p_q / norm_at_samples)

    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m_1", "delta_m_2", "mlow_1", "mlow_2"]
        vars.remove("mmin")
        vars = set(vars).difference(self.kwargs.keys())
        return vars


class OnePeakBrokenPowerLawSmoothedMassDistribution(BrokenPowerLawPlusPeaksSmoothedMassDistribution):
    primary_model = two_component_double_power_law_primary_mass


class TwoPeakBrokenPowerLawSmoothedMassDistribution(BrokenPowerLawPlusPeaksSmoothedMassDistribution):
    primary_model = three_component_double_power_law_primary_mass


class ThreePeakBrokenPowerLawSmoothedMassDistribution(BrokenPowerLawPlusPeaksSmoothedMassDistribution):
    primary_model = four_component_double_power_law_primary_mass


def _setup_interpolant(nodes, values, kind="cubic", backend=None):
    from cached_interpolate import RegularCachingInterpolant as CachingInterpolant

    if backend is None:
        backend = xp
    nodes = to_numpy(nodes)
    interpolant = CachingInterpolant(nodes, nodes, kind=kind, backend=backend)
    interpolant.conversion = backend.array(interpolant.conversion)
    return partial(interpolant, backend.array(values))
