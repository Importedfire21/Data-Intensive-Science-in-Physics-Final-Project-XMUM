"""
Weighted least-squares fitting for the bouncing-ball linearised model:

    ln(Delta t_n) = n * ln(e) + ln(A),   A = 2 * sqrt(2 * h0 / g)

and extraction of g (with propagated uncertainty) from the fitted intercept,
plus a PDG-style weighted combination across independent height trials.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearFitResult:
    slope: float
    slope_err: float
    intercept: float
    intercept_err: float
    chi2: float
    ndf: int
    chi2_ndf: float
    residuals: np.ndarray
    fitted_y: np.ndarray


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> LinearFitResult:
    """Weighted least-squares fit of y = slope * x + intercept.

    Uses the standard closed-form weighted linear regression (equivalent to
    a 2-parameter chi-square minimisation), valid for y_err all > 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    w = 1.0 / y_err**2
    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x**2)
    Sxy = np.sum(w * x * y)

    delta = S * Sxx - Sx**2
    slope = (S * Sxy - Sx * Sy) / delta
    intercept = (Sxx * Sy - Sx * Sxy) / delta

    slope_err = np.sqrt(S / delta)
    intercept_err = np.sqrt(Sxx / delta)

    fitted_y = slope * x + intercept
    residuals = y - fitted_y
    chi2 = float(np.sum((residuals / y_err) ** 2))
    ndf = len(x) - 2
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

    return LinearFitResult(
        slope=slope, slope_err=slope_err,
        intercept=intercept, intercept_err=intercept_err,
        chi2=chi2, ndf=ndf, chi2_ndf=chi2_ndf,
        residuals=residuals, fitted_y=fitted_y,
    )


def reliability_weight(n: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Down-weighting factor for bounce number n, reflecting decreasing
    confidence in later bounces beyond pure timing precision.

    Two independent effects make late bounces less trustworthy, only one of
    which the timing-uncertainty propagation in pipeline.fit_bounce_sequence
    already captures on its own:

    1. Timing precision degrades in *relative* terms as Delta t_n shrinks
       (already captured: sigma_ln_dt = sigma_dt_n / dt_n grows as dt_n
       shrinks with n).
    2. The model itself (constant coefficient of restitution, purely
       vertical bounces) is more likely to have broken down by the time a
       later bounce is reached -- amplitude has decayed further (worse
       detection SNR), and small perturbations (spin, an off-vertical
       bounce) compound with each impact. This is NOT captured by timing
       uncertainty alone, and was observed directly: checking whether
       Delta t_n decreases monotonically (as the model requires) showed
       real violations that become more common at higher n.

    reliability_weight(n) = 1 / n**power models effect 2 as a simple,
    monotonically decreasing multiplicative weight -- n=1 (the first
    interval, presumably the most reliable) keeps full weight, and each
    successive bounce counts for less. power=1.0 is a moderate default;
    power=0 disables this (pure timing-based weighting only), higher power
    down-weights later bounces more aggressively.

    This is a simple, explainable heuristic, not a first-principles model of
    restitution decay -- treat it as a systematic knob to be justified and
    reported, not as a precisely-derived correction.
    """
    n = np.asarray(n, dtype=float)
    return 1.0 / n**power


def resolution_uncertainty(resolution: float) -> float:
    """Standard uncertainty of a quantity known only to within one
    measurement "resolution" unit (e.g. one audio sample period), treating
    the true value as uniformly ("rectangularly") distributed within that
    unit -- the standard Type B evaluation for a digitised/quantised
    measurement (JCGM 100:2008, GUM, sec. 4.3.7): for a uniform
    distribution of half-width a = resolution/2, the standard deviation is
    a/sqrt(3) = resolution/(2*sqrt(3)) = resolution/sqrt(12).

    This replaces an earlier ad hoc resolution/2 convention used for the
    per-impact timing uncertainty -- resolution/2 is the *half-width* of
    the uniform distribution, not its standard deviation, and overstates
    sigma by a factor of sqrt(3) (~1.73x). Using the correct sqrt(12)
    denominator gives a smaller, more rigorous timing uncertainty and
    correspondingly larger chi^2/ndf values downstream (less claimed
    uncertainty makes the same residuals look relatively larger) --
    report that honestly if/when it happens, rather than treating a good
    chi^2/ndf under the old, looser convention as validation of anything.
    """
    return resolution / np.sqrt(12)


def g_stat_syst_from_intercept(intercept: float, intercept_err: float,
                                h0: float, h0_err: float) -> tuple[float, float, float]:
    """Convert the fitted ln(A) intercept into g, with its statistical
    (timing/fit precision) and systematic (height measurement) uncertainty
    contributions returned SEPARATELY rather than combined in quadrature.

    A = exp(intercept);  g = 8 * h0 / A^2 = 8 * h0 * exp(-2 * intercept)

    Uncertainty propagation (independent errors, first order):
      dg/d(intercept) = -2 * g   -> statistical (varies trial-to-trial)
      dg/dh0           = g / h0  -> systematic (same h0 for every repeat
                                    at a given height -- fully correlated,
                                    must not be averaged down like a
                                    statistical uncertainty; see
                                    pipeline.combine_repeats_stat_syst)

    Returns (g, g_stat_err, g_syst_err). g_from_intercept() below is the
    same calculation with the two combined in quadrature, kept for
    call sites (e.g. the notebooks) that only need one number.
    """
    A = np.exp(intercept)
    g = 8.0 * h0 / A**2

    dg_dintercept = -2.0 * g
    dg_dh0 = g / h0

    g_stat = abs(dg_dintercept * intercept_err)
    g_syst = abs(dg_dh0 * h0_err)
    return float(g), float(g_stat), float(g_syst)


def g_from_intercept(intercept: float, intercept_err: float,
                      h0: float, h0_err: float) -> tuple[float, float]:
    """Convert the fitted ln(A) intercept into g and a single combined
    (statistical + systematic added in quadrature) uncertainty. Prefer
    g_stat_syst_from_intercept() when the two need to stay separate (e.g.
    to report g = value +/- stat +/- syst, or to combine repeats/heights
    correctly -- see pipeline.combine_repeats_stat_syst /
    combine_heights_stat_syst).
    """
    g, g_stat, g_syst = g_stat_syst_from_intercept(intercept, intercept_err, h0, h0_err)
    return g, float(np.sqrt(g_stat**2 + g_syst**2))


@dataclass
class CombinationResult:
    g_mean: float
    g_mean_err: float
    chi2: float
    ndf: int
    chi2_ndf: float
    pull_per_point: np.ndarray


def pdg_combine(g_values: np.ndarray, g_errors: np.ndarray) -> CombinationResult:
    """PDG-style inverse-variance weighted average with a chi-square
    compatibility test across the individual measurements.
    """
    g_values = np.asarray(g_values, dtype=float)
    g_errors = np.asarray(g_errors, dtype=float)

    weights = 1.0 / g_errors**2
    g_mean = np.sum(weights * g_values) / np.sum(weights)
    g_mean_err = 1.0 / np.sqrt(np.sum(weights))

    pulls = (g_values - g_mean) / g_errors
    chi2 = float(np.sum(pulls**2))
    ndf = len(g_values) - 1
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

    return CombinationResult(
        g_mean=float(g_mean), g_mean_err=float(g_mean_err),
        chi2=chi2, ndf=ndf, chi2_ndf=chi2_ndf, pull_per_point=pulls,
    )


@dataclass
class CombinationResultStatSyst:
    g_mean: float
    stat_err: float
    syst_err: float
    chi2: float
    ndf: int
    chi2_ndf: float
    pull_per_point: np.ndarray

    @property
    def total_err(self) -> float:
        """Combined stat+syst in quadrature -- for call sites (plots,
        pulls vs g_ref) that need a single number rather than the
        breakdown. Prefer quoting stat and syst separately in the report.
        """
        return float(np.sqrt(self.stat_err**2 + self.syst_err**2))


def combine_repeats_stat_syst(g_values: np.ndarray, g_stat_errors: np.ndarray,
                               g_syst_errors: np.ndarray) -> CombinationResultStatSyst:
    """Combine repeated drop trials AT THE SAME HEIGHT into one g, keeping
    statistical and systematic uncertainty separate.

    The statistical part (timing/fit precision) is a genuinely independent
    draw per trial, so it is inverse-variance weighted and combined exactly
    like pdg_combine -- it shrinks with more repeats, as it should.

    The systematic part (height measurement uncertainty) is NOT
    independent per trial: every repeat at this height shares the exact
    same physical h0 and therefore the exact same h0_err. Averaging N
    fully-correlated copies of the same systematic does not shrink it by
    sqrt(N) the way averaging N independent statistical draws does -- so
    it must be evaluated once (here, from the combined g and the common
    relative height uncertainty), not inverse-variance-combined alongside
    the statistical part.
    """
    g_values = np.asarray(g_values, dtype=float)
    g_stat_errors = np.asarray(g_stat_errors, dtype=float)
    g_syst_errors = np.asarray(g_syst_errors, dtype=float)

    weights = 1.0 / g_stat_errors**2
    g_mean = np.sum(weights * g_values) / np.sum(weights)
    stat_err = 1.0 / np.sqrt(np.sum(weights))

    pulls = (g_values - g_mean) / g_stat_errors
    chi2 = float(np.sum(pulls**2))
    ndf = len(g_values) - 1
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

    # Same relative height uncertainty applies to every trial at this
    # height (h0, h0_err are shared) -- evaluate the systematic once at
    # the combined g rather than inverse-variance-combining N copies of
    # what is really one systematic.
    rel_syst = np.mean(g_syst_errors / g_values)
    syst_err = abs(g_mean * rel_syst)

    return CombinationResultStatSyst(
        g_mean=float(g_mean), stat_err=float(stat_err), syst_err=float(syst_err),
        chi2=chi2, ndf=ndf, chi2_ndf=chi2_ndf, pull_per_point=pulls,
    )


def combine_heights_stat_syst(g_values: np.ndarray, stat_errors: np.ndarray,
                               syst_errors: np.ndarray) -> CombinationResultStatSyst:
    """Combine several heights' (g, stat, syst) results into one overall
    result. Unlike combine_repeats_stat_syst, each height's height
    measurement is independent of the others (a different physical
    distance, separately measured) -- so unlike the within-height case,
    the systematic contributions here ARE independent per height and are
    propagated like an ordinary uncertainty through the same
    inverse-variance weights used for the statistical combination, rather
    than treated as one shared value.
    """
    g_values = np.asarray(g_values, dtype=float)
    stat_errors = np.asarray(stat_errors, dtype=float)
    syst_errors = np.asarray(syst_errors, dtype=float)

    weights = 1.0 / stat_errors**2
    norm = np.sum(weights)
    g_mean = np.sum(weights * g_values) / norm
    stat_err = 1.0 / np.sqrt(norm)

    rel_weights = weights / norm
    syst_err = float(np.sqrt(np.sum((rel_weights * syst_errors) ** 2)))

    pulls = (g_values - g_mean) / stat_errors
    chi2 = float(np.sum(pulls**2))
    ndf = len(g_values) - 1
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

    return CombinationResultStatSyst(
        g_mean=float(g_mean), stat_err=float(stat_err), syst_err=syst_err,
        chi2=chi2, ndf=ndf, chi2_ndf=chi2_ndf, pull_per_point=pulls,
    )


def pull(value: float, value_err: float, reference: float) -> float:
    """Standard pull of a measurement against a reference value."""
    return (value - reference) / value_err
