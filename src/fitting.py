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


def g_from_intercept(intercept: float, intercept_err: float,
                      h0: float, h0_err: float) -> tuple[float, float]:
    """Convert the fitted ln(A) intercept into g and its propagated
    uncertainty, given the drop height h0 (with its own uncertainty).

    A = exp(intercept);  g = 8 * h0 / A^2 = 8 * h0 * exp(-2 * intercept)

    Uncertainty propagation (independent errors, first order):
      dg/d(intercept) = -2 * g
      dg/dh0           = g / h0
    """
    A = np.exp(intercept)
    g = 8.0 * h0 / A**2

    dg_dintercept = -2.0 * g
    dg_dh0 = g / h0

    g_err = np.sqrt((dg_dintercept * intercept_err) ** 2 + (dg_dh0 * h0_err) ** 2)
    return float(g), float(g_err)


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


def pull(value: float, value_err: float, reference: float) -> float:
    """Standard pull of a measurement against a reference value."""
    return (value - reference) / value_err
