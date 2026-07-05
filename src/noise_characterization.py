"""
Characterise the microphone/room noise floor from a phyphox Audio Amplitude
background recording, and assess the temporal resolution this recording mode
offers for bounce-timing measurements.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NoiseProfile:
    n_samples: int
    n_nan: int
    mean_db: float
    std_db: float
    median_db: float
    p95_db: float
    max_db: float
    mean_dt_s: float
    median_dt_s: float
    std_dt_s: float
    min_dt_s: float
    max_dt_s: float
    recommended_threshold_db: float


def characterise(df: pd.DataFrame, k_sigma: float = 5.0, robust: bool = False) -> NoiseProfile:
    """Compute noise-floor statistics and sampling-interval statistics.

    **This must be run on a recording that contains no bounce/impact signal**
    -- a dedicated quiet background capture. Running it on a trial recording
    that includes the actual bounce(s) contaminates mean/std with the very
    transient you're trying to detect, which can push the threshold above
    real peaks (this happened during pipeline validation -- see
    characterise_excluding_window() for the one-capture-only fallback).

    k_sigma: number of standard deviations above the mean noise level used
    to set the default bounce-detection threshold (mean + k_sigma * std).
    A conservative k_sigma (>=5) keeps false positives from room noise low;
    tune against real trial recordings once collected.

    robust: if True, use the median and a MAD-based robust sigma estimate
    instead of mean/std. Use this whenever the input might not be perfectly
    clean (e.g. a "quiet" capture that still has a stray knock in it) --
    median/MAD are far less sensitive to a handful of outlier samples than
    mean/std are.
    """
    spl = df["spl_db"].to_numpy()
    t = df["time_s"].to_numpy()

    n_nan = int(np.isnan(spl).sum())
    spl_clean = spl[~np.isnan(spl)]

    dt = np.diff(t)
    dt = dt[dt > 0]

    if robust:
        center_db = float(np.median(spl_clean))
        mad = float(np.median(np.abs(spl_clean - center_db)))
        spread_db = 1.4826 * mad  # MAD-to-sigma conversion for a Gaussian core
    else:
        center_db = float(np.mean(spl_clean))
        spread_db = float(np.std(spl_clean, ddof=1))

    return NoiseProfile(
        n_samples=len(spl),
        n_nan=n_nan,
        mean_db=center_db,
        std_db=spread_db,
        median_db=float(np.median(spl_clean)),
        p95_db=float(np.percentile(spl_clean, 95)),
        max_db=float(np.max(spl_clean)),
        mean_dt_s=float(np.mean(dt)),
        median_dt_s=float(np.median(dt)),
        std_dt_s=float(np.std(dt, ddof=1)),
        min_dt_s=float(np.min(dt)),
        max_dt_s=float(np.max(dt)),
        recommended_threshold_db=center_db + k_sigma * spread_db,
    )


def characterise_excluding_window(df: pd.DataFrame, exclude_center_s: float,
                                   exclude_width_s: float, k_sigma: float = 5.0,
                                   robust: bool = True) -> NoiseProfile:
    """Fallback for when you only have ONE capture that contains both quiet
    background and the signal (e.g. an Audio Scope buffer too short to
    record a separate background) -- exclude a window around the known/
    suspected transient before computing noise statistics, so the threshold
    is calibrated from quiet samples only.

    exclude_center_s / exclude_width_s: the [center - width/2, center +
    width/2] window to drop. A reasonable first guess for exclude_center_s
    is the time of the global maximum |amplitude|.
    """
    lo = exclude_center_s - exclude_width_s / 2
    hi = exclude_center_s + exclude_width_s / 2
    quiet = df.loc[(df["time_s"] < lo) | (df["time_s"] > hi)]
    return characterise(quiet, k_sigma=k_sigma, robust=robust)


def resolvable_bounce_count(profile: NoiseProfile, A_estimate_s: float, e_estimate: float,
                             nyquist_factor: float = 2.0) -> int:
    """Estimate how many inter-bounce intervals n can be reliably resolved
    before Delta t_n = A * e^n drops below `nyquist_factor` times the median
    sampling interval of this recording mode.

    This directly formalises the "peak detection algorithm" risk noted in
    the assignment brief: at ~8 Hz (Audio Amplitude mode), late-time bounces
    of a fast-restituting ball can alias or be missed entirely.
    """
    min_resolvable_dt = nyquist_factor * profile.median_dt_s
    n = 1
    while True:
        dt_n = A_estimate_s * e_estimate ** n
        if dt_n < min_resolvable_dt:
            return n - 1
        n += 1
        if n > 100:
            return 100


def summary_text(profile: NoiseProfile) -> str:
    lines = [
        f"Samples: {profile.n_samples} (NaN at start: {profile.n_nan})",
        f"Noise floor: mean = {profile.mean_db:.2f} dB, std = {profile.std_db:.2f} dB, "
        f"median = {profile.median_db:.2f} dB, 95th pct = {profile.p95_db:.2f} dB, "
        f"max transient = {profile.max_db:.2f} dB",
        f"Sampling interval: mean = {profile.mean_dt_s*1000:.1f} ms, "
        f"median = {profile.median_dt_s*1000:.1f} ms, "
        f"std = {profile.std_dt_s*1000:.1f} ms, "
        f"range = [{profile.min_dt_s*1000:.1f}, {profile.max_dt_s*1000:.1f}] ms",
        f"Recommended bounce-detection threshold: {profile.recommended_threshold_db:.2f} dB "
        f"(mean + k*std)",
    ]
    return "\n".join(lines)
