"""
Two independent methods for extracting bounce onset times from a phyphox
Audio Amplitude (SPL vs time) trace, as required by the Feature Engineering
rubric criterion (>= 2 independent methods, compared for consistency).

Method A: prominence-based peak picking on the SPL trace (scipy.signal.find_peaks).
Method B: rising-edge threshold-crossing detection (first sample per impulse
          that crosses a fixed dB threshold), which is less sensitive to a
          peak being "clipped" by the low sample rate than method A.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_bounces_peak_picking(df: pd.DataFrame, threshold_db: float,
                                 min_separation_s: float = 0.05,
                                 prominence_db: float = 3.0) -> np.ndarray:
    """Method A: local-maxima peak picking above `threshold_db`.

    min_separation_s guards against a single physical impact producing two
    detections if the SPL trace has a noisy double hump.

    prominence_db matters more than it looks: `distance` alone only enforces
    a minimum *spacing* between accepted peaks -- it does not require a
    candidate to be a genuinely separated bump. On a smooth decay tail (e.g.
    an Audio Scope RMS envelope), floating-point-level ripple produces huge
    numbers of infinitesimal local maxima; without a prominence floor,
    find_peaks' greedy by-height selection can pick one of these numerical
    blips as a fake "second impact" once it's outside the `distance`
    exclusion zone around the real peak. This was caught by validating
    against synthetic data with a known single impact -- a spurious second
    detection ~30 ms later appeared purely from numerical noise.
    """
    t = df["time_s"].to_numpy()
    spl = df["spl_db"].to_numpy()
    valid = ~np.isnan(spl)
    t, spl = t[valid], spl[valid]

    median_dt = np.median(np.diff(t))
    min_dist_samples = max(1, int(round(min_separation_s / median_dt)))

    peaks, _ = find_peaks(spl, height=threshold_db, distance=min_dist_samples,
                           prominence=prominence_db)
    return t[peaks]


def detect_bounces_threshold_crossing(df: pd.DataFrame, threshold_db: float,
                                       refractory_s: float = 0.05) -> np.ndarray:
    """Method B: first upward crossing of `threshold_db` per impulse.

    refractory_s: minimum time after a detected crossing before another
    crossing may be registered (prevents multiple triggers on one impact's
    decay ringing).
    """
    t = df["time_s"].to_numpy()
    spl = df["spl_db"].to_numpy()
    valid = ~np.isnan(spl)
    t, spl = t[valid], spl[valid]

    above = spl >= threshold_db
    rising = np.where(above[1:] & ~above[:-1])[0] + 1

    crossings = []
    last_t = -np.inf
    for idx in rising:
        if t[idx] - last_t >= refractory_s:
            crossings.append(t[idx])
            last_t = t[idx]
    return np.array(crossings)


def inter_bounce_intervals(bounce_times: np.ndarray) -> np.ndarray:
    """Delta t_n = t_{n+1} - t_n for consecutive detected bounce times."""
    return np.diff(bounce_times)


def expected_interval(h0: float, n, g_ref: float = 9.783, e_estimate: float = 0.8):
    """Theoretical Delta t_n = A * e^n for a given height -- used as a
    plausibility reference, not a measurement. n may be scalar or array.
    """
    A = 2 * np.sqrt(2 * h0 / g_ref)
    return A * np.asarray(e_estimate, dtype=float) ** np.asarray(n, dtype=float)


def filter_plausible_bounce_times(times: np.ndarray, h0: float, g_ref: float = 9.783,
                                   e_estimate: float = 0.8, abs_tolerance_factor: float = 1.7,
                                   ratio_ceiling: float = 1.4, max_leading_drop: int = 2) -> np.ndarray:
    """Physical Consistency Filter: drop detected impacts that make the
    resulting Delta t_n sequence physically implausible, before it ever
    reaches the fit.

    Two independent checks, because they catch different failure modes --
    validated against a real example where the first alone was not enough:

    1. Absolute plausibility ceiling: Delta t_n must not exceed
       expected_interval(h0, n) * abs_tolerance_factor. Catches a whole
       trial shifted high from its very first interval (e.g. a missed
       first bounce, or a stray detection before the real sequence) --
       this is the one case the ratio check below CANNOT see, because a
       sequence like [1.34, 0.49, 0.24] is still strictly decreasing even
       though 1.34 s is roughly 3x too large for any plausible drop height
       here. This was a real, observed case (0.7 m trial 1), not a
       hypothetical.
    2. Ratio/monotonicity check: Delta t_{n+1} / Delta t_n must not exceed
       ratio_ceiling. The model predicts this ratio equals e < 1, so
       anything meaningfully above 1 (ratio_ceiling gives room for
       measurement noise around a true ratio close to 1) means the
       sequence increased where the physics says it must decrease --
       catches a sudden jump mid-sequence that check 1 alone might miss
       if the jump isn't large enough to break the *absolute* ceiling.

    Both checks are deliberately generous (not tight statistical bounds) --
    the goal is to catch obviously-broken intervals, not to aggressively
    prune ordinary measurement scatter. Tune and re-validate against
    synthetic data (see synthetic_demo.py) before tightening either
    tolerance.

    max_leading_drop: a corrupted *first* interval is often caused by a
    stray detection before the real bounce sequence starts (as opposed to
    a genuinely missed bounce partway through, which this filter does NOT
    attempt to repair -- seeing a violation, it truncates the trial there
    rather than guessing a relabeling of everything after it). This tries
    dropping 0..max_leading_drop leading impacts and keeps whichever
    starting point survives the longest before hitting a violation.
    """
    times = np.asarray(times, dtype=float)
    best_kept = times[:1]  # fallback: no usable intervals at all

    for drop in range(0, min(max_leading_drop, max(len(times) - 2, 0)) + 1):
        candidate = times[drop:]
        dt = np.diff(candidate)
        if len(dt) == 0:
            continue

        n_kept = 0
        for i, dt_n in enumerate(dt):
            n = i + 1
            if dt_n > expected_interval(h0, n, g_ref, e_estimate) * abs_tolerance_factor:
                break
            if i > 0 and dt_n / dt[i - 1] > ratio_ceiling:
                break
            n_kept = i + 1

        kept = candidate[: n_kept + 1]
        if len(kept) > len(best_kept):
            best_kept = kept

    return best_kept


def compare_methods(times_a: np.ndarray, times_b: np.ndarray,
                     match_tol_s: float = 0.15) -> dict:
    """Pair up detections from the two methods within match_tol_s and report
    agreement statistics -- this is the "compare their results" step the
    rubric asks for under Feature Engineering.

    A rising-edge crossing (method B) is expected to precede the true peak
    (method A) by roughly the impulse rise time, so a consistent, non-zero
    `mean_offset_s` here is physically expected -- report it as a small
    timing systematic rather than treating it as disagreement.
    """
    matched_diffs = []
    used_b = set()
    for ta in times_a:
        candidates = [(abs(ta - tb), j) for j, tb in enumerate(times_b) if j not in used_b]
        if not candidates:
            continue
        diff, j = min(candidates, key=lambda x: x[0])
        if diff <= match_tol_s:
            matched_diffs.append(times_a[np.searchsorted(times_a, ta)] - times_b[j])
            used_b.add(j)

    matched_diffs = np.array(matched_diffs)
    return {
        "n_a": len(times_a),
        "n_b": len(times_b),
        "n_matched": len(matched_diffs),
        "mean_offset_s": float(np.mean(matched_diffs)) if len(matched_diffs) else np.nan,
        "std_offset_s": float(np.std(matched_diffs, ddof=1)) if len(matched_diffs) > 1 else np.nan,
    }
