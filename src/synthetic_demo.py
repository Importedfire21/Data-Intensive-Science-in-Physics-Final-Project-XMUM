"""
SYNTHETIC DATA ONLY -- for validating the analysis pipeline before real drop
trials are collected. Do not present outputs from this module as measured
results in the report; they exist to prove the code is correct end-to-end.
"""
import numpy as np
import pandas as pd


def generate_synthetic_trial(h0: float, g_true: float = 9.783, e: float = 0.78,
                              n_bounces: int = 8, sample_dt: float = 0.123,
                              noise_db_std: float = 4.3, noise_db_mean: float = -52.3,
                              impulse_db: float = 45.0, impulse_width_s: float = 0.08,
                              lead_in_s: float = 2.0, lead_out_s: float = 2.0,
                              seed: int | None = None) -> pd.DataFrame:
    """Simulate a phyphox Audio Amplitude trace for a ball dropped from h0,
    bouncing with a constant coefficient of restitution e (simplification;
    real balls have a slightly velocity-dependent e -- see systematics
    discussion in the report).

    lead_in_s / lead_out_s mimic the data-collection protocol's requirement
    of >= 2 s of stationary silence before/after the drop, so the same
    trim_and_clean() step used on real trials can be exercised here too.

    Returns a DataFrame shaped like a real phyphox export: time_s, spl_db.
    """
    rng = np.random.default_rng(seed)

    A = 2 * np.sqrt(2 * h0 / g_true)
    t_first_impact = lead_in_s + np.sqrt(2 * h0 / g_true)

    bounce_times = [t_first_impact]
    for n in range(1, n_bounces + 1):
        bounce_times.append(bounce_times[-1] + A * e**n)
    bounce_times = np.array(bounce_times)

    t_end = bounce_times[-1] + lead_out_s
    t = np.arange(0, t_end, sample_dt)
    t += rng.normal(0, sample_dt * 0.15, size=t.shape)
    t = np.sort(t)

    spl = rng.normal(noise_db_mean, noise_db_std, size=t.shape)
    for bt in bounce_times:
        spl += impulse_db * np.exp(-0.5 * ((t - bt) / impulse_width_s) ** 2)

    return pd.DataFrame({"time_s": t, "spl_db": spl})


def generate_multi_drop_recording(heights: list[float], gap_s: float = 4.0,
                                   g_true: float = 9.783, e: float = 0.78,
                                   seed: int | None = None, **trial_kwargs) -> pd.DataFrame:
    """Concatenate several single-drop traces (one per height, in order)
    separated by `gap_s` of pure background noise, to simulate recording
    multiple trials in one continuous phyphox capture. Used only to validate
    `segmentation.segment_recording` against a known ground truth (known
    heights, known order, known trial count) before it is trusted on real
    multi-drop recordings.
    """
    rng_master = np.random.default_rng(seed)
    noise_db_mean = trial_kwargs.get("noise_db_mean", -52.3)
    noise_db_std = trial_kwargs.get("noise_db_std", 4.3)
    sample_dt = trial_kwargs.get("sample_dt", 0.123)

    chunks = []
    t_offset = 0.0
    true_bounds = []
    for h0 in heights:
        seed_i = int(rng_master.integers(0, 1_000_000))
        trial = generate_synthetic_trial(h0=h0, g_true=g_true, e=e, seed=seed_i, **trial_kwargs)
        trial = trial.copy()
        trial["time_s"] += t_offset
        chunks.append(trial)
        true_bounds.append((trial["time_s"].min(), trial["time_s"].max()))

        gap_t = np.arange(0, gap_s, sample_dt) + trial["time_s"].max() + sample_dt
        gap_spl = rng_master.normal(noise_db_mean, noise_db_std, size=gap_t.shape)
        chunks.append(pd.DataFrame({"time_s": gap_t, "spl_db": gap_spl}))

        t_offset = gap_t.max() + sample_dt if len(gap_t) else trial["time_s"].max()

    full = pd.concat(chunks, ignore_index=True).sort_values("time_s").reset_index(drop=True)
    full.attrs["true_segment_bounds"] = true_bounds
    full.attrs["true_heights"] = heights
    return full


def generate_synthetic_scope_trial(h0: float, g_true: float = 9.783, e: float = 0.78,
                                    n_bounces: int = 8, sample_dt: float = 0.001,
                                    noise_amp: float = 0.02, burst_amp: float = 1.0,
                                    decay_tau: float = 0.02, f0: float = 150.0,
                                    lead_in_s: float = 0.2, lead_out_s: float = 0.2,
                                    seed: int | None = None) -> pd.DataFrame:
    """SYNTHETIC raw-waveform (Audio Scope-like) trial: a low-amplitude noise
    floor plus a decaying-sinusoid burst at each bounce impact. Used only to
    validate envelope.compute_envelope() + the rest of the pipeline against
    a known ground truth before a real Audio Scope export exists.

    f0 is kept well below the Nyquist frequency implied by sample_dt so the
    burst is not aliased into nonsense; the pipeline itself never assumes a
    particular sample_dt (it measures dt directly from the timestamps), so
    this can be changed freely to match whatever your real export turns out
    to use.

    Returns a DataFrame shaped like a raw phyphox export: 'Time (s)', 'Data'
    (matches one of io_utils._SCOPE_VALUE_CANDIDATES).
    """
    rng = np.random.default_rng(seed)

    A = 2 * np.sqrt(2 * h0 / g_true)
    t_first_impact = lead_in_s + np.sqrt(2 * h0 / g_true)

    bounce_times = [t_first_impact]
    for n in range(1, n_bounces + 1):
        bounce_times.append(bounce_times[-1] + A * e**n)
    bounce_times = np.array(bounce_times)

    t_end = bounce_times[-1] + lead_out_s
    t = np.arange(0, t_end, sample_dt)

    data = rng.normal(0, noise_amp, size=t.shape)
    for bt in bounce_times:
        mask = t >= bt
        dt_since = t[mask] - bt
        data[mask] += burst_amp * np.exp(-dt_since / decay_tau) * np.sin(2 * np.pi * f0 * dt_since)

    return pd.DataFrame({"Time (s)": t, "Data": data})


def generate_synthetic_scope_background(duration_s: float = 0.5, sample_dt: float = 0.001,
                                         noise_amp: float = 0.02,
                                         seed: int | None = None) -> pd.DataFrame:
    """SYNTHETIC quiet-only raw-waveform recording (no bounces), for
    calibrating a detection threshold the same principled way Section 1 does
    for Audio Amplitude: from a dedicated background capture, never from the
    trial recording that contains the actual signal.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, sample_dt)
    data = rng.normal(0, noise_amp, size=t.shape)
    return pd.DataFrame({"Time (s)": t, "Data": data})
