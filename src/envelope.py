"""
Convert a raw Audio Scope waveform trace into an SPL-like envelope
(time_s, spl_db) so the rest of the pipeline -- noise characterisation,
cleaning, bounce detection, segmentation, fitting -- can operate identically
regardless of whether the input was phyphox's Audio Amplitude (already an
envelope) or Audio Scope (raw oscillating waveform).
"""
import numpy as np
import pandas as pd


def compute_envelope(df: pd.DataFrame, window_s: float = 0.01, value_col: str = "amplitude",
                      mode: str = "rms", db_reference: float | None = None) -> pd.DataFrame:
    """Turn a raw waveform trace into a dB-like envelope.

    window_s: rolling window used to turn oscillating raw samples into a
    smooth impact-energy trace. Pick this from the *sample interval* of your
    Audio Scope export, not guessed blindly: it should span several raw
    samples (so RMS/max is meaningful) but be much shorter than the
    shortest inter-bounce interval you need to resolve. E.g. at a 1 ms
    sample interval, window_s=0.01 (10 samples, 10 ms window) is a
    reasonable starting point; tune against a real bounce trial once you
    have one (the impact pulse should show up as a clean, few-window-wide
    bump, not a single spike or a smeared-out plateau).

    mode: 'rms' (root-mean-square within the window) or 'abs_max' (rolling
    max absolute amplitude). 'rms' is smoother and usually preferable.

    db_reference: reference level for the dB conversion
    (20*log10(level/db_reference)). If None, uses this recording's own 5th
    percentile RMS level (i.e. 0 dB ~ near-silence *for this recording*).
    That is sufficient for relative, single-recording threshold detection,
    but means absolute dB values are NOT comparable across recordings taken
    at different microphone gain -- pass an explicit reference if you need
    that.
    """
    t = df["time_s"].to_numpy()
    x = df[value_col].to_numpy(dtype=float)

    dt = float(np.median(np.diff(t)))
    win_samples = max(1, int(round(window_s / dt)))

    if mode == "rms":
        power = x ** 2
        kernel = np.ones(win_samples) / win_samples
        level = np.sqrt(np.convolve(power, kernel, mode="same"))
    elif mode == "abs_max":
        level = pd.Series(np.abs(x)).rolling(win_samples, center=True, min_periods=1).max().to_numpy()
    else:
        raise ValueError(f"Unknown mode: {mode!r}; use 'rms' or 'abs_max'")

    level = np.clip(level, 1e-12, None)
    if db_reference is None:
        db_reference = max(float(np.percentile(level, 5)), 1e-12)

    spl_db = 20 * np.log10(level / db_reference)
    return pd.DataFrame({"time_s": t, "spl_db": spl_db})
