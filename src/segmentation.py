"""
Split a single continuous phyphox recording that contains multiple ball-drop
trials (recorded back-to-back without stopping/restarting the app) into
per-trial segments, using the silent gap between trials as the split point.
"""
import numpy as np
import pandas as pd

import bounce_detection as bd


def segment_by_gaps(event_times: np.ndarray, min_gap_s: float) -> list[np.ndarray]:
    """Split sorted event (impulse) times into groups separated by gaps
    >= min_gap_s. Each group is the set of impulse times belonging to one
    contiguous drop trial (a ball bouncing to rest before the next drop).
    """
    if len(event_times) == 0:
        return []
    event_times = np.sort(event_times)
    gaps = np.diff(event_times)
    split_idx = np.where(gaps >= min_gap_s)[0] + 1
    return list(np.split(event_times, split_idx))


def segment_recording(df: pd.DataFrame, threshold_db: float, min_gap_s: float,
                       pad_s: float = 0.5, min_separation_s: float = 0.05) -> list[pd.DataFrame]:
    """Detect every impulse in a (possibly multi-drop) recording, group them
    into per-trial segments by inter-event silence, and return the
    corresponding DataFrame slices in chronological order.

    min_gap_s must be chosen larger than the largest expected inter-bounce
    interval within one trial (~ A for the first bounce, so up to ~1.3 s for
    h0 = 2 m) but smaller than the deliberate pause you leave between drops
    when recording multiple trials in one file. The data-collection protocol
    asks for >= 3 s of stillness between drops for exactly this reason.
    """
    all_times = bd.detect_bounces_peak_picking(df, threshold_db, min_separation_s=min_separation_s)
    groups = segment_by_gaps(all_times, min_gap_s)

    segments = []
    for g in groups:
        t_start = g[0] - pad_s
        t_end = g[-1] + pad_s
        mask = (df["time_s"] >= t_start) & (df["time_s"] <= t_end)
        segments.append(df.loc[mask].reset_index(drop=True))
    return segments


def segmentation_report(df: pd.DataFrame, threshold_db: float, min_gap_s: float) -> str:
    all_times = bd.detect_bounces_peak_picking(df, threshold_db)
    groups = segment_by_gaps(all_times, min_gap_s)
    lines = [f"Detected {len(groups)} trial segment(s) from {len(all_times)} total impulses "
             f"(min_gap_s = {min_gap_s}):"]
    for i, g in enumerate(groups):
        lines.append(f"  segment {i}: {len(g)} impulses, "
                      f"t = [{g[0]:.2f}, {g[-1]:.2f}] s, span = {g[-1]-g[0]:.2f} s")
    return "\n".join(lines)
