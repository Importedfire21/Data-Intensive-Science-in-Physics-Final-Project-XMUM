"""
Data cleaning utilities. The main issue this addresses: starting/stopping a
phyphox recording involves reaching for the phone, tapping the screen, and
moving around the lab -- this contaminates the first and last several
seconds of every recording (background *and* trial) with handling noise
that has nothing to do with the room's ambient floor or the ball bounces.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    n_before: int
    n_after: int
    retention_rate: float
    trim_start_s: float
    trim_end_s: float
    n_nan_dropped: int


def trim_and_clean(df: pd.DataFrame, trim_start_s: float = 10.0,
                    trim_end_s: float = 10.0) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop the first `trim_start_s` and last `trim_end_s` seconds of a
    recording (handling noise from starting/stopping phyphox) and any NaN
    rows (phyphox writes NaN for the first couple of samples before its
    envelope filter has enough history).

    Quantitative criterion: fixed time window, chosen from the observed
    duration of handling noise in the raw background recording (see
    justification in the report / notebook markdown). Returns the cleaned
    DataFrame plus a report with the retention rate the rubric asks for.
    """
    n_before = len(df)

    t = df["time_s"]
    t_max = t.max()
    mask_time = (t >= trim_start_s) & (t <= t_max - trim_end_s)
    trimmed = df.loc[mask_time].copy()

    n_nan = int(trimmed["spl_db"].isna().sum())
    cleaned = trimmed.dropna(subset=["spl_db"]).reset_index(drop=True)

    n_after = len(cleaned)
    report = CleaningReport(
        n_before=n_before,
        n_after=n_after,
        retention_rate=n_after / n_before if n_before else np.nan,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
        n_nan_dropped=n_nan,
    )
    return cleaned, report


def cleaning_summary_text(report: CleaningReport) -> str:
    return (
        f"Trimmed first {report.trim_start_s:.0f} s and last {report.trim_end_s:.0f} s "
        f"(handling noise); dropped {report.n_nan_dropped} NaN samples.\n"
        f"Retained {report.n_after} / {report.n_before} samples "
        f"({report.retention_rate*100:.1f}%)."
    )
