"""
Data ingestion helpers for phyphox exports (Audio Amplitude and Audio Scope
modes). Both are funnelled to the same unified shape -- time_s, spl_db --
via load_recording(), so the rest of the pipeline never needs to know which
recording mode was used.
"""
from pathlib import Path

import pandas as pd

import envelope as envelope_mod

# Candidate raw-value column headers seen across phyphox Audio Scope exports.
# "Recording (a.u.)" is confirmed from a real export (phyphox 1.2.0, Samsung
# SM-A556E); the others are kept as fallbacks for other phone/app versions.
# Add to this list if your export uses yet another header -- load_audio_scope()
# raises a clear error naming the actual columns if none match, so this is a
# one-line fix, not a silent bug.
_SCOPE_VALUE_CANDIDATES = ["Recording (a.u.)", "Data", "Amplitude", "Sound (a.u.)", "Level", "Value"]

# phyphox's Audio Scope .xls export is a multi-sheet workbook; the samples
# live on a sheet named "Audio data" (confirmed from a real export), with
# "Metadata Device" / "Metadata Time" sheets alongside it that we don't need.
_SCOPE_DATA_SHEET_CANDIDATES = ["Audio data", "Data", "Sheet1"]

# Time column headers seen in phyphox exports, mapped to the multiplier
# needed to convert to seconds. Audio Amplitude uses seconds; the real Audio
# Scope export uses milliseconds.
_TIME_COLUMN_SCALES = {"Time (s)": 1.0, "Time (ms)": 1e-3}


def _read_any(path: str | Path) -> pd.DataFrame:
    """Read the first/only sheet of a simple (single-table) export."""
    path = Path(path)
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _read_scope_sheet(path: str | Path) -> pd.DataFrame:
    """Read the samples sheet from a (possibly multi-sheet) Audio Scope
    export, trying known sheet names before falling back to the first sheet.
    CSV exports are assumed to be single-table already.
    """
    path = Path(path)
    if path.suffix.lower() not in (".xls", ".xlsx"):
        return pd.read_csv(path)

    xls = pd.ExcelFile(path)
    for name in _SCOPE_DATA_SHEET_CANDIDATES:
        if name in xls.sheet_names:
            return pd.read_excel(xls, sheet_name=name)
    return pd.read_excel(xls, sheet_name=xls.sheet_names[0])


def _standardise_time(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever known time column is present to time_s, applying
    the correct unit conversion (e.g. Audio Scope's 'Time (ms)')."""
    for col, scale in _TIME_COLUMN_SCALES.items():
        if col in df.columns:
            df = df.rename(columns={col: "time_s"})
            if scale != 1.0:
                df["time_s"] = df["time_s"] * scale
            return df
    raise ValueError(
        f"Could not find a recognised time column. Columns present: {list(df.columns)}. "
        f"Add the correct header (and its to-seconds scale factor) to "
        f"io_utils._TIME_COLUMN_SCALES."
    )


def load_audio_amplitude(path: str | Path) -> pd.DataFrame:
    """Load a phyphox 'Audio Amplitude' export (.xls/.xlsx/.csv).

    Expected columns: 'Time (s)', 'Sound pressure level (dB)'.
    Returns a DataFrame with standardised column names: time_s, spl_db.
    """
    df = _read_any(path)
    df = _standardise_time(df)
    df = df.rename(columns={"Sound pressure level (dB)": "spl_db"})
    return df[["time_s", "spl_db"]]


def load_scope_raw(path: str | Path) -> pd.DataFrame:
    """Load a phyphox Audio Scope export WITHOUT computing an envelope --
    returns the raw (time_s, amplitude) samples, e.g. for plotting the raw
    waveform alongside its processed envelope.
    """
    df = _read_scope_sheet(path)
    df = _standardise_time(df)

    value_col = next((c for c in _SCOPE_VALUE_CANDIDATES if c in df.columns), None)
    if value_col is None:
        raise ValueError(
            f"Could not find a raw-value column in {path}. Columns present: "
            f"{list(df.columns)}. Add the correct header to "
            f"io_utils._SCOPE_VALUE_CANDIDATES."
        )
    return df.rename(columns={value_col: "amplitude"})[["time_s", "amplitude"]]


def load_audio_scope(path: str | Path, window_s: float = 0.01,
                      mode: str = "rms") -> pd.DataFrame:
    """Load a phyphox 'Audio Scope' (raw waveform) export and convert it to
    an Audio-Amplitude-like envelope (time_s, spl_db) via envelope.compute_envelope.

    Handles both a simple single-table export and the real multi-sheet .xls
    workbook (sheets "Audio data" / "Metadata Device" / "Metadata Time") --
    only the samples sheet is used.

    Note: a real Audio Scope export was found to cover only ~0.5 s of audio
    regardless of how long the phyphox recording session ran -- it appears
    to export the oscilloscope's rolling display buffer, not a continuous
    multi-second recording. Plan trials around a single bounce per capture
    (see the data-collection protocol) unless you find a phyphox buffer-size
    setting that extends this.

    window_s / mode: passed straight to envelope.compute_envelope(); tune
    window_s to your Audio Scope's actual sample interval (see that
    function's docstring).
    """
    df = _read_scope_sheet(path)
    df = _standardise_time(df)

    value_col = next((c for c in _SCOPE_VALUE_CANDIDATES if c in df.columns), None)
    if value_col is None:
        raise ValueError(
            f"Could not find a raw-value column in {path}. Columns present: "
            f"{list(df.columns)}. Add the correct header to "
            f"io_utils._SCOPE_VALUE_CANDIDATES."
        )
    df = df.rename(columns={value_col: "amplitude"})

    return envelope_mod.compute_envelope(df[["time_s", "amplitude"]], window_s=window_s, mode=mode)


def load_recording(path: str | Path, kind: str = "auto", **scope_kwargs) -> pd.DataFrame:
    """Unified loader: inspect the file's columns and dispatch to the right
    loader, always returning the standard (time_s, spl_db) shape.

    kind: 'auto' (detect from columns), 'amplitude', or 'scope'.
    scope_kwargs: forwarded to load_audio_scope (window_s, mode) when the
    file is (or is treated as) an Audio Scope export.
    """
    if kind == "amplitude":
        return load_audio_amplitude(path)
    if kind == "scope":
        return load_audio_scope(path, **scope_kwargs)

    path = Path(path)
    if path.suffix.lower() in (".xls", ".xlsx") and "Audio data" in pd.ExcelFile(path).sheet_names:
        return load_audio_scope(path, **scope_kwargs)

    header = _read_any(path)
    if "Sound pressure level (dB)" in header.columns:
        return load_audio_amplitude(path)
    if any(c in header.columns for c in _SCOPE_VALUE_CANDIDATES):
        return load_audio_scope(path, **scope_kwargs)

    raise ValueError(
        f"Could not auto-detect recording type for {path}. Columns present: "
        f"{list(header.columns)}. Pass kind='amplitude' or kind='scope' explicitly, "
        f"updating the column-name candidates in io_utils.py if needed."
    )


def load_trial_batch(raw_dir: str | Path, pattern: str = "*.xls", kind: str = "auto",
                      **scope_kwargs) -> dict[str, pd.DataFrame]:
    """Load every matching file in raw_dir into a dict keyed by filename stem.

    Use this once real drop-trial recordings are collected, e.g.:
        trials = load_trial_batch("data/raw", pattern="height_*.xls")
    """
    raw_dir = Path(raw_dir)
    return {f.stem: load_recording(f, kind=kind, **scope_kwargs)
            for f in sorted(raw_dir.glob(pattern))}
