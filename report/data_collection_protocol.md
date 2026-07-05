# Data Collection Protocol — Bouncing Ball Method (g measurement)

This protocol is designed to feed directly into `notebooks/analysis.ipynb`.
Follow the file-naming and logging conventions below so the notebook needs
no changes beyond filling in the height table.

## 1. Equipment

- Hard ball with high, consistent coefficient of restitution (e.g. a
  superball or a hard rubber/steel ball). Avoid soft/foam balls — their
  restitution is too low and too variable, and the acoustic impact is weak.
- Hard, non-resonant, level surface (e.g. a tiled or concrete floor; avoid
  hollow wooden platforms, which ring and confuse peak detection).
- Smartphone running phyphox. Two tools are used together (see Section 4a):
  **Audio Amplitude** for the full multi-bounce sequence per height, and
  **Audio Scope** for a high-resolution cross-check of the first bounce.
- Tape measure or laser distance meter (mm resolution) for drop height.
- A rigid drop guide (e.g. a vertical rod/tube, or dropping along a wall
  with height marks) to make the release height reproducible and to
  minimise ball spin/lateral drift.

## 2. Pre-session setup

1. Place the phone in its measurement position (same position it will stay
   in for the whole session) and start a background recording of the empty
   room for several minutes with the phone in place, nobody moving.
2. This background file characterises the room's noise floor and sets the
   bounce-detection threshold (`notebooks/analysis.ipynb`, Section 1).
   Save it as `data/raw/background_noise.xls`.
3. **Do not talk, walk, or touch the phone during this recording** except
   to start/stop it — the first and last ~10 s are trimmed automatically to
   remove handling noise, so keep those brief, but avoid extended noise
   throughout since it inflates the estimated noise floor and desensitises
   the detection threshold.

## 3. Drop heights and trial count

- Use **at least 8 distinct drop heights**, spanning 0.3 m to 1.5 m (a wider
  spread and more points give a better-constrained fit and a more
  convincing scatter plot for the report).
- Measure height from the **bottom of the ball** to the impact surface, at
  the moment of release.
- Record height uncertainty from your measuring method (e.g. ±3-5 mm for a
  tape measure read carefully, more if you're estimating "roughly where the
  ball left your fingers").
- At each height, record **at least 2–3 repeat drops** if time allows —
  this lets you assess drop-to-drop variability (a real statistical
  uncertainty source) separately from the within-trial fit uncertainty.

## 4. Recording: choose ONE workflow per session

**Workflow A — one file per height (recommended for your first attempts):**
start a fresh phyphox recording for each height, drop once the trace is
clean, and stop the recording ~1–2 s after the ball visibly/audibly comes
to rest. Export as `height_<mm>.xls`, e.g. `height_030.xls` for 0.30 m,
`height_150.xls` for 1.50 m.

**Workflow B — several heights in one continuous recording:** start one
recording, then for each height: stand still and silent for **>= 3 s**,
drop the ball, let it come to rest, stand still and silent for **>= 3 s**
again before the next drop. This silence is what `segment_recording()` in
the notebook uses to automatically split the file into per-trial segments —
without it, either trials get merged or a trial's own bounces get cut.
**Log the exact order of heights as you drop them** (e.g. in a notebook or
voice memo) — segmentation recovers timing and count, not which height was
used for which segment. Export as `data/raw/all_heights.xls`.

You can mix workflows across different sessions, but keep the log of which
file corresponds to which height(s) unambiguous either way.

## 4a. Optional: Audio Scope cross-check of the first bounce

Testing confirmed that phyphox's **Audio Scope** export only covers ~0.5 s
of audio regardless of how long the recording session runs (it appears to
save the oscilloscope's rolling display buffer, not a continuous
recording) — enough for the drop and the first bounce or two, not a full
multi-bounce sequence. Use it as a **second, independent method** for the
Feature Engineering rubric criterion, not as a replacement for Audio
Amplitude:

1. For a few representative heights (small heights work best — the whole
   drop + first bounce must fit in ~0.5 s), start an Audio Scope recording
   timed so the drop happens within the first ~50-100 ms of the buffer.
2. Export as `scope_height_<mm>.xls`.
3. Also record a **dedicated quiet Audio Scope background** (no drop, same
   room/position) exactly as in Section 2 — noise/threshold calibration
   must always come from a quiet-only recording, never from the trial file
   that contains the actual bounce. A background this short may itself
   still contain incidental room noise (it did in initial testing); if so,
   lower `k_sigma` in `noise_characterization.characterise()` and expect a
   correspondingly higher false-positive risk — report this as a limitation
   rather than silently tuning it away.
4. In the notebook, compare Δt₁ extracted from the Audio Scope capture
   against Δt₁ from the corresponding Audio Amplitude capture at the same
   height (Section 3 of `notebooks/analysis.ipynb`). At small heights,
   expect Audio Amplitude to sometimes **miss the true first bounce
   entirely** (its ~120 ms sampling interval aliases against a fast
   interval) and report the interval to the *second* bounce instead — this
   is a real, reportable finding, not just a validation nuisance.

Real Audio Scope exports are multi-sheet `.xls` workbooks (`Audio data`,
`Metadata Device`, `Metadata Time`) with columns `Time (ms)` and
`Recording (a.u.)` (confirmed from a Samsung SM-A556E export, phyphox
1.2.0) — `io_utils.load_recording()` already handles this format.

## 5. Per-trial checklist

- [ ] Height measured and logged (with uncertainty)
- [ ] Drop released cleanly (minimal spin, no lateral push)
- [ ] Recording captures the full bounce sequence to rest, plus >= 1 s
      of quiet before the first impact and after the last audible bounce
- [ ] No talking/footsteps/background noise during the trial
- [ ] File saved/exported with the agreed naming convention

## 6. What to log alongside the raw files

Keep a simple table (a spreadsheet or the lab notebook) with, per trial:

| trial file | height (m) | height unc. (m) | ball used | surface | notes (e.g. "clean drop", "slight spin") |
|---|---|---|---|---|---|

This table is what feeds the `trial_files` / `heights` dictionaries in the
notebook's "Real trial data" section, and is also the raw-data table that
belongs in the report's Appendix.

## 7. Sanity checks before moving to analysis

- Re-run the noise characterisation notebook cell on the background file —
  confirm the recommended threshold sits clearly above the ambient floor
  but well below the loudness of an actual ball impact (spot-check one
  trial file by eye/ear).
- For Workflow B recordings, always inspect the segmentation diagnostic
  plot (`process_multi_drop_recording`) before trusting the numbers — it
  must show exactly as many segments as drops you performed.
