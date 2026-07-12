# Measuring Local *g* via the Bouncing-Ball Method

PHY408 (Data Intensive Science) final project. A ball is dropped repeatedly
from a range of heights and its bounce sounds are recorded with a
smartphone microphone (phyphox). Successive inter-bounce intervals decay
geometrically, `Δt_n = A·e^n`, so a weighted linear fit of `ln(Δt_n)` vs.
bounce number `n` recovers `A = 2√(2h₀/g)` and hence the local
gravitational acceleration `g = 8h₀/A²`.

**Headline result (Run 1, 10 heights, 0.315–1.215 m):**
`g = 9.70 ± 0.36 (stat) ± 0.09 (syst) m/s²`, pull `-0.24σ` vs. the
reference value `g_ref = 9.783 m/s²`.

## Quick start

```bash
python main.py config_run1.yaml
```

Add a new height's phyphox export by dropping the file into `data_dir`
(see the relevant `config*.yaml`) and adding one `height: filename` line
under `height_files` — no code changes needed. Re-run the same command.

## Repository layout

```
main.py                        config-driven pipeline runner (entry point)
combine_sessions.py             pools two independent sessions' trials per height
run1_filter_optimization.py     grid-searches the Physical Consistency Filter's
                                tolerances directly on Run 1 (cross-height chi2/ndf
                                objective); produces the values in config_run1.yaml
optimize_filter_params.py       earlier, superseded exploration: tunes the same
                                filter on the pilot dataset only (train/validation
                                split) -- kept for the negative result it produced
                                (see report Appendix A: it overfit and did not
                                generalise to Run 1)
config.yaml               pilot/test-data run (10 heights x ~5 repeats)
config_run1.yaml          primary dataset (10 heights x ~15-18 repeats)
config_run2.yaml          a second, independent person's dataset (benchmark only)

src/
  io_utils.py              load phyphox Audio Amplitude / Audio Scope exports
  cleaning.py              trim handling noise, drop NaNs
  noise_characterization.py  noise-floor stats, detection threshold, aliasing ceiling
  bounce_detection.py      two independent bounce-detection methods + Physical
                           Consistency Filter (screens implausible Delta t_n sequences)
  segmentation.py          split one continuous recording into per-drop trials
  fitting.py               weighted least squares, g extraction, stat/syst
                           uncertainty propagation, PDG-style combination
  pipeline.py              orchestrates the above per trial / per height
  synthetic_demo.py        SYNTHETIC data generators, used only for validation

data/                     raw phyphox exports + processed interval summaries
figures/                  all generated plots, namespaced per dataset
notebooks/                 pipeline walkthrough + synthetic validation notebooks
```

## Method summary

1. **Clean** — trim start/end handling noise, drop NaNs.
2. **Threshold** — set from a dedicated quiet background recording
   (`mean + 5σ` of the noise floor), never reused across sessions.
3. **Detect bounces** — two independent methods (peak-picking and
   threshold-crossing), cross-checked against each other.
4. **Segment** — split a continuous multi-drop recording into per-trial
   groups.
5. **Physical Consistency Filter** — screen each trial's own
   `Δt_n` sequence for physical plausibility (absolute ceiling + ratio
   ceiling). Tolerances are selected by a grid search run directly on the
   reported dataset (`run1_filter_optimization.py`), using cross-height
   `chi2/ndf` — not agreement with the reference value — as the objective,
   so the selection isn't circular. An earlier attempt tuned these on the
   pilot dataset instead as a held-out split; it overfit and didn't
   generalise (see `optimize_filter_params.py` and report Appendix A).
6. **Fit** — weighted linear regression of `ln(Δt_n)` vs. `n` per trial,
   with a shared-slope rescue for trials too thin (1 interval) to fit
   alone.
7. **Combine** — statistical uncertainty is inverse-variance weighted
   across repeats and heights (GUM/JCGM quantisation uncertainty on
   timing); systematic (height) uncertainty is evaluated once per height
   (shared, correlated across repeats) and combined independently across
   heights.

Every pipeline component was validated against synthetic data with known
ground truth before being trusted on real recordings — see
`src/synthetic_demo.py` and `notebooks/analysis.ipynb`.

## Notes

- The written report and presentation slides for this course are kept
  local (not part of this public repository) per the submission
  requirements.
- `config.yaml` (pilot dataset) and `config_run2.yaml` (an independent
  student's dataset, same method) are kept for validation/benchmark
  purposes only — the reported result comes from `config_run1.yaml`.
