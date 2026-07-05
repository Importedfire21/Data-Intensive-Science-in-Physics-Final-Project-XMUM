# Process Overview: Measuring *g* via the Bouncing-Ball Method

This document explains, end to end, how the analysis pipeline in this
project works, why each step exists, and everything that has actually been
found running it against real data — including things that went wrong and
why. It is the exhaustive technical reference; for a short, results-focused
status snapshot, see `report/project_overview.md` instead.

**Current state in one sentence:** the pipeline (validated against
synthetic ground truth throughout) is config-driven via `main.py`, uses
reliability-weighted per-trial fitting plus a Physical Consistency Filter
(§4.5a–4.7) that squeezes real information out of thin/imperfect trials,
tracks statistical and systematic uncertainty **separately** throughout
(§4.6a), and now has **two datasets behind it**: an initial pilot
("test-data run", §8) that found and fixed most of the pipeline's bugs but
never reached good cross-height compatibility, and a deliberately
recollected, larger dataset ("Full Data Run 1", §11) whose central value
agrees well with the reference but whose cross-height
$\chi^2/\mathrm{ndf}$ is honestly worse than an earlier, looser-uncertainty
version of this analysis reported — $g=9.58\pm0.36\,\mathrm{(stat)}
\pm0.08\,\mathrm{(syst)}\ \mathrm{m/s^2}$ (total $\pm0.37$),
$\chi^2/\mathrm{ndf}=3.41$, across all 10 heights (§11.2, §4.6a explains
why this number changed).

## 1. Physical model

A ball dropped from height $h_0$ bounces repeatedly, losing a fraction of
its energy each impact (coefficient of restitution $e < 1$). The time
between consecutive bounces shrinks geometrically:

$$\Delta t_n = A\,e^{n}, \qquad A = 2\sqrt{\frac{2h_0}{g}}$$

Taking logs linearises this:

$$\ln(\Delta t_n) = n\ln(e) + \ln A$$

A weighted linear fit of $\ln(\Delta t_n)$ against bounce number $n$ gives
the intercept $\ln A$, from which:

$$g = \frac{8h_0}{A^2}$$

Repeating at several heights (and several repeat drops per height) gives
independent estimates of $g$, combined via inverse-variance weighting —
the same method the Particle Data Group uses for world-average constants:

$$\bar g = \frac{\sum_i g_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}, \qquad
\sigma_{\bar g} = \frac{1}{\sqrt{\sum_i 1/\sigma_i^2}}$$

with a $\chi^2$ compatibility check across the combined measurements:
values that disagree by more than their quoted uncertainties allow push
$\chi^2/\mathrm{ndf}$ above 1, which is a warning that something
unmodelled is going on — not just a bookkeeping statistic.

## 2. Two recording modes, one unified pipeline

phyphox offers two microphone tools, with very different tradeoffs:

| | Audio Amplitude | Audio Scope |
|---|---|---|
| What it records | dB sound-level envelope | raw microphone waveform |
| Sample interval (measured, real device) | ~123 ms | ~21 μs (≈48 kHz) |
| Duration per capture | full session | **~0.5 s only** (confirmed empirically — see §6) |
| Role in this project | main multi-bounce sequence per trial | single high-resolution first-bounce cross-check |

Both are converted to the same internal shape — a `(time_s, spl_db)`
table — so every downstream step (cleaning, detection, fitting) is
identical regardless of source. Audio Scope's raw waveform is converted to
an amplitude envelope via a rolling RMS window (`src/envelope.py`) before
it looks like an Audio Amplitude trace. This is why the codebase never
needs an `if scope: ... else: ...` branch past the ingestion step.

Everything actually analysed in Runs 1 (§11) is Audio Amplitude; Audio
Scope was explored on earlier real files (§6, findings 1–3) but has not
yet been re-run against Full Data Run 1.

## 3. Codebase map

```
src/
  io_utils.py                load/auto-detect Audio Amplitude & Audio Scope exports
  envelope.py                 raw waveform -> RMS envelope (for Audio Scope)
  cleaning.py                  trim start/end handling noise, drop NaNs
  noise_characterization.py    noise-floor stats, detection threshold, resolvable-bounce-count
  bounce_detection.py          two independent peak-detection methods + agreement check + Physical Consistency Filter
  segmentation.py               split one continuous recording into per-trial groups
  fitting.py                    weighted least squares, g extraction, PDG combination, reliability weighting
  pipeline.py                   process_trial() / process_height_recording() / process_multi_drop_recording()
  synthetic_demo.py             SYNTHETIC data generators, for validation only

config.yaml                     test-data run config (pilot dataset, §8)
config_run1.yaml                Full Data Run 1 config (current primary dataset, §11)
main.py                         config-driven runner; usage: python main.py [config_file]
                                 NOTE: no-argument `python main.py` defaults to config.yaml
                                 (the OLD test-data config), not config_run1.yaml -- always
                                 pass the config explicitly when working with Run 1.

notebooks/
  analysis.ipynb                full pipeline + synthetic validation (submission notebook)
  real_data_analysis.ipynb      real data only, no synthetic cells (pre-Run-1 exploration)
build_notebook.py               assembles analysis.ipynb from cell definitions
build_real_notebook.py          assembles real_data_analysis.ipynb

data/
  raw/                          test-data run: real phyphox exports (background, Scope tests, 10 height trials)
  raw/run1/                     Full Data Run 1: its own background + 10 height trials
  processed/, processed/run1/   derived tables (interval_summary.csv) per dataset -- regenerated by main.py

figures/, figures/run1/         PNGs saved by main.py and the notebooks, per dataset:
  background_before_cleaning.png, noise_floor.png   background diagnostics
  raw_traces/{h0}_raw_trace_peaks.png                full trial trace + every raw detected impact
  per_height/{h0}_segments.png                       trial-boundary diagram for that height
  per_height/{h0}_trial{i}[_rescued].png              each trial's own fit + residual plot
  intervals/{h0}_dt_intervals.png                     Delta t_1/2/3 across that height's repeats
  dt_intervals_combined.png, g_vs_height.png          cross-height summary plots
  uncertainty_budget.png                              stat vs. syst sigma_g per height (bar chart)

report/
  data_collection_protocol.md   how to record trials so they slot into the pipeline
  process_overview.md           this document (full technical detail)
  project_overview.md           short results-focused status snapshot
  results_summary.csv           test-data run's per-height + overall g
  run1_results_summary.csv      Full Data Run 1's per-height + overall g (the current headline numbers)
```

**Two ways to run the pipeline, for two different purposes:**

- **`python main.py <config>`** — the fast-iteration tool. Add a new
  height's file to the config's `data_dir`, add one `height: filename`
  line to `height_files`, re-run. No code or notebook edits needed. Every
  real number in §8 and §11 came from this.
- **The notebooks** — the polished, explanatory deliverable for the actual
  report/submission (pipeline walkthrough + synthetic validation prose).
  Neither `.ipynb` is hand-edited; both are generated from their `build_*.py`
  script:
  ```
  python build_notebook.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
  ```
  (same pattern for `build_real_notebook.py`). These predate Full Data Run
  1 and have not been regenerated against it — they demonstrate the
  pipeline and its synthetic validation, not the current headline result.

### Config-driven, multi-run design

`main.py` takes every path and parameter from a YAML config rather than
hardcoding them, specifically so a second (or third) data-collection pass
doesn't require touching code: `config_run1.yaml` duplicates
`config.yaml`'s structure with `data_dir`, `figures_dir`, `results_csv`,
and `processed_dir` all pointed at a `run1`-namespaced set of paths, so
the two datasets' outputs can never overwrite or mix with each other. Any
future run (Run 2, etc.) follows the same pattern: copy the config, point
its four output/input paths at a new namespace, populate `height_files`.

## 4. Step-by-step pipeline

### 4.1 Ingestion (`io_utils.load_recording`)

Auto-detects Audio Amplitude vs Audio Scope from the file's columns/sheets
and returns a standard `(time_s, spl_db)` DataFrame either way.

- **Audio Amplitude**: columns `Time (s)`, `Sound pressure level (dB)`.
- **Audio Scope**: a multi-sheet `.xls` workbook (`Audio data` /
  `Metadata Device` / `Metadata Time`), with the samples on the `Audio
  data` sheet using columns `Time (ms)`, `Recording (a.u.)` — confirmed
  from a real export (phyphox 1.2.0, Samsung SM-A556E). The raw waveform is
  passed through `envelope.compute_envelope()` (rolling RMS over a
  configurable window) to produce a comparable dB-like trace.

Column-name and sheet-name candidates are kept as small, documented lists
at the top of `io_utils.py` specifically so a different phone/phyphox
version's export can be supported by adding one line, not by rewriting the
loader.

### 4.2 Cleaning (`cleaning.trim_and_clean`)

Drops the first/last N seconds of a recording and any NaN rows, and
reports a retention rate. The N seconds matters for a real, physical
reason: starting/stopping a phyphox recording means reaching for the
phone, which contaminates the signal with handling noise that has nothing
to do with the room's ambient floor or a ball bounce. The trim window is
recording-length-dependent — ~10 s each side for a several-minute
background capture is negligible; the same 10 s would delete all the real
bounce data from a much shorter trial file, so trials use ~1.5–2 s instead
(both values are `cleaning.*_trim_s` entries in the config).

### 4.3 Noise characterisation & threshold (`noise_characterization.py`)

`characterise()` computes the noise floor (mean/std, or a MAD-based robust
estimate for `robust=True`) from a **dedicated quiet recording**, and sets
a detection threshold at `mean + k_sigma * std`. This must never be
computed from a recording that contains the actual signal — doing so
contaminates the estimate (§7, bug #2). Nor should it be reused across
recording *sessions* without checking — §11.1 documents a real case where
reusing an old background's threshold for a new session badly miscalibrated
detection, because the new session's ambient noise floor was genuinely
~8 dB louder. `characterise_excluding_window()` is the fallback for when
only one capture (signal + surrounding quiet) exists at all, as with a
single short Audio Scope buffer.

`resolvable_bounce_count(profile, A_estimate, e_estimate, nyquist_factor)`
answers a different question: given the recording's *actual measured*
sampling interval, how many bounces can be resolved before $\Delta t_n$
drops below a Nyquist-like floor and aliases? This is computed **per
height** (via the config's `restitution_estimate` and each height's own
$A$ estimate) and used as that height's `max_bounce_n` ceiling, rather
than one fixed number applied to every height — a fixed cap is either too
strict for large, slow-bouncing drops or too lax for small, fast ones.

### 4.4 Bounce detection (`bounce_detection.py`)

Two independent methods, because the rubric requires the key observable to
be extracted at least two ways and compared:

- **Method A — peak picking**: local maxima above the threshold
  (`scipy.signal.find_peaks`), with both a minimum-separation and a
  **prominence** requirement. The prominence filter is not cosmetic — see
  §7, bug #3, which explains a real failure mode it prevents.
- **Method B — threshold crossing**: first rising edge above the
  threshold per impulse, with a refractory period.

`compare_methods()` matches detections between the two and reports a mean
timing offset. A small, *consistent* non-zero offset (Method B leads
Method A by roughly the impulse's rise time) is physically expected and
should be reported as a timing systematic, not treated as disagreement.

### 4.5 Multi-drop segmentation (`segmentation.py`)

If several drops are recorded in one continuous file (the actual recording
pattern used for every height trial in both datasets — see §6, §11),
`segment_by_gaps()` splits the single list of already-detected impact
times into per-trial groups wherever the gap between consecutive impacts
exceeds `min_gap_s` (i.e. the ball has come to rest and the experimenter is
resetting for the next drop). Any group with fewer than
`min_impacts_per_group` (default 2) impacts is dropped — a lone detection
cannot form any $\Delta t$ at all, and is far more likely to be an isolated
noise transient than a genuine, otherwise-silent trial.

**Trial count is not assumed.** Every "N bounce" filename was expected to
contain N repeated drops, and most do — but the pipeline counts usable
segments from the data itself for every height rather than hard-coding a
number, because both datasets have produced at least one file where the
"expected" count didn't quite match what the data actually supported
(test-data run: 1.1 m segmented into 6, not 5, §6 finding 5).

**A stable segment count across a range of `min_gap_s` is itself a data
quality signal, not just a technical detail.** The test-data run's files
were stable across `min_gap_s` = 1.5–3.0 s; §11.1 found that Run 1's
pacing was *not* as cleanly bimodal for at least one file, which mattered
for interpreting that file's results.

**Important implementation detail:** detection runs exactly **once**, on
the whole cleaned recording; segmentation only *groups* those results. It
never re-runs peak detection on a cropped slice of the data. See §7, bug
#4 for the real, data-validated reason this distinction matters.

### 4.5a Physical Consistency Filter (`bounce_detection.filter_plausible_bounce_times`)

Segmentation only decides *which* detected impacts belong to which trial —
it says nothing about whether a trial's own impacts are physically
sensible. A missed bounce (detection skips one real impact) merges two
real intervals into one apparent $\Delta t_n$ that is far too large; that
single corrupted point is enough to badly bias an otherwise-good trial's
fit (§9.1, first example found this way). This stage screens for exactly
that, before any interval reaches the fit, using two independent checks
(chosen deliberately over more elaborate options — see the note below):

1. **Absolute plausibility ceiling.** $\Delta t_n$ must not exceed
   $A e^n \times$ `abs_tolerance_factor` (the theoretical model prediction,
   scaled by a generous tolerance). This is the *only* check of the two
   that can catch a corrupted-but-still-decreasing sequence: a real example
   found in this data, $[\Delta t_1,\Delta t_2,\Delta t_3] = [1.34, 0.49,
   0.24]\,\mathrm{s}$, is still strictly *decreasing* even though 1.34 s is
   roughly 2–3× too large for any plausible drop height here — a pure
   ratio/monotonicity check alone would not flag it.
2. **Ratio ceiling.** $\Delta t_{n+1}/\Delta t_n$ must not exceed
   `ratio_ceiling` (the model predicts this ratio equals $e<1$, so a value
   meaningfully above 1 means the sequence increased where the physics says
   it must decrease). Catches a sudden jump mid-sequence.

Both tolerances are deliberately loose — the goal is to catch obviously
broken intervals, not to prune ordinary measurement scatter — and both were
tuned against **synthetic** data before touching real trials:
`ratio_ceiling=1.4` and `abs_tolerance_factor=1.7` together give a ~1.5%
false-positive rate under a stress-test 15% per-interval noise level, while
both real corrupted-interval patterns found in the test-data run (a stray
leading blip inflating $\Delta t_1$, and a genuine mid-sequence jump) are
still caught cleanly. In Full Data Run 1, this same filter routinely trims
5–11 corrupted intervals per height (§11) out of ~15–18 detected trials —
a real, active part of getting that dataset's good result, not a vestige
of test-run bug-fixing.

If the very first interval is the corrupted one (as in the example above),
truncating from the start would throw the whole trial away. The filter
instead tries dropping 0, 1, or 2 *leading* impacts (a stray detection
before the real bounce sequence starts is a plausible cause) and keeps
whichever starting point survives the longest run of plausible intervals.
**What it deliberately does not attempt:** repairing a genuinely missed
bounce *mid-sequence* by relabelling every later interval's bounce number
by one — that requires guessing how many bounces were missed, which is
unverifiable from the dB envelope alone. Once a mid-sequence violation is
found, the trial is simply truncated there, keeping only the impacts before
it.

*Why not the fancier alternatives (confidence scores, morphological
peak-shape filtering, an automatic re-detect-and-refit feedback loop)?*
These were considered and set aside as disproportionate for this project's
scope — the two-check design above already fixes the concrete failures
found in this data, is fully explainable in a couple of sentences, and was
itself validated the same way as everything else in this pipeline
(synthetic ground truth first, §5).

### 4.6 Fitting and g extraction (`fitting.py`, `pipeline.fit_bounce_sequence`)

Weighted least-squares fit of $\ln(\Delta t_n)$ vs $n$ (closed-form,
2-parameter) for each trial independently. Weights come from a per-impact
timing uncertainty $\sigma_t = \Delta t_\mathrm{sample}/\sqrt{12}$
(§4.6a), propagated through the interval difference and the log,
**optionally further inflated by `fitting.reliability_weight(n, power)`**
(§4.7). The fit reports slope, intercept, their uncertainties,
$\chi^2/\mathrm{ndf}$, and residuals.

`fitting.g_stat_syst_from_intercept()` propagates the intercept
uncertainty and the height uncertainty into $g$'s **statistical** and
**systematic** contributions *separately* (§4.6a) via
$g = 8h_0/A^2 = 8h_0 e^{-2\cdot\mathrm{intercept}}$; `g_from_intercept()`
is the same calculation with the two combined in quadrature, kept only for
call sites (the notebooks) that need a single number.

### 4.6a Statistical vs. systematic uncertainty, and the height/timing error models

Two structurally different uncertainty sources feed into every $g$
estimate, and from this version of the pipeline onward they are tracked
**separately** rather than added into one blended number at the point of
computation:

**Statistical — per-impact timing precision.** Each detected impact time
is only known to within one sample of the recording (`sample_dt`, ≈123 ms
for Audio Amplitude). Treating the true impact time as uniformly
("rectangularly") distributed within that sample — the standard Type B
evaluation for a digitised/quantised measurement (GUM/JCGM 100:2008
§4.3.7) — gives a standard uncertainty of
$\sigma_t = \Delta t_\mathrm{sample}/\sqrt{12}$
(`fitting.resolution_uncertainty`). This **replaces an earlier ad hoc
$\Delta t_\mathrm{sample}/2$ convention**, which was actually the uniform
distribution's *half-width*, not its standard deviation, and overstated
$\sigma_t$ by a factor of $\sqrt{3}\approx1.73\times$. Because this is a
uniform rescaling of every weight within a trial's weighted least-squares
fit, it leaves the fitted slope/intercept/$g$ **exactly unchanged**
(verified against synthetic data with known ground truth — central values
matched $g_\mathrm{true}$ to machine precision before and after the
change) while shrinking every *reported* statistical uncertainty by that
same $1/\sqrt3$ factor. Since real-world scatter between trials obviously
did not shrink to match, $\chi^2/\mathrm{ndf}$ **increased** correspondingly
wherever residual scatter — not just quoted uncertainty — dominates
(§11.2's overall $\chi^2/\mathrm{ndf}$ went from 1.13 to 3.41 this way).
This is being reported as-is: the earlier, better-looking
$\chi^2/\mathrm{ndf}$ was an artefact of an over-generous uncertainty
convention, not evidence the data were better behaved.

Because it is an independent draw per trial (different detected impact
times each drop), the statistical term is combined inverse-variance
weighted across repeats at a height, and shrinks with more repeats —
`fitting.combine_repeats_stat_syst`.

**Systematic — drop-height measurement.** Height is assumed known to
$h_0 \pm 0.02\,\mathrm{m}$ (`height_uncertainty_m: 0.02` in both configs)
for every trial. Two points about this term specifically:

- *Ball-radius correction is not needed.* Heights were measured from the
  **bottom of the ball to the surface**, not centre-of-mass to surface.
  With this convention the centre of mass starts at $h_0+r$ and ends at
  $r$ once the ball's bottom reaches the surface — the radius contributes
  equally to both ends and cancels, so the centre of mass still falls
  exactly $h_0$ regardless of ball radius. This was explicitly considered,
  not overlooked, and needs no additional correction term.
- *It does not average down the way the statistical term does.* Every
  repeat drop at a given height shares the exact same $h_0$ and $h_0$
  measurement — the systematic is fully correlated across those repeats,
  not an independent draw. `combine_repeats_stat_syst` therefore evaluates
  it **once** per height (from the combined $g$ and the common relative
  height uncertainty) rather than inverse-variance-combining $N$ copies of
  what is really one number — doing the latter would understate it by a
  spurious $\sqrt N$. When combining *across* heights instead,
  `combine_heights_stat_syst` treats the systematic as independent again
  (each height is a separately measured distance), propagating it through
  the same inverse-variance weights as the statistical part.

Every per-trial and per-height result (`pipeline.fit_bounce_sequence`,
`_fit_single_interval_with_shared_slope`, `combine_height_result`) now
carries `g_stat_err` and `g_syst_err` as separate fields, and
`fitting.CombinationResultStatSyst` (with a `.total_err` convenience
property for the quadrature sum) is the return type used everywhere a
combined value is needed. `main.py` reports both terms in its per-height
and overall print statements, in `results_csv`'s
`g_stat_err`/`g_syst_err`/`g_total_err` columns, and visually in a
dedicated `figures/{run}/uncertainty_budget.png` bar chart comparing the
two contributions per height (see §4.8).

### 4.7 Handling trials with too few intervals to fit alone

A 2-parameter line needs ≥2 points (≥2 intervals, ≥3 impacts) to be
determined at all. A trial with only 2 detected impacts (1 interval)
is *unidentifiable* on its own -- the original pipeline reported `g=nan`
for it and discarded it entirely, wasting a real, if limited, measurement.
Two changes address this without simply forcing more data through a
model that no longer fits it:

- **`fitting.reliability_weight(n, power)` = $1/n^{\text{power}}$**
  smoothly down-weights later bounces *within* a trial's own fit, on top
  of the timing-based weight it already had. This reflects a real,
  separate risk from pure timing precision: the restitution assumption
  and detection SNR both degrade further with each successive bounce (see
  §6, finding 6). A pure pooling approach (every trial's points merged
  into one shared-intercept fit) was tried first and **rejected** — one
  bad interval in one trial then corrupts the fit for every trial, whereas
  keeping trials independent lets a bad trial's own $\chi^2$/uncertainty
  absorb its own anomaly instead of contaminating its neighbours.
- **Shared-slope rescue** (`pipeline._fit_single_interval_with_shared_slope`):
  for a trial with *exactly* 1 interval, the slope ($\ln e$) is borrowed
  from that height's other, well-constrained trials (inverse-variance
  averaged), and only the intercept is solved from the thin trial's single
  point. Its uncertainty combines the point's own timing uncertainty with
  the borrowed slope's uncertainty -- a rescued trial is never claimed to
  be as precise as a properly-constrained one, and is labelled
  `method: shared_slope_rescue` in every result so it's never silently
  conflated with a normal fit.

Both were validated against synthetic data (including a deliberately
truncated "thin" trial) before use on real data: the rescue mechanism
recovers a sensible g with appropriately larger uncertainty, and does not
crash or silently misbehave when no well-constrained sibling trial exists
(it is skipped and reported, not guessed). In Full Data Run 1 this rescues
1–7 trials per height depending on height (§11's table).

### 4.8 Combination (`fitting.combine_repeats_stat_syst` / `combine_heights_stat_syst`) and the `main.py` workflow

Inverse-variance weighted average of the statistical term (with the
systematic term handled as described in §4.6a), used twice: across trials
within one height (`pipeline.combine_height_result`, via
`combine_repeats_stat_syst`), and across the per-height results for the
final number (`combine_heights_stat_syst`, in `main.py`). Both calls also
produce a $\chi^2/\mathrm{ndf}$ compatibility statistic computed from the
statistical uncertainty (the systematic contributes no trial-to-trial
scatter to check against). **A small formal uncertainty on the combined
value does not mean the combination is trustworthy** — §8 is a real,
worked example of this going wrong, and §4.6a/§11.2 are a second example
in the opposite direction (tighter, more honest uncertainties revealing
real disagreement that a looser convention had hidden). `pdg_combine` (the
older, single-uncertainty version) remains in `fitting.py` only for the
notebooks' backward compatibility — `main.py` and `pipeline.py` no longer
call it.

`main.py` runs this whole chain end to end from its config file: loads the
background and every configured height file, computes each height's own
`max_bounce_n` (§4.3), fits and combines each height
(`process_height_recording` + `combine_height_result`), combines across
heights, prints a verdict based on the overall $\chi^2/\mathrm{ndf}$, and
saves, per height:

- `figures/{run}/background_before_cleaning.png`, `noise_floor.png` —
  background diagnostics (once per run, not per height).
- `figures/{run}/raw_traces/{h0}_raw_trace_peaks.png` — the full cleaned
  trial trace, the detection threshold, and every *raw* detected impact
  (before the consistency filter runs) — a visual sanity check independent
  of segmentation/fitting.
- `figures/{run}/per_height/{h0}_segments.png` and
  `{h0}_trial{i}[_rescued].png` (only if `plots.save_per_height: true` —
  off by default since it is verbose across many heights/trials) — the
  segmentation diagram and every individual trial's own fit + residual
  plot.
- `figures/{run}/intervals/{h0}_dt_intervals.png` and
  `dt_intervals_combined.png` — $\Delta t_1$/$\Delta t_2$/$\Delta t_3$
  diagnostics, per height and across all heights.
- `data/processed/{run}/interval_summary.csv` — every trial's raw
  $\Delta t_1$, $\Delta t_2$, $\Delta t_3$, with its height, label, and fit
  method.
- `figures/{run}/g_vs_height.png` — per-height and overall $g$, with two
  nested error bars (thin/outer = stat+syst total, thick/inner =
  statistical only) so the systematic's relative contribution is visible
  directly on the summary plot, not just in the CSV.
- `figures/{run}/uncertainty_budget.png` — a bar chart of statistical vs.
  systematic $\sigma_g$ per height, the dedicated view of the §4.6a
  uncertainty split requested for the error analysis.
- `report/{run}_results_summary.csv` — the final per-height and overall
  result, now with `g_stat_err`/`g_syst_err`/`g_total_err` as separate
  columns rather than one blended `g_err`.

## 5. Validation strategy: synthetic data with known ground truth

Before trusting any of the above on real recordings, every piece is
exercised against **synthetic** data generated from the known physical
model (`synthetic_demo.py`): known true $g$, known bounce times, known
drop heights, realistic injected noise. Because the answer is known in
advance, recovery error can be checked directly and bugs show up as "the
code returns the wrong number" rather than "the code returns *a* number
that might be wrong for reasons indistinguishable from real physical
messiness." Synthetic numbers are never reported as measurements —
`synthetic_demo.py`'s docstring says so explicitly, and every notebook
cell using it is labelled.

This is not a formality: **every bug in §7, the pooling-vs-per-trial
design decision in §4.7, and the Physical Consistency Filter's tolerance
values in §4.5a were all settled by synthetic validation** before they
could affect a real result.

## 6. Real data collected: the test-data run

| File(s) | What it is |
|---|---|
| `background_noise.xls` | Audio Amplitude, quiet room, ~10 min — sets the detection threshold used everywhere else in this dataset |
| `scope_noisy_one_bounce_test.xls` | Audio Scope, one bounce, deliberately noisy room — SNR feasibility test |
| `scope_0.3m_test.xls` | Audio Scope, single 0.3 m drop, known height, clean isolated spike |
| `amp_{0.3, 0.4, ..., 1.2}m_5bounce_test.xls` | Audio Amplitude, ~5 repeated drops per height, 10 heights spanning 0.3–1.2 m in 0.1 m steps |

Key findings, roughly in the order they were discovered:

1. **Audio Scope's export is a ~0.5 s rolling buffer**, not the full
   session, regardless of how long phyphox recorded — confirmed by
   comparing a real export's sample count/duration (24000 samples, 500 ms)
   against the session metadata (a ~10 s START-to-PAUSE window). This caps
   Audio Scope to a single-impact (or, for small enough heights,
   occasionally two-impact) role in this project — see §2.
2. **A single confirmed impact cannot give g** without a known height
   ($h_0$ appears directly in $g=8h_0/A^2$) and without a second confirmed
   impact (needed for any $\Delta t$ at all). The 0.3 m Audio Scope test,
   with a known height and one clean isolated spike, gave only an
   illustrative single-point free-fall estimate
   ($g \approx 6.6\ \mathrm{m/s^2}$, biased by the unverifiable assumption
   that "recording start = release") — explicitly not a citable result.
3. **Predicting the full bounce timeline** for that 0.3 m case showed that
   no second bounce could ever fit inside a 0.5 s Audio Scope buffer for
   any plausible coefficient of restitution (0.70–0.90) — turning a
   qualitative limitation into a quantitative one: the maximum drop height
   for even a 2nd bounce to fit is ≈0.17–0.20 m.
4. Every "5 bounce" Audio Amplitude filename turned out to mean **5
   repeated drops from that height**, not one drop bouncing 5 times —
   confirmed by segmentation being stable (same trial count) across a wide
   range of `min_gap_s` (1.5–3.0 s), with multi-second silent gaps clearly
   visible between trials in the raw trace.
5. **1.1 m genuinely segments into 6 usable trials**, not 5 — all 6 have
   ≥2 real, well-separated impacts. The pipeline handles this by counting
   trials per file rather than assuming 5 everywhere (§4.5).
6. **Checking whether $\Delta t_n$ actually decreases monotonically**
   (as the model strictly requires) across the full 10-height dataset
   showed violations at *every* height once all detected impacts are used
   without restriction — motivating both the per-height aliasing ceiling
   (§4.3) and the reliability down-weighting (§4.7).
7. **The raw $\Delta t_1$/$\Delta t_2$/$\Delta t_3$ values, averaged across
   repeats, follow the expected physical trend with height** (roughly
   increasing, since $A\propto\sqrt{h_0}$). This matters: it shows the bad
   g values in §8 are not caused by generally-broken detection, but by
   specific corrupted intervals in specific trials (§9.1) plus genuine
   restitution variability (§9.2).

## 7. Bugs caught during validation (and why each one mattered)

1. **Threshold contamination from handling noise.** Computing the
   background noise profile *before* trimming start/end handling noise
   inflated the estimated std by >2×, desensitising the detection
   threshold well past where it should sit. Fixed by always cleaning
   before characterising.
2. **Threshold contamination from the signal itself.** An early version
   computed the noise profile from the *trial* recording (which contains
   the bounce), not a dedicated background. The loud transient inflated
   the std enough to push the threshold above real peaks, causing zero
   detections. Fixed by requiring a separate quiet reference (or
   `characterise_excluding_window()` when only one capture exists at all).
3. **False double-detection from floating-point ripple.** On a smooth RMS
   decay tail, tiny numerical ripple creates enormous numbers of
   infinitesimal local maxima. `scipy.find_peaks`'s `distance` parameter
   only enforces *spacing* between accepted peaks — it does not require a
   candidate to be a genuinely separated bump. Without a prominence floor,
   the algorithm could pick a numerical blip 30 ms after a real peak as a
   fake second impact, once that blip fell outside the `distance`
   exclusion zone. Fixed by adding a `prominence_db` requirement.
4. **Segmentation re-detection edge effect.** An early version re-ran peak
   detection on each *sliced* per-trial DataFrame rather than reusing
   whole-file detections. This silently dropped real bounces near a
   segment's boundary, because `scipy.find_peaks`'s prominence calculation
   depends on the array's edges — cropping the array changes the computed
   prominence of peaks near the cut. Fixed by detecting once on the full
   recording and only grouping the resulting times for segmentation.
5. **Pure interval pooling looked appealing but was worse.** Pooling every
   trial's $(n, \Delta t_n)$ points into one shared-intercept fit was
   tried as a way to use thin trials; validated against synthetic data, it
   turned out that a single bad interval in any one trial corrupts the
   *shared* fit for every trial, whereas independent per-trial fits
   contain that trial's own anomaly within its own inflated uncertainty.
   Replaced with the per-trial + shared-slope-rescue design in §4.7.
6. **Reused background threshold across recording sessions (Full Data Run
   1, §11.1).** Reusing an old session's background to calibrate a new
   session's threshold silently assumed the ambient noise floor hadn't
   changed. It had — by ~8 dB — causing severe over-detection (92 "impacts"
   where ~35–40 were expected) and a badly wrong initial result
   ($g=3.17\pm0.82$, pull $-8\sigma$). Fixed by always recording (and
   using) a dedicated background per session; see §11.1 for the full
   diagnostic process, which is worth reading even though the fix itself
   is simple, because the *symptoms* (a continuous, non-bimodal gap
   distribution; wildly varying $\Delta t_1$) initially looked like a data
   quality problem with the drops themselves, not a threshold problem.

Every one of these was caught by comparing pipeline output against
**synthetic data with a known answer**, or by direct, systematic
comparison against a known-good reference (the old background, in bug #6),
before the corresponding real-data section was trusted. None were found by
inspection alone.

## 8. Test-data run: final status (superseded as a result by Run 1, §11)

Per-height combined $g$ from `python main.py config.yaml` (reliability-
weighted, per-height `max_bounce_n` from `resolvable_bounce_count`,
Physical Consistency Filter enabled at `abs_tolerance_factor=1.7`,
`ratio_ceiling=1.4`, ±2 cm assumed height uncertainty, resolution-based
timing uncertainty — §4.6a; see `report/results_summary.csv` for the live
numbers):

| $h_0$ (m) | $g$ (m/s²) | stat | syst | $\chi^2/\mathrm{ndf}$ | Pull vs. $g_\mathrm{ref}$ | Trials (full + rescued) |
|---|---|---|---|---|---|---|
| 0.3 | 8.75  | ±3.03 | ±0.58 | 0.22 | −0.34σ | 5 + 0 |
| 0.4 | 5.92  | ±1.78 | ±0.30 | 0.93 | −2.14σ | 3 + 2 |
| 0.5 | 5.32  | ±1.74 | ±0.21 | 0.69 | −2.54σ | 5 + 0 |
| 0.6 | 10.85 | ±1.59 | ±0.36 | 1.07 | +0.66σ | 4 + 1 |
| 0.7 | 15.44 | ±2.91 | ±0.44 | 1.84 | +1.92σ | 4 + 1 |
| 0.8 | 11.33 | ±2.32 | ±0.28 | 1.22 | +0.66σ | 4 + 1 |
| 0.9 | **3.42** | ±0.61 | ±0.08 | 3.37 | −10.30σ | 5 + 0 |
| 1.0 | 10.24 | ±2.38 | ±0.20 | 1.38 | +0.19σ | 5 + 0 |
| 1.1 | **3.06** | ±0.45 | ±0.06 | 7.34 | −14.78σ | 5 + 1 |
| 1.2 | 12.80 | ±2.27 | ±0.21 | 0.82 | +1.33σ | 5 + 0 |

**Overall combined: $g = 4.39\pm0.33\,\mathrm{(stat)}\pm0.04\,
\mathrm{(syst)}\ \mathrm{m/s^2}$ (total $\pm0.33$), $\chi^2/\mathrm{ndf} =
8.21$, pull $= -16.4\sigma$ vs. $g_\mathrm{ref}=9.783$.** Both the
compatibility score and the pull are worse than an earlier, looser-
uncertainty version of this table reported ($\chi^2/\mathrm{ndf}=2.74$,
pull $-9.6\sigma$) — expected, since tighter uncertainties (§4.6a) reveal
the same underlying scatter more starkly rather than changing it. This
**must not** be quoted as a result regardless of which uncertainty
convention is used. Central values are unchanged from before: several
heights (0.6, 0.8, 1.0, 1.2 m) still land close to the reference, while
0.9 m and 1.1 m remain badly off (§9). This run's value going forward is
as a **pipeline-validation and lessons-learned dataset** (§9, §7 bug #6's
lead-up), not as a source of final numbers — Full Data Run 1 (§11)
replaces it for that purpose.

## 9. Two distinct root causes behind the test-data run's incompatibility

### 9.1 Corrupted individual intervals (fixed by the Physical Consistency Filter, §4.5a)

The per-height interval plots made this mechanism visible. Example,
h0 = 0.7 m, before filtering:

| Trial | $\Delta t_1$ (s) | $\Delta t_2$ (s) | $\Delta t_3$ (s) |
|---|---|---|---|
| 0 | 0.36 | 0.61 | **1.20** |
| 1 | **1.34** | 0.49 | 0.24 |
| 2 | 0.49 | 0.36 | 0.48 |
| 3 | 0.36 | 0.48 | 0.36 |
| 4 | 0.49 | 0.36 | 0.23 |

Trial 1's $\Delta t_1=1.34\,\mathrm{s}$ is roughly 3× every sibling trial's
$\Delta t_1$, while its own $\Delta t_2$/$\Delta t_3$ look completely
normal — almost certainly a missed first bounce merging two real intervals
into one apparent one. The same pattern, at a narrower margin (~1.9×),
appeared at 0.8 m and had slipped past an initial, too-generous
`abs_tolerance_factor=2.0` — tightening it to 1.7 (§4.5a) caught it too,
with **zero additional false-positive cost** measured against synthetic
noise, and fixed that height's combined g from $5.49\pm0.96$
(pull $-4.5\sigma$) to $11.33\pm4.02$ (pull $0.38\sigma$).

### 9.2 Genuine trial-to-trial restitution variability (NOT something to filter out)

1.1 m's remaining trials, post-filter:

| Trial | $\Delta t_1$ (s) | $\Delta t_2$ (s) | $\Delta t_3$ (s) | Implied $e=\Delta t_2/\Delta t_1$ |
|---|---|---|---|---|
| 0 | 0.370 | 0.362 | 0.352 | **0.98** |
| 1 | 0.611 | 0.371 | 0.249 | 0.61 |
| 2 | 1.122 | 0.618 | 0.370 | 0.55 |
| 3 | 0.596 | 0.609 | 0.359 | 1.02 (increasing!) |
| 5 | 0.493 | 0.485 | 0.244 | 0.98 |

Trials 0 and 5 show almost **no decay at all** across three bounces
(implied $e\approx0.98$), while trials 1 and 2 decay much faster
($e\approx0.55$–$0.61$) — a real, large spread in the *effective*
coefficient of restitution between repeats at the *same* height, not a
detection error. This is exactly the "coefficient of restitution is not
perfectly constant" risk the assignment brief calls out as a key
consideration for this method, now seen directly in the data rather than
just anticipated.

**Why this breaks the fit, not just adds scatter:** a near-flat decay
($e\approx1$) means the fitted slope is close to zero, which makes the
fitted *intercept* (and therefore $g=8h_0e^{-2\cdot\mathrm{intercept}}$)
extremely sensitive to small timing noise — extrapolating a nearly flat
line back to $n=0$ amplifies noise far more than extrapolating a steeply
decaying one does. This is a **statistical/systematic uncertainty source
to report**, not a data-quality problem to filter away.

**Open question carried into Run 1:** with ~3× more repeats per height,
does this restitution spread average out into a well-behaved combined
value (as §11's results suggest, at least in aggregate), or does it persist
per-height and just get masked by higher statistics? §11's per-height
$\chi^2/\mathrm{ndf}$ values (all ≤ 0.81) suggest the latter is not
dominating Run 1 the way it dominated 0.9/1.1 m here — but this has not
been checked as explicitly as §9.2's table does for the test-data run.

## 10. What changed between the test-data run and Full Data Run 1

Before reading §11's results, it's worth being explicit about what's
actually different, since several changes happened together:

1. **A dedicated background recording** for the new session, discovered to
   be necessary the hard way (§11.1) rather than planned from the start.
2. **~15–18 repeats per height instead of ~5** — roughly 3× the statistical
   power per height, so individual imperfect trials have much less leverage
   over a height's combined value.
3. **The Physical Consistency Filter and reliability weighting were already
   built** (developed against the test-data run) and applied to Run 1 from
   the start, rather than retrofitted afterward.

Because all three changed at once, §11 cannot cleanly attribute *how much*
of the improvement comes from each. The repeat-count increase (2) is the
most likely dominant factor given how the test-data run's problems traced
back to individual corrupted/anomalous trials having outsized influence
with only 5 repeats to average over — but this is a reasoned judgement, not
something isolated experimentally.

## 11. Full Data Run 1: a cleaner recollection

Ten heights (0.3–1.2 m, 0.1 m steps), ~15–18 repeated drops per height,
`config_run1.yaml` / `data/raw/run1/`. Files are named
`amp_{h0}m_{N}bounce.xls` where N (15–17) is the number of repeats
recorded in that file, matching the "one continuous file per height,
several repeats inside it" pattern from the test-data run.

### 11.1 First result was badly wrong — and why

The very first Run 1 file processed (0.3 m, reusing the test-data run's
background for threshold calibration, per an initial "no new background
yet" decision) gave $g=3.17\pm0.82\ \mathrm{m/s^2}$ (pull $-8\sigma$) — an
immediate red flag given the instruction to verify new data before
building on it. Diagnosis, in the order it happened:

1. **Segmentation looked unstable in a new way.** Unlike the test-data
   run's clean, wide stable range of `min_gap_s`, this file's segment count
   shifted continuously (21 → 20 → 19 → 17 → 15 across `min_gap_s` = 1.0 s
   → 3.0 s) with no plateau. One "trial" merged 24 impacts across 14.4 s —
   clearly several real drops merged together.
2. **Visual inspection of the raw trace showed clean, well-separated
   bursts** — roughly matching the expected repeat count by eye — which
   contradicted the segmentation instability and pointed at detection,
   not the underlying drops, as the problem.
3. **Comparing this session's actual quiet-trough SPL level against the
   reused background's calibration revealed the root cause:** the new
   session's quiet floor measured ≈−44 dB directly from the trial file's
   own pauses, versus the reused background's −52.8 dB. The reused
   threshold (−42.88 dB, `mean + 5σ` of the *old* background) only cleared
   the *new* session's actual noise by ~1–3σ instead of the intended 5σ,
   causing widespread spurious detections in what should have been quiet
   gaps — which explains both the inflated impact count and the unstable,
   non-bimodal segmentation gap structure (spurious detections fill in
   what should have been a clean silent gap between trials).
4. **A quick pseudo-threshold estimated from the trial file's own quiet
   windows** (not a substitute for a real background, but a fast check)
   immediately produced clean, stable 15-segment behaviour and a
   compatible result ($g=13.17\pm3.97$, pull $0.85\sigma$) — confirming the
   drops themselves were fine and the threshold was the entire problem.
5. **A dedicated ~200 s background was then recorded for the session**
   (`background_run1.xls`) and used properly. Its calibrated noise floor
   (mean $-49.21$ dB) sat between the old background and the ad hoc
   estimate — reinforcing that neither borrowing an old calibration nor
   improvising one from a few quiet seconds inside a trial file is a
   substitute for a real, dedicated recording. With it: threshold
   $-38.20$ dB, 17 usable trials, $g=13.39\pm3.31$ (pull $1.09\sigma$) —
   the pull quoted here is under the pipeline's original, looser timing-
   uncertainty convention at the time this diagnosis was carried out;
   under the current $\sigma_t=\Delta t_\mathrm{sample}/\sqrt{12}$ +
   ±2 cm-height convention (§4.6a) this same 0.3 m trial set gives
   $g=13.39\pm1.91\,\mathrm{(stat)}\pm0.89\,\mathrm{(syst)}$, pull
   $+1.71\sigma$ — the central value and qualitative conclusion (threshold
   fix resolved the problem) are identical; only the quoted precision
   changed.

This whole episode is §7's bug #6. The general lesson: **a detection
threshold calibrated for one recording session should not be assumed valid
for another, even in "the same room" — verify against the new session's
own quiet segments before trusting any downstream result.**

### 11.2 Full 10-height result

`python main.py config_run1.yaml`, same methodology as §4 throughout
(per-height `max_bounce_n`, Physical Consistency Filter at
`abs_tolerance_factor=1.7`/`ratio_ceiling=1.4`, reliability-weighted
fitting, shared-slope rescue for thin trials, ±2 cm height uncertainty and
resolution-based timing uncertainty per §4.6a), against the properly
calibrated background from §11.1:

| $h_0$ (m) | $g$ (m/s²) | stat | syst | $\chi^2/\mathrm{ndf}$ | Pull vs. $g_\mathrm{ref}$ | Trials (full + rescued) |
|---|---|---|---|---|---|---|
| 0.3 | 13.39 | ±1.91 | ±0.89 | 1.46 | +1.71σ | 10 + 7 |
| 0.4 | 5.77  | ±1.18 | ±0.29 | 1.05 | −3.31σ | 13 + 1 |
| 0.5 | 8.10  | ±1.09 | ±0.32 | 2.42 | −1.47σ | 10 + 5 |
| 0.6 | 8.79  | ±1.34 | ±0.29 | 2.41 | −0.73σ | 14 + 1 |
| 0.7 | 11.63 | ±1.50 | ±0.33 | 0.87 | +1.20σ | 16 + 0 |
| 0.8 | 11.29 | ±1.33 | ±0.28 | 0.77 | +1.11σ | 15 + 0 |
| 0.9 | 8.51  | ±1.00 | ±0.19 | 1.66 | −1.25σ | 15 + 1 |
| 1.0 | 11.59 | ±1.15 | ±0.23 | 1.98 | +1.53σ | 16 + 2 |
| 1.1 | 8.65  | ±0.84 | ±0.16 | 2.25 | −1.32σ | 16 + 1 |
| 1.2 | 11.71 | ±0.94 | ±0.20 | 0.71 | +2.00σ | 15 + 2 |

**Overall combined: $g = 9.58\pm0.36\,\mathrm{(stat)}\pm0.08\,
\mathrm{(syst)}\ \mathrm{m/s^2}$ (total $\pm0.37$), $\chi^2/\mathrm{ndf} =
3.41$, pull $= -0.54\sigma$ vs. $g_\mathrm{ref}=9.783$.** The central
value's agreement with the reference is essentially unchanged from an
earlier, looser-uncertainty version of this table
($g=9.58\pm0.62$, $\chi^2/\mathrm{ndf}=1.13$, pull $-0.32\sigma$) — but the
formal cross-height compatibility is not: $\chi^2/\mathrm{ndf}=3.41$ is
above the usual "good" threshold. This is the direct, expected consequence
of §4.6a's timing-uncertainty correction ($\sigma_t=\Delta
t_\mathrm{sample}/\sqrt3$ smaller than the old convention), not a change
in the underlying data or fits — every per-height central $g$ above is
numerically identical to the previous version of this table. Individual
heights are still mostly within ~2σ (worst case 0.4 m at $-3.31\sigma$,
1.2 m at $+2.00\sigma$), but this is now a materially more honest, and
less comfortable, picture than "every height within ~1σ." Both facts
belong in the report: the central estimate is credible, and the
cross-height scatter is real and larger than pure timing precision alone
would predict (§11.3 discusses candidate explanations).

### 11.3 What's still not fully resolved

- **Cross-height $\chi^2/\mathrm{ndf}=3.41$ needs an explanation, not just
  a bigger error bar.** With statistical and systematic uncertainty now
  separated (§4.6a, `figures/run1/uncertainty_budget.png`), the
  statistical term is confirmed to dominate every height's budget (syst is
  ~15–25% of stat throughout) — so the disagreement is not simply "height
  wasn't measured precisely enough." The two candidates already on record
  are genuine per-height restitution variability (§9.2, observed directly
  in the test-data run) and residual detection/segmentation noise the
  Physical Consistency Filter doesn't catch; neither has been isolated
  specifically for Run 1 yet.
- **Individual repeat precision is still limited.** Most repeats resolve
  only 2–4 usable bounce intervals (the same aliasing/reliability
  constraints as §4.3/§4.7 predict) — the combined values' tightness comes
  from averaging ~15–18 repeats, not from any single precise trial.
- **§9.2's restitution-variability question is not re-checked per height
  for Run 1** — this is now a more pressing open item than it was under
  the old uncertainty convention, given §11.2's worse $\chi^2/\mathrm{ndf}$.
- **Height uncertainty (±2 cm) and ball-radius exclusion are now
  documented (§4.6a)**, but ±2 cm is still a stated tolerance rather than
  a per-height re-measurement.
- **No Audio Scope cross-check has been run against Run 1 data.**

## 12. Next steps

1. **Treat Full Data Run 1 (§11) as the dataset for the final report.**
   The test-data run (§8–§9) is valuable as a documented pipeline-
   validation and lessons-learned history, not as a competing result.
2. **Statistical + systematic uncertainty budget is now built** (§4.6a,
   §11.2's table, `figures/run1/uncertainty_budget.png`) — the remaining
   work is *interpreting* it, specifically explaining §11.2's
   $\chi^2/\mathrm{ndf}=3.41$ rather than further tuning the uncertainty
   model itself.
3. **Investigate the cross-height disagreement directly** — check §9.2's
   restitution-variability effect per height for Run 1 specifically
   (implied $e=\Delta t_2/\Delta t_1$ per trial, as §9.2's table did for
   the test-data run), since this is now the leading unverified
   explanation for §11.2's worse-than-before compatibility.
4. **Optionally extend Run 1 toward 1.5 m** — the ≥8-height target is
   already met (10 heights, 0.3–1.2 m), but the assignment's suggested
   range runs to 1.5 m.
5. **Consider an Audio Scope first-bounce cross-check** (§2, §4.4) at one
   or two Run 1 heights, as the second independent feature-extraction
   method the rubric asks for — not yet done against this dataset.
6. **Decide how to present the test-data run in the report** — likely as
   a brief "development/validation" note (it is what surfaced most of the
   bugs in §7) rather than as a result in its own right.
7. Write the report's Results section from §11.2's table, and the
   Conclusion's honest-assessment section from §11.3's caveats plus §7's
   bug history (a legitimate, specific answer to "what went wrong and how
   did you know").
