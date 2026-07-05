# Project Overview: Measuring Local *g* via the Bouncing-Ball Method

Status snapshot and headline results. For the detailed pipeline
architecture, validation history, and full bug/finding log, see
`report/process_overview.md`. For how to record a trial so it slots
straight into the pipeline, see `report/data_collection_protocol.md`.

## Headline result (Run 1, 10 heights)

$$\boxed{g = 9.58 \pm 0.36\ \mathrm{(stat)} \pm 0.08\ \mathrm{(syst)}\ \mathrm{m/s^2}}$$

Total uncertainty (stat+syst in quadrature): $\pm 0.37\ \mathrm{m/s^2}$.
Pull vs. $g_\mathrm{ref}=9.783\ \mathrm{m/s^2}$ is $-0.54\sigma$ — the
central value still agrees well with the reference, **but**
$\chi^2/\mathrm{ndf} = 3.41$ across the 10 heights, which is formally bad
compatibility (see "Statistical vs. systematic uncertainty" below for why
this got worse, and honestly, since the last version of this doc).

| $h_0$ (m) | $g$ (m/s²) | stat | syst | $\chi^2/\mathrm{ndf}$ | Pull (σ) | Trials (full + rescued) |
|---|---|---|---|---|---|---|
| 0.3 | 13.39 | ±1.91 | ±0.89 | 1.46 | +1.71 | 10 + 7 |
| 0.4 | 5.77  | ±1.18 | ±0.29 | 1.05 | −3.31 | 13 + 1 |
| 0.5 | 8.10  | ±1.09 | ±0.32 | 2.42 | −1.47 | 10 + 5 |
| 0.6 | 8.79  | ±1.34 | ±0.29 | 2.41 | −0.73 | 14 + 1 |
| 0.7 | 11.63 | ±1.50 | ±0.33 | 0.87 | +1.20 | 16 + 0 |
| 0.8 | 11.29 | ±1.33 | ±0.28 | 0.77 | +1.11 | 15 + 0 |
| 0.9 | 8.51  | ±1.00 | ±0.19 | 1.66 | −1.25 | 15 + 1 |
| 1.0 | 11.59 | ±1.15 | ±0.23 | 1.98 | +1.53 | 16 + 2 |
| 1.1 | 8.65  | ±0.84 | ±0.16 | 2.25 | −1.32 | 16 + 1 |
| 1.2 | 11.71 | ±0.94 | ±0.20 | 0.71 | +2.00 | 15 + 2 |

Every individual height still lands within ~2σ of the reference (only
0.4 m exceeds it, at −3.31σ), but the per-height and cross-height
$\chi^2/\mathrm{ndf}$ values are noticeably worse than an earlier version
of this table — that is a real, reported effect of tightening the timing
uncertainty (see below), not a new problem with the data.
See `figures/run1/g_vs_height.png` and `figures/run1/uncertainty_budget.png`
for the plots and `report/run1_results_summary.csv` for the machine-readable
table.

## Statistical vs. systematic uncertainty, and the ±2 cm height error

Two uncertainty sources are now tracked and combined **separately**, not
lumped into one number:

- **Statistical**: timing/fit precision from the linear regression's
  intercept. It is independent per trial (different detected impact times
  each drop), so repeats are combined inverse-variance-weighted and this
  term genuinely shrinks with more repeats — e.g. 0.3 m's per-trial
  statistical uncertainty is much larger than the ±1.91 shown above, which
  is already the ~17-trial combined value.
- **Systematic**: drop-height measurement uncertainty, assumed
  **±2 cm** for every height (`height_uncertainty_m: 0.02` in both
  configs, replacing an earlier placeholder ±5 mm). Every repeat at a
  given height shares the exact same $h_0$ and $h_0$-uncertainty, so this
  term is fully correlated across those repeats and must **not** be
  averaged down the way the statistical term is — `combine_repeats_stat_syst`
  in `src/fitting.py` evaluates it once per height rather than
  inverse-variance-combining N copies of what is really one number. When
  combining *across* heights, each height's systematic *is* independent
  (a separately measured height), so it does shrink there, through the
  same weights as the statistical part
  (`combine_heights_stat_syst`).
- **Height measurement convention**: heights were measured **bottom of the
  ball to the surface**, not center-of-mass to surface. With this
  convention no ball-radius correction is needed — the center of mass
  starts at $h_0 + r$ and ends at $r$ when the ball's bottom touches the
  surface, so it still falls exactly $h_0$ regardless of the ball's
  radius. This was explicitly considered and is correctly excluded, not
  overlooked.
- **Timing uncertainty formula corrected**: per-impact timing uncertainty
  is now $\sigma_t = \Delta t_\mathrm{sample}/\sqrt{12}$ (the standard
  deviation of a uniform distribution over one sample interval — GUM/JCGM
  100:2008 §4.3.7 Type B quantisation uncertainty), replacing an earlier
  ad hoc $\Delta t_\mathrm{sample}/2$ convention, which was really the
  distribution's *half-width*, not its standard deviation, and overstated
  $\sigma_t$ by a factor of $\sqrt{3}\approx1.73\times$. This change is a
  uniform rescaling of every weight within a trial's fit, so it leaves
  central $g$ values completely unchanged (verified against synthetic
  data with known ground truth) but shrinks statistical uncertainties by
  that same $1/\sqrt{3}$ factor — and since real-world scatter didn't
  shrink along with it, $\chi^2/\mathrm{ndf}$ correspondingly inflated
  (by ~3× where residuals dominate). This is being reported honestly as a
  more rigorous, smaller claimed uncertainty revealing real scatter that a
  looser convention was masking — not treated as a new problem with the
  data or hidden by reverting to the old formula.

## Two datasets exist — know which is which

1. **Test-data run** (`config.yaml`, `data/raw/*.xls`, 10 heights × ~5
   repeats): the original pilot dataset, used to build and stress-test the
   pipeline. Its combined result is **not reliable** (cross-height
   $\chi^2/\mathrm{ndf}$ never got below ~2.7 even after the Physical
   Consistency Filter) — useful for having found and fixed real bugs
   (§7 of `process_overview.md`), not for quoting a final $g$.
2. **Full Data Run 1** (`config_run1.yaml`, `data/raw/run1/*.xls`, 10
   heights × ~15–17 repeats): a more careful, higher-repeat-count
   recollection with its own dedicated background recording. This is the
   dataset behind the headline result above, and the one to build the
   final report from.

Run each with `python main.py config.yaml` or `python main.py
config_run1.yaml` respectively — **`python main.py` with no argument
defaults to the test-data config**, so always pass the config explicitly
when working with Run 1.

## Method, in one paragraph

A ball dropped from height $h_0$ bounces with inter-bounce intervals
$\Delta t_n = A e^n$ ($A = 2\sqrt{2h_0/g}$, $e$ = coefficient of
restitution). A weighted fit of $\ln(\Delta t_n)$ vs. bounce number $n$
gives $g$ from the fitted intercept. Repeated drops per height are
combined (inverse-variance weighted, PDG-style), and heights are combined
again for the final result. Full derivation in `process_overview.md` §1.

## Pipeline, in one paragraph

Audio Amplitude recordings (phyphox) are cleaned (handling-noise trimmed),
thresholded against a dedicated quiet background, peak-detected (two
independent methods), segmented into per-trial repeats, screened by a
**Physical Consistency Filter** (drops/truncates individual corrupted
intervals — e.g. a missed bounce — using both an absolute plausibility
ceiling and an inter-interval ratio check), fit per trial with
reliability-weighted bounces (later bounces count less), and combined.
Every step was validated against synthetic data with known ground truth
before being trusted on real recordings. Full detail in
`process_overview.md` §2–§7; the filter specifically in §4.5a.

## Why Run 1 is so much cleaner than the test run

- **A dedicated background recording**, taken for this session rather than
  reused — the test run's later heights suffered from a reused/mismatched
  noise floor at points; process_overview.md §9.1 documents a concrete
  case (missed bounce inflating $\Delta t_1$ to ~3× plausible) that a
  mismatched threshold makes more likely, not less.
- **~15–17 repeats per height instead of ~5** — far more statistical
  power per height, so a handful of imperfect trials no longer dominate a
  height's combined value the way they could with only 5.
- **The Physical Consistency Filter and reliability weighting**, absent
  from the pipeline's first version, now routinely trim 5–11 corrupted
  intervals per height before they can bias a fit.

## Known remaining caveats (be honest about these in the report)

- **Height uncertainty is a flat assumed ±2 cm** for every height
  (`height_uncertainty_m: 0.02` in both configs) — a stated measurement
  tolerance, not an independently re-derived value per height. Ball-radius
  contribution is correctly excluded (heights were measured bottom-of-ball
  to surface, see above), so no additional correction is owed there.
- **Cross-height $\chi^2/\mathrm{ndf}$ is now 3.41 (bad by the usual <1.3
  rule of thumb)**, up from 1.13 in the previous (looser-uncertainty)
  version of this analysis — this is the direct, expected consequence of
  moving to the correct $\sigma_t=\Delta t_\mathrm{sample}/\sqrt{12}$
  timing-uncertainty formula (see above): tighter honest uncertainties
  made real height-to-height scatter visible that a looser convention was
  masking. The central combined value is unaffected and still agrees with
  $g_\mathrm{ref}$ to $-0.54\sigma$, but the *individual-height*
  agreement is now honestly worse — 0.4 m sits at $-3.31\sigma$. Report
  both facts; do not quietly keep the old formula because it looked
  better.
- **`restitution_estimate=0.8`** is only used to size the per-height
  aliasing ceiling (`resolvable_bounce_count`) and the consistency
  filter's plausibility check — it is not fitted from the data, and real
  $e$ varies both within and across heights (see the test run's 0.9/1.1 m
  finding in `process_overview.md` §9.2, a genuine physical effect, not a
  bug).
- **Individual repeat statistical uncertainties are still large relative
  to the combined per-height value** — most repeats resolve only 2–4
  usable bounce intervals. The combined values' tightness comes from
  averaging ~15 repeats, not from any single precise measurement.
- **This is Audio Amplitude only.** No Audio Scope cross-check has been
  run against Run 1 data yet (Section 3 of `notebooks/analysis.ipynb`
  demonstrates the method on synthetic + earlier real data).

## What's left before writing the final report

1. **Optionally extend Run 1 toward 1.5 m** — the protocol's ≥8-height
   target is already met (10 heights, 0.3–1.2 m), but the assignment's
   suggested range runs to 1.5 m; more heights only help the cross-height
   fit.
2. **Investigate the now-worse cross-height $\chi^2/\mathrm{ndf}$
   (3.41)** — with the uncertainty budget now built (stat vs. syst
   separated, see `figures/run1/uncertainty_budget.png`), the next honest
   step is understanding *why* heights disagree this much (candidates:
   genuine per-height restitution variability, as already documented for
   the test run in §9.2; residual detection/segmentation noise not fully
   caught by the Physical Consistency Filter) rather than further
   uncertainty-model tuning.
3. **Decide what to do with the test-data run in the report** — likely
   worth keeping as a "lessons learned" / methodology-validation appendix
   rather than a result, given its poor compatibility.
4. **Consider an Audio Scope first-bounce cross-check** at one or two Run
   1 heights, per `process_overview.md` §2/§4.4, as the second independent
   feature-extraction method the rubric asks for.
5. Write the report's Results section from the Run 1 numbers above, and
   the Conclusion's honest-assessment section from the caveats above.

## Where everything lives

```
config.yaml, config_run1.yaml     analysis configs (test data vs. Run 1)
main.py                            run: python main.py [config_run1.yaml]
data/raw/, data/raw/run1/          raw phyphox exports, per dataset
data/processed/, data/processed/run1/   interval_summary.csv per dataset
figures/, figures/run1/            all plots, per dataset (see subfolders:
                                    per_height/, intervals/, raw_traces/;
                                    g_vs_height.png and
                                    uncertainty_budget.png at top level)
report/results_summary.csv         test-data run results table
report/run1_results_summary.csv    Run 1 results table (the headline numbers)
report/process_overview.md         full technical methodology + history
report/data_collection_protocol.md how to record a trial correctly
report/project_overview.md         this document
notebooks/analysis.ipynb           polished pipeline walkthrough + synthetic validation
notebooks/real_data_analysis.ipynb early real-data exploration (pre-Run 1)
```
