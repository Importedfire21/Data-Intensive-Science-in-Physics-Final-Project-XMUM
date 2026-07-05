"""
Top-level per-trial and per-recording pipeline functions, shared by
notebooks/analysis.ipynb (full pipeline + synthetic validation) and
notebooks/real_data_analysis.ipynb (real data only), so the logic exists in
exactly one place.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import cleaning
import bounce_detection as bd
import fitting
import segmentation


def _save_current_fig(plot_dir, plot_name):
    if plot_dir and plot_name:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{plot_name}.png"), dpi=150, bbox_inches="tight")


def fit_bounce_sequence(times, sample_dt, h0, h0_err, max_bounce_n=None,
                         reliability_power=0.0, label="", make_plots=True,
                         plot_dir=None, plot_name=None):
    """Core fit step: given already-detected impact times (and the
    recording's sample interval, for the timing-uncertainty estimate),
    build Delta t_n, fit ln(Delta t_n) vs n, and extract g.

    Detection must happen exactly once, on the full recording -- re-running
    peak detection on a cropped slice of the same data can give a different
    (usually fewer) set of detections, because scipy's prominence
    calculation depends on the array's edges. This was caught by comparing
    whole-file detection against per-segment re-detection on real data: a
    genuine third bounce was silently dropped when detection was re-run on
    a segment slice instead of reusing the whole-file detection.

    max_bounce_n: hard ceiling on which intervals are used at all. Prefer
    passing a *per-height* value from
    noise_characterization.resolvable_bounce_count() rather than one fixed
    number for every height -- beyond that point Delta t_n is aliasing
    against the sampling interval, which is a systematic bias, not just
    added noise, and down-weighting alone does not fix a systematic bias.

    reliability_power: within the surviving (non-aliased) intervals, applies
    fitting.reliability_weight(n, power=...) to smoothly down-weight later
    bounces further, reflecting non-timing risks (restitution not staying
    constant, spin, amplitude decay hurting detection precision) that
    compound with n. power=0 (default here) disables this -- pass a
    positive value (e.g. 1.0) to enable it explicitly.
    """
    dt_n = bd.inter_bounce_intervals(times)
    n = np.arange(1, len(dt_n) + 1)

    if max_bounce_n is not None:
        keep = n <= max_bounce_n
        n, dt_n = n[keep], dt_n[keep]

    ln_dt = np.log(dt_n)
    # Timing uncertainty per bounce from the recording's sampling
    # resolution (GUM-style quantisation uncertainty, sample_dt/sqrt(12) --
    # see fitting.resolution_uncertainty), propagated to Delta t_n
    # (difference of two times, so the two samples' uncertainties add in
    # quadrature: sqrt(2)*sigma_t) and then to ln(dt).
    sigma_t = fitting.resolution_uncertainty(sample_dt)
    sigma_dt_n = np.sqrt(2) * sigma_t
    sigma_ln_dt = sigma_dt_n / dt_n

    if reliability_power:
        reliability = fitting.reliability_weight(n, power=reliability_power)
        sigma_ln_dt = sigma_ln_dt / np.sqrt(reliability)

    fit = fitting.weighted_linear_fit(n, ln_dt, sigma_ln_dt)
    g_val, g_stat_err, g_syst_err = fitting.g_stat_syst_from_intercept(
        fit.intercept, fit.intercept_err, h0, h0_err)
    g_err = float(np.sqrt(g_stat_err**2 + g_syst_err**2))

    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].errorbar(n, ln_dt, yerr=sigma_ln_dt, fmt="o", label="data")
        axes[0].plot(n, fit.fitted_y, "C1-", label="weighted fit")
        axes[0].set_xlabel("Bounce number $n$")
        axes[0].set_ylabel(r"$\ln(\Delta t_n$ / s$)$")
        axes[0].set_title(f"{label}: linearised fit")
        axes[0].legend()

        axes[1].errorbar(n, fit.residuals, yerr=sigma_ln_dt, fmt="o", color="C2")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel("Bounce number $n$")
        axes[1].set_ylabel("Residual")
        axes[1].set_title(f"{label}: fit residuals")

        plt.tight_layout()
        _save_current_fig(plot_dir, plot_name)
        plt.show()
        plt.close(fig)

    return {
        "label": label, "h0": h0, "h0_err": h0_err,
        "fit": fit, "g": g_val, "g_err": g_err,
        "g_stat_err": g_stat_err, "g_syst_err": g_syst_err,
        "n_used": len(n), "bounce_times": times, "method": "own_fit",
    }


def _fit_single_interval_with_shared_slope(times, sample_dt, h0, h0_err,
                                            shared_slope, shared_slope_err,
                                            label="", make_plots=False,
                                            plot_dir=None, plot_name=None):
    """Rescue a "thin" trial that detected only 2 impacts (1 interval) --
    too few to fit its own slope AND intercept (a 2-parameter line needs
    >= 2 points). Borrows the coefficient-of-restitution slope (ln e) from
    this height's other, well-constrained trials instead of discarding the
    single real data point this trial does have.

    intercept = ln(Delta t_1) - shared_slope * 1

    The intercept's uncertainty combines this point's own timing
    uncertainty with the shared slope's uncertainty (evaluated at n=1) --
    a trial rescued this way is never claimed to be as precise as a
    properly-constrained one.
    """
    dt_n = bd.inter_bounce_intervals(times)
    if len(dt_n) != 1:
        raise ValueError("_fit_single_interval_with_shared_slope expects exactly 1 interval")

    n = 1.0
    ln_dt1 = np.log(dt_n[0])
    sigma_t = fitting.resolution_uncertainty(sample_dt)
    sigma_dt1 = np.sqrt(2) * sigma_t
    sigma_ln_dt1 = sigma_dt1 / dt_n[0]

    intercept = ln_dt1 - shared_slope * n
    intercept_err = np.sqrt(sigma_ln_dt1**2 + (n * shared_slope_err) ** 2)

    g_val, g_stat_err, g_syst_err = fitting.g_stat_syst_from_intercept(
        intercept, intercept_err, h0, h0_err)
    g_err = float(np.sqrt(g_stat_err**2 + g_syst_err**2))

    if make_plots:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar([n], [ln_dt1], yerr=[sigma_ln_dt1], fmt="o", label="data (1 point)")
        n_line = np.array([0.5, 1.5])
        ax.plot(n_line, shared_slope * n_line + intercept, "C1--",
                label="borrowed slope, fitted intercept")
        ax.set_xlabel("Bounce number $n$")
        ax.set_ylabel(r"$\ln(\Delta t_n$ / s$)$")
        ax.set_title(f"{label}: rescued (shared-slope) fit")
        ax.legend(fontsize=8)
        plt.tight_layout()
        _save_current_fig(plot_dir, plot_name)
        plt.show()
        plt.close(fig)

    return {
        "label": label, "h0": h0, "h0_err": h0_err,
        "fit": None, "g": g_val, "g_err": g_err,
        "g_stat_err": g_stat_err, "g_syst_err": g_syst_err,
        "n_used": 1, "bounce_times": times, "method": "shared_slope_rescue",
    }


def process_trial(df, h0, h0_err, threshold_db, max_bounce_n=None, label="",
                   make_plots=True, trim_start_s=1.5, trim_end_s=1.5,
                   min_separation_s=0.05, reliability_power=0.0):
    """Full single-trial pipeline: clean (trim handling noise), detect
    bounces (2 methods), build Delta t_n, fit ln(Delta t_n) vs n, extract g.

    min_separation_s (also used as the threshold-crossing refractory period)
    must be tuned to the recording mode: ~0.05 s is reasonable for Audio
    Amplitude's coarse envelope, but a high-resolution Audio Scope envelope
    can resolve bounces much closer together -- lower this (e.g. to 0.005-
    0.01 s) once using Audio Scope data, or fast late-time bounces will be
    silently merged/dropped.
    """
    df, trial_clean_report = cleaning.trim_and_clean(df, trim_start_s=trim_start_s,
                                                       trim_end_s=trim_end_s)

    times_a = bd.detect_bounces_peak_picking(df, threshold_db, min_separation_s=min_separation_s)
    times_b = bd.detect_bounces_threshold_crossing(df, threshold_db, refractory_s=min_separation_s)
    agreement = bd.compare_methods(times_a, times_b)

    sample_dt = np.median(np.diff(df["time_s"].dropna().to_numpy()))
    result = fit_bounce_sequence(times_a, sample_dt, h0, h0_err, max_bounce_n=max_bounce_n,
                                  reliability_power=reliability_power, label=label,
                                  make_plots=make_plots)
    result["agreement"] = agreement
    result["clean_report"] = trial_clean_report
    return result


def process_height_recording(df, h0, h0_err, threshold_db, min_gap_s=1.5,
                              trim_start_s=1.5, trim_end_s=1.5, max_bounce_n=None,
                              reliability_power=1.0, make_plots=True, min_separation_s=0.05,
                              min_impacts_per_group=2, label=None, plot_dir=None,
                              apply_consistency_filter=True, g_ref=9.783, e_estimate=0.8,
                              filter_abs_tolerance=2.0, filter_ratio_ceiling=1.4):
    """Clean a multi-drop recording of repeated trials AT THE SAME HEIGHT,
    detect every impact once, group into per-trial segments (however many
    the data actually supports -- not assumed in advance), run each trial's
    intervals through the Physical Consistency Filter, fit every
    well-constrained trial (>= 2 intervals) independently, rescue any thin
    trial (exactly 1 interval) using a slope borrowed from the
    well-constrained trials, and PDG-combine every trial's resulting g into
    one per-height result.

    This is the preferred entry point for a "repeated drops at one height"
    recording. Two things it deliberately does NOT do, based on validating
    against synthetic data first:

    - It does not pool every trial's raw (n, Delta t_n) points into one
      single shared-intercept fit. That was tried and rejected: one bad
      interval in one trial then corrupts the fit for every trial, whereas
      fitting each trial independently lets a bad trial's own chi^2/
      uncertainty absorb its own anomaly, protecting the rest.
    - It does not use one fixed max_bounce_n for every height. Prefer
      computing it per height from noise_characterization.resolvable_bounce_count()
      and passing that in -- a fixed cap is either too strict for large,
      slow-bouncing drops or too lax for small, fast ones.

    apply_consistency_filter: run bounce_detection.filter_plausible_bounce_times
    on each trial before fitting -- drops/truncates a trial's own detected
    impacts if they produce a physically implausible Delta t_n sequence
    (see that function's docstring). g_ref/e_estimate/filter_* are forwarded
    to it. Disable to compare against the unfiltered result.

    min_impacts_per_group: default 2 -- a trial needs at least 2 detected
    impacts (the "first and second impact" minimum) to contribute anything
    at all, even via the shared-slope rescue. A lone single-impact "group"
    has zero intervals and is dropped as noise, not a usable trial.
    """
    if label is None:
        label = f"h0={h0} m"

    df_clean, report = cleaning.trim_and_clean(df, trim_start_s=trim_start_s, trim_end_s=trim_end_s)
    print(cleaning.cleaning_summary_text(report))

    all_times = bd.detect_bounces_peak_picking(df_clean, threshold_db, min_separation_s=min_separation_s)
    raw_groups = segmentation.segment_by_gaps(all_times, min_gap_s)

    if apply_consistency_filter:
        filtered_groups = []
        n_trimmed = 0
        for g in raw_groups:
            kept = bd.filter_plausible_bounce_times(
                g, h0, g_ref=g_ref, e_estimate=e_estimate,
                abs_tolerance_factor=filter_abs_tolerance, ratio_ceiling=filter_ratio_ceiling)
            if len(kept) < len(g):
                n_trimmed += 1
            filtered_groups.append(kept)
        if n_trimmed:
            print(f"Physical Consistency Filter trimmed {n_trimmed} trial(s) with an "
                  f"implausible Delta t_n (see bounce_detection.filter_plausible_bounce_times).")
        raw_groups = filtered_groups

    n_dropped = sum(1 for g in raw_groups if len(g) < min_impacts_per_group)
    groups = [g for g in raw_groups if len(g) >= min_impacts_per_group]
    if n_dropped:
        print(f"Dropped {n_dropped} group(s) with < {min_impacts_per_group} impacts "
              f"(treated as noise, not a usable trial).")
    print(f"Segmented into {len(groups)} usable trial(s) at h0={h0} m.")

    if make_plots:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(df_clean["time_s"], df_clean["spl_db"], lw=0.4, color="0.5")
        for i, g in enumerate(groups):
            ax.axvspan(g[0] - 0.3, g[-1] + 0.3, alpha=0.25, color=f"C{i % 10}", label=f"trial {i}")
        ax.axhline(threshold_db, color="k", ls="--", lw=0.8, label="threshold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SPL (dB)")
        ax.set_title(f"{label}: detected trial boundaries")
        ax.legend(fontsize=7, ncol=min(len(groups) + 1, 6))
        plt.tight_layout()
        _save_current_fig(plot_dir, f"{h0}_segments")
        plt.show()
        plt.close(fig)

    sample_dt = np.median(np.diff(df_clean["time_s"].dropna().to_numpy()))

    # Pass 1: fit every trial with >= 2 intervals (>= 3 impacts) independently.
    full_results, thin_groups = [], []
    for i, times in enumerate(groups):
        n_intervals = len(times) - 1
        if n_intervals >= 2:
            res = fit_bounce_sequence(times, sample_dt, h0, h0_err, max_bounce_n=max_bounce_n,
                                       reliability_power=reliability_power,
                                       label=f"{label} trial {i}", make_plots=make_plots,
                                       plot_dir=plot_dir, plot_name=f"{h0}_trial{i}")
            full_results.append(res)
        else:
            thin_groups.append((i, times))

    # Pass 2: rescue thin (1-interval) trials using a slope borrowed from
    # the well-constrained trials above, if any exist.
    rescued_results = []
    if thin_groups and full_results:
        slopes = np.array([r["fit"].slope for r in full_results])
        slope_errs = np.array([r["fit"].slope_err for r in full_results])
        slope_combo = fitting.pdg_combine(slopes, slope_errs)  # reuse: same inverse-variance average
        shared_slope, shared_slope_err = slope_combo.g_mean, slope_combo.g_mean_err
        for i, times in thin_groups:
            res = _fit_single_interval_with_shared_slope(
                times, sample_dt, h0, h0_err, shared_slope, shared_slope_err,
                label=f"{label} trial {i} (rescued)", make_plots=make_plots,
                plot_dir=plot_dir, plot_name=f"{h0}_trial{i}_rescued")
            rescued_results.append(res)
    elif thin_groups:
        print(f"{len(thin_groups)} thin (1-interval) trial(s) could not be rescued -- "
              f"no well-constrained trial at this height to borrow a slope from.")

    all_results = full_results + rescued_results
    return {
        "label": label, "h0": h0, "h0_err": h0_err,
        "trials": all_results,
        "n_trials_detected": len(groups),
        "n_trials_full": len(full_results),
        "n_trials_rescued": len(rescued_results),
        "clean_report": report,
    }


def combine_height_result(height_result):
    """Combine every trial's g within one process_height_recording() result
    into a single per-height g with statistical and systematic uncertainty
    kept separate (see fitting.combine_repeats_stat_syst) and a chi^2/ndf
    across trials (computed from the statistical uncertainty only, since
    the systematic is shared/correlated across all trials at this height
    and contributes no trial-to-trial scatter).
    """
    g_vals = np.array([r["g"] for r in height_result["trials"]])
    g_stat_errs = np.array([r["g_stat_err"] for r in height_result["trials"]])
    g_syst_errs = np.array([r["g_syst_err"] for r in height_result["trials"]])
    finite = np.isfinite(g_vals) & np.isfinite(g_stat_errs) & np.isfinite(g_syst_errs)
    combo = fitting.combine_repeats_stat_syst(
        g_vals[finite], g_stat_errs[finite], g_syst_errs[finite])
    return combo, int((~finite).sum())


def process_multi_drop_recording(df, heights, heights_err, threshold_db, min_gap_s=2.5,
                                  trim_start_s=10.0, trim_end_s=10.0, max_bounce_n=5,
                                  make_plots=True, min_separation_s=0.05, min_impacts_per_group=2):
    """Clean a multi-drop recording, detect every impact ONCE on the full
    recording, group the detections into per-trial segments by inter-impact
    gap, and fit each group. Use this when DIFFERENT heights are recorded
    in one file (the trial count must be known in advance and match
    `heights`); use process_height_recording() instead for repeated drops
    at the SAME height, which does not require knowing the trial count
    ahead of time and rescues thin (1-interval) trials.

    `heights` must be given in the same chronological order the drops were
    actually performed in -- log this at recording time, since segmentation
    recovers *count* and *timing* but has no way to know the physical
    height used for a given segment.

    min_impacts_per_group: groups with fewer impacts than this are dropped
    before matching against `heights` -- a lone single-impact "group" cannot
    form any Delta t (needs >= 2 points) and is far more likely to be an
    isolated noise transient than a real, otherwise-silent drop trial.
    """
    df_clean, report = cleaning.trim_and_clean(df, trim_start_s=trim_start_s, trim_end_s=trim_end_s)
    print(cleaning.cleaning_summary_text(report))

    all_times = bd.detect_bounces_peak_picking(df_clean, threshold_db, min_separation_s=min_separation_s)
    raw_groups = segmentation.segment_by_gaps(all_times, min_gap_s)
    n_dropped = sum(1 for g in raw_groups if len(g) < min_impacts_per_group)
    groups = [g for g in raw_groups if len(g) >= min_impacts_per_group]
    if n_dropped:
        print(f"Dropped {n_dropped} group(s) with < {min_impacts_per_group} impacts "
              f"(treated as noise, not a usable trial).")
    print(f"Segmented into {len(groups)} usable trial(s); expected {len(heights)} "
          f"(from the heights list you supplied).")

    if make_plots:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(df_clean["time_s"], df_clean["spl_db"], lw=0.4, color="0.5")
        for i, g in enumerate(groups):
            ax.axvspan(g[0] - 0.3, g[-1] + 0.3, alpha=0.25, color=f"C{i % 10}", label=f"segment {i}")
        ax.axhline(threshold_db, color="k", ls="--", lw=0.8, label="threshold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SPL (dB)")
        ax.set_title("Multi-drop recording: detected segment boundaries "
                      "(verify this matches the number of drops you performed!)")
        ax.legend(fontsize=7, ncol=min(len(groups) + 1, 6))
        plt.tight_layout()
        plt.show()

    if len(groups) != len(heights):
        raise ValueError(
            f"Segmentation found {len(groups)} trial(s) but {len(heights)} heights were "
            f"supplied -- inspect the plot above, then adjust min_gap_s (too small -> a trial's "
            f"own bounces get split; too large -> two trials get merged) before proceeding."
        )

    sample_dt = np.median(np.diff(df_clean["time_s"].dropna().to_numpy()))
    results = []
    for times, h0, h0_err in zip(groups, heights, heights_err):
        res = fit_bounce_sequence(times, sample_dt, h0, h0_err, max_bounce_n=max_bounce_n,
                                   label=f"h0={h0} m", make_plots=make_plots)
        results.append(res)
    return results
