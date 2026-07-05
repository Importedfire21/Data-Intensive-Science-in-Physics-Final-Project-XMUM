"""
Config-driven entry point for the bouncing-ball g analysis.

Usage:
    python main.py [config.yaml]

To add a new height trial: drop its phyphox Audio Amplitude export into
data_dir (see config.yaml) and add one `height: filename` line under
height_files -- no code or notebook edits needed. Re-run this script.

This is the fast-iteration tool for processing real data as it comes in.
notebooks/analysis.ipynb remains the polished, explanatory deliverable
(pipeline description + synthetic validation) for the actual report/
submission; this script and that notebook share every function in
src/pipeline.py, src/fitting.py, etc., so results are consistent between
the two -- only the presentation differs.
"""
import csv
import os
import sys
from pathlib import Path

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # script context, not a notebook -- never block on plt.show()
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

import io_utils
import cleaning
import noise_characterization as noise
import bounce_detection as bd
import fitting
from pipeline import process_height_recording, combine_height_result


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    data_dir = Path(cfg["data_dir"])
    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["results_csv"]).parent.mkdir(parents=True, exist_ok=True)

    g_ref = cfg["g_ref"]
    h_err = cfg["height_uncertainty_m"]
    e_est = cfg["restitution_estimate"]

    clean_cfg = cfg["cleaning"]
    det_cfg = cfg["detection"]
    seg_cfg = cfg["segmentation"]
    fit_cfg = cfg["fitting"]
    plot_cfg = cfg.get("plots", {})
    filt_cfg = cfg.get("consistency_filter", {"enabled": False})

    # --- Background / threshold ---
    bg_raw = io_utils.load_audio_amplitude(str(data_dir / cfg["background_file"]))
    bg, bg_report = cleaning.trim_and_clean(
        bg_raw, trim_start_s=clean_cfg["background_trim_s"], trim_end_s=clean_cfg["background_trim_s"])
    profile = noise.characterise(bg, k_sigma=det_cfg["k_sigma"])
    print("=== Background ===")
    print(cleaning.cleaning_summary_text(bg_report))
    print(noise.summary_text(profile))
    print()

    # Background: raw trace (before cleaning) with trimmed regions shaded.
    trim_s = clean_cfg["background_trim_s"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(bg_raw["time_s"], bg_raw["spl_db"], lw=0.5)
    ax.axvspan(0, trim_s, color="C3", alpha=0.15, label="trimmed (start handling noise)")
    ax.axvspan(bg_raw["time_s"].max() - trim_s, bg_raw["time_s"].max(), color="C3", alpha=0.15,
               label="trimmed (end handling noise)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sound pressure level (dB)")
    ax.set_title("Background: RAW trace (before cleaning)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "background_before_cleaning.png", dpi=150)
    plt.close(fig)

    # Background: cleaned trace + threshold, and noise-floor histogram.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(bg["time_s"], bg["spl_db"], lw=0.5)
    axes[0].axhline(profile.mean_db, color="C1", label=f"mean = {profile.mean_db:.1f} dB")
    axes[0].axhline(profile.recommended_threshold_db, color="C3", ls="--",
                     label=f"threshold = {profile.recommended_threshold_db:.1f} dB")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Sound pressure level (dB)")
    axes[0].set_title("Background: cleaned trace")
    axes[0].legend(fontsize=8)

    axes[1].hist(bg["spl_db"].dropna(), bins=40, color="C0", alpha=0.8)
    axes[1].axvline(profile.recommended_threshold_db, color="C3", ls="--")
    axes[1].set_xlabel("Sound pressure level (dB)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Noise-floor distribution (cleaned)")
    plt.tight_layout()
    plt.savefig(figures_dir / "noise_floor.png", dpi=150)
    plt.close(fig)
    print(f"Saved background plots: {figures_dir / 'background_before_cleaning.png'}, "
          f"{figures_dir / 'noise_floor.png'}")
    print()

    # --- Per-height processing ---
    height_files = cfg["height_files"]
    per_height = {}

    for h0 in sorted(height_files):
        fname = height_files[h0]
        path = data_dir / fname
        if not path.exists():
            print(f"!! Skipping h0={h0} m: file not found: {path}")
            continue

        A_est = 2 * np.sqrt(2 * h0 / g_ref)
        n_max = noise.resolvable_bounce_count(profile, A_est, e_est,
                                               nyquist_factor=fit_cfg["nyquist_factor"])

        print(f"=== h0 = {h0} m ({fname}), resolvable_bounce_count -> max_bounce_n={n_max} ===")
        raw = io_utils.load_audio_amplitude(str(path))

        # Raw time-curve + peak-finding plot: full cleaned trial trace, the
        # detection threshold, and every raw detected impact (before the
        # Physical Consistency Filter runs) -- a visual sanity check of
        # what the detector is seeing, independent of segmentation/fitting.
        trial_clean, _ = cleaning.trim_and_clean(
            raw, trim_start_s=clean_cfg["trial_trim_s"], trim_end_s=clean_cfg["trial_trim_s"])
        raw_impacts = bd.detect_bounces_peak_picking(
            trial_clean, profile.recommended_threshold_db, min_separation_s=det_cfg["min_separation_s"])

        raw_traces_dir = figures_dir / "raw_traces"
        raw_traces_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(10, trial_clean["time_s"].max() / 8), 4))
        ax.plot(trial_clean["time_s"], trial_clean["spl_db"], lw=0.5, color="0.4")
        ax.axhline(profile.recommended_threshold_db, color="C3", ls="--",
                   label=f"threshold = {profile.recommended_threshold_db:.1f} dB")
        for t in raw_impacts:
            ax.axvline(t, color="g", alpha=0.3, lw=0.8)
        ax.plot([], [], color="g", alpha=0.5, label=f"{len(raw_impacts)} raw detected impacts")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SPL (dB)")
        ax.set_title(f"h0={h0} m ({fname}): raw trace + peak finding")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(raw_traces_dir / f"{h0}_raw_trace_peaks.png", dpi=130)
        plt.close(fig)
        print(f"  Saved raw trace + peaks plot: {raw_traces_dir / f'{h0}_raw_trace_peaks.png'}")

        result = process_height_recording(
            raw, h0=h0, h0_err=h_err, threshold_db=profile.recommended_threshold_db,
            min_gap_s=seg_cfg["min_gap_s"], trim_start_s=clean_cfg["trial_trim_s"],
            trim_end_s=clean_cfg["trial_trim_s"], max_bounce_n=n_max,
            reliability_power=fit_cfg["reliability_power"],
            make_plots=plot_cfg.get("save_per_height", False),
            min_separation_s=det_cfg["min_separation_s"],
            min_impacts_per_group=seg_cfg["min_impacts_per_group"],
            plot_dir=str(figures_dir / "per_height") if plot_cfg.get("save_per_height", False) else None,
            apply_consistency_filter=filt_cfg.get("enabled", False),
            g_ref=g_ref, e_estimate=e_est,
            filter_abs_tolerance=filt_cfg.get("abs_tolerance_factor", 2.0),
            filter_ratio_ceiling=filt_cfg.get("ratio_ceiling", 1.4),
        )
        combo, n_excluded = combine_height_result(result)
        per_height[h0] = (result, combo)

        print(f"  trials: {result['n_trials_full']} full + {result['n_trials_rescued']} rescued "
              f"({n_excluded} unusable)")
        print(f"  g = {combo.g_mean:.2f} +/- {combo.stat_err:.2f} (stat) +/- {combo.syst_err:.2f} (syst) m/s^2, "
              f"chi2/ndf = {combo.chi2_ndf:.2f}, "
              f"pull vs g_ref = {fitting.pull(combo.g_mean, combo.total_err, g_ref):.2f} sigma")
        print()

    if not per_height:
        print("No height files were found -- check config.yaml's data_dir/height_files.")
        return

    # --- Combine across heights ---
    heights_arr = np.array(sorted(per_height))
    g_arr = np.array([per_height[h][1].g_mean for h in heights_arr])
    stat_err_arr = np.array([per_height[h][1].stat_err for h in heights_arr])
    syst_err_arr = np.array([per_height[h][1].syst_err for h in heights_arr])

    overall = fitting.combine_heights_stat_syst(g_arr, stat_err_arr, syst_err_arr)
    pull_overall = fitting.pull(overall.g_mean, overall.total_err, g_ref)

    print("=== Combined across all heights ===")
    print(f"g = {overall.g_mean:.2f} +/- {overall.stat_err:.2f} (stat) +/- {overall.syst_err:.2f} (syst) m/s^2 "
          f"[total {overall.total_err:.2f}]")
    print(f"chi2/ndf = {overall.chi2_ndf:.2f}, pull vs g_ref = {pull_overall:.2f} sigma")
    if overall.chi2_ndf > 2.0:
        print("VERDICT: BAD compatibility across heights -- do not quote this as the final result "
              "without resolving the disagreement first.")
    elif overall.chi2_ndf > 1.3:
        print("VERDICT: some tension across heights -- treat the combined value cautiously.")
    else:
        print("VERDICT: good compatibility across heights.")

    # --- Interval diagnostics: Delta t_1 (impact 1->2), Delta t_2 (2->3), Delta t_3 (3->4) ---
    interval_rows = []
    for h0 in heights_arr:
        result, _ = per_height[h0]
        for trial in result["trials"]:
            times = np.asarray(trial["bounce_times"])
            dt = np.diff(times)
            interval_rows.append({
                "height_m": h0,
                "trial_label": trial["label"],
                "method": trial["method"],
                "dt1_s": dt[0] if len(dt) >= 1 else "",
                "dt2_s": dt[1] if len(dt) >= 2 else "",
                "dt3_s": dt[2] if len(dt) >= 3 else "",
            })

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    interval_csv_path = processed_dir / "interval_summary.csv"
    with open(interval_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["height_m", "trial_label", "method", "dt1_s", "dt2_s", "dt3_s"])
        writer.writeheader()
        for row in interval_rows:
            writer.writerow(row)
    print(f"Saved interval summary: {interval_csv_path}")

    def _col(rows, key):
        return np.array([r[key] if r[key] != "" else np.nan for r in rows], dtype=float)

    # One plot per height: Delta t_1/2/3 across that height's repeats.
    per_height_plot_dir = figures_dir / "intervals"
    per_height_plot_dir.mkdir(parents=True, exist_ok=True)
    for h0 in heights_arr:
        rows = [r for r in interval_rows if r["height_m"] == h0]
        trial_idx = np.arange(len(rows))
        dt1, dt2, dt3 = _col(rows, "dt1_s"), _col(rows, "dt2_s"), _col(rows, "dt3_s")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trial_idx, dt1, "o-", label=r"$\Delta t_1$ (impact 1$\to$2)")
        if np.any(~np.isnan(dt2)):
            ax.plot(trial_idx, dt2, "s-", label=r"$\Delta t_2$ (impact 2$\to$3)")
        if np.any(~np.isnan(dt3)):
            ax.plot(trial_idx, dt3, "^-", label=r"$\Delta t_3$ (impact 3$\to$4)")
        ax.set_xlabel("Trial index (repeat)")
        ax.set_ylabel(r"$\Delta t$ (s)")
        ax.set_title(f"h0={h0} m: inter-bounce intervals across repeats")
        ax.set_xticks(trial_idx)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(per_height_plot_dir / f"{h0}_dt_intervals.png", dpi=150)
        plt.close(fig)
    print(f"Saved per-height interval plots: {per_height_plot_dir}\\*.png")

    # One combined plot: Delta t_1/2/3 (mean +/- std across repeats) vs height.
    means, stds = {1: [], 2: [], 3: []}, {1: [], 2: [], 3: []}
    for h0 in heights_arr:
        rows = [r for r in interval_rows if r["height_m"] == h0]
        for k, key in [(1, "dt1_s"), (2, "dt2_s"), (3, "dt3_s")]:
            vals = _col(rows, key)
            vals = vals[~np.isnan(vals)]
            means[k].append(np.mean(vals) if len(vals) else np.nan)
            stds[k].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {1: "o-", 2: "s-", 3: "^-"}
    labels = {1: r"$\Delta t_1$ (impact 1$\to$2)", 2: r"$\Delta t_2$ (impact 2$\to$3)", 3: r"$\Delta t_3$ (impact 3$\to$4)"}
    for k in (1, 2, 3):
        if np.any(~np.isnan(means[k])):
            ax.errorbar(heights_arr, means[k], yerr=stds[k], fmt=markers[k], capsize=3, label=labels[k])
    ax.set_xlabel("Drop height $h_0$ (m)")
    ax.set_ylabel(r"$\Delta t$ (s), mean $\pm$ std across repeats")
    ax.set_title("Inter-bounce intervals vs. drop height (all heights combined)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "dt_intervals_combined.png", dpi=150)
    plt.close(fig)
    print(f"Saved combined interval plot: {figures_dir / 'dt_intervals_combined.png'}")

    total_err_arr = np.sqrt(stat_err_arr**2 + syst_err_arr**2)

    # --- Summary plot ---
    if plot_cfg.get("save_summary", True):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Outer error bar = stat+syst in quadrature (total); inner, thicker
        # bar = statistical only -- shows how much of each height's total
        # uncertainty budget is the (shared, non-shrinking) height systematic.
        ax.errorbar(heights_arr, g_arr, yerr=total_err_arr, fmt="o", capsize=4, color="C0",
                    label="per-height combined g (stat+syst)")
        ax.errorbar(heights_arr, g_arr, yerr=stat_err_arr, fmt="none", capsize=3, elinewidth=2.5,
                    color="C0", alpha=0.6, label="statistical only")
        ax.axhline(g_ref, color="C3", ls="--", label=f"g_ref = {g_ref} m/s^2")
        ax.axhline(overall.g_mean, color="C1", ls=":", label=f"overall combined = {overall.g_mean:.2f} m/s^2")
        ax.fill_between(ax.get_xlim(), overall.g_mean - overall.total_err, overall.g_mean + overall.total_err,
                         color="C1", alpha=0.1)
        ax.set_xlabel("Drop height $h_0$ (m)")
        ax.set_ylabel("$g$ (m/s$^2$)")
        ax.set_title(f"g vs. drop height ({len(heights_arr)} heights, reliability-weighted pooling)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / "g_vs_height.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved summary plot: {figures_dir / 'g_vs_height.png'}")

        # --- Uncertainty budget plot: stat vs syst magnitude per height ---
        fig, ax = plt.subplots(figsize=(7, 4.5))
        width = 0.35 * (heights_arr[1] - heights_arr[0]) if len(heights_arr) > 1 else 0.03
        ax.bar(heights_arr - width / 2, stat_err_arr, width=width, label="statistical (timing/fit)", color="C0")
        ax.bar(heights_arr + width / 2, syst_err_arr, width=width, label="systematic (height)", color="C3")
        ax.set_xlabel("Drop height $h_0$ (m)")
        ax.set_ylabel(r"$\sigma_g$ (m/s$^2$)")
        ax.set_title("Uncertainty budget: statistical vs. systematic contribution per height")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / "uncertainty_budget.png", dpi=150)
        plt.close(fig)
        print(f"Saved uncertainty budget plot: {figures_dir / 'uncertainty_budget.png'}")

    # --- Results CSV ---
    with open(cfg["results_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["height_m", "g", "g_stat_err", "g_syst_err", "g_total_err",
                          "chi2_ndf", "n_trials_full", "n_trials_rescued"])
        for h0 in heights_arr:
            result, combo = per_height[h0]
            writer.writerow([h0, f"{combo.g_mean:.4f}", f"{combo.stat_err:.4f}", f"{combo.syst_err:.4f}",
                              f"{combo.total_err:.4f}", f"{combo.chi2_ndf:.3f}",
                              result["n_trials_full"], result["n_trials_rescued"]])
        writer.writerow([])
        writer.writerow(["overall", f"{overall.g_mean:.4f}", f"{overall.stat_err:.4f}", f"{overall.syst_err:.4f}",
                          f"{overall.total_err:.4f}", f"{overall.chi2_ndf:.3f}", "", ""])
    print(f"Saved results table: {cfg['results_csv']}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
