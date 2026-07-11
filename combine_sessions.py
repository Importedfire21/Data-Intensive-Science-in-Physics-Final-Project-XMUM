"""
Pool the test-data run (config.yaml) and Full Data Run 1 (config_run1.yaml)
into one combined result, increasing the sample size beyond either session
alone.

This is valid because the two sessions are genuinely independent repeats of
the same experiment by the same person, with the same apparatus (including
the same +1.5 cm height-measurement offset, confirmed and corrected here for
both sessions) -- not just two summary numbers averaged together. For each
nominal height, EVERY individual trial's own fit (full or shared-slope
rescued) from BOTH sessions is pooled into one inverse-variance combination,
exactly the way repeats within a single session are combined
(fitting.combine_repeats_stat_syst) -- then heights are combined across the
full 0.3-1.2 m range as usual (fitting.combine_heights_stat_syst).

The test-data run's own, separately documented issues (see
report/process_overview.md Sec 8-9: corrupted intervals already handled by
the Physical Consistency Filter; genuine restitution variability at 0.9/1.1 m
that is NOT filtered, since it's real physics) are inherited into the pool
as-is -- this is not swept away by pooling, it is diluted by Run 1's larger
sample and should be reported honestly in the combined result's own
chi2/ndf, not hidden.

Usage: python combine_sessions.py
"""
import csv
from pathlib import Path

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

import io_utils
import cleaning
import noise_characterization as noise
from pipeline import process_height_recording
import fitting

HEIGHT_OFFSET_M = 0.015  # +1.5 cm, confirmed for both sessions


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_profile(cfg, data_dir):
    bg_raw = io_utils.load_audio_amplitude(str(data_dir / cfg["background_file"]))
    bg, _ = cleaning.trim_and_clean(
        bg_raw, trim_start_s=cfg["cleaning"]["background_trim_s"],
        trim_end_s=cfg["cleaning"]["background_trim_s"])
    return noise.characterise(bg, k_sigma=cfg["detection"]["k_sigma"])


def session_trials_for_height(cfg, data_dir, profile, lookup_key, h0_corrected, g_ref, h_err, e_est):
    fname = cfg["height_files"][lookup_key]
    path = data_dir / fname
    if not path.exists():
        return []
    raw = io_utils.load_audio_amplitude(str(path))
    A_est = 2 * np.sqrt(2 * h0_corrected / g_ref)
    n_max = noise.resolvable_bounce_count(profile, A_est, e_est, nyquist_factor=cfg["fitting"]["nyquist_factor"])
    filt_cfg = cfg.get("consistency_filter", {"enabled": False})
    result = process_height_recording(
        raw, h0=h0_corrected, h0_err=h_err, threshold_db=profile.recommended_threshold_db,
        min_gap_s=cfg["segmentation"]["min_gap_s"], trim_start_s=cfg["cleaning"]["trial_trim_s"],
        trim_end_s=cfg["cleaning"]["trial_trim_s"], max_bounce_n=n_max,
        reliability_power=cfg["fitting"]["reliability_power"], make_plots=False,
        min_separation_s=cfg["detection"]["min_separation_s"],
        min_impacts_per_group=cfg["segmentation"]["min_impacts_per_group"],
        apply_consistency_filter=filt_cfg.get("enabled", False), g_ref=g_ref, e_estimate=e_est,
        filter_abs_tolerance=filt_cfg.get("abs_tolerance_factor", 2.0),
        filter_ratio_ceiling=filt_cfg.get("ratio_ceiling", 1.4),
    )
    return result["trials"]


def main():
    root = Path(__file__).parent
    cfg_test = load_config(root / "config.yaml")
    cfg_run1 = load_config(root / "config_run1.yaml")

    g_ref = cfg_run1["g_ref"]
    h_err = cfg_run1["height_uncertainty_m"]
    e_est = cfg_run1["restitution_estimate"]

    dir_test = root / cfg_test["data_dir"]
    dir_run1 = root / cfg_run1["data_dir"]
    profile_test = build_profile(cfg_test, dir_test)
    profile_run1 = build_profile(cfg_run1, dir_run1)

    nominal_heights = sorted(cfg_test["height_files"])
    figures_dir = root / "figures" / "combined"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(root / "report").mkdir(parents=True, exist_ok=True)

    per_height = {}
    print("=== Pooled per-height results (test-data + Run 1, height-corrected) ===")
    for h0_nom in nominal_heights:
        h0_corr = round(h0_nom + HEIGHT_OFFSET_M, 3)
        # config.yaml's height_files keys are nominal (0.3); config_run1.yaml's
        # are already the corrected values (0.315) -- look each up accordingly.
        trials_test = session_trials_for_height(cfg_test, dir_test, profile_test, h0_nom, h0_corr, g_ref, h_err, e_est)
        trials_run1 = session_trials_for_height(cfg_run1, dir_run1, profile_run1, h0_corr, h0_corr, g_ref, h_err, e_est)
        pooled = trials_test + trials_run1

        g_vals = np.array([r["g"] for r in pooled])
        g_stat = np.array([r["g_stat_err"] for r in pooled])
        g_syst = np.array([r["g_syst_err"] for r in pooled])
        finite = np.isfinite(g_vals) & np.isfinite(g_stat) & np.isfinite(g_syst)
        combo = fitting.combine_repeats_stat_syst(g_vals[finite], g_stat[finite], g_syst[finite])
        per_height[h0_corr] = (combo, len(trials_test), len(trials_run1))

        pull = fitting.pull(combo.g_mean, combo.total_err, g_ref)
        print(f"h0={h0_corr:.3f} m (nominal {h0_nom} m): n_test={len(trials_test)} n_run1={len(trials_run1)} "
              f"n_pooled={finite.sum()} -> g={combo.g_mean:.2f} +/- {combo.stat_err:.2f} (stat) "
              f"+/- {combo.syst_err:.2f} (syst), chi2/ndf={combo.chi2_ndf:.2f}, pull={pull:.2f} sigma")

    heights_arr = np.array(sorted(per_height))
    g_arr = np.array([per_height[h][0].g_mean for h in heights_arr])
    stat_arr = np.array([per_height[h][0].stat_err for h in heights_arr])
    syst_arr = np.array([per_height[h][0].syst_err for h in heights_arr])
    overall = fitting.combine_heights_stat_syst(g_arr, stat_arr, syst_arr)
    pull_overall = fitting.pull(overall.g_mean, overall.total_err, g_ref)

    print("\n=== Combined across all heights (pooled) ===")
    print(f"g = {overall.g_mean:.2f} +/- {overall.stat_err:.2f} (stat) +/- {overall.syst_err:.2f} (syst) "
          f"m/s^2 [total {overall.total_err:.2f}]")
    print(f"chi2/ndf = {overall.chi2_ndf:.2f}, pull vs g_ref = {pull_overall:.2f} sigma")

    # --- results CSV ---
    csv_path = root / "report" / "combined_results_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["height_m", "g", "g_stat_err", "g_syst_err", "g_total_err", "chi2_ndf",
                          "n_trials_test", "n_trials_run1", "n_trials_pooled"])
        for h0 in heights_arr:
            combo, n_test, n_run1 = per_height[h0]
            writer.writerow([f"{h0:.3f}", f"{combo.g_mean:.4f}", f"{combo.stat_err:.4f}",
                              f"{combo.syst_err:.4f}", f"{combo.total_err:.4f}", f"{combo.chi2_ndf:.3f}",
                              n_test, n_run1, n_test + n_run1])
        writer.writerow([])
        writer.writerow(["overall", f"{overall.g_mean:.4f}", f"{overall.stat_err:.4f}",
                          f"{overall.syst_err:.4f}", f"{overall.total_err:.4f}", f"{overall.chi2_ndf:.3f}", "", "", ""])
    print(f"\nSaved: {csv_path}")

    # --- summary plot ---
    total_err_arr = np.sqrt(stat_arr**2 + syst_arr**2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(heights_arr, g_arr, yerr=total_err_arr, fmt="o", capsize=4, color="C0", label="pooled g (stat+syst)")
    ax.errorbar(heights_arr, g_arr, yerr=stat_arr, fmt="none", capsize=3, elinewidth=2.5, color="C0", alpha=0.6,
                label="statistical only")
    ax.axhline(g_ref, color="C3", ls="--", label=f"g_ref = {g_ref} m/s^2")
    ax.axhline(overall.g_mean, color="C1", ls=":", label=f"overall pooled = {overall.g_mean:.2f} m/s^2")
    ax.set_xlabel("Drop height $h_0$ (m, corrected)")
    ax.set_ylabel("$g$ (m/s$^2$)")
    ax.set_title(f"g vs. height, pooled test-data + Run 1 ({len(heights_arr)} heights)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "g_vs_height_pooled.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {figures_dir / 'g_vs_height_pooled.png'}")


if __name__ == "__main__":
    main()
