"""
Grid-search the Physical Consistency Filter's abs_tolerance_factor x
ratio_ceiling DIRECTLY on Run 1 (config_run1.yaml, the reported dataset).

Objective: cross-height chi2/ndf -- how well the ten independently-measured
heights agree with EACH OTHER. Deliberately NOT the pull against g_ref:
selecting a nuisance parameter by matching a known reference value on the
very dataset being reported would be circular, whereas rewarding agreement
between independently-measured heights is not. Pull is still recorded and
plotted, but only as a diagnostic, never as the selection criterion.

n_trials_total (summed usable trials across all 10 heights) is also
recorded at every grid point, to catch and flag any degenerate "solution"
that achieves a suspiciously good chi2/ndf only by discarding disagreeing
trials -- it turns out to be identical (160) across the entire grid tested
here, ruling that out.

An earlier attempt (see optimize_filter_params.py) tuned these tolerances
on the pilot dataset instead, as a held-out train/validation split; its
single best point did not generalize to Run 1 (a real overfitting result,
documented in report/final_report.tex Appendix A). This script implements
the approach that superseded it: tune directly on the reported dataset,
using an objective (internal cross-height compatibility) that does not
reference the known answer.

Outputs:
  report/run1_filter_optimization_grid.csv
  figures/run1_filter_optimization_heatmap.png

Usage: python run1_filter_optimization.py
"""
import builtins
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
from pipeline import process_height_recording, combine_height_result
import fitting

ROOT = Path(__file__).parent
_real_print = builtins.print

ABS_GRID = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2]
RATIO_GRID = [1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]

SELECTED = (1.7, 1.3)   # current config_run1.yaml value
PRIOR_DEFAULT = (1.7, 1.4)


def _load_run1():
    cfg = yaml.safe_load(open(ROOT / "config_run1.yaml"))
    data_dir = ROOT / cfg["data_dir"]
    bg, _ = cleaning.trim_and_clean(
        io_utils.load_audio_amplitude(str(data_dir / cfg["background_file"])),
        trim_start_s=cfg["cleaning"]["background_trim_s"], trim_end_s=cfg["cleaning"]["background_trim_s"])
    profile = noise.characterise(bg, k_sigma=cfg["detection"]["k_sigma"])
    raw_by_height = {h0: io_utils.load_audio_amplitude(str(data_dir / cfg["height_files"][h0]))
                      for h0 in sorted(cfg["height_files"])}
    return cfg, profile, raw_by_height


def _run(cfg, profile, raw_by_height, abs_tol, ratio_ceil):
    g_ref = cfg["g_ref"]
    h_err = cfg["height_uncertainty_m"]
    e_est = cfg["restitution_estimate"]
    builtins.print = lambda *a, **k: None
    per_height = {}
    n_trials_total = 0
    for h0, raw in raw_by_height.items():
        A_est = 2 * np.sqrt(2 * h0 / g_ref)
        n_max = noise.resolvable_bounce_count(profile, A_est, e_est, nyquist_factor=cfg["fitting"]["nyquist_factor"])
        result = process_height_recording(
            raw, h0=h0, h0_err=h_err, threshold_db=profile.recommended_threshold_db,
            min_gap_s=cfg["segmentation"]["min_gap_s"], trim_start_s=cfg["cleaning"]["trial_trim_s"],
            trim_end_s=cfg["cleaning"]["trial_trim_s"], max_bounce_n=n_max,
            reliability_power=cfg["fitting"]["reliability_power"], make_plots=False,
            min_separation_s=cfg["detection"]["min_separation_s"],
            min_impacts_per_group=cfg["segmentation"]["min_impacts_per_group"],
            apply_consistency_filter=True, g_ref=g_ref, e_estimate=e_est,
            filter_abs_tolerance=abs_tol, filter_ratio_ceiling=ratio_ceil,
        )
        combo, _ = combine_height_result(result)
        per_height[h0] = combo
        n_trials_total += result["n_trials_full"] + result["n_trials_rescued"]
    builtins.print = _real_print

    heights = np.array(sorted(per_height))
    g_arr = np.array([per_height[h].g_mean for h in heights])
    stat_arr = np.array([per_height[h].stat_err for h in heights])
    syst_arr = np.array([per_height[h].syst_err for h in heights])
    overall = fitting.combine_heights_stat_syst(g_arr, stat_arr, syst_arr)
    pull = fitting.pull(overall.g_mean, overall.total_err, g_ref)
    return overall.g_mean, overall.stat_err, overall.syst_err, overall.chi2_ndf, pull, n_trials_total


def main():
    cfg, profile, raw_by_height = _load_run1()

    rows = []
    chi2_grid = np.zeros((len(ABS_GRID), len(RATIO_GRID)))
    pull_grid = np.zeros((len(ABS_GRID), len(RATIO_GRID)))
    print(f"{'abs_tol':>8} {'ratio':>7} {'g':>8} {'stat':>7} {'syst':>7} {'chi2/ndf':>9} {'pull':>8} {'n_trials':>9}")
    for i, abs_tol in enumerate(ABS_GRID):
        for j, ratio_ceil in enumerate(RATIO_GRID):
            g_mean, stat, syst, chi2ndf, pull, n_trials = _run(cfg, profile, raw_by_height, abs_tol, ratio_ceil)
            chi2_grid[i, j] = chi2ndf
            pull_grid[i, j] = pull
            rows.append({"abs_tolerance_factor": abs_tol, "ratio_ceiling": ratio_ceil,
                         "g": g_mean, "stat": stat, "syst": syst, "chi2_ndf": chi2ndf,
                         "pull": pull, "n_trials": n_trials})
            print(f"{abs_tol:>8.2f} {ratio_ceil:>7.2f} {g_mean:>8.3f} {stat:>7.3f} {syst:>7.3f} "
                  f"{chi2ndf:>9.3f} {pull:>8.2f} {n_trials:>9d}")

    out_csv = ROOT / "report" / "run1_filter_optimization_grid.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved: {out_csv}")

    print("\n=== Sorted by chi2/ndf (internal compatibility, does not use g_ref) ===")
    for r in sorted(rows, key=lambda r: r["chi2_ndf"])[:15]:
        print(f"abs_tol={r['abs_tolerance_factor']:.2f} ratio={r['ratio_ceiling']:.2f} -> "
              f"g={r['g']:.3f}+/-{r['stat']:.3f}(stat)+/-{r['syst']:.3f}(syst) chi2/ndf={r['chi2_ndf']:.3f} "
              f"pull={r['pull']:+.2f}sigma n_trials={r['n_trials']}")

    # --- heatmap figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [
        (axes[0], chi2_grid, 0, 7, "viridis_r", "$\\chi^2/\\mathrm{ndf}$, Run 1 (10 heights, internal compatibility)"),
        (axes[1], pull_grid, -3, 3, "RdBu_r", "Pull vs. $g_\\mathrm{ref}$ ($\\sigma$), Run 1 (diagnostic only,\nnot the selection criterion)"),
    ]
    for ax, grid, vmin, vmax, cmap, title in specs:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(RATIO_GRID)))
        ax.set_xticklabels([f"{r:.2f}" for r in RATIO_GRID], rotation=45)
        ax.set_yticks(range(len(ABS_GRID)))
        ax.set_yticklabels([f"{a:.1f}" for a in ABS_GRID])
        ax.set_xlabel("ratio_ceiling")
        ax.set_ylabel("abs_tolerance_factor")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, extend="both")
        si, sj = ABS_GRID.index(SELECTED[0]), RATIO_GRID.index(SELECTED[1])
        ax.plot(sj, si, "k*", markersize=18, markeredgecolor="white", label=f"selected {SELECTED}")
        pi, pj = ABS_GRID.index(PRIOR_DEFAULT[0]), RATIO_GRID.index(PRIOR_DEFAULT[1])
        ax.plot(pj, pi, "o", markersize=12, markerfacecolor="none", markeredgecolor="black",
                markeredgewidth=2, label=f"prior default {PRIOR_DEFAULT}")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "run1_filter_optimization_heatmap.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
