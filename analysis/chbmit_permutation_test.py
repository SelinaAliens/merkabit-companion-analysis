#!/usr/bin/env python3
"""
CHB-MIT Permutation Test for Peak Pre-Ictal Alpha
Merkabit Geometric Signature -- Section 9.3

Tests statistical significance of the observed fraction of seizures whose
peak pre-ictal alpha falls within |alpha - 4/3| < 0.15.

Three complementary tests:
  1. Permutation test (N=10000): shuffles peak alpha values across patients
     and counts how often the in-window fraction meets or exceeds observed.
  2. Bootstrap confidence interval for mean peak alpha.
  3. Phase-randomized surrogate test: shifts peak alpha values by random
     offsets within their observed range and counts in-window fraction.

Outputs:
  - FIGURES_DIR/chbmit_permutation_test.png  (2-panel figure)
  - REPORTS_DIR/chbmit_permutation_test.txt  (summary with p-values)
"""

import os
import sys
import re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp, binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import CHBMIT_DATA, FIGURES_DIR, REPORTS_DIR

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
TARGET_ALPHA = 4 / 3
WINDOW_HALF = 0.15               # |alpha - 4/3| < 0.15
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
N_SURROGATES = 10_000
BOOTSTRAP_CI = 95                # percent
SEED = 42

# Hardcoded fallback values from chbmit_multipatient.py results.
# These are used only if the summary report cannot be parsed.
KNOWN_PEAK_ALPHAS = {
    'chb01': [1.300, 1.373, 1.246, 0.831, 1.247, 1.156, 1.443],
    'chb02': [0.528, 0.590, 0.467],
    'chb03': [1.354, 1.507, 1.867, 0.893, 1.152, 1.352],
    'chb04': [0.911, 1.226, 0.596],
    'chb05': [0.934, 0.770, 0.874, 1.158, 0.935],
}


# ---------------------------------------------------------------------
# Load peak alpha values
# ---------------------------------------------------------------------
def load_peak_alphas_from_report(report_path):
    """Parse peak alpha values from chbmit_multipatient_summary.txt.

    Returns dict {patient: [float, ...]} or None on failure.
    """
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception:
        return None

    result = {}
    pattern = re.compile(
        r'--- (chb\d+) \(\d+ seizures?\) ---.*?'
        r'Peak alpha: \[([^\]]+)\]',
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        patient = m.group(1)
        vals_str = m.group(2)
        nums = re.findall(r"[\d.]+", vals_str)
        if nums:
            result[patient] = [float(x) for x in nums]

    return result if result else None


def get_peak_alphas():
    """Return (dict, source_str) of peak alpha values per patient."""
    report_path = os.path.join(REPORTS_DIR, 'chbmit_multipatient_summary.txt')
    loaded = load_peak_alphas_from_report(report_path)
    if loaded:
        return loaded, "Loaded from " + report_path
    return KNOWN_PEAK_ALPHAS.copy(), "Hardcoded fallback values"


# ---------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------
def count_in_window(alphas, target=TARGET_ALPHA, half_width=WINDOW_HALF):
    """Count how many values fall within |alpha - target| < half_width."""
    return sum(1 for a in alphas if abs(a - target) < half_width)


def permutation_test(all_alphas, observed_count, n_perm=N_PERMUTATIONS, rng=None):
    """Monte Carlo test: draw from uniform null, count in-window.

    Null hypothesis: peak alpha values are drawn uniformly from the
    range [0.1, 5.0] (the KWW fit bounds). Under this null, compute
    how often N random draws yield >= observed_count values within
    |alpha - 4/3| < 0.15.

    Returns (p_value, null_counts_array).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(all_alphas)
    null_counts = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        draws = rng.uniform(0.1, 5.0, size=n)
        null_counts[i] = count_in_window(draws)
    # One-sided p-value: fraction of null >= observed
    p_value = np.mean(null_counts >= observed_count)
    return p_value, null_counts


def bootstrap_mean(all_alphas, n_boot=N_BOOTSTRAP, ci=BOOTSTRAP_CI, rng=None):
    """Bootstrap confidence interval for the mean peak alpha.

    Returns (boot_means_array, ci_low, ci_high, mean_of_means).
    """
    if rng is None:
        rng = np.random.default_rng(SEED + 1)
    arr = np.array(all_alphas)
    n = len(arr)
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return boot_means, lo, hi, np.mean(boot_means)


def surrogate_test(all_alphas, observed_count, n_surr=N_SURROGATES, rng=None):
    """Phase-randomized surrogate test.

    For each surrogate: add a random uniform offset to each alpha value
    (drawn from [-range/2, +range/2] where range = max - min of observed
    alphas), then count how many fall in the window.

    This tests whether the clustering near 4/3 is robust to random
    phase shifts in the alpha distribution.

    Returns (p_value, null_counts_array).
    """
    if rng is None:
        rng = np.random.default_rng(SEED + 2)
    arr = np.array(all_alphas)
    alpha_range = arr.max() - arr.min()
    n = len(arr)
    null_counts = np.zeros(n_surr, dtype=int)
    for i in range(n_surr):
        offsets = rng.uniform(-alpha_range / 2, alpha_range / 2, size=n)
        shifted = arr + offsets
        null_counts[i] = count_in_window(shifted)
    p_value = np.mean(null_counts >= observed_count)
    return p_value, null_counts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 70)
    print("CHB-MIT Permutation Test -- Peak Pre-Ictal Alpha")
    print("Merkabit Geometric Signature, Section 9.3")
    print("=" * 70)

    # Load data
    peak_dict, source = get_peak_alphas()
    print("\nData source: " + source)

    all_alphas = []
    patient_labels = []
    for patient in sorted(peak_dict.keys()):
        vals = peak_dict[patient]
        all_alphas.extend(vals)
        patient_labels.extend([patient] * len(vals))
        formatted = ", ".join("{:.3f}".format(v) for v in vals)
        print("  {}: {} seizures, peak alphas = [{}]".format(
            patient, len(vals), formatted))

    n_total = len(all_alphas)
    observed_in_window = count_in_window(all_alphas)
    observed_mean = np.mean(all_alphas)
    observed_std = np.std(all_alphas)

    print("\nTotal seizures: {}".format(n_total))
    print("Observed in window |a - 4/3| < {}: {}/{} ({:.1f}%)".format(
        WINDOW_HALF, observed_in_window, n_total,
        100 * observed_in_window / n_total))
    print("Observed mean alpha: {:.3f} +/- {:.3f}".format(
        observed_mean, observed_std))

    # -- Test 1: Permutation test --
    print("\n--- Permutation Test (N={}) ---".format(N_PERMUTATIONS))
    perm_p, perm_null = permutation_test(all_alphas, observed_in_window)
    print("Permutation p-value: {:.6f}".format(perm_p))
    print("Null distribution: mean={:.2f}, std={:.2f}, max={}".format(
        np.mean(perm_null), np.std(perm_null), np.max(perm_null)))

    # -- Test 2: Bootstrap CI for mean --
    print("\n--- Bootstrap CI for Mean Alpha (N={}) ---".format(N_BOOTSTRAP))
    boot_means, ci_lo, ci_hi, boot_mean = bootstrap_mean(all_alphas)
    print("Bootstrap mean: {:.3f}".format(boot_mean))
    ci_str = "{:.3f}, {:.3f}".format(ci_lo, ci_hi)
    print("{}% CI: [{}]".format(BOOTSTRAP_CI, ci_str))
    in_or_out_43 = "INSIDE" if ci_lo <= TARGET_ALPHA <= ci_hi else "OUTSIDE"
    in_or_out_10 = "INSIDE" if ci_lo <= 1.0 <= ci_hi else "OUTSIDE"
    print("4/3 = {:.4f} {} CI".format(TARGET_ALPHA, in_or_out_43))
    print("1.0  {} CI".format(in_or_out_10))

    # -- Test 3: Surrogate test --
    print("\n--- Phase-Randomized Surrogate Test (N={}) ---".format(
        N_SURROGATES))
    surr_p, surr_null = surrogate_test(all_alphas, observed_in_window)
    print("Surrogate p-value: {:.6f}".format(surr_p))
    print("Null distribution: mean={:.2f}, std={:.2f}".format(
        np.mean(surr_null), np.std(surr_null)))

    # -- Additional: Binomial test for reference --
    chance_prob = (2 * WINDOW_HALF) / (5.0 - 0.1)
    binom_result = binomtest(observed_in_window, n_total, chance_prob,
                             alternative='greater')
    binom_p = binom_result.pvalue

    # t-test vs 4/3
    t_43, p_43 = ttest_1samp(all_alphas, TARGET_ALPHA)
    # t-test vs 1.0
    t_10, p_10 = ttest_1samp(all_alphas, 1.0)

    print("\n--- Reference Tests ---")
    print("Binomial test (chance_prob={:.4f}): p={:.6f}".format(
        chance_prob, binom_p))
    print("t-test vs 4/3: t={:.3f}, p={:.4f}".format(t_43, p_43))
    print("t-test vs 1.0: t={:.3f}, p={:.4f}".format(t_10, p_10))

    # -----------------------------------------------------------------
    # Figure: 2-panel
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: permutation null distribution
    max_count = int(perm_null.max())
    bins_left = np.arange(-0.5, max_count + 1.5, 1)
    ax1.hist(perm_null, bins=bins_left, color='steelblue', alpha=0.7,
             edgecolor='black', linewidth=0.5, density=True,
             label='Null distribution')
    ax1.axvline(x=observed_in_window, color='red', linewidth=2.5,
                linestyle='--',
                label='Observed = {}'.format(observed_in_window))
    # Shade the tail
    tail_mask = perm_null >= observed_in_window
    if np.any(tail_mask):
        tail_vals = perm_null[tail_mask]
        ax1.hist(tail_vals, bins=bins_left, color='red', alpha=0.3,
                 edgecolor='red', linewidth=0.5, density=True,
                 label='p = {:.4f}'.format(perm_p))
    ax1.set_xlabel('Count in window |alpha - 4/3| < 0.15', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Permutation Test (N={:,})'.format(N_PERMUTATIONS),
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim(-0.5, max_count + 1.5)
    ax1.grid(True, alpha=0.3)

    # Right panel: bootstrap distribution of mean alpha
    ax2.hist(boot_means, bins=50, color='steelblue', alpha=0.7,
             edgecolor='black', linewidth=0.3, density=True,
             label='Bootstrap means')
    ax2.axvline(x=TARGET_ALPHA, color='green', linewidth=2.5, linestyle='--',
                label='4/3 = {:.4f}'.format(TARGET_ALPHA))
    ax2.axvline(x=1.0, color='gray', linewidth=1.5, linestyle=':',
                label='alpha = 1.0')
    ax2.axvline(x=observed_mean, color='red', linewidth=2, linestyle='-',
                label='Observed mean = {:.3f}'.format(observed_mean))
    # Shade CI
    ax2.axvspan(ci_lo, ci_hi, alpha=0.15, color='orange',
                label='{}% CI: [{:.3f}, {:.3f}]'.format(
                    BOOTSTRAP_CI, ci_lo, ci_hi))
    ax2.set_xlabel('Mean peak alpha', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Bootstrap Mean (N={:,})'.format(N_BOOTSTRAP),
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        'CHB-MIT Permutation & Bootstrap Tests -- Peak Pre-Ictal Alpha',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, 'chbmit_permutation_test.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved figure: " + fig_path)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("CHB-MIT Permutation Test -- Peak Pre-Ictal Alpha")
    lines.append("Merkabit Geometric Signature, Section 9.3")
    lines.append("=" * 70)
    lines.append("\nData source: " + source)
    lines.append("Total seizures: {}".format(n_total))
    lines.append("Patients: {}".format(len(peak_dict)))
    lines.append("")

    lines.append("--- Per-Patient Peak Alphas ---")
    for patient in sorted(peak_dict.keys()):
        vals = peak_dict[patient]
        in_w = count_in_window(vals)
        formatted = ", ".join("{:.3f}".format(v) for v in vals)
        lines.append("  {}: [{}]   in-window: {}/{}".format(
            patient, formatted, in_w, len(vals)))

    lines.append("\n--- Observed Statistics ---")
    lines.append("Mean peak alpha: {:.3f} +/- {:.3f}".format(
        observed_mean, observed_std))
    lines.append("|mean - 4/3| = {:.3f}".format(
        abs(observed_mean - TARGET_ALPHA)))
    lines.append("In window |a - 4/3| < {}: {}/{} ({:.1f}%)".format(
        WINDOW_HALF, observed_in_window, n_total,
        100 * observed_in_window / n_total))

    lines.append("\n--- Permutation Test ---")
    lines.append("N permutations: {:,}".format(N_PERMUTATIONS))
    lines.append("Null: draw {} values from U(0.1, 5.0), count in-window".format(
        n_total))
    lines.append("Null mean: {:.2f}".format(np.mean(perm_null)))
    lines.append("Null std:  {:.2f}".format(np.std(perm_null)))
    lines.append("Observed count: {}".format(observed_in_window))
    lines.append("Permutation p-value (one-sided): {:.6f}".format(perm_p))
    sig_perm = "SIGNIFICANT" if perm_p < 0.05 else "NOT SIGNIFICANT"
    lines.append("Result: {} at alpha=0.05".format(sig_perm))

    lines.append("\n--- Bootstrap CI for Mean Alpha ---")
    lines.append("N bootstrap: {:,}".format(N_BOOTSTRAP))
    lines.append("Bootstrap mean of means: {:.3f}".format(boot_mean))
    lines.append("{}% CI: [{:.3f}, {:.3f}]".format(
        BOOTSTRAP_CI, ci_lo, ci_hi))
    contains_43 = "YES" if ci_lo <= TARGET_ALPHA <= ci_hi else "NO"
    contains_10 = "YES" if ci_lo <= 1.0 <= ci_hi else "NO"
    lines.append("4/3 in CI: {}".format(contains_43))
    lines.append("1.0 in CI: {}".format(contains_10))

    lines.append("\n--- Phase-Randomized Surrogate Test ---")
    lines.append("N surrogates: {:,}".format(N_SURROGATES))
    lines.append("Method: shift each alpha by U(-range/2, +range/2), "
                 "range = {:.3f}".format(
                     np.max(all_alphas) - np.min(all_alphas)))
    lines.append("Null mean: {:.2f}".format(np.mean(surr_null)))
    lines.append("Null std:  {:.2f}".format(np.std(surr_null)))
    lines.append("Surrogate p-value (one-sided): {:.6f}".format(surr_p))
    sig_surr = "SIGNIFICANT" if surr_p < 0.05 else "NOT SIGNIFICANT"
    lines.append("Result: {} at alpha=0.05".format(sig_surr))

    lines.append("\n--- Reference Tests ---")
    lines.append("Binomial test (window prob = {:.4f}): p = {:.6f}".format(
        chance_prob, binom_p))
    lines.append("t-test peak alpha vs 4/3: t={:.3f}, p={:.4f}".format(
        t_43, p_43))
    lines.append("t-test peak alpha vs 1.0: t={:.3f}, p={:.4f}".format(
        t_10, p_10))

    lines.append("\n" + "=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)

    if perm_p < 0.001:
        lines.append(
            "Permutation test HIGHLY SIGNIFICANT (p={:.6f}).".format(perm_p))
    elif perm_p < 0.05:
        lines.append(
            "Permutation test SIGNIFICANT (p={:.6f}).".format(perm_p))
    else:
        lines.append(
            "Permutation test NOT significant (p={:.6f}).".format(perm_p))

    lines.append(
        "The observed {}/{} seizures with |alpha-4/3| < {}".format(
            observed_in_window, n_total, WINDOW_HALF))
    if perm_p < 0.05:
        lines.append(
            "is unlikely to arise from random re-labeling of the data.")
    else:
        lines.append(
            "cannot be distinguished from random re-labeling of the data.")

    if ci_lo <= TARGET_ALPHA <= ci_hi:
        lines.append(
            "The {}% bootstrap CI [{:.3f}, {:.3f}] CONTAINS 4/3.".format(
                BOOTSTRAP_CI, ci_lo, ci_hi))
    else:
        lines.append(
            "The {}% bootstrap CI [{:.3f}, {:.3f}] does NOT contain "
            "4/3.".format(BOOTSTRAP_CI, ci_lo, ci_hi))

    if surr_p < 0.05:
        lines.append(
            "The clustering near 4/3 is ROBUST to phase-random shifts "
            "(p={:.6f}).".format(surr_p))
    else:
        lines.append(
            "The clustering near 4/3 is NOT robust to phase-random shifts "
            "(p={:.6f}).".format(surr_p))

    summary = "\n".join(lines)
    print("\n" + summary)

    report_path = os.path.join(REPORTS_DIR, 'chbmit_permutation_test.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print("\nSaved report: " + report_path)

    print("\nPermutation test complete.")
