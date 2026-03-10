#!/usr/bin/env python3
"""
CHB-MIT Seizure Classification by KWW Signature
================================================
Paper Section 9.3 -- Seizure-level feature analysis

Classifies each seizure from the multi-patient CHB-MIT analysis by its
KWW signature strength and identifies which features predict a strong
cooperative (alpha ~ 4/3) signal.

Classification scheme:
  - strong_signal : |peak_alpha - 4/3| < 0.15   (in cooperative window)
  - above_lock    : peak_alpha > 1.036 but not in cooperative window
  - weak_signal   : peak_alpha <= 1.036

Uses hardcoded results from chbmit_multipatient.py (24 seizures, 5 patients).

March 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import CHBMIT_DATA, FIGURES_DIR, REPORTS_DIR

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
TARGET_ALPHA = 4 / 3
COOP_WINDOW = 0.15       # |peak_alpha - 4/3| threshold for strong signal
LOCK_THRESHOLD = 1.036   # first lock threshold (from Phase 1 analysis)

# -----------------------------------------------------------------
# Hardcoded seizure results from chbmit_multipatient.py
# Columns:
#   patient, file, sz_start, sz_end,
#   alpha_pre, alpha_near, alpha_int, alpha_ict,
#   peak_alpha, tau_pre, r2_pre
# -----------------------------------------------------------------
SEIZURE_DATA = [
    # chb01 -- 7 seizures
    ('chb01', 'chb01_03.edf', 2996, 3036, 0.644, 0.896, 1.006, 0.155, 1.300, 0.89, 0.92),
    ('chb01', 'chb01_04.edf', 1467, 1494, 0.712, 0.954, 1.006, 0.412, 1.373, 1.12, 0.88),
    ('chb01', 'chb01_15.edf', 1732, 1772, 0.589, 0.823, 1.006, 0.298, 1.246, 0.76, 0.91),
    ('chb01', 'chb01_16.edf', 1015, 1066, 0.501, 0.445, 1.006, 1.907, 0.831, 0.65, 0.85),
    ('chb01', 'chb01_18.edf', 1720, 1810, 0.678, 0.912, 0.496, 0.789, 1.247, 0.95, 0.89),
    ('chb01', 'chb01_21.edf',  327,  420, 0.534, 0.867, 0.496, 0.534, 1.156, 0.51, 0.87),
    ('chb01', 'chb01_26.edf', 1862, 1963, 0.756, 1.123, 0.496, 1.458, 1.443, 1.32, 0.93),
    # chb02 -- 3 seizures
    ('chb02', 'chb02_16.edf',   130,  212, 0.412, 0.389, 0.623, 0.345, 0.528, 0.45, 0.78),
    ('chb02', 'chb02_16+.edf', 2972, 3053, 0.467, 0.512, 0.623, 0.289, 0.590, 0.52, 0.81),
    ('chb02', 'chb02_19.edf',  3369, 3378, 0.356, 0.345, 0.623, 0.167, 0.467, 0.38, 0.72),
    # chb03 -- 6 seizures
    ('chb03', 'chb03_01.edf',  362,  414, 1.123, 1.245, 0.867, 0.789, 1.354, 0.92, 0.91),
    ('chb03', 'chb03_02.edf',  731,  796, 1.267, 1.389, 0.867, 0.923, 1.507, 1.05, 0.89),
    ('chb03', 'chb03_03.edf',  432,  501, 1.445, 1.678, 0.867, 1.123, 1.867, 1.23, 0.85),
    ('chb03', 'chb03_04.edf', 2162, 2214, 0.712, 0.789, 0.867, 0.567, 0.893, 0.67, 0.83),
    ('chb03', 'chb03_34.edf', 1982, 2029, 0.923, 1.045, 0.867, 0.812, 1.152, 0.89, 0.88),
    ('chb03', 'chb03_35.edf', 2592, 2656, 1.089, 1.234, 0.867, 0.934, 1.352, 0.98, 0.90),
    # chb04 -- 3 seizures
    ('chb04', 'chb04_05.edf', 7804, 7853, 0.678, 0.812, 0.534, 0.456, 0.911, 0.72, 0.84),
    ('chb04', 'chb04_08.edf', 6446, 6557, 0.912, 1.067, 0.534, 0.723, 1.226, 0.95, 0.88),
    ('chb04', 'chb04_28.edf', 1679, 1781, 0.445, 0.523, 0.534, 0.312, 0.596, 0.48, 0.79),
    # chb05 -- 5 seizures
    ('chb05', 'chb05_06.edf',  417,  532, 0.712, 0.834, 0.612, 0.534, 0.934, 0.78, 0.86),
    ('chb05', 'chb05_13.edf', 1086, 1196, 0.589, 0.623, 0.612, 0.445, 0.770, 0.62, 0.82),
    ('chb05', 'chb05_16.edf', 2317, 2413, 0.656, 0.756, 0.612, 0.389, 0.874, 0.71, 0.84),
    ('chb05', 'chb05_17.edf', 2451, 2571, 0.889, 1.023, 0.612, 0.734, 1.158, 0.92, 0.89),
    ('chb05', 'chb05_22.edf', 2348, 2465, 0.723, 0.834, 0.612, 0.567, 0.935, 0.79, 0.85),
]

COLUMN_NAMES = [
    'patient', 'file', 'sz_start', 'sz_end',
    'alpha_pre', 'alpha_near', 'alpha_int', 'alpha_ict',
    'peak_alpha', 'tau_pre', 'r2_pre',
]


# -----------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------
def classify_seizure(peak_alpha):
    """Classify a seizure by its peak KWW alpha value."""
    if abs(peak_alpha - TARGET_ALPHA) < COOP_WINDOW:
        return 'strong_signal'
    elif peak_alpha > LOCK_THRESHOLD:
        return 'above_lock'
    else:
        return 'weak_signal'


def build_feature_table(data):
    """
    Build a list of feature dictionaries from raw seizure data.

    Each entry contains the original columns plus derived features:
      - seizure_duration
      - pre_ictal_available
      - alpha_contrast  (alpha_near - alpha_int)
      - alpha_rise      (peak_alpha - alpha_pre)
      - classification
    """
    records = []
    for row in data:
        d = dict(zip(COLUMN_NAMES, row))

        # Derived features
        d['seizure_duration'] = d['sz_end'] - d['sz_start']
        d['pre_ictal_available'] = min(d['sz_start'], 300)
        d['alpha_contrast'] = d['alpha_near'] - d['alpha_int']
        d['alpha_rise'] = d['peak_alpha'] - d['alpha_pre']
        d['classification'] = classify_seizure(d['peak_alpha'])

        records.append(d)
    return records


def feature_importance(records, feature_names):
    """
    For each feature, compute descriptive statistics for the strong-signal
    vs weak-signal groups, Mann-Whitney U p-value, and Spearman correlation
    with peak_alpha.

    Returns a list of dicts sorted by Mann-Whitney p-value (ascending).
    """
    strong = [r for r in records if r['classification'] == 'strong_signal']
    weak = [r for r in records if r['classification'] == 'weak_signal']

    results = []
    for feat in feature_names:
        vals_strong = np.array([r[feat] for r in strong])
        vals_weak = np.array([r[feat] for r in weak])
        all_vals = np.array([r[feat] for r in records])
        all_peak = np.array([r['peak_alpha'] for r in records])

        # Descriptive stats
        entry = {
            'feature': feat,
            'strong_mean': np.mean(vals_strong) if len(vals_strong) > 0 else np.nan,
            'strong_median': np.median(vals_strong) if len(vals_strong) > 0 else np.nan,
            'weak_mean': np.mean(vals_weak) if len(vals_weak) > 0 else np.nan,
            'weak_median': np.median(vals_weak) if len(vals_weak) > 0 else np.nan,
        }

        # Mann-Whitney U (strong vs weak)
        if len(vals_strong) >= 2 and len(vals_weak) >= 2:
            stat, pval = mannwhitneyu(vals_strong, vals_weak, alternative='two-sided')
            entry['mw_stat'] = stat
            entry['mw_pval'] = pval
        else:
            entry['mw_stat'] = np.nan
            entry['mw_pval'] = np.nan

        # Spearman correlation with peak_alpha
        if len(all_vals) >= 3:
            rho, sp_pval = spearmanr(all_vals, all_peak)
            entry['spearman_rho'] = rho
            entry['spearman_pval'] = sp_pval
        else:
            entry['spearman_rho'] = np.nan
            entry['spearman_pval'] = np.nan

        results.append(entry)

    # Sort by Mann-Whitney p-value (most significant first)
    results.sort(key=lambda x: x['mw_pval'] if not np.isnan(x['mw_pval']) else 999)
    return results


def patient_stratification(records):
    """
    Compute per-patient classification breakdown and consistency scores.

    Returns a dict: patient -> {n_seizures, n_strong, n_above_lock, n_weak,
                                frac_strong, frac_above_lock, consistency}
    """
    patients = sorted(set(r['patient'] for r in records))
    strat = {}
    for pat in patients:
        pat_recs = [r for r in records if r['patient'] == pat]
        n = len(pat_recs)
        n_strong = sum(1 for r in pat_recs if r['classification'] == 'strong_signal')
        n_above = sum(1 for r in pat_recs if r['classification'] == 'above_lock')
        n_weak = sum(1 for r in pat_recs if r['classification'] == 'weak_signal')

        frac_strong = n_strong / n
        frac_above_lock = (n_strong + n_above) / n

        # Consistency score: fraction of seizures in the dominant class
        dominant_count = max(n_strong, n_above, n_weak)
        consistency = dominant_count / n

        strat[pat] = {
            'n_seizures': n,
            'n_strong': n_strong,
            'n_above_lock': n_above,
            'n_weak': n_weak,
            'frac_strong': frac_strong,
            'frac_above_lock': frac_above_lock,
            'consistency': consistency,
        }
    return strat


# -----------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------
def make_figure(records, feat_results, strat, fig_path):
    """Generate the 4-panel classification figure."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('CHB-MIT Seizure Classification by KWW Signature',
                 fontsize=14, fontweight='bold', y=0.98)

    class_colors = {
        'strong_signal': '#2ca02c',
        'above_lock': '#ff7f0e',
        'weak_signal': '#d62728',
    }
    class_labels = {
        'strong_signal': 'Strong (|a-4/3|<0.15)',
        'above_lock': 'Above lock (a>1.036)',
        'weak_signal': 'Weak (a<=1.036)',
    }

    # -- Panel 1: Feature importance (Mann-Whitney -log10 p-values) --
    ax1 = axes[0, 0]
    feat_names = [f['feature'] for f in feat_results]
    pvals = [f['mw_pval'] for f in feat_results]
    neg_log_p = [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in pvals]

    y_pos = np.arange(len(feat_names))
    bar_colors = ['#1f77b4' if nlp > -np.log10(0.05) else '#aec7e8' for nlp in neg_log_p]
    ax1.barh(y_pos, neg_log_p, color=bar_colors, edgecolor='k', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feat_names, fontsize=8)
    ax1.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax1.axvline(-np.log10(0.01), color='darkred', linestyle=':', linewidth=1, label='p=0.01')
    ax1.set_xlabel(r'$-\log_{10}(p)$  [Mann-Whitney U]', fontsize=10)
    ax1.set_title('Feature Importance: Strong vs Weak Signal', fontsize=11)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.invert_yaxis()

    # -- Panel 2: Patient x signal-type heatmap (confusion-matrix style) --
    ax2 = axes[0, 1]
    patients = sorted(strat.keys())
    class_order = ['strong_signal', 'above_lock', 'weak_signal']
    matrix = np.zeros((len(patients), len(class_order)), dtype=int)
    for i, pat in enumerate(patients):
        matrix[i, 0] = strat[pat]['n_strong']
        matrix[i, 1] = strat[pat]['n_above_lock']
        matrix[i, 2] = strat[pat]['n_weak']

    im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    ax2.set_xticks(range(len(class_order)))
    ax2.set_xticklabels(['Strong', 'Above lock', 'Weak'], fontsize=9)
    ax2.set_yticks(range(len(patients)))
    ax2.set_yticklabels(patients, fontsize=9)
    ax2.set_xlabel('Signal Classification', fontsize=10)
    ax2.set_ylabel('Patient', fontsize=10)
    ax2.set_title('Patient x Classification Matrix', fontsize=11)
    for i in range(len(patients)):
        for j in range(len(class_order)):
            val = matrix[i, j]
            text_color = 'white' if val >= 3 else 'black'
            ax2.text(j, i, str(val), ha='center', va='center',
                     fontsize=12, fontweight='bold', color=text_color)
    fig.colorbar(im, ax=ax2, shrink=0.7, label='Count')

    # -- Panel 3: alpha_contrast vs peak_alpha scatter --
    ax3 = axes[1, 0]
    for cls in class_order:
        subset = [r for r in records if r['classification'] == cls]
        if not subset:
            continue
        x = [r['alpha_contrast'] for r in subset]
        y = [r['peak_alpha'] for r in subset]
        ax3.scatter(x, y, c=class_colors[cls], label=class_labels[cls],
                    s=60, edgecolors='k', linewidth=0.5, alpha=0.85, zorder=3)

    ax3.axhline(TARGET_ALPHA, color='gray', linestyle='--', linewidth=1,
                label=r'$\alpha = 4/3$', zorder=1)
    ax3.axhspan(TARGET_ALPHA - COOP_WINDOW, TARGET_ALPHA + COOP_WINDOW,
                color='green', alpha=0.08, zorder=0, label='Cooperative window')
    ax3.axhline(LOCK_THRESHOLD, color='orange', linestyle=':', linewidth=1,
                label='Lock threshold (1.036)', zorder=1)
    ax3.set_xlabel(r'$\alpha_{contrast}$  (near-onset $-$ interictal)', fontsize=10)
    ax3.set_ylabel(r'Peak $\alpha$', fontsize=10)
    ax3.set_title('Alpha Contrast vs Peak Alpha', fontsize=11)
    ax3.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # -- Panel 4: Patient-level stacked bar chart --
    ax4 = axes[1, 1]
    x_pos = np.arange(len(patients))
    width = 0.6
    bottoms = np.zeros(len(patients))

    for cls in class_order:
        counts = []
        for pat in patients:
            if cls == 'strong_signal':
                counts.append(strat[pat]['n_strong'])
            elif cls == 'above_lock':
                counts.append(strat[pat]['n_above_lock'])
            else:
                counts.append(strat[pat]['n_weak'])
        counts = np.array(counts, dtype=float)
        ax4.bar(x_pos, counts, width, bottom=bottoms,
                color=class_colors[cls], edgecolor='k', linewidth=0.5,
                label=class_labels[cls])
        bottoms += counts

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(patients, fontsize=9)
    ax4.set_xlabel('Patient', fontsize=10)
    ax4.set_ylabel('Number of Seizures', fontsize=10)
    ax4.set_title('Seizure Classification per Patient', fontsize=11)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim(0, max(bottoms) + 1)
    for i, pat in enumerate(patients):
        total = strat[pat]['n_seizures']
        ax4.text(i, bottoms[i] + 0.15, str(int(total)),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


# -----------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------
def write_report(records, feat_results, strat, report_path):
    """Write the text summary report."""

    strong = [r for r in records if r['classification'] == 'strong_signal']
    above = [r for r in records if r['classification'] == 'above_lock']
    weak = [r for r in records if r['classification'] == 'weak_signal']

    lines = []
    lines.append("=" * 72)
    lines.append("CHB-MIT Seizure Classification by KWW Signature")
    lines.append("Paper Section 9.3 -- Feature Importance Analysis")
    lines.append("=" * 72)
    lines.append("")

    # -- Overall summary --
    lines.append("CLASSIFICATION SUMMARY")
    lines.append("-" * 40)
    n_total = len(records)
    n_strong = len(strong)
    n_above = len(above)
    n_weak = len(weak)
    lines.append(f"Total seizures analysed:  {n_total}")
    lines.append(f"  Strong signal (|a-4/3|<0.15):  {n_strong}  "
                 f"({100*n_strong/n_total:.1f}%)")
    lines.append(f"  Above lock (a>1.036):           {n_above}  "
                 f"({100*n_above/n_total:.1f}%)")
    lines.append(f"  Weak signal (a<=1.036):         {n_weak}  "
                 f"({100*n_weak/n_total:.1f}%)")
    lines.append("")

    # -- Seizure-by-seizure table --
    lines.append("SEIZURE-LEVEL DETAIL")
    lines.append("-" * 40)
    header = (f"{'Patient':<8} {'File':<18} {'Peak_a':>7} {'Class':>14} "
              f"{'a_pre':>7} {'a_near':>7} {'a_int':>7} {'a_ict':>7} "
              f"{'tau':>6} {'R2':>5} {'dur':>5} {'contrast':>9} {'rise':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in records:
        cls_map = {
            'strong_signal': 'STRONG',
            'above_lock': 'ABOVE_LOCK',
            'weak_signal': 'WEAK',
        }
        cls_short = cls_map[r['classification']]
        line = (f"{r['patient']:<8} {r['file']:<18} {r['peak_alpha']:7.3f} "
                f"{cls_short:>14} {r['alpha_pre']:7.3f} {r['alpha_near']:7.3f} "
                f"{r['alpha_int']:7.3f} {r['alpha_ict']:7.3f} "
                f"{r['tau_pre']:6.2f} {r['r2_pre']:5.2f} "
                f"{r['seizure_duration']:5d} "
                f"{r['alpha_contrast']:9.3f} {r['alpha_rise']:6.3f}")
        lines.append(line)
    lines.append("")

    # -- Feature importance --
    lines.append("FEATURE IMPORTANCE (Strong vs Weak Signal)")
    lines.append("-" * 40)
    lines.append(f"{'Feature':<22} {'Strong_mean':>12} {'Weak_mean':>10} "
                 f"{'MW_p':>10} {'Spear_rho':>10} {'Sp_p':>10}")
    lines.append("-" * 76)
    for f in feat_results:
        sig_flag = ""
        if not np.isnan(f['mw_pval']):
            if f['mw_pval'] < 0.001:
                sig_flag = " ***"
            elif f['mw_pval'] < 0.01:
                sig_flag = " **"
            elif f['mw_pval'] < 0.05:
                sig_flag = " *"
        lines.append(
            f"{f['feature']:<22} {f['strong_mean']:12.4f} {f['weak_mean']:10.4f} "
            f"{f['mw_pval']:10.4f} {f['spearman_rho']:10.4f} "
            f"{f['spearman_pval']:10.4f}{sig_flag}"
        )
    lines.append("")
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("")

    # -- Patient stratification --
    lines.append("PATIENT STRATIFICATION")
    lines.append("-" * 40)
    lines.append(f"{'Patient':<8} {'N_sz':>5} {'Strong':>7} {'Above':>7} "
                 f"{'Weak':>7} {'%Strong':>8} {'%AbvLock':>9} {'Consist':>8}")
    lines.append("-" * 62)
    for pat in sorted(strat.keys()):
        s = strat[pat]
        lines.append(
            f"{pat:<8} {s['n_seizures']:5d} {s['n_strong']:7d} "
            f"{s['n_above_lock']:7d} {s['n_weak']:7d} "
            f"{100*s['frac_strong']:7.1f}% {100*s['frac_above_lock']:8.1f}% "
            f"{s['consistency']:8.2f}"
        )
    lines.append("")

    # -- Key findings --
    lines.append("KEY FINDINGS")
    lines.append("-" * 40)

    # Best-discriminating features
    top_feats = [f for f in feat_results
                 if not np.isnan(f['mw_pval']) and f['mw_pval'] < 0.05]
    if top_feats:
        lines.append(f"Significant features (p<0.05): {len(top_feats)}")
        for f in top_feats:
            lines.append(f"  {f['feature']}: MW p={f['mw_pval']:.4f}, "
                         f"rho={f['spearman_rho']:.3f}")
    else:
        lines.append("No features reached p<0.05 (Mann-Whitney, strong vs weak).")

    # Strongest Spearman correlations
    sorted_by_rho = sorted(feat_results,
                           key=lambda x: abs(x['spearman_rho'])
                           if not np.isnan(x['spearman_rho']) else 0,
                           reverse=True)
    lines.append("")
    lines.append("Top 3 Spearman correlations with peak_alpha:")
    for f in sorted_by_rho[:3]:
        lines.append(f"  {f['feature']}: rho={f['spearman_rho']:.3f}, "
                     f"p={f['spearman_pval']:.4f}")

    # Patient with strongest signal
    best_pat = max(strat, key=lambda p: strat[p]['frac_strong'])
    bp = strat[best_pat]
    lines.append("")
    lines.append(f"Strongest patient: {best_pat} "
                 f"({100*bp['frac_strong']:.0f}% in cooperative window, "
                 f"consistency={bp['consistency']:.2f})")

    # Patient with no signal
    no_signal = [p for p in strat if strat[p]['n_strong'] == 0]
    if no_signal:
        no_sig_str = ", ".join(no_signal)
        lines.append(f"No-signal patients: {no_sig_str}")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append("The cooperative cascade signal (alpha -> 4/3) is patient-dependent.")
    lines.append("chb01 shows the strongest signal (5/7 seizures in window),")
    lines.append("while chb02 shows no signal. This is consistent with seizure")
    lines.append("type variability: the cooperative threshold mechanism may require")
    lines.append("specific network topology (focal onset with sufficient cortical")
    lines.append("recruitment) to produce the characteristic KWW signature.")
    lines.append("")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(report_text)
    print(f"Report saved: {report_path}")
    return report_text


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    print("=" * 60)
    print("CHB-MIT Seizure Classification by KWW Signature")
    print("=" * 60)

    # 1. Build feature table
    records = build_feature_table(SEIZURE_DATA)
    n_patients = len(set(r['patient'] for r in records))
    print(f"\nLoaded {len(records)} seizures from {n_patients} patients")

    # Classification counts
    for cls in ['strong_signal', 'above_lock', 'weak_signal']:
        n = sum(1 for r in records if r['classification'] == cls)
        print(f"  {cls}: {n}")

    # 2. Feature importance analysis
    feature_names = [
        'peak_alpha', 'alpha_pre', 'alpha_near', 'alpha_int', 'alpha_ict',
        'tau_pre', 'r2_pre', 'seizure_duration', 'pre_ictal_available',
        'alpha_contrast', 'alpha_rise',
    ]
    feat_results = feature_importance(records, feature_names)

    print("\nFeature importance (sorted by MW p-value):")
    for f in feat_results:
        print(f"  {f['feature']:<22} MW_p={f['mw_pval']:.4f}  "
              f"rho={f['spearman_rho']:.3f}")

    # 3. Patient stratification
    strat = patient_stratification(records)

    print("\nPatient stratification:")
    for pat in sorted(strat.keys()):
        s = strat[pat]
        print(f"  {pat}: {s['n_strong']}/{s['n_seizures']} strong, "
              f"{s['n_above_lock']}/{s['n_seizures']} above_lock, "
              f"consistency={s['consistency']:.2f}")

    # 4. Generate figure
    fig_path = os.path.join(FIGURES_DIR, 'chbmit_seizure_classification.png')
    make_figure(records, feat_results, strat, fig_path)

    # 5. Write report
    report_path = os.path.join(REPORTS_DIR, 'chbmit_seizure_classification.txt')
    report_text = write_report(records, feat_results, strat, report_path)

    print("\n" + report_text)


if __name__ == '__main__':
    main()
