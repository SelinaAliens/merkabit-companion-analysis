"""
Meta-Analysis and Forest Plot for Cross-Platform KWW Measurements
===================================================================
Paper Section 9.3 / Figure 7

Computes inverse-variance weighted fixed-effects and DerSimonian-Laird
random-effects meta-analysis for all 14 cooperative-window measurements
from Table 9, plus 3 negative controls.

Generates:
  - Forest plot (Figure 7)
  - Summary statistics text report
  - JSON data file with all computed values

March 2026
"""

import sys
import os
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import FIGURES_DIR, REPORTS_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ============================================================================
# DATA: All 14 measurements from Table 9 (cooperative window only)
# ============================================================================
MEASUREMENTS = [
    {'name': 'Xiang Z0 (clean DTC)', 'alpha': 1.344, 'se': 0.081, 'domain': 'Quantum', 'state': 'Condensate'},
    {'name': 'Zhang Q3 (Ge/SiGe QD)', 'alpha': 1.355, 'se': 0.221, 'domain': 'Quantum', 'state': 'Solid'},
    {'name': 'Tsoukalas 2026 (Ge/SiGe)', 'alpha': 1.442, 'se': 0.306, 'domain': 'Quantum', 'state': 'Solid'},
    {'name': 'Mn₃Sn 570ps (Ogawa)', 'alpha': 1.283, 'se': 0.098, 'domain': 'Magnetic', 'state': 'Magnetic'},
    {'name': 'ASDEX Idiv tail', 'alpha': 1.273, 'se': 0.123, 'domain': 'Plasma', 'state': 'Plasma'},
    {'name': 'ASDEX ∇Ti recovery', 'alpha': 1.432, 'se': 0.1, 'domain': 'Plasma', 'state': 'Plasma'},
    {'name': 'Dorian wind (TS→Cat5)', 'alpha': 1.435, 'se': 0.322, 'domain': 'Atmosphere', 'state': 'Fluid'},
    {'name': 'Dorian pressure', 'alpha': 1.473, 'se': 0.467, 'domain': 'Atmosphere', 'state': 'Fluid'},
    {'name': 'IBM brisbane T2', 'alpha': 1.323, 'se': 0.066, 'domain': 'Quantum', 'state': 'Condensate'},
    {'name': 'CHB-MIT chb01 peak α', 'alpha': 1.228, 'se': 0.105, 'domain': 'Biological', 'state': 'Biological'},
    {'name': 'HYT meditation α/γ', 'alpha': 1.361, 'se': 0.028, 'domain': 'Neural', 'state': 'Neural'},
    {'name': 'Colloidal gel q=1.60', 'alpha': 1.35, 'se': 0.07, 'domain': 'Liquid', 'state': 'Liquid'},
    {'name': 'SDSS S82+ZTF near-Edd', 'alpha': 1.253, 'se': 0.066, 'domain': 'Astrophysical', 'state': 'Plasma'},
    {'name': 'Perseus ICM (Chandra)', 'alpha': 1.353, 'se': 0.106, 'domain': 'Astrophysical', 'state': 'Plasma'},
]

# Negative controls (plotted separately, excluded from meta-analysis)
CONTROLS = [
    {'name': 'Mi et al. g=0.97 (MBL)', 'alpha': 0.822, 'se': 0.006, 'domain': 'Quantum', 'state': 'Condensate'},
    {'name': 'Xiang dis=0.8 (MBL)', 'alpha': 0.72, 'se': 0.1, 'domain': 'Quantum', 'state': 'Condensate'},
    {'name': 'Xiang dis=0.4 (crossover)', 'alpha': 0.95, 'se': 0.1, 'domain': 'Quantum', 'state': 'Condensate'},
]

# Cooperative window bounds
COOP_LO = 4 / 3 - 0.15   # 1.1833...
COOP_HI = 4 / 3 + 0.15   # 1.4833...
ALPHA_REF = 4 / 3         # 1.3333...

# Domain colours
DOMAIN_COLORS = {
    'Quantum': '#4477AA',
    'Plasma': '#EE6677',
    'Atmosphere': '#CCBB44',
    'Biological': '#228833',
    'Neural': '#228833',
    'Liquid': '#AA3377',
    'Magnetic': '#66CCEE',
    'Astrophysical': '#FF8800',
}


# ==============================================================================
# META-ANALYSIS COMPUTATIONS
# ==============================================================================

def meta_analysis(measurements, label=''):
    """
    Fixed-effects (inverse-variance) and DerSimonian-Laird random-effects
    meta-analysis for a list of {'alpha', 'se'} dicts.

    Returns dict with all computed statistics.
    """
    k = len(measurements)
    alphas = np.array([m['alpha'] for m in measurements])
    ses    = np.array([m['se']    for m in measurements])

    # --- Fixed-effects ---
    w = 1.0 / ses**2                                    # inverse-variance weights
    w_sum = np.sum(w)
    alpha_wm = np.sum(w * alphas) / w_sum               # weighted mean
    se_wm = 1.0 / np.sqrt(w_sum)                        # SE of weighted mean
    ci_lo = alpha_wm - 1.96 * se_wm                     # 95% CI
    ci_hi = alpha_wm + 1.96 * se_wm

    # --- Cochran Q ---
    Q = np.sum(w * (alphas - alpha_wm)**2)
    df = k - 1
    Q_p = 1.0 - stats.chi2.cdf(Q, df)

    # --- I-squared ---
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # --- DerSimonian-Laird tau-squared ---
    C = w_sum - np.sum(w**2) / w_sum                    # scaling constant
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # --- Random-effects (if tau2 > 0) ---
    if tau2 > 0:
        w_re = 1.0 / (ses**2 + tau2)
        w_re_sum = np.sum(w_re)
        alpha_re = np.sum(w_re * alphas) / w_re_sum
        se_re = 1.0 / np.sqrt(w_re_sum)
        ci_re_lo = alpha_re - 1.96 * se_re
        ci_re_hi = alpha_re + 1.96 * se_re
    else:
        w_re = w.copy()
        alpha_re = alpha_wm
        se_re = se_wm
        ci_re_lo = ci_lo
        ci_re_hi = ci_hi

    # --- Test vs 4/3 (fixed-effects) ---
    z_vs_43 = (alpha_wm - ALPHA_REF) / se_wm
    p_vs_43 = 2.0 * (1.0 - stats.norm.cdf(abs(z_vs_43)))

    # --- Test vs 4/3 (random-effects) ---
    z_re_vs_43 = (alpha_re - ALPHA_REF) / se_re
    p_re_vs_43 = 2.0 * (1.0 - stats.norm.cdf(abs(z_re_vs_43)))

    # --- Weight percentages ---
    w_pct = w / w_sum * 100.0
    w_re_pct = w_re / np.sum(w_re) * 100.0

    result = {
        'label': label,
        'k': k,
        'alphas': alphas.tolist(),
        'ses': ses.tolist(),
        'weights_fe': w.tolist(),
        'weight_pct_fe': w_pct.tolist(),
        'weights_re': w_re.tolist(),
        'weight_pct_re': w_re_pct.tolist(),
        'alpha_wm': float(alpha_wm),
        'se_wm': float(se_wm),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'Q': float(Q),
        'Q_df': int(df),
        'Q_p': float(Q_p),
        'I2': float(I2),
        'tau2': float(tau2),
        'alpha_re': float(alpha_re),
        'se_re': float(se_re),
        'ci_re_lo': float(ci_re_lo),
        'ci_re_hi': float(ci_re_hi),
        'z_vs_43': float(z_vs_43),
        'p_vs_43': float(p_vs_43),
        'z_re_vs_43': float(z_re_vs_43),
        'p_re_vs_43': float(p_re_vs_43),
    }
    return result


def format_report(res, header=''):
    """Format meta-analysis results as human-readable text."""
    lines = []
    if header:
        lines.append(header)
        lines.append('=' * len(header))
    lines.append(f"  k = {res['k']} measurements")
    lines.append(f"  Fixed-effects weighted mean:  {res['alpha_wm']:.4f} +/- {res['se_wm']:.4f}")
    lines.append(f"  95% CI: [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    lines.append(f"  Cochran Q = {res['Q']:.2f}  (df = {res['Q_df']}, p = {res['Q_p']:.4f})")
    lines.append(f"  I^2 = {res['I2']:.1f}%")
    lines.append(f"  DerSimonian-Laird tau^2 = {res['tau2']:.6f}")
    if res['tau2'] > 0:
        lines.append(f"  Random-effects weighted mean: {res['alpha_re']:.4f} +/- {res['se_re']:.4f}")
        lines.append(f"  RE 95% CI: [{res['ci_re_lo']:.4f}, {res['ci_re_hi']:.4f}]")
    else:
        lines.append("  Random-effects: identical to fixed-effects (tau^2 = 0)")
    lines.append(f"  Test vs 4/3 (FE): z = {res['z_vs_43']:.3f}, p = {res['p_vs_43']:.4f}")
    lines.append(f"  Test vs 4/3 (RE): z = {res['z_re_vs_43']:.3f}, p = {res['p_re_vs_43']:.4f}")
    lines.append('')
    return chr(10).join(lines)


# ============================================================================
# FOREST PLOT
# ============================================================================

def make_forest_plot(measurements, controls, res_all, res_kww, outpath):
    """
    Publication-quality forest plot (Figure 7).

    Layout (bottom to top):
        - Diamond for weighted mean (all 14)
        - Diamond for weighted mean (13 KWW)
        - 14 measurements (index 0 at bottom)
        - 3 negative controls (gray)
    """
    n_meas = len(measurements)
    n_ctrl = len(controls)
    n_rows = n_meas + n_ctrl + 3

    fig, ax = plt.subplots(figsize=(12, 0.55 * n_rows + 1.5))

    y_diamond_all = 0
    y_diamond_kww = 1
    y_gap_bottom  = 2
    y_meas_start  = 3
    y_sep         = y_meas_start + n_meas
    y_ctrl_start  = y_sep + 1

    # --- Cooperative window shaded band ---
    ax.axvspan(COOP_LO, COOP_HI, color='#c8e6c9', alpha=0.45, zorder=0,
               label='Cooperative window')

    # --- Reference line at 4/3 ---
    ax.axvline(ALPHA_REF, color='#333333', ls='--', lw=1.0, zorder=1,
               label=r'$\alpha = 4/3$')

    # --- Compute max weight for marker scaling ---
    w_fe = np.array(res_all['weight_pct_fe'])
    max_w = w_fe.max()

    # --- Plot measurements ---
    y_labels = []
    y_positions = []
    annotations = []

    for i, m in enumerate(measurements):
        y = y_meas_start + i
        a_val = m['alpha']
        se = m['se']
        ci_lo_i = a_val - 1.96 * se
        ci_hi_i = a_val + 1.96 * se
        col = DOMAIN_COLORS.get(m['domain'], '#888888')
        w_pct = w_fe[i]

        # CI line
        ax.plot([ci_lo_i, ci_hi_i], [y, y], color=col, lw=1.5, zorder=2)

        # Square marker (area proportional to weight)
        marker_size = 4.0 + 10.0 * (w_pct / max_w)
        ax.plot(a_val, y, 's', color=col, markersize=marker_size, zorder=3,
                markeredgecolor='black', markeredgewidth=0.5)

        y_labels.append(m['name'])
        y_positions.append(y)
        annotations.append(f"{a_val:.3f} +/- {se:.3f}  [{w_pct:.1f}%]")

    # --- Plot negative controls (gray) ---
    for j, c in enumerate(controls):
        y = y_ctrl_start + j
        a_val = c['alpha']
        se = c['se']
        ci_lo_j = a_val - 1.96 * se
        ci_hi_j = a_val + 1.96 * se

        ax.plot([ci_lo_j, ci_hi_j], [y, y], color='#999999', lw=1.5, zorder=2)
        ax.plot(a_val, y, 's', color='#999999', markersize=6, zorder=3,
                markeredgecolor='black', markeredgewidth=0.5)

        y_labels.append(c['name'])
        y_positions.append(y)
        annotations.append(f"{a_val:.3f} +/- {se:.3f}  [ctrl]")

    # --- Separator line ---
    ax.axhline(y_sep, color='#aaaaaa', lw=0.8, ls='-', zorder=1)

    # --- Diamond for weighted mean (all 14) ---
    _draw_diamond(ax, y_diamond_all, res_all['alpha_wm'], res_all['ci_lo'],
                  res_all['ci_hi'], color='#D32F2F')
    y_labels.append('FE mean (all 14)')
    y_positions.append(y_diamond_all)
    annotations.append(
        f"{res_all['alpha_wm']:.3f} +/- {res_all['se_wm']:.3f}  [100%]"
    )

    # --- Diamond for weighted mean (13 KWW) ---
    _draw_diamond(ax, y_diamond_kww, res_kww['alpha_wm'], res_kww['ci_lo'],
                  res_kww['ci_hi'], color='#1565C0')
    y_labels.append('FE mean (13 KWW)')
    y_positions.append(y_diamond_kww)
    annotations.append(
        f"{res_kww['alpha_wm']:.3f} +/- {res_kww['se_wm']:.3f}  [100%]"
    )

    # --- Right-side annotations ---
    x_annot = 2.08
    for yp, txt in zip(y_positions, annotations):
        ax.text(x_annot, yp, txt, va='center', ha='left', fontsize=7.5,
                fontfamily='monospace')

    # --- Axis formatting ---
    ax.set_xlim(0.55, 2.55)
    all_y = sorted(y_positions)
    ax.set_ylim(all_y[0] - 0.8, all_y[-1] + 0.8)

    ax.set_xlabel(r'Stretched exponent $\alpha$', fontsize=11)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.5)

    # Column header
    ax.text(x_annot, all_y[-1] + 0.6, r'$\alpha$ +/- SE  [wt%]', va='center',
            ha='left', fontsize=8, fontfamily='monospace', fontweight='bold')

    # --- Bottom statistics text ---
    stat_text = (
        f"All 14: Q = {res_all['Q']:.2f} (df={res_all['Q_df']}, "
        f"p = {res_all['Q_p']:.2f}),  "
        "I$^2$ = " + f"{res_all['I2']:.1f}%,  "
        "$\\tau^2$ = " + f"{res_all['tau2']:.4f},  "
        f"p vs 4/3 = {res_all['p_vs_43']:.2f}\n"
        f"13 KWW: Q = {res_kww['Q']:.2f} (df={res_kww['Q_df']}, "
        f"p = {res_kww['Q_p']:.2f}),  "
        "I$^2$ = " + f"{res_kww['I2']:.1f}%,  "
        f"p vs 4/3 = {res_kww['p_vs_43']:.2f}"
    )
    ax.text(0.5, -0.10, stat_text, transform=ax.transAxes, fontsize=8,
            va='top', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#cccccc'))

    # --- Section labels ---
    ax.text(0.56, y_ctrl_start + n_ctrl - 1 + 0.55, 'Negative controls',
            fontsize=8, fontstyle='italic', color='#666666', va='bottom')
    ax.text(0.56, y_meas_start + n_meas - 1 + 0.55,
            'Cooperative-window measurements', fontsize=8, fontstyle='italic',
            color='#333333', va='bottom')

    # --- Legend for domain colours ---
    from matplotlib.lines import Line2D
    legend_elements = []
    seen = set()
    for m in measurements:
        d = m['domain']
        if d not in seen:
            seen.add(d)
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=DOMAIN_COLORS[d], markersize=8,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label=d))
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#999999',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Control'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
              framealpha=0.9, ncol=2)

    ax.set_title('Cross-Platform KWW Meta-Analysis (Figure 7)', fontsize=12,
                 fontweight='bold', pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, lw=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved forest plot: {outpath}")


def _draw_diamond(ax, y, center, ci_lo, ci_hi, color='red'):
    """Draw a diamond marker at the given y position spanning [ci_lo, ci_hi]."""
    half_h = 0.3
    xs = [ci_lo, center, ci_hi, center, ci_lo]
    ys = [y, y + half_h, y, y - half_h, y]
    ax.fill(xs, ys, color=color, alpha=0.7, zorder=4)
    ax.plot(xs, ys, color='black', lw=0.8, zorder=5)



# ============================================================================
# MAIN
# ============================================================================

def main():
    print("======================================================================")
    print("META-ANALYSIS: Cross-Platform KWW Stretched Exponent")
    print("======================================================================")
    print()

    # ------------------------------------------------------------------
    # 1. All 14 measurements
    # ------------------------------------------------------------------
    res_all = meta_analysis(MEASUREMENTS, label='All 14 measurements')
    report_all = format_report(res_all, 'ALL 14 MEASUREMENTS')
    print(report_all)

    # ------------------------------------------------------------------
    # 2. 13-KWW subset (exclude meditation ratio, index 10)
    # ------------------------------------------------------------------
    kww_subset = [m for i, m in enumerate(MEASUREMENTS) if i != 10]
    res_kww = meta_analysis(kww_subset, label='13 KWW measurements')
    report_kww = format_report(res_kww, '13 KWW MEASUREMENTS (excl. meditation ratio)')
    print(report_kww)
    # ------------------------------------------------------------------
    # 3. Per-measurement detail table
    # ------------------------------------------------------------------
    w_fe = np.array(res_all['weight_pct_fe'])
    detail_lines = []
    detail_lines.append("PER-MEASUREMENT DETAIL")
    detail_lines.append("=" * 90)
    hdr = f"{'#':>2}  {'Name':<32}  {'alpha':>6}  {'SE':>6}  {'95% CI':>17}  {'wt%':>6}  {'Domain':<12}"
    detail_lines.append(hdr)
    detail_lines.append("-" * 90)
    for i, m in enumerate(MEASUREMENTS):
        ci = f"[{m['alpha']-1.96*m['se']:.3f}, {m['alpha']+1.96*m['se']:.3f}]"
        detail_lines.append(
            f"{i+1:>2}  {m['name']:<32}  {m['alpha']:>6.3f}  {m['se']:>6.3f}  "
            f"{ci:>17}  {w_fe[i]:>5.1f}%  {m['domain']:<12}"
        )
    detail_lines.append("-" * 90)
    detail_lines.append('')

    detail_lines.append("NEGATIVE CONTROLS")
    detail_lines.append("-" * 90)
    for j, c in enumerate(CONTROLS):
        ci = f"[{c['alpha']-1.96*c['se']:.3f}, {c['alpha']+1.96*c['se']:.3f}]"
        detail_lines.append(
            f" C{j+1}  {c['name']:<32}  {c['alpha']:>6.3f}  {c['se']:>6.3f}  "
            f"{ci:>17}  {'--':>6}  {c['domain']:<12}"
        )
    detail_lines.append("-" * 90)
    detail_lines.append('')
    detail_text = chr(10).join(detail_lines)
    print(detail_text)

    # ------------------------------------------------------------------
    # 4. Comparison block
    # ------------------------------------------------------------------
    compare_lines = []
    compare_lines.append("COMPARISON: 14 vs 13 KWW")
    compare_lines.append("=" * 50)
    compare_lines.append(f"  {'':20} {'All 14':>12}  {'13 KWW':>12}")
    compare_lines.append(f"  {'Weighted mean':20} {res_all['alpha_wm']:>12.4f}  {res_kww['alpha_wm']:>12.4f}")
    compare_lines.append(f"  {'SE':20} {res_all['se_wm']:>12.4f}  {res_kww['se_wm']:>12.4f}")
    compare_lines.append(f"  {'95% CI lo':20} {res_all['ci_lo']:>12.4f}  {res_kww['ci_lo']:>12.4f}")
    compare_lines.append(f"  {'95% CI hi':20} {res_all['ci_hi']:>12.4f}  {res_kww['ci_hi']:>12.4f}")
    compare_lines.append(f"  {'Q':20} {res_all['Q']:>12.2f}  {res_kww['Q']:>12.2f}")
    compare_lines.append(f"  {'Q p-value':20} {res_all['Q_p']:>12.4f}  {res_kww['Q_p']:>12.4f}")
    compare_lines.append(f"  {'I^2 (%)':20} {res_all['I2']:>12.1f}  {res_kww['I2']:>12.1f}")
    compare_lines.append(f"  {'tau^2':20} {res_all['tau2']:>12.6f}  {res_kww['tau2']:>12.6f}")
    compare_lines.append(f"  {'p vs 4/3 (FE)':20} {res_all['p_vs_43']:>12.4f}  {res_kww['p_vs_43']:>12.4f}")
    compare_lines.append(f"  {'p vs 4/3 (RE)':20} {res_all['p_re_vs_43']:>12.4f}  {res_kww['p_re_vs_43']:>12.4f}")
    compare_lines.append('')
    compare_text = chr(10).join(compare_lines)
    print(compare_text)

    # ------------------------------------------------------------------
    # 5. Save text report
    # ------------------------------------------------------------------
    report_path = os.path.join(REPORTS_DIR, 'meta_analysis_summary.txt')
    import datetime
    with open(report_path, 'w', encoding='utf-8') as fout:
        fout.write("CROSS-PLATFORM KWW META-ANALYSIS SUMMARY\n")
        fout.write("Paper Section 9.3 / Figure 7\n")
        fout.write("Generated: " + datetime.datetime.now().isoformat() + "\n")
        fout.write("=" * 70 + "\n\n")
        fout.write(report_all + "\n")
        fout.write(report_kww + "\n")
        fout.write(detail_text + "\n")
        fout.write(compare_text + "\n")
    print(f"  Saved report: {report_path}")

    # ------------------------------------------------------------------
    # 6. Save JSON data
    # ------------------------------------------------------------------
    json_data = {
        'measurements': MEASUREMENTS,
        'controls': CONTROLS,
        'meta_all_14': res_all,
        'meta_13_kww': res_kww,
        'reference_alpha': float(ALPHA_REF),
        'cooperative_window': [float(COOP_LO), float(COOP_HI)],
    }
    json_path = os.path.join(REPORTS_DIR, 'meta_analysis_data.json')
    with open(json_path, 'w', encoding='utf-8') as fout:
        json.dump(json_data, fout, indent=2, ensure_ascii=False)
    print(f"  Saved JSON:   {json_path}")

    # ------------------------------------------------------------------
    # 7. Generate forest plot
    # ------------------------------------------------------------------
    fig_path = os.path.join(FIGURES_DIR, 'meta_analysis_forest_plot.png')
    make_forest_plot(MEASUREMENTS, CONTROLS, res_all, res_kww, fig_path)

    print()
    print("======================================================================")
    print("DONE")
    print("======================================================================")


if __name__ == '__main__':
    main()
