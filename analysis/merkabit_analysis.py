"""
Merkabit Framework Analysis of Mi et al. 2022 Time Crystal Dataset
==================================================================
Fits stretched exponential envelopes: A_env(n) = A0 * exp(-(n/n*)^alpha)
to autocorrelator decay data from the Google Sycamore DTC experiment.

Primary question: Does alpha ~ 1.3 appear at or near the DTC phase boundary?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MI_2022_DATA, FIGURES_DIR, REPORTS_DIR

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUT = FIGURES_DIR
os.makedirs(OUT, exist_ok=True)
DATA = MI_2022_DATA

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def stretched_exp(n, A0, n_star, alpha):
    """Stretched exponential decay envelope."""
    return A0 * np.exp(-(n / n_star) ** alpha)

def extract_envelope(signal):
    """Extract amplitude envelope from oscillating autocorrelator.
    Method: Take absolute value of the signal (works for period-2 subharmonic).
    """
    return np.abs(signal)

def extract_envelope_peak(signal):
    """Extract envelope by taking local maxima of |signal|.
    More robust for noisy data with period-2 oscillation.
    """
    abs_sig = np.abs(signal)
    # For period-2 oscillation, every other point is a "peak"
    # Take max of consecutive pairs
    n_pairs = len(abs_sig) // 2
    envelope = np.array([max(abs_sig[2*i], abs_sig[2*i+1]) for i in range(n_pairs)])
    n_vals = np.arange(n_pairs) * 2  # corresponding cycle numbers
    return n_vals, envelope

def fit_envelope(n_vals, envelope, method='direct'):
    """Fit stretched exponential to envelope data.
    Returns: (A0, n_star, alpha, R2, param_errors)
    """
    # Remove zeros/nans
    mask = (envelope > 0) & np.isfinite(envelope)
    n_fit = n_vals[mask]
    env_fit = envelope[mask]

    if len(n_fit) < 4:
        return None

    try:
        # Initial guesses
        A0_guess = env_fit[0] if len(env_fit) > 0 else 1.0
        n_star_guess = len(n_fit) / 2
        alpha_guess = 1.0

        popt, pcov = curve_fit(
            stretched_exp, n_fit, env_fit,
            p0=[A0_guess, n_star_guess, alpha_guess],
            bounds=([0, 0.1, 0.01], [2.0, 10000, 10.0]),
            maxfev=50000
        )

        # R-squared
        residuals = env_fit - stretched_exp(n_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((env_fit - np.mean(env_fit))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Parameter uncertainties
        perr = np.sqrt(np.diag(pcov))

        return {
            'A0': popt[0], 'n_star': popt[1], 'alpha': popt[2],
            'R2': R2,
            'A0_err': perr[0], 'n_star_err': perr[1], 'alpha_err': perr[2]
        }
    except Exception as e:
        return None

# ============================================================
# ANALYSIS 1: ENVELOPE DECAY ACROSS PHASE DIAGRAM
# ============================================================

def analysis_1():
    print("=" * 70)
    print("ANALYSIS 1: ENVELOPE DECAY ACROSS PHASE DIAGRAM")
    print("=" * 70)

    results = {}

    # --- fig_2c: Averaged autocorrelator at g=0.60 and g=0.97 ---
    df_2c = pd.read_csv(os.path.join(DATA, 'fig_2c.csv'), encoding='utf-8-sig')
    print(f"\nfig_2c columns: {list(df_2c.columns)}")
    print(f"fig_2c shape: {df_2c.shape}")

    # A_0_60, A_60 are the initial-state-corrected and raw autocorrelator for g=0.60
    # A_0_97, A_97 are for g=0.97
    n_cycles = np.arange(len(df_2c))

    for g_label, col_A0, col_A in [('0.60', 'A_0_60', 'A_60'), ('0.97', 'A_0_97', 'A_97')]:
        signal_A0 = df_2c[col_A0].values  # Initial state contribution (envelope)
        signal_A = df_2c[col_A].values     # Full autocorrelator (oscillating)

        # The A_0 column is already the envelope (non-oscillating)
        # The A column oscillates with period 2

        # Method A: Fit the A_0 (envelope) directly
        env_A0 = signal_A0.copy()
        # Skip first point (n=0, A=1 trivially)
        n_fit = n_cycles[1:]
        env_fit = env_A0[1:]

        result = fit_envelope(n_fit, env_fit)
        if result:
            results[f'g={g_label}_A0_envelope'] = result
            print(f"\n  g = {g_label} (A_0 envelope, fig_2c):")
            print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
            print(f"    n*    = {result['n_star']:.2f} +/- {result['n_star_err']:.2f}")
            print(f"    A0    = {result['A0']:.4f}")
            print(f"    R^2   = {result['R2']:.6f}")

        # Method B: Extract envelope from oscillating signal |A(n)|
        abs_A = np.abs(signal_A)
        result_B = fit_envelope(n_fit, abs_A[1:])
        if result_B:
            results[f'g={g_label}_absA_envelope'] = result_B
            print(f"\n  g = {g_label} (|A| envelope, fig_2c):")
            print(f"    alpha = {result_B['alpha']:.4f} +/- {result_B['alpha_err']:.4f}")
            print(f"    n*    = {result_B['n_star']:.2f} +/- {result_B['n_star_err']:.2f}")
            print(f"    R^2   = {result_B['R2']:.6f}")

    # --- fig_3a: MBL (g=0.97) vs Prethermal (g=0.94) ---
    df_3a = pd.read_csv(os.path.join(DATA, 'fig_3a.csv'), encoding='utf-8-sig')
    print(f"\nfig_3a columns: {list(df_3a.columns)}")
    n_3a = np.arange(len(df_3a))

    for col in df_3a.columns:
        signal = df_3a[col].values
        env = np.abs(signal)
        result = fit_envelope(n_3a[1:], env[1:])
        if result:
            # Determine g value
            if 'mbl' in col:
                g_val = '0.97 (MBL)'
            else:
                g_val = '0.94 (prethermal)'
            results[f'{col}'] = result
            print(f"\n  {col} (g ~ {g_val}):")
            print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
            print(f"    n*    = {result['n_star']:.2f} +/- {result['n_star_err']:.2f}")
            print(f"    R^2   = {result['R2']:.6f}")

    # --- fig_s8: Data vs simulation at g=0.60 and g=0.97 ---
    df_s8 = pd.read_csv(os.path.join(DATA, 'fig_s8.csv'), encoding='utf-8-sig')
    n_s8 = np.arange(len(df_s8))

    for col, g_val in [('data_60', '0.60'), ('data_97', '0.97')]:
        signal = df_s8[col].values
        env = np.abs(signal)
        result = fit_envelope(n_s8[1:], env[1:])
        if result:
            results[f'fig_s8_{col}'] = result
            print(f"\n  fig_s8 {col} (g = {g_val}):")
            print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
            print(f"    n*    = {result['n_star']:.2f} +/- {result['n_star_err']:.2f}")
            print(f"    R^2   = {result['R2']:.6f}")

    # --- Plot alpha vs g ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Envelope fits
    ax = axes[0]
    # Plot averaged autocorrelator envelopes
    for g_label, col_A0 in [('0.60', 'A_0_60'), ('0.97', 'A_0_97')]:
        signal = df_2c[col_A0].values
        n_vals = np.arange(len(signal))
        ax.plot(n_vals, signal, 'o-', markersize=2, label=f'g={g_label} (A\u2080 envelope)')

        key = f'g={g_label}_A0_envelope'
        if key in results:
            r = results[key]
            n_dense = np.linspace(0, len(signal)-1, 200)
            ax.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
                   '--', linewidth=2,
                   label=f'  fit: \u03b1={r["alpha"]:.3f}\u00b1{r["alpha_err"]:.3f}, n*={r["n_star"]:.1f}')

    ax.set_xlabel('Floquet cycle n')
    ax.set_ylabel('Autocorrelator envelope A\u2080(n)')
    ax.set_title('Stretched Exponential Fits to Autocorrelator Envelopes')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Right: Alpha summary
    ax = axes[1]
    g_vals_plot = []
    alpha_vals_plot = []
    alpha_errs_plot = []
    labels_plot = []

    # Collect all results
    data_points = [
        (0.60, results.get('g=0.60_A0_envelope'), 'g=0.60 (avg, A\u2080)'),
        (0.60, results.get('g=0.60_absA_envelope'), 'g=0.60 (avg, |A|)'),
        (0.94, results.get('prethermal_neel'), 'g\u22480.94 (pretherm, N\u00e9el)'),
        (0.94, results.get('prethermal_ground'), 'g\u22480.94 (pretherm, ground)'),
        (0.94, results.get('prethermal_random'), 'g\u22480.94 (pretherm, random)'),
        (0.97, results.get('g=0.97_A0_envelope'), 'g=0.97 (avg, A\u2080)'),
        (0.97, results.get('g=0.97_absA_envelope'), 'g=0.97 (avg, |A|)'),
        (0.97, results.get('mbl_neel'), 'g=0.97 (MBL, N\u00e9el)'),
        (0.97, results.get('mbl_ground'), 'g=0.97 (MBL, ground)'),
        (0.97, results.get('mbl_random'), 'g=0.97 (MBL, random)'),
    ]

    for g, r, label in data_points:
        if r is not None:
            g_vals_plot.append(g)
            alpha_vals_plot.append(r['alpha'])
            alpha_errs_plot.append(r['alpha_err'])
            labels_plot.append(label)

    ax.errorbar(g_vals_plot, alpha_vals_plot, yerr=alpha_errs_plot,
               fmt='o', markersize=8, capsize=5, capthick=2, color='navy')
    for i, label in enumerate(labels_plot):
        ax.annotate(label, (g_vals_plot[i], alpha_vals_plot[i]),
                   textcoords="offset points", xytext=(10, 5), fontsize=6, alpha=0.8)

    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Merkabit prediction \u03b1=1.3')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Simple exponential \u03b1=1')
    ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='Gaussian \u03b1=2')
    ax.set_xlabel('g (driving parameter)')
    ax.set_ylabel('Stretch exponent \u03b1')
    ax.set_title('Alpha vs g \u2014 Phase Diagram')
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'analysis1_alpha_vs_g.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: analysis1_alpha_vs_g.png")

    return results


# ============================================================
# ANALYSIS 2: INDIVIDUAL QUBIT ALPHA VARIATION
# ============================================================

def analysis_2():
    print("\n" + "=" * 70)
    print("ANALYSIS 2: INDIVIDUAL QUBIT ALPHA VARIATION")
    print("=" * 70)

    results = {}

    # --- fig_s7: Per-qubit autocorrelator data ---
    df_s7 = pd.read_csv(os.path.join(DATA, 'fig_s7.csv'), encoding='utf-8-sig')
    print(f"\nfig_s7 columns (first 5): {list(df_s7.columns)[:5]}")
    print(f"fig_s7 shape: {df_s7.shape}")

    n_s7 = np.arange(len(df_s7))

    # Parse column structure: nn_mbl_1..20 (Neel initial, MBL), n_mbl_1..20,
    # nn_thermal_1..20 (Neel initial, thermal), n_thermal_1..20
    categories = {
        'MBL_Neel': [f'nn_mbl_{i}' for i in range(1, 21)],
        'MBL_random': [f'n_mbl_{i}' for i in range(1, 21)],
        'Thermal_Neel': [f'nn_thermal_{i}' for i in range(1, 21)],
        'Thermal_random': [f'n_thermal_{i}' for i in range(1, 21)],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (cat_name, cols) in enumerate(categories.items()):
        ax = axes[idx // 2][idx % 2]
        alphas = []
        alpha_errs = []
        n_stars = []
        R2s = []
        qubit_ids = []

        for qi, col in enumerate(cols):
            if col not in df_s7.columns:
                continue
            signal = df_s7[col].values
            env = np.abs(signal)
            result = fit_envelope(n_s7[1:], env[1:])
            if result:
                alphas.append(result['alpha'])
                alpha_errs.append(result['alpha_err'])
                n_stars.append(result['n_star'])
                R2s.append(result['R2'])
                qubit_ids.append(qi + 1)

        alphas = np.array(alphas)
        alpha_errs = np.array(alpha_errs)
        qubit_ids = np.array(qubit_ids)

        if len(alphas) > 0:
            mean_alpha = np.mean(alphas)
            std_alpha = np.std(alphas)
            cv = std_alpha / mean_alpha * 100 if mean_alpha > 0 else 0

            g_label = '0.97' if 'MBL' in cat_name else '0.60'
            init_label = 'N\u00e9el' if 'Neel' in cat_name else 'random'

            results[cat_name] = {
                'alphas': alphas, 'alpha_errs': alpha_errs,
                'qubit_ids': qubit_ids,
                'mean_alpha': mean_alpha, 'std_alpha': std_alpha,
                'cv_percent': cv,
                'n_stars': np.array(n_stars), 'R2s': np.array(R2s)
            }

            print(f"\n  {cat_name} (g={g_label}, {init_label} initial state):")
            print(f"    Mean alpha = {mean_alpha:.4f} +/- {std_alpha:.4f}")
            print(f"    CV = {cv:.1f}%")
            print(f"    Alpha range: [{alphas.min():.4f}, {alphas.max():.4f}]")
            print(f"    Mean R^2 = {np.mean(R2s):.4f}")

            # Find qubits closest to 1.3
            dist_to_13 = np.abs(alphas - 1.3)
            closest_idx = np.argsort(dist_to_13)[:3]
            print(f"    Qubits closest to alpha=1.3:")
            for ci in closest_idx:
                print(f"      Q{qubit_ids[ci]}: alpha = {alphas[ci]:.4f} +/- {alpha_errs[ci]:.4f}")

            # Plot
            ax.errorbar(qubit_ids, alphas, yerr=alpha_errs,
                       fmt='o-', markersize=6, capsize=3, color='navy')
            ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Merkabit \u03b1=1.3')
            ax.axhline(y=mean_alpha, color='green', linestyle='-', alpha=0.5,
                      label=f'Mean \u03b1={mean_alpha:.3f}')
            ax.fill_between(qubit_ids, mean_alpha - std_alpha, mean_alpha + std_alpha,
                          alpha=0.15, color='green')
            ax.set_xlabel('Qubit position')
            ax.set_ylabel('\u03b1 (stretch exponent)')
            ax.set_title(f'{cat_name} (g={g_label})\nMean \u03b1={mean_alpha:.3f}, CV={cv:.1f}%')
            ax.legend(fontsize=8)
            ax.set_xlim(0, 21)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'analysis2_qubit_alpha_variation.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: analysis2_qubit_alpha_variation.png")

    # --- Also do fig_2d for 20-qubit data ---
    print("\n  --- fig_2d: Per-qubit autocorrelators (20 qubits) ---")

    # fig_2d_right: g=0.97, each row is one disorder instance (20 rows), columns are cycles
    df_2d_r = pd.read_csv(os.path.join(DATA, 'fig_2d_right.csv'), encoding='utf-8-sig')
    print(f"  fig_2d_right shape: {df_2d_r.shape}")
    # Each row is one qubit (20 qubits), columns are Floquet cycles
    # The header says "data_97_2d" - 20 rows x 101 columns

    n_qubits_2d = df_2d_r.shape[0]
    n_cycles_2d = df_2d_r.shape[1]
    n_vals_2d = np.arange(n_cycles_2d)

    alphas_2d = []
    alpha_errs_2d = []
    for qi in range(n_qubits_2d):
        signal = df_2d_r.iloc[qi].values.astype(float)
        env = np.abs(signal)
        result = fit_envelope(n_vals_2d[1:], env[1:])
        if result:
            alphas_2d.append(result['alpha'])
            alpha_errs_2d.append(result['alpha_err'])
        else:
            alphas_2d.append(np.nan)
            alpha_errs_2d.append(np.nan)

    alphas_2d = np.array(alphas_2d)
    valid = ~np.isnan(alphas_2d)
    if valid.sum() > 0:
        mean_a = np.nanmean(alphas_2d)
        std_a = np.nanstd(alphas_2d)
        print(f"\n  fig_2d g=0.97 (20 qubits):")
        print(f"    Mean alpha = {mean_a:.4f} +/- {std_a:.4f}")
        print(f"    CV = {std_a/mean_a*100:.1f}%")
        print(f"    Range: [{np.nanmin(alphas_2d):.4f}, {np.nanmax(alphas_2d):.4f}]")

        results['fig_2d_right_g097'] = {
            'alphas': alphas_2d, 'mean_alpha': mean_a, 'std_alpha': std_a,
            'cv_percent': std_a/mean_a*100
        }

    # fig_2d_left: g=0.60
    df_2d_l = pd.read_csv(os.path.join(DATA, 'fig_2d_left.csv'), encoding='utf-8-sig')
    print(f"\n  fig_2d_left shape: {df_2d_l.shape}")

    n_qubits_2d_l = df_2d_l.shape[0]
    n_cycles_2d_l = df_2d_l.shape[1]
    n_vals_2d_l = np.arange(n_cycles_2d_l)

    alphas_2d_l = []
    for qi in range(n_qubits_2d_l):
        signal = df_2d_l.iloc[qi].values.astype(float)
        env = np.abs(signal)
        result = fit_envelope(n_vals_2d_l[1:], env[1:])
        if result:
            alphas_2d_l.append(result['alpha'])
        else:
            alphas_2d_l.append(np.nan)

    alphas_2d_l = np.array(alphas_2d_l)
    if (~np.isnan(alphas_2d_l)).sum() > 0:
        mean_al = np.nanmean(alphas_2d_l)
        std_al = np.nanstd(alphas_2d_l)
        print(f"\n  fig_2d g=0.60 (20 qubits):")
        print(f"    Mean alpha = {mean_al:.4f} +/- {std_al:.4f}")
        print(f"    CV = {std_al/mean_al*100:.1f}%")
        results['fig_2d_left_g060'] = {
            'alphas': alphas_2d_l, 'mean_alpha': mean_al, 'std_alpha': std_al
        }

    return results


# ============================================================
# ANALYSIS 3: TIME-REVERSAL ECHO PROTOCOL
# ============================================================

def analysis_3():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: TIME-REVERSAL (ECHO) PROTOCOL")
    print("=" * 70)

    results = {}

    # fig_4c: A_0 (forward), A (full with echo), A/A_0 (normalized)
    df_4c = pd.read_csv(os.path.join(DATA, 'fig_4c.csv'), encoding='utf-8-sig')
    print(f"\nfig_4c columns: {list(df_4c.columns)}")
    print(f"fig_4c shape: {df_4c.shape}")

    n_4c = np.arange(len(df_4c))

    for col in df_4c.columns:
        signal = df_4c[col].values
        env = np.abs(signal)
        result = fit_envelope(n_4c[1:], env[1:])
        if result:
            results[f'fig4c_{col}'] = result
            print(f"\n  {col}:")
            print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
            print(f"    n*    = {result['n_star']:.2f} +/- {result['n_star_err']:.2f}")
            print(f"    R^2   = {result['R2']:.6f}")

    # fig_s10_a: Ancilla-based time-reversal (prethermal)
    df_s10a = pd.read_csv(os.path.join(DATA, 'fig_s10_a.csv'), encoding='utf-8-sig')
    print(f"\nfig_s10_a columns: {list(df_s10a.columns)}")
    n_s10a = np.arange(len(df_s10a))

    for col in df_s10a.columns:
        signal = df_s10a[col].values
        env = np.abs(signal)
        result = fit_envelope(n_s10a[1:], env[1:])
        if result:
            results[f's10a_{col}'] = result
            print(f"\n  {col} (ancilla prethermal):")
            print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
            print(f"    n*    = {result['n_star']:.2f} +/- {result['n_star_err']:.2f}")
            print(f"    R^2   = {result['R2']:.6f}")

    # fig_s10_c: Ancilla-based time-reversal (thermal)
    df_s10c = pd.read_csv(os.path.join(DATA, 'fig_s10_c.csv'), encoding='utf-8-sig')
    n_s10c = np.arange(len(df_s10c))

    for col in df_s10c.columns:
        signal = df_s10c[col].values
        env = np.abs(signal)
        result = fit_envelope(n_s10c[1:], env[1:])
        if result:
            results[f's10c_{col}'] = result
            print(f"\n  {col} (ancilla thermal):")
            print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
            print(f"    n*    = {result['n_star']:.2f} +/- {result['n_star_err']:.2f}")
            print(f"    R^2   = {result['R2']:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: fig_4c raw data + fits
    ax = axes[0]
    for col in df_4c.columns:
        signal = df_4c[col].values
        ax.plot(n_4c, np.abs(signal), 'o-', markersize=2, label=f'|{col}|')
        key = f'fig4c_{col}'
        if key in results:
            r = results[key]
            n_dense = np.linspace(1, len(signal)-1, 200)
            ax.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
                   '--', linewidth=2, label=f'  \u03b1={r["alpha"]:.3f}')
    ax.set_xlabel('Floquet cycle n')
    ax.set_ylabel('|Autocorrelator|')
    ax.set_title('Time-Reversal Protocol (fig 4c)')
    ax.legend(fontsize=7)

    # Middle: Ancilla prethermal
    ax = axes[1]
    for col in df_s10a.columns:
        signal = df_s10a[col].values
        ax.plot(n_s10a, np.abs(signal), 'o-', markersize=2, label=f'|{col}|')
        key = f's10a_{col}'
        if key in results:
            r = results[key]
            n_dense = np.linspace(1, len(signal)-1, 200)
            ax.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
                   '--', linewidth=2, label=f'  \u03b1={r["alpha"]:.3f}')
    ax.set_xlabel('Floquet cycle n')
    ax.set_ylabel('|Signal|')
    ax.set_title('Ancilla Prethermal (g\u22480.94)')
    ax.legend(fontsize=7)

    # Right: Ancilla thermal
    ax = axes[2]
    for col in df_s10c.columns:
        signal = df_s10c[col].values
        ax.plot(n_s10c, np.abs(signal), 'o-', markersize=2, label=f'|{col}|')
        key = f's10c_{col}'
        if key in results:
            r = results[key]
            n_dense = np.linspace(1, len(signal)-1, 200)
            ax.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
                   '--', linewidth=2, label=f'  \u03b1={r["alpha"]:.3f}')
    ax.set_xlabel('Floquet cycle n')
    ax.set_ylabel('|Signal|')
    ax.set_title('Ancilla Thermal (g\u22480.60)')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'analysis3_time_reversal.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: analysis3_time_reversal.png")

    return results


# ============================================================
# ANALYSIS 4: DISORDER DEPENDENCE
# ============================================================

def analysis_4():
    print("\n" + "=" * 70)
    print("ANALYSIS 4: DISORDER DEPENDENCE \u2014 Alpha Distribution")
    print("=" * 70)

    results = {}

    # fig_3b: 500 disorder instances
    # prethermal_autocorrelators, mbl_autocorrelators, energies_mbl, energies_prethermal
    df_3b = pd.read_csv(os.path.join(DATA, 'fig_3b.csv'), encoding='utf-8-sig')
    print(f"\nfig_3b columns: {list(df_3b.columns)}")
    print(f"fig_3b shape: {df_3b.shape}")

    # These are the late-time (cycle 40) autocorrelator values for 500 disorder instances
    # Not time-series! Each row is one disorder instance's late-time value.
    # We need TIME SERIES per disorder instance. These columns are scalar values.

    # Let me check fig_4d which has more structure
    df_4d = pd.read_csv(os.path.join(DATA, 'fig_4d.csv'), encoding='utf-8-sig')
    print(f"\nfig_4d columns: {list(df_4d.columns)}")
    print(f"fig_4d shape: {df_4d.shape}")

    # K_2, K_0, K_20 — 500 rows. These are likely the ancilla-based measurements
    # at different scrambling depths K. Not time-series either.

    # fig_s10_b: prethermal list (500 instances at K=0,5,20)
    df_s10b = pd.read_csv(os.path.join(DATA, 'fig_s10_b.csv'), encoding='utf-8-sig')
    print(f"\nfig_s10_b columns: {list(df_s10b.columns)}")
    print(f"fig_s10_b shape: {df_s10b.shape}")

    # fig_s10_d: thermal list
    df_s10d = pd.read_csv(os.path.join(DATA, 'fig_s10_d.csv'), encoding='utf-8-sig')
    print(f"\nfig_s10_d columns: {list(df_s10d.columns)}")
    print(f"fig_s10_d shape: {df_s10d.shape}")

    # The disorder dependence is best captured through fig_2d which has per-qubit data
    # that implicitly represents different "disorder environments" along the chain.

    # For proper disorder analysis, we can treat each qubit in fig_s7 as sampling
    # a different local disorder configuration. With 20 qubits this gives a distribution.

    # Re-use fig_s7 per-qubit results
    df_s7 = pd.read_csv(os.path.join(DATA, 'fig_s7.csv'), encoding='utf-8-sig')
    n_s7 = np.arange(len(df_s7))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (cat_label, prefix, g_val) in enumerate([
        ('MBL N\u00e9el (g=0.97)', 'nn_mbl', '0.97'),
        ('MBL random (g=0.97)', 'n_mbl', '0.97'),
        ('Thermal N\u00e9el (g=0.60)', 'nn_thermal', '0.60'),
        ('Thermal random (g=0.60)', 'n_thermal', '0.60'),
    ]):
        ax = axes[idx // 2][idx % 2]
        alphas = []
        for qi in range(1, 21):
            col = f'{prefix}_{qi}'
            if col in df_s7.columns:
                signal = df_s7[col].values
                env = np.abs(signal)
                result = fit_envelope(n_s7[1:], env[1:])
                if result and result['R2'] > 0.5:
                    alphas.append(result['alpha'])

        alphas = np.array(alphas)
        if len(alphas) > 0:
            mean_a = np.mean(alphas)
            std_a = np.std(alphas)
            in_merkabit = np.sum((alphas >= 1.1) & (alphas <= 1.5))
            frac_merkabit = in_merkabit / len(alphas) * 100

            results[cat_label] = {
                'alphas': alphas, 'mean': mean_a, 'std': std_a,
                'frac_in_merkabit_range': frac_merkabit,
                'n_in_merkabit_range': int(in_merkabit),
                'n_total': len(alphas)
            }

            print(f"\n  {cat_label}:")
            print(f"    N instances (qubits): {len(alphas)}")
            print(f"    Mean alpha: {mean_a:.4f} +/- {std_a:.4f}")
            print(f"    Fraction in [1.1, 1.5]: {frac_merkabit:.1f}% ({in_merkabit}/{len(alphas)})")

            ax.hist(alphas, bins=np.linspace(0, 4, 25), color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=1.3, color='red', linestyle='--', linewidth=2, label='Merkabit \u03b1=1.3')
            ax.axvspan(1.1, 1.5, alpha=0.15, color='red', label='Merkabit range [1.1, 1.5]')
            ax.axvline(x=mean_a, color='green', linestyle='-', linewidth=2, label=f'Mean={mean_a:.2f}')
            ax.set_xlabel('\u03b1 (stretch exponent)')
            ax.set_ylabel('Count')
            ax.set_title(f'{cat_label}\n{frac_merkabit:.0f}% in Merkabit range')
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'analysis4_disorder_alpha_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: analysis4_disorder_alpha_distribution.png")

    # Also analyze the late-time autocorrelator distribution from fig_3b
    print("\n  --- Late-time autocorrelator distribution (fig_3b) ---")
    for col in ['mbl_autocorrelators', 'prethermal_autocorrelators']:
        vals = df_3b[col].dropna().values
        print(f"\n  {col}:")
        print(f"    N = {len(vals)}")
        print(f"    Mean |A| = {np.mean(np.abs(vals)):.4f} +/- {np.std(np.abs(vals)):.4f}")
        print(f"    Median |A| = {np.median(np.abs(vals)):.4f}")
        results[f'late_time_{col}'] = {
            'mean_abs': np.mean(np.abs(vals)),
            'std_abs': np.std(np.abs(vals)),
            'median_abs': np.median(np.abs(vals))
        }

    return results


# ============================================================
# ANALYSIS 5: NOISE SPECTRAL INDEX
# ============================================================

def analysis_5():
    print("\n" + "=" * 70)
    print("ANALYSIS 5: NOISE SPECTRAL INDEX")
    print("=" * 70)

    # The dataset doesn't contain multiple sequence types (Ramsey/echo/CPMG).
    # However, we can extract an effective beta from the relationship between
    # alpha and the noise spectral density. For stretched exponential decay
    # with alpha, the noise PSD exponent beta relates as:
    # alpha = 1 + beta for 1/f^beta noise (in the filter function framework)
    # More precisely: alpha = (1+beta)/(1+beta) = ... this is model-dependent.

    # We can try: The time-reversal data gives us both the "Ramsey-like" (forward)
    # and "echo-like" (time-reversed) decay. The ratio of their decay rates
    # encodes the noise spectrum.

    results = {}

    # From fig_4c: A_0 is forward, A_divided_A_0 is the echo-normalized signal
    df_4c = pd.read_csv(os.path.join(DATA, 'fig_4c.csv'), encoding='utf-8-sig')
    n_4c = np.arange(len(df_4c))

    signal_fwd = df_4c['A_0'].values  # Forward (Ramsey-like)
    signal_echo = df_4c['A_divided_A_0'].values  # Echo-normalized (echo-like)

    result_fwd = fit_envelope(n_4c[1:], np.abs(signal_fwd[1:]))
    result_echo = fit_envelope(n_4c[1:], np.abs(signal_echo[1:]))

    if result_fwd and result_echo:
        alpha_fwd = result_fwd['alpha']
        alpha_echo = result_echo['alpha']
        n_star_fwd = result_fwd['n_star']
        n_star_echo = result_echo['n_star']

        # For 1/f^beta noise:
        # T2_echo / T2_Ramsey scales as ~N^(beta/(1+beta)) or similar
        # n_star ratio gives us a constraint on beta
        ratio = n_star_echo / n_star_fwd if n_star_fwd > 0 else 0

        # Simple model: In filter function theory,
        # For Ramsey: decay ~ exp(-(t/T2*)^(1+beta))
        # For echo: decay ~ exp(-(t/T2_echo)^(1+beta)) with T2_echo > T2*
        # The ratio T2_echo/T2* = 2^(beta/(1+beta))  for 1/f^beta noise

        # From alpha: alpha ≈ 1+beta for pure 1/f^beta noise (Ramsey)
        # So beta_from_alpha = alpha - 1

        beta_from_fwd = alpha_fwd - 1
        beta_from_echo = alpha_echo - 1

        print(f"\n  Forward (Ramsey-like):")
        print(f"    alpha = {alpha_fwd:.4f}, implied beta = {beta_from_fwd:.4f}")
        print(f"    n* = {n_star_fwd:.2f}")

        print(f"\n  Echo-normalized (echo-like):")
        print(f"    alpha = {alpha_echo:.4f}, implied beta = {beta_from_echo:.4f}")
        print(f"    n* = {n_star_echo:.2f}")

        print(f"\n  n*_echo / n*_fwd ratio = {ratio:.4f}")

        # Alternative beta estimate from n* ratio
        # Assuming 1/f^beta: ratio = 2^(beta/(1+beta))
        # => beta/(1+beta) = log2(ratio)
        # => beta = log2(ratio) / (1 - log2(ratio))
        if ratio > 1:
            log2_ratio = np.log2(ratio)
            if log2_ratio < 1:
                beta_from_ratio = log2_ratio / (1 - log2_ratio)
                print(f"  Beta from n* ratio (1/f model): {beta_from_ratio:.4f}")
                results['beta_from_ratio'] = beta_from_ratio

        results['alpha_forward'] = alpha_fwd
        results['alpha_echo'] = alpha_echo
        results['beta_from_fwd_alpha'] = beta_from_fwd
        results['beta_from_echo_alpha'] = beta_from_echo
        results['n_star_ratio'] = ratio

    # Also check the fig_3c data for system size scaling
    df_3c = pd.read_csv(os.path.join(DATA, 'fig_3c.csv'), encoding='utf-8-sig')
    print(f"\n  fig_3c (system size dependence):")
    print(f"  columns: {list(df_3c.columns)}")
    n_3c = np.arange(len(df_3c))

    for col in df_3c.columns:
        signal = df_3c[col].values
        env = np.abs(signal)
        result = fit_envelope(n_3c[1:], env[1:])
        if result:
            print(f"    {col}: alpha = {result['alpha']:.4f}, n* = {result['n_star']:.2f}, R^2 = {result['R2']:.4f}")
            results[f'fig3c_{col}'] = result

    print("\n  NOTE: Full noise spectral analysis (Ramsey/echo/CPMG comparison)")
    print("  is limited because the dataset does not contain multiple")
    print("  sequence types. The estimates above are model-dependent.")

    return results


# ============================================================
# COMPREHENSIVE ALPHA SUMMARY TABLE
# ============================================================

def generate_summary(r1, r2, r3, r4, r5):
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ALPHA SUMMARY")
    print("=" * 70)

    # Collect all alpha values with their g values
    all_results = []

    # From Analysis 1
    for key, r in r1.items():
        if 'g=0.60' in key:
            g = 0.60
        elif 'g=0.97' in key or 'mbl' in key:
            g = 0.97
        elif 'prethermal' in key:
            g = 0.94
        elif 'fig_s8_data_60' in key:
            g = 0.60
        elif 'fig_s8_data_97' in key:
            g = 0.97
        else:
            g = None
        if g is not None:
            all_results.append({
                'source': f'A1: {key}', 'g': g,
                'alpha': r['alpha'], 'alpha_err': r['alpha_err'],
                'n_star': r['n_star'], 'R2': r['R2']
            })

    # From Analysis 3
    for key, r in r3.items():
        all_results.append({
            'source': f'A3: {key}', 'g': 0.97,
            'alpha': r['alpha'], 'alpha_err': r['alpha_err'],
            'n_star': r['n_star'], 'R2': r['R2']
        })

    # Print sorted by g
    all_results.sort(key=lambda x: (x['g'], x['source']))

    print(f"\n{'Source':<45} {'g':>5} {'alpha':>8} {'\u00b1err':>8} {'n*':>8} {'R\u00b2':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['source']:<45} {r['g']:>5.2f} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {r['n_star']:>8.2f} {r['R2']:>8.4f}")

    # THE KEY QUESTION
    print("\n" + "=" * 70)
    print("THE KEY QUESTION: Is alpha ~ 1.3 found anywhere?")
    print("=" * 70)

    closest = min(all_results, key=lambda x: abs(x['alpha'] - 1.3))
    print(f"\n  Closest to alpha=1.3:")
    print(f"    Source: {closest['source']}")
    print(f"    g = {closest['g']:.2f}")
    print(f"    alpha = {closest['alpha']:.4f} \u00b1 {closest['alpha_err']:.4f}")
    print(f"    Distance from 1.3: {abs(closest['alpha'] - 1.3):.4f}")

    # Check per-qubit results from Analysis 2
    print("\n  Per-qubit analysis (Analysis 2):")
    for cat_name, data in r2.items():
        if isinstance(data, dict) and 'alphas' in data:
            alphas = data['alphas']
            if hasattr(alphas, '__len__'):
                in_range = np.sum((alphas >= 1.1) & (alphas <= 1.5))
                closest_q = alphas[np.argmin(np.abs(alphas - 1.3))] if len(alphas) > 0 else None
                if closest_q is not None:
                    print(f"    {cat_name}: closest qubit alpha = {closest_q:.4f}, "
                          f"{in_range}/{len(alphas)} in [1.1, 1.5]")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Separate by g value
    alphas_060 = [r['alpha'] for r in all_results if r['g'] == 0.60]
    alphas_094 = [r['alpha'] for r in all_results if r['g'] == 0.94]
    alphas_097 = [r['alpha'] for r in all_results if r['g'] == 0.97]

    if alphas_060:
        mean_060 = np.mean(alphas_060)
        print(f"\n  g = 0.60 (thermal phase): mean alpha = {mean_060:.4f}")
    if alphas_094:
        mean_094 = np.mean(alphas_094)
        print(f"  g = 0.94 (prethermal DTC): mean alpha = {mean_094:.4f}")
    if alphas_097:
        mean_097 = np.mean(alphas_097)
        print(f"  g = 0.97 (MBL DTC):        mean alpha = {mean_097:.4f}")

    # Save summary to file
    with open(os.path.join(REPORTS_DIR, 'summary.txt'), 'w') as f:
        f.write("MERKABIT ANALYSIS: Mi et al. 2022 Time Crystal Dataset\n")
        f.write("=" * 70 + "\n\n")

        f.write("KEY QUESTION: Does alpha ~ 1.3 appear in the DTC phase boundary?\n\n")

        f.write(f"{'Source':<45} {'g':>5} {'alpha':>8} {'\u00b1err':>8} {'n*':>8} {'R\u00b2':>8}\n")
        f.write("-" * 90 + "\n")
        for r in all_results:
            f.write(f"{r['source']:<45} {r['g']:>5.2f} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {r['n_star']:>8.2f} {r['R2']:>8.4f}\n")

        f.write(f"\n\nClosest to alpha=1.3: {closest['source']}, alpha={closest['alpha']:.4f}, g={closest['g']:.2f}\n")

        f.write("\n\nPER-QUBIT ALPHA DISTRIBUTIONS:\n")
        for cat_name, data in r2.items():
            if isinstance(data, dict) and 'mean_alpha' in data:
                f.write(f"  {cat_name}: mean={data['mean_alpha']:.4f}, std={data.get('std_alpha', data.get('std', 0)):.4f}\n")

        f.write("\n\nNOISE SPECTRAL INDEX (Analysis 5):\n")
        for key, val in r5.items():
            if isinstance(val, (int, float)):
                f.write(f"  {key}: {val:.4f}\n")

        if alphas_060:
            f.write(f"\n\ng=0.60 (thermal): mean alpha = {np.mean(alphas_060):.4f}\n")
        if alphas_094:
            f.write(f"g=0.94 (prethermal DTC): mean alpha = {np.mean(alphas_094):.4f}\n")
        if alphas_097:
            f.write(f"g=0.97 (MBL DTC): mean alpha = {np.mean(alphas_097):.4f}\n")

    print(f"\n  Saved: summary.txt")

    return all_results


# ============================================================
# MASTER FIGURE: Everything on one page
# ============================================================

def master_figure(r1, r2, r3, r5):
    """Create a comprehensive figure summarizing all analyses."""
    fig = plt.figure(figsize=(20, 16))

    # Panel 1: Raw autocorrelator data + fits (fig_2c)
    ax1 = fig.add_subplot(3, 3, 1)
    df_2c = pd.read_csv(os.path.join(DATA, 'fig_2c.csv'), encoding='utf-8-sig')
    n_2c = np.arange(len(df_2c))
    for col, color, label in [('A_0_60', 'blue', 'g=0.60'), ('A_0_97', 'red', 'g=0.97')]:
        ax1.plot(n_2c, df_2c[col].values, 'o', markersize=1.5, color=color, alpha=0.5)
        key = f'g={col.split("_")[-1][0]}.{col.split("_")[-1][1:]}_A0_envelope'
        g_key = f'g=0.{col.split("_")[-1]}_A0_envelope'
        if g_key in r1:
            r = r1[g_key]
            n_dense = np.linspace(1, 100, 200)
            ax1.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
                    '-', color=color, linewidth=2, label=f'{label}: \u03b1={r["alpha"]:.2f}')
    ax1.set_xlabel('Floquet cycle')
    ax1.set_ylabel('A\u2080(n)')
    ax1.set_title('Autocorrelator Envelopes')
    ax1.legend(fontsize=7)

    # Panel 2: Alpha vs g summary
    ax2 = fig.add_subplot(3, 3, 2)
    g_vals = []
    alpha_vals = []
    alpha_errs = []
    for key, r in r1.items():
        if '_A0_' in key or '_absA_' not in key:
            if 'g=0.60' in key:
                g = 0.60
            elif 'g=0.97' in key or 'mbl' in key:
                g = 0.97
            elif 'prethermal' in key:
                g = 0.94
            else:
                continue
            g_vals.append(g)
            alpha_vals.append(r['alpha'])
            alpha_errs.append(r['alpha_err'])
    ax2.errorbar(g_vals, alpha_vals, yerr=alpha_errs, fmt='o', markersize=8,
                capsize=4, color='navy')
    ax2.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='Merkabit \u03b1=1.3')
    ax2.set_xlabel('g')
    ax2.set_ylabel('\u03b1')
    ax2.set_title('Alpha vs g (Phase Diagram)')
    ax2.legend()
    ax2.set_xlim(0.5, 1.05)

    # Panel 3: Per-qubit alpha at g=0.97
    ax3 = fig.add_subplot(3, 3, 3)
    if 'MBL_Neel' in r2 and 'alphas' in r2['MBL_Neel']:
        data = r2['MBL_Neel']
        ax3.errorbar(data['qubit_ids'], data['alphas'], yerr=data['alpha_errs'],
                    fmt='o-', markersize=5, capsize=3, color='navy')
        ax3.axhline(y=1.3, color='red', linestyle='--', linewidth=2)
        ax3.axhline(y=data['mean_alpha'], color='green', linestyle='-', alpha=0.5,
                   label=f'Mean={data["mean_alpha"]:.2f}')
        ax3.set_xlabel('Qubit position')
        ax3.set_ylabel('\u03b1')
        ax3.set_title(f'Per-Qubit \u03b1 (g=0.97, N\u00e9el)\nCV={data["cv_percent"]:.1f}%')
        ax3.legend(fontsize=7)

    # Panel 4: Time-reversal comparison
    ax4 = fig.add_subplot(3, 3, 4)
    df_4c = pd.read_csv(os.path.join(DATA, 'fig_4c.csv'), encoding='utf-8-sig')
    n_4c = np.arange(len(df_4c))
    for col, color in [('A_0', 'blue'), ('A_divided_A_0', 'red')]:
        ax4.plot(n_4c, np.abs(df_4c[col].values), 'o', markersize=1.5, color=color, alpha=0.5)
        key = f'fig4c_{col}'
        if key in r3:
            r = r3[key]
            n_dense = np.linspace(1, 100, 200)
            ax4.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
                    '-', color=color, linewidth=2,
                    label=f'{col}: \u03b1={r["alpha"]:.2f}')
    ax4.set_xlabel('Floquet cycle')
    ax4.set_ylabel('|Signal|')
    ax4.set_title('Time-Reversal Protocol')
    ax4.legend(fontsize=7)

    # Panel 5: Alpha histogram (MBL Neel)
    ax5 = fig.add_subplot(3, 3, 5)
    if 'MBL_Neel' in r2 and 'alphas' in r2['MBL_Neel']:
        alphas = r2['MBL_Neel']['alphas']
        ax5.hist(alphas, bins=12, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(x=1.3, color='red', linestyle='--', linewidth=2, label='\u03b1=1.3')
        ax5.set_xlabel('\u03b1')
        ax5.set_ylabel('Count')
        ax5.set_title('\u03b1 Distribution (g=0.97, MBL N\u00e9el)')
        ax5.legend()

    # Panel 6: Thermal vs DTC comparison
    ax6 = fig.add_subplot(3, 3, 6)
    if 'Thermal_Neel' in r2 and 'MBL_Neel' in r2:
        cats = ['Thermal_Neel', 'Thermal_random', 'MBL_Neel', 'MBL_random']
        means = [r2[c]['mean_alpha'] for c in cats if c in r2]
        stds = [r2[c]['std_alpha'] for c in cats if c in r2]
        labels = [c for c in cats if c in r2]
        colors = ['lightblue', 'lightblue', 'salmon', 'salmon']
        x = np.arange(len(means))
        ax6.bar(x, means, yerr=stds, color=colors[:len(means)], edgecolor='black', capsize=5)
        ax6.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='Merkabit \u03b1=1.3')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax6.set_ylabel('Mean \u03b1')
        ax6.set_title('Thermal vs DTC Phase')
        ax6.legend()

    # Panel 7: fig_3a raw data
    ax7 = fig.add_subplot(3, 3, 7)
    df_3a = pd.read_csv(os.path.join(DATA, 'fig_3a.csv'), encoding='utf-8-sig')
    n_3a = np.arange(len(df_3a))
    for col in df_3a.columns:
        ax7.plot(n_3a, np.abs(df_3a[col].values), '-', linewidth=1, alpha=0.7, label=col)
    ax7.set_xlabel('Floquet cycle')
    ax7.set_ylabel('|A(n)|')
    ax7.set_title('MBL vs Prethermal (|Autocorrelator|)')
    ax7.legend(fontsize=5, ncol=2)

    # Panel 8: Phase diagram from fig_5
    ax8 = fig.add_subplot(3, 3, 8)
    df_5 = pd.read_csv(os.path.join(DATA, 'fig_5.csv'), encoding='utf-8-sig')
    for col in ['s_8', 's_12', 's_16', 's_20']:
        ax8.plot(df_5['g'].values, df_5[col].values, 'o-', markersize=4, label=col)
    ax8.set_xlabel('g')
    ax8.set_ylabel('Spin glass order parameter')
    ax8.set_title('DTC Phase Diagram (fig 5)')
    ax8.legend(fontsize=7)
    ax8.axvline(x=0.83, color='gray', linestyle=':', alpha=0.5, label='~phase boundary')

    # Panel 9: Summary text
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = "KEY FINDINGS:\n\n"

    alphas_060 = [r['alpha'] for key, r in r1.items() if 'g=0.60' in key and 'absA' not in key]
    alphas_094 = [r['alpha'] for key, r in r1.items() if 'prethermal' in key]
    alphas_097 = [r['alpha'] for key, r in r1.items() if ('g=0.97' in key or 'mbl' in key) and 'absA' not in key]

    if alphas_060:
        summary_text += f"g=0.60 (thermal): \u03b1 \u2248 {np.mean(alphas_060):.3f}\n"
    if alphas_094:
        summary_text += f"g=0.94 (prethermal): \u03b1 \u2248 {np.mean(alphas_094):.3f}\n"
    if alphas_097:
        summary_text += f"g=0.97 (MBL DTC): \u03b1 \u2248 {np.mean(alphas_097):.3f}\n\n"

    all_alphas = alphas_060 + alphas_094 + alphas_097
    if all_alphas:
        closest = min(all_alphas, key=lambda x: abs(x - 1.3))
        summary_text += f"Closest to \u03b1=1.3: {closest:.3f}\n"
        summary_text += f"Distance: {abs(closest-1.3):.3f}\n\n"

    if 'alpha_forward' in r5:
        summary_text += f"Noise \u03b2 (from \u03b1_fwd): {r5['beta_from_fwd_alpha']:.3f}\n"
        summary_text += f"Noise \u03b2 (from \u03b1_echo): {r5['beta_from_echo_alpha']:.3f}\n"

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Merkabit Framework Analysis: Mi et al. 2022 DTC Dataset\n'
                'Does the stretched exponent \u03b1 \u2248 1.3 appear at the DTC phase boundary?',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT, 'master_figure.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: master_figure.png")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    print("MERKABIT FRAMEWORK ANALYSIS OF MI ET AL. 2022 TIME CRYSTAL DATASET")
    print("=" * 70)
    print(f"Data directory: {DATA}")
    print(f"Output directory: {OUT}")
    print()

    r1 = analysis_1()
    r2 = analysis_2()
    r3 = analysis_3()
    r4 = analysis_4()
    r5 = analysis_5()
    all_results = generate_summary(r1, r2, r3, r4, r5)
    master_figure(r1, r2, r3, r5)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Results saved to: {OUT}")
    print("=" * 70)
