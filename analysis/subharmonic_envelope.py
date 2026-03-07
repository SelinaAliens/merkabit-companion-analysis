"""
Subharmonic Envelope Analysis — Merkabit Framework
====================================================
The DTC autocorrelator oscillates at period 2: A(n) ~ (-1)^n * envelope(n)

The SUBHARMONIC ENVELOPE is extracted by:
  - Even cycles: A(0), A(2), A(4), ... → positive branch
  - Odd cycles:  A(1), A(3), A(5), ... → negative branch (negate to get envelope)

Fitting alpha to these envelopes separately gives the DTC order parameter decay,
which is the cleanest observable for the Merkabit comparison.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, io
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUT = FIGURES_DIR
DATA = MI_2022_DATA

def stretched_exp(n, A0, n_star, alpha):
    return A0 * np.exp(-(n / n_star) ** alpha)

def fit_stretched(n_vals, env, label=''):
    """Fit stretched exponential, return dict or None."""
    mask = (env > 1e-4) & np.isfinite(env)
    n_f = n_vals[mask]
    e_f = env[mask]
    if len(n_f) < 4:
        return None
    try:
        popt, pcov = curve_fit(
            stretched_exp, n_f, e_f,
            p0=[e_f[0], len(n_f), 1.0],
            bounds=([0, 0.1, 0.01], [2.0, 10000, 10.0]),
            maxfev=100000
        )
        res = e_f - stretched_exp(n_f, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((e_f - np.mean(e_f))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {'A0': popt[0], 'n_star': popt[1], 'alpha': popt[2],
                'R2': R2, 'A0_err': perr[0], 'n_star_err': perr[1], 'alpha_err': perr[2],
                'n_fit': n_f, 'env_fit': e_f}
    except:
        return None

def extract_subharmonic_envelope(signal):
    """Extract the subharmonic (period-2) envelope.

    In the DTC phase, A(n) ~ (-1)^n * E(n) where E(n) is the envelope.

    Even cycles (n=0,2,4,...): A(n) > 0, these trace the positive envelope
    Odd cycles  (n=1,3,5,...): A(n) < 0, -A(n) traces the positive envelope

    Returns:
        n_even, env_even: even-cycle indices and values
        n_odd, env_odd:   odd-cycle indices and values
        n_combined, env_combined: all points mapped to positive envelope
    """
    n_all = np.arange(len(signal))

    even_mask = (n_all % 2 == 0)
    odd_mask = (n_all % 2 == 1)

    n_even = n_all[even_mask]
    env_even = signal[even_mask]  # Should be positive in DTC

    n_odd = n_all[odd_mask]
    env_odd = -signal[odd_mask]   # Negate to get positive envelope

    # Combined: map both branches to the envelope
    # Use the MAGNITUDE — this handles both DTC (oscillating) and thermal (decaying)
    n_combined = n_all
    env_combined = np.abs(signal)
    # But for subharmonic specifically, alternate the sign
    env_subharmonic = signal.copy()
    env_subharmonic[odd_mask] = -env_subharmonic[odd_mask]

    return {
        'n_even': n_even, 'env_even': env_even,
        'n_odd': n_odd, 'env_odd': env_odd,
        'n_all': n_all, 'env_subharmonic': env_subharmonic
    }


print("=" * 70)
print("SUBHARMONIC ENVELOPE ANALYSIS")
print("=" * 70)

# ============================================================
# 1. AVERAGED AUTOCORRELATOR (fig_2c) — g=0.60 and g=0.97
# ============================================================

df_2c = pd.read_csv(os.path.join(DATA, 'fig_2c.csv'), encoding='utf-8-sig')

print("\n1. AVERAGED AUTOCORRELATOR — SUBHARMONIC ENVELOPE")
print("-" * 60)

results_sub = {}

for g_label, col_raw in [('0.60', 'A_60'), ('0.97', 'A_97')]:
    signal = df_2c[col_raw].values
    n_all = np.arange(len(signal))

    sub = extract_subharmonic_envelope(signal)

    # The subharmonic envelope: E(n) = (-1)^n * A(n), should decay monotonically
    env_sub = sub['env_subharmonic']

    print(f"\n  g = {g_label} ({col_raw}):")
    print(f"    Raw signal: A(0)={signal[0]:.4f}, A(1)={signal[1]:.4f}, A(2)={signal[2]:.4f}")
    print(f"    Subharmonic envelope: E(0)={env_sub[0]:.4f}, E(1)={env_sub[1]:.4f}, E(2)={env_sub[2]:.4f}")

    # Fit to ALL subharmonic envelope points (skip n=0)
    result_all = fit_stretched(n_all[1:], env_sub[1:])
    if result_all:
        print(f"    Full subharmonic envelope (n=1..100):")
        print(f"      alpha = {result_all['alpha']:.4f} +/- {result_all['alpha_err']:.4f}")
        print(f"      n*    = {result_all['n_star']:.2f} +/- {result_all['n_star_err']:.2f}")
        print(f"      A0    = {result_all['A0']:.4f}")
        print(f"      R^2   = {result_all['R2']:.6f}")
        results_sub[f'g={g_label}_sub_all'] = result_all

    # Fit even branch only (n=2,4,6,...)
    n_even = sub['n_even'][1:]  # skip n=0
    env_even = sub['env_even'][1:]
    result_even = fit_stretched(n_even, env_even)
    if result_even:
        print(f"    Even-cycle branch (n=2,4,6,...):")
        print(f"      alpha = {result_even['alpha']:.4f} +/- {result_even['alpha_err']:.4f}")
        print(f"      n*    = {result_even['n_star']:.2f}")
        print(f"      R^2   = {result_even['R2']:.6f}")
        results_sub[f'g={g_label}_even'] = result_even

    # Fit odd branch (n=1,3,5,...), negated
    n_odd = sub['n_odd']
    env_odd = sub['env_odd']
    result_odd = fit_stretched(n_odd, env_odd)
    if result_odd:
        print(f"    Odd-cycle branch (-A(1), -A(3), ...):")
        print(f"      alpha = {result_odd['alpha']:.4f} +/- {result_odd['alpha_err']:.4f}")
        print(f"      n*    = {result_odd['n_star']:.2f}")
        print(f"      R^2   = {result_odd['R2']:.6f}")
        results_sub[f'g={g_label}_odd'] = result_odd


# ============================================================
# 2. PRETHERMAL (g~0.94) from fig_3a
# ============================================================

print("\n\n2. PRETHERMAL DTC (g~0.94) — SUBHARMONIC ENVELOPE")
print("-" * 60)

df_3a = pd.read_csv(os.path.join(DATA, 'fig_3a.csv'), encoding='utf-8-sig')

for col in ['prethermal_neel', 'prethermal_ground', 'prethermal_random']:
    signal = df_3a[col].values
    n_all = np.arange(len(signal))
    sub = extract_subharmonic_envelope(signal)
    env_sub = sub['env_subharmonic']

    result = fit_stretched(n_all[1:], env_sub[1:])
    if result:
        print(f"\n  {col}:")
        print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
        print(f"    n*    = {result['n_star']:.2f}")
        print(f"    R^2   = {result['R2']:.6f}")
        results_sub[f'g=0.94_{col}'] = result

# Also MBL (g=0.97) from fig_3a for consistency with shorter data
for col in ['mbl_neel', 'mbl_ground', 'mbl_random']:
    signal = df_3a[col].values
    n_all = np.arange(len(signal))
    sub = extract_subharmonic_envelope(signal)
    env_sub = sub['env_subharmonic']

    result = fit_stretched(n_all[1:], env_sub[1:])
    if result:
        print(f"\n  {col} (g=0.97, 61 cycles):")
        print(f"    alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}")
        print(f"    n*    = {result['n_star']:.2f}")
        print(f"    R^2   = {result['R2']:.6f}")
        results_sub[f'g=0.97_{col}_3a'] = result


# ============================================================
# 3. PER-QUBIT SUBHARMONIC ENVELOPE (fig_s7)
# ============================================================

print("\n\n3. PER-QUBIT SUBHARMONIC ENVELOPE")
print("-" * 60)

df_s7 = pd.read_csv(os.path.join(DATA, 'fig_s7.csv'), encoding='utf-8-sig')
n_s7 = np.arange(len(df_s7))

per_qubit_results = {}

for phase, prefix, g_val in [
    ('MBL DTC (g=0.97) Neel', 'nn_mbl', 0.97),
    ('MBL DTC (g=0.97) random', 'n_mbl', 0.97),
    ('Thermal (g=0.60) Neel', 'nn_thermal', 0.60),
    ('Thermal (g=0.60) random', 'n_thermal', 0.60),
]:
    print(f"\n  {phase}:")
    alphas = []
    alpha_errs = []
    n_stars = []
    R2s = []
    qids = []

    for qi in range(1, 21):
        col = f'{prefix}_{qi}'
        signal = df_s7[col].values
        sub = extract_subharmonic_envelope(signal)
        env_sub = sub['env_subharmonic']

        result = fit_stretched(n_s7[1:], env_sub[1:])
        if result and result['R2'] > 0.9:
            alphas.append(result['alpha'])
            alpha_errs.append(result['alpha_err'])
            n_stars.append(result['n_star'])
            R2s.append(result['R2'])
            qids.append(qi)

    alphas = np.array(alphas)
    alpha_errs = np.array(alpha_errs)
    n_stars = np.array(n_stars)
    qids = np.array(qids)

    if len(alphas) > 0:
        per_qubit_results[phase] = {
            'alphas': alphas, 'alpha_errs': alpha_errs,
            'qids': qids, 'n_stars': n_stars, 'R2s': np.array(R2s)
        }

        mean_a = np.mean(alphas)
        std_a = np.std(alphas)
        cv = std_a / mean_a * 100

        print(f"    Valid fits: {len(alphas)}/20")
        print(f"    Mean alpha = {mean_a:.4f} +/- {std_a:.4f}")
        print(f"    CV = {cv:.1f}%")
        print(f"    Range: [{alphas.min():.4f}, {alphas.max():.4f}]")
        print(f"    Mean n* = {np.mean(n_stars):.2f}")
        print(f"    Mean R^2 = {np.mean(R2s):.4f}")

        # Distance to 1.3
        dist = np.abs(alphas - 1.3)
        closest = np.argsort(dist)[:3]
        print(f"    Closest to alpha=1.3:")
        for ci in closest:
            print(f"      Q{qids[ci]:2d}: alpha = {alphas[ci]:.4f} +/- {alpha_errs[ci]:.4f}")

        # In Merkabit range
        in_range = np.sum((alphas >= 1.1) & (alphas <= 1.5))
        print(f"    In [1.1, 1.5]: {in_range}/{len(alphas)} ({in_range/len(alphas)*100:.0f}%)")

        # Edge vs bulk
        edge_a = [alphas[i] for i, q in enumerate(qids) if q <= 3 or q >= 18]
        bulk_a = [alphas[i] for i, q in enumerate(qids) if 8 <= q <= 13]
        if edge_a and bulk_a:
            print(f"    Edge (Q1-3,Q18-20): mean alpha = {np.mean(edge_a):.4f}")
            print(f"    Bulk (Q8-13):       mean alpha = {np.mean(bulk_a):.4f}")


# ============================================================
# 4. PER-QUBIT from fig_2d (raw 20-qubit traces)
# ============================================================

print("\n\n4. PER-QUBIT SUBHARMONIC ENVELOPE FROM fig_2d")
print("-" * 60)

# fig_2d_right: g=0.97, 20 qubits x 101 cycles
df_2d_r = pd.read_csv(os.path.join(DATA, 'fig_2d_right.csv'), encoding='utf-8-sig')
print(f"\n  fig_2d_right: {df_2d_r.shape[0]} qubits x {df_2d_r.shape[1]} cycles")

alphas_2d_97 = []
errs_2d_97 = []
for qi in range(df_2d_r.shape[0]):
    signal = df_2d_r.iloc[qi].values.astype(float)
    n_all = np.arange(len(signal))
    sub = extract_subharmonic_envelope(signal)
    env_sub = sub['env_subharmonic']

    result = fit_stretched(n_all[1:], env_sub[1:])
    if result and result['R2'] > 0.9:
        alphas_2d_97.append(result['alpha'])
        errs_2d_97.append(result['alpha_err'])
    else:
        alphas_2d_97.append(np.nan)
        errs_2d_97.append(np.nan)

alphas_2d_97 = np.array(alphas_2d_97)
errs_2d_97 = np.array(errs_2d_97)
valid = ~np.isnan(alphas_2d_97)

if valid.sum() > 0:
    mean_a = np.nanmean(alphas_2d_97)
    std_a = np.nanstd(alphas_2d_97)
    print(f"\n  g=0.97 (fig_2d, 20 qubits, subharmonic envelope):")
    print(f"    Valid fits: {valid.sum()}/20")
    print(f"    Mean alpha = {mean_a:.4f} +/- {std_a:.4f}")
    print(f"    CV = {std_a/mean_a*100:.1f}%")
    print(f"    Range: [{np.nanmin(alphas_2d_97):.4f}, {np.nanmax(alphas_2d_97):.4f}]")

    for qi in range(20):
        if valid[qi]:
            tag = '*' if 1.1 <= alphas_2d_97[qi] <= 1.5 else ' '
            print(f"    {tag} Q{qi+1:2d}: alpha = {alphas_2d_97[qi]:.4f} +/- {errs_2d_97[qi]:.4f}")

# fig_2d_left: g=0.60, 20 qubits x 26 cycles
df_2d_l = pd.read_csv(os.path.join(DATA, 'fig_2d_left.csv'), encoding='utf-8-sig')
print(f"\n  fig_2d_left: {df_2d_l.shape[0]} qubits x {df_2d_l.shape[1]} cycles")

alphas_2d_60 = []
errs_2d_60 = []
for qi in range(df_2d_l.shape[0]):
    signal = df_2d_l.iloc[qi].values.astype(float)
    n_all = np.arange(len(signal))
    sub = extract_subharmonic_envelope(signal)
    env_sub = sub['env_subharmonic']

    result = fit_stretched(n_all[1:], env_sub[1:])
    if result and result['R2'] > 0.8:  # lower threshold, only 26 points
        alphas_2d_60.append(result['alpha'])
        errs_2d_60.append(result['alpha_err'])
    else:
        alphas_2d_60.append(np.nan)
        errs_2d_60.append(np.nan)

alphas_2d_60 = np.array(alphas_2d_60)
valid_60 = ~np.isnan(alphas_2d_60)
if valid_60.sum() > 0:
    print(f"\n  g=0.60 (fig_2d, 20 qubits, subharmonic envelope):")
    print(f"    Valid fits: {valid_60.sum()}/20")
    print(f"    Mean alpha = {np.nanmean(alphas_2d_60):.4f} +/- {np.nanstd(alphas_2d_60):.4f}")


# ============================================================
# 5. TIME-REVERSAL SUBHARMONIC ENVELOPE
# ============================================================

print("\n\n5. TIME-REVERSAL PROTOCOL — SUBHARMONIC ENVELOPE")
print("-" * 60)

df_4c = pd.read_csv(os.path.join(DATA, 'fig_4c.csv'), encoding='utf-8-sig')

for col in ['A_0', 'A', 'A_divided_A_0']:
    signal = df_4c[col].values
    n_all = np.arange(len(signal))
    sub = extract_subharmonic_envelope(signal)
    env_sub = sub['env_subharmonic']

    result = fit_stretched(n_all[1:], env_sub[1:])
    if result:
        tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
        print(f"  {tag} {col}: alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}, "
              f"n* = {result['n_star']:.2f}, R^2 = {result['R2']:.4f}")
        results_sub[f'echo_{col}'] = result


# ============================================================
# 6. SYSTEM-SIZE TRACES (fig_3c) — SUBHARMONIC
# ============================================================

print("\n\n6. SYSTEM-SIZE TRACES — SUBHARMONIC ENVELOPE")
print("-" * 60)

df_3c = pd.read_csv(os.path.join(DATA, 'fig_3c.csv'), encoding='utf-8-sig')
n_3c = np.arange(len(df_3c))

for col in df_3c.columns:
    signal = df_3c[col].values
    sub = extract_subharmonic_envelope(signal)
    env_sub = sub['env_subharmonic']

    result = fit_stretched(n_3c[1:], env_sub[1:])
    if result and result['R2'] > 0.8:
        tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
        print(f"  {tag} {col}: alpha = {result['alpha']:.4f}, n* = {result['n_star']:.1f}, R^2 = {result['R2']:.4f}")
        results_sub[f'fig3c_{col}'] = result


# ============================================================
# 7. ANCILLA MEASUREMENTS — SUBHARMONIC
# ============================================================

print("\n\n7. ANCILLA MEASUREMENTS — SUBHARMONIC ENVELOPE")
print("-" * 60)

for fname, label in [('fig_s10_a.csv', 'Prethermal (g~0.94)'),
                      ('fig_s10_c.csv', 'Thermal (g~0.60)')]:
    df = pd.read_csv(os.path.join(DATA, fname), encoding='utf-8-sig')
    n_vals = np.arange(len(df))
    print(f"\n  {label}:")
    for col in df.columns:
        signal = df[col].values
        sub = extract_subharmonic_envelope(signal)
        env_sub = sub['env_subharmonic']
        result = fit_stretched(n_vals[1:], env_sub[1:])
        if result:
            tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
            print(f"    {tag} {col}: alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}, "
                  f"n* = {result['n_star']:.2f}, R^2 = {result['R2']:.4f}")
            results_sub[f'{fname}_{col}'] = result


# ============================================================
# COMPREHENSIVE SUMMARY TABLE
# ============================================================

print("\n\n" + "=" * 70)
print("SUBHARMONIC ENVELOPE: COMPREHENSIVE ALPHA SUMMARY")
print("=" * 70)

# Collect the key high-confidence results
summary = []

# g=0.60 averaged
k = 'g=0.60_sub_all'
if k in results_sub:
    r = results_sub[k]
    summary.append(('g=0.60', 'Thermal (averaged)', r['alpha'], r['alpha_err'], r['n_star'], r['R2']))

# g=0.94 prethermal
for init in ['neel', 'ground', 'random']:
    k = f'g=0.94_prethermal_{init}'
    if k in results_sub:
        r = results_sub[k]
        summary.append(('g=0.94', f'Prethermal ({init})', r['alpha'], r['alpha_err'], r['n_star'], r['R2']))

# g=0.97 averaged
k = 'g=0.97_sub_all'
if k in results_sub:
    r = results_sub[k]
    summary.append(('g=0.97', 'MBL DTC (averaged)', r['alpha'], r['alpha_err'], r['n_star'], r['R2']))

# g=0.97 from fig_3a
for init in ['neel', 'ground', 'random']:
    k = f'g=0.97_mbl_{init}_3a'
    if k in results_sub:
        r = results_sub[k]
        summary.append(('g=0.97', f'MBL DTC ({init}, 3a)', r['alpha'], r['alpha_err'], r['n_star'], r['R2']))

# Even/odd branches
for branch in ['even', 'odd']:
    for g in ['0.60', '0.97']:
        k = f'g={g}_{branch}'
        if k in results_sub:
            r = results_sub[k]
            summary.append((f'g={g}', f'{branch}-cycle branch', r['alpha'], r['alpha_err'], r['n_star'], r['R2']))

# Echo
for col in ['A_0', 'A', 'A_divided_A_0']:
    k = f'echo_{col}'
    if k in results_sub:
        r = results_sub[k]
        summary.append(('g=0.97', f'Echo {col}', r['alpha'], r['alpha_err'], r['n_star'], r['R2']))

print(f"\n{'g':<8} {'Source':<30} {'alpha':>8} {'+-err':>8} {'n*':>8} {'R2':>8}")
print("-" * 80)
for g, src, a, ae, ns, r2 in sorted(summary, key=lambda x: (x[0], x[1])):
    tag = ' *' if 1.1 <= a <= 1.5 else '  '
    print(f"{g:<8} {src:<30} {a:>8.4f} {ae:>8.4f} {ns:>8.2f} {r2:>8.4f}{tag}")


# ============================================================
# KEY COMPARISON: alpha at g=0.60 vs g=0.94 vs g=0.97
# ============================================================

print("\n\n" + "=" * 70)
print("THE KEY COMPARISON: SUBHARMONIC alpha BY PHASE")
print("=" * 70)

# Collect best estimates per g
best = {}
for g_val, key_pattern in [
    (0.60, 'g=0.60_sub_all'),
    (0.97, 'g=0.97_sub_all'),
]:
    if key_pattern in results_sub:
        best[g_val] = results_sub[key_pattern]

if 'g=0.94_prethermal_neel' in results_sub:
    best[0.94] = results_sub['g=0.94_prethermal_neel']

print(f"\n  Best subharmonic envelope alpha at each g:")
for g in sorted(best.keys()):
    r = best[g]
    dist = abs(r['alpha'] - 1.3)
    print(f"    g = {g:.2f}: alpha = {r['alpha']:.4f} +/- {r['alpha_err']:.4f}  "
          f"(n*={r['n_star']:.1f}, R2={r['R2']:.4f})  |alpha-1.3| = {dist:.4f}")

# Per-qubit means
print(f"\n  Per-qubit subharmonic alpha (fig_s7):")
for phase, data in per_qubit_results.items():
    mean_a = np.mean(data['alphas'])
    std_a = np.std(data['alphas'])
    dist = abs(mean_a - 1.3)
    in_range = np.sum((data['alphas'] >= 1.1) & (data['alphas'] <= 1.5))
    print(f"    {phase}: mean = {mean_a:.4f} +/- {std_a:.4f}  "
          f"|mean-1.3| = {dist:.4f}  [{in_range}/{len(data['alphas'])} in Merkabit range]")

if valid.sum() > 0:
    mean_2d = np.nanmean(alphas_2d_97)
    std_2d = np.nanstd(alphas_2d_97)
    in_range_2d = np.sum((alphas_2d_97[valid] >= 1.1) & (alphas_2d_97[valid] <= 1.5))
    print(f"\n  Per-qubit subharmonic alpha (fig_2d, g=0.97):")
    print(f"    Mean = {mean_2d:.4f} +/- {std_2d:.4f}  "
          f"|mean-1.3| = {abs(mean_2d-1.3):.4f}  [{in_range_2d}/{valid.sum()} in Merkabit range]")


# ============================================================
# PUBLICATION FIGURE
# ============================================================

fig = plt.figure(figsize=(22, 18))

# Panel A: Raw autocorrelator + subharmonic envelope + fit at g=0.97
ax = fig.add_subplot(3, 3, 1)
signal_97 = df_2c['A_97'].values
n = np.arange(len(signal_97))
sub_97 = extract_subharmonic_envelope(signal_97)

ax.plot(n, signal_97, 'o', markersize=2, color='gray', alpha=0.4, label='Raw A(n)')
ax.plot(n, sub_97['env_subharmonic'], 's', markersize=3, color='#d62728', alpha=0.7,
        label='Subharmonic env E(n)')

k = 'g=0.97_sub_all'
if k in results_sub:
    r = results_sub[k]
    n_dense = np.linspace(1, 100, 200)
    ax.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
           '-', color='#d62728', linewidth=2.5,
           label=f'Fit: alpha={r["alpha"]:.3f}+/-{r["alpha_err"]:.3f}')

ax.set_xlabel('Floquet cycle n', fontsize=11)
ax.set_ylabel('Signal', fontsize=11)
ax.set_title('A. g=0.97 (MBL DTC) Subharmonic Envelope', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.set_ylim(-1.1, 1.1)

# Panel B: Same for g=0.60
ax = fig.add_subplot(3, 3, 2)
signal_60 = df_2c['A_60'].values
n60 = np.arange(len(signal_60))
sub_60 = extract_subharmonic_envelope(signal_60)

ax.plot(n60, signal_60, 'o', markersize=2, color='gray', alpha=0.4, label='Raw A(n)')
ax.plot(n60, sub_60['env_subharmonic'], 's', markersize=3, color='#1f77b4', alpha=0.7,
        label='Subharmonic env E(n)')

k = 'g=0.60_sub_all'
if k in results_sub:
    r = results_sub[k]
    n_dense = np.linspace(1, 100, 200)
    ax.plot(n_dense, stretched_exp(n_dense, r['A0'], r['n_star'], r['alpha']),
           '-', color='#1f77b4', linewidth=2.5,
           label=f'Fit: alpha={r["alpha"]:.3f}+/-{r["alpha_err"]:.3f}')

ax.set_xlabel('Floquet cycle n', fontsize=11)
ax.set_ylabel('Signal', fontsize=11)
ax.set_title('B. g=0.60 (Thermal) Subharmonic Envelope', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# Panel C: Alpha vs g — THE KEY PLOT
ax = fig.add_subplot(3, 3, 3)

g_plot = []
a_plot = []
ae_plot = []
labels_ax = []

# Averaged subharmonic
for g, label in [(0.60, 'Averaged'), (0.97, 'Averaged')]:
    if g in best:
        g_plot.append(g)
        a_plot.append(best[g]['alpha'])
        ae_plot.append(best[g]['alpha_err'])
        labels_ax.append(f'g={g:.2f} avg')

# Prethermal
if 0.94 in best:
    g_plot.append(0.94)
    a_plot.append(best[0.94]['alpha'])
    ae_plot.append(best[0.94]['alpha_err'])
    labels_ax.append('g=0.94 pretherm')

# Per-qubit means
for phase, data in per_qubit_results.items():
    if '0.97' in phase and 'Neel' in phase:
        g_plot.append(0.975)
        a_plot.append(np.mean(data['alphas']))
        ae_plot.append(np.std(data['alphas']))
        labels_ax.append('g=0.97 per-qubit (Neel)')
    elif '0.60' in phase and 'Neel' in phase:
        g_plot.append(0.605)
        a_plot.append(np.mean(data['alphas']))
        ae_plot.append(np.std(data['alphas']))
        labels_ax.append('g=0.60 per-qubit (Neel)')

ax.errorbar(g_plot, a_plot, yerr=ae_plot, fmt='o', markersize=10, capsize=6,
           capthick=2, color='navy', zorder=5)
for i, lab in enumerate(labels_ax):
    ax.annotate(lab, (g_plot[i], a_plot[i]), textcoords="offset points",
               xytext=(8, 8), fontsize=7, alpha=0.8)

ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
           label='Merkabit alpha=1.3')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.3)

# Phase boundary
df_5 = pd.read_csv(os.path.join(DATA, 'fig_5.csv'), encoding='utf-8-sig')
g_5 = df_5['g'].values
s_20 = df_5['s_20'].values
ds = np.diff(s_20) / np.diff(g_5)
g_boundary = (g_5[:-1] + g_5[1:])[np.argmax(ds)] / 2
ax.axvline(x=g_boundary, color='orange', linestyle=':', linewidth=2, alpha=0.7,
           label=f'Phase boundary g~{g_boundary:.2f}')

ax.set_xlabel('g (driving parameter)', fontsize=12)
ax.set_ylabel('alpha (subharmonic envelope)', fontsize=12)
ax.set_title('C. Subharmonic Alpha vs g', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0.50, 1.05)
ax.set_ylim(0, 2.5)

# Panel D: Per-qubit alpha at g=0.97 (subharmonic)
ax = fig.add_subplot(3, 3, 4)
key = 'MBL DTC (g=0.97) Neel'
if key in per_qubit_results:
    d = per_qubit_results[key]
    ax.errorbar(d['qids'], d['alphas'], yerr=d['alpha_errs'],
               fmt='o-', markersize=6, capsize=3, color='#d62728')
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, alpha=0.7)
    mean_a = np.mean(d['alphas'])
    ax.axhline(y=mean_a, color='green', linestyle='-', alpha=0.5,
              label=f'Mean={mean_a:.3f}')
    ax.fill_between(d['qids'], mean_a-np.std(d['alphas']),
                   mean_a+np.std(d['alphas']), alpha=0.1, color='green')
    ax.set_xlabel('Qubit position')
    ax.set_ylabel('alpha')
    ax.set_title(f'D. Per-Qubit Subharmonic alpha (g=0.97, Neel)\n'
                f'CV={np.std(d["alphas"])/mean_a*100:.1f}%', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 21)

# Panel E: Per-qubit alpha at g=0.60
ax = fig.add_subplot(3, 3, 5)
key = 'Thermal (g=0.60) Neel'
if key in per_qubit_results:
    d = per_qubit_results[key]
    ax.errorbar(d['qids'], d['alphas'], yerr=d['alpha_errs'],
               fmt='o-', markersize=6, capsize=3, color='#1f77b4')
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, alpha=0.7)
    mean_a = np.mean(d['alphas'])
    ax.axhline(y=mean_a, color='green', linestyle='-', alpha=0.5,
              label=f'Mean={mean_a:.3f}')
    ax.fill_between(d['qids'], mean_a-np.std(d['alphas']),
                   mean_a+np.std(d['alphas']), alpha=0.1, color='green')
    ax.set_xlabel('Qubit position')
    ax.set_ylabel('alpha')
    ax.set_title(f'E. Per-Qubit Subharmonic alpha (g=0.60, Neel)\n'
                f'CV={np.std(d["alphas"])/mean_a*100:.1f}%', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 21)

# Panel F: Histogram comparison
ax = fig.add_subplot(3, 3, 6)
for key, color, label in [
    ('MBL DTC (g=0.97) Neel', '#d62728', 'g=0.97 MBL (Neel)'),
    ('Thermal (g=0.60) Neel', '#1f77b4', 'g=0.60 Thermal (Neel)'),
]:
    if key in per_qubit_results:
        d = per_qubit_results[key]
        ax.hist(d['alphas'], bins=15, alpha=0.5, color=color, edgecolor='black',
               label=label, density=True)
ax.axvline(x=1.3, color='red', linestyle='--', linewidth=2.5, label='Merkabit alpha=1.3')
ax.axvspan(1.1, 1.5, alpha=0.1, color='red')
ax.set_xlabel('alpha', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('F. Alpha Distribution by Phase', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# Panel G: fig_2d per-qubit at g=0.97
ax = fig.add_subplot(3, 3, 7)
if valid.sum() > 0:
    qids_2d = np.arange(1, 21)
    ax.errorbar(qids_2d[valid], alphas_2d_97[valid], yerr=errs_2d_97[valid],
               fmt='s-', markersize=6, capsize=3, color='#d62728')
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, alpha=0.7)
    m = np.nanmean(alphas_2d_97)
    ax.axhline(y=m, color='green', linestyle='-', alpha=0.5, label=f'Mean={m:.3f}')
    ax.set_xlabel('Qubit position')
    ax.set_ylabel('alpha')
    ax.set_title(f'G. fig_2d Per-Qubit Subharmonic alpha (g=0.97)\n'
                f'Mean={m:.3f}+/-{np.nanstd(alphas_2d_97):.3f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 21)

# Panel H: Even vs Odd branch comparison
ax = fig.add_subplot(3, 3, 8)
branches = []
for g in ['0.60', '0.97']:
    for b in ['even', 'odd', 'sub_all']:
        k = f'g={g}_{b}'
        if k in results_sub:
            r = results_sub[k]
            branches.append((g, b, r['alpha'], r['alpha_err'], r['R2']))

if branches:
    x_pos = np.arange(len(branches))
    colors = ['#1f77b4' if '0.60' in b[0] else '#d62728' for b in branches]
    ax.bar(x_pos, [b[2] for b in branches], yerr=[b[3] for b in branches],
          color=colors, edgecolor='black', capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'g={b[0]}\n{b[1]}' for b in branches], fontsize=8)
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='alpha=1.3')
    ax.set_ylabel('alpha')
    ax.set_title('H. Even/Odd/Combined Branch Comparison', fontsize=12, fontweight='bold')
    ax.legend()

# Panel I: Summary text
ax = fig.add_subplot(3, 3, 9)
ax.axis('off')

txt = "SUBHARMONIC ENVELOPE: KEY FINDINGS\n"
txt += "=" * 40 + "\n\n"

for g in sorted(best.keys()):
    r = best[g]
    txt += f"g={g:.2f}: alpha = {r['alpha']:.4f} +/- {r['alpha_err']:.4f}\n"
    txt += f"        n* = {r['n_star']:.1f}, R2 = {r['R2']:.4f}\n\n"

txt += "\nPer-qubit (Neel, subharmonic):\n"
for key in ['MBL DTC (g=0.97) Neel', 'Thermal (g=0.60) Neel']:
    if key in per_qubit_results:
        d = per_qubit_results[key]
        txt += f"  {key}:\n"
        txt += f"    mean = {np.mean(d['alphas']):.4f} +/- {np.std(d['alphas']):.4f}\n"

if valid.sum() > 0:
    txt += f"\nfig_2d (g=0.97, subharmonic):\n"
    txt += f"  mean = {np.nanmean(alphas_2d_97):.4f} +/- {np.nanstd(alphas_2d_97):.4f}\n"

# The verdict
txt += "\n" + "=" * 40 + "\n"
if 0.97 in best:
    a97 = best[0.97]['alpha']
    if abs(a97 - 1.3) < 0.15:
        txt += f"\nVERDICT: alpha(g=0.97) = {a97:.3f}\n"
        txt += "CLOSE TO 1.3 - Merkabit signature\n"
        txt += "detected in MBL DTC phase!"
    else:
        txt += f"\nVERDICT: alpha(g=0.97) = {a97:.3f}\n"
        txt += f"|alpha - 1.3| = {abs(a97-1.3):.3f}\n"

ax.text(0.05, 0.95, txt, transform=ax.transAxes,
       fontsize=9, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Subharmonic Envelope Analysis: Merkabit Framework x DTC Phase Diagram\n'
            'E(n) = (-1)^n A(n) fitted to A0*exp(-(n/n*)^alpha)',
            fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(FIGURES_DIR, 'subharmonic_envelope_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: subharmonic_envelope_analysis.png")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL VERDICT: SUBHARMONIC ENVELOPE ALPHA")
print("=" * 70)

if best:
    print("\n  Subharmonic envelope alpha at each g:")
    for g in sorted(best.keys()):
        r = best[g]
        print(f"    g = {g:.2f}: alpha = {r['alpha']:.4f} +/- {r['alpha_err']:.4f}")

    if 0.97 in best and 0.94 in best and 0.60 in best:
        a60 = best[0.60]['alpha']
        a94 = best[0.94]['alpha']
        a97 = best[0.97]['alpha']

        print(f"\n  Does alpha=1.3 appear specifically in MBL DTC (g=0.97)?")
        print(f"    alpha(0.97) = {a97:.4f}, |alpha - 1.3| = {abs(a97-1.3):.4f}")

        print(f"\n  Does it fall away at g=0.94 and g=0.60?")
        print(f"    alpha(0.94) = {a94:.4f}, |alpha - 1.3| = {abs(a94-1.3):.4f}")
        print(f"    alpha(0.60) = {a60:.4f}, |alpha - 1.3| = {abs(a60-1.3):.4f}")

        if abs(a97 - 1.3) < abs(a94 - 1.3) and abs(a97 - 1.3) < abs(a60 - 1.3):
            print(f"\n  ==> alpha IS closest to 1.3 at g=0.97 (MBL DTC)")
            if abs(a97 - 1.3) < 0.15:
                print(f"  ==> AND within 0.15 of 1.3 — Merkabit-DTC connection supported")
            else:
                print(f"  ==> BUT distance {abs(a97-1.3):.3f} is substantial")
        else:
            closest_g = min(best.keys(), key=lambda g: abs(best[g]['alpha'] - 1.3))
            print(f"\n  ==> alpha is closest to 1.3 at g={closest_g:.2f}, not at g=0.97")

# Save report
with open(os.path.join(REPORTS_DIR, 'subharmonic_report.txt'), 'w', encoding='utf-8') as f:
    f.write("SUBHARMONIC ENVELOPE ANALYSIS REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write("Method: E(n) = (-1)^n * A(n), fitted to A0*exp(-(n/n*)^alpha)\n")
    f.write("This extracts the DTC order parameter decay envelope specifically.\n\n")

    f.write("HIGH-CONFIDENCE RESULTS:\n")
    f.write(f"{'g':<8} {'alpha':>10} {'+-err':>10} {'n*':>10} {'R2':>10}\n")
    f.write("-" * 50 + "\n")
    for g in sorted(best.keys()):
        r = best[g]
        f.write(f"{g:<8.2f} {r['alpha']:>10.4f} {r['alpha_err']:>10.4f} {r['n_star']:>10.2f} {r['R2']:>10.6f}\n")

    f.write("\n\nPER-QUBIT RESULTS (Neel initial state, subharmonic envelope):\n")
    for key in sorted(per_qubit_results.keys()):
        d = per_qubit_results[key]
        f.write(f"\n  {key}:\n")
        f.write(f"    Mean alpha = {np.mean(d['alphas']):.4f} +/- {np.std(d['alphas']):.4f}\n")
        f.write(f"    CV = {np.std(d['alphas'])/np.mean(d['alphas'])*100:.1f}%\n")
        for i in range(len(d['qids'])):
            tag = '*' if 1.1 <= d['alphas'][i] <= 1.5 else ' '
            f.write(f"    {tag} Q{d['qids'][i]:2d}: alpha = {d['alphas'][i]:.4f} +/- {d['alpha_errs'][i]:.4f}\n")

    if valid.sum() > 0:
        f.write(f"\n\nfig_2d PER-QUBIT (g=0.97, subharmonic):\n")
        f.write(f"  Mean alpha = {np.nanmean(alphas_2d_97):.4f} +/- {np.nanstd(alphas_2d_97):.4f}\n")
        for qi in range(20):
            if valid[qi]:
                tag = '*' if 1.1 <= alphas_2d_97[qi] <= 1.5 else ' '
                f.write(f"  {tag} Q{qi+1:2d}: alpha = {alphas_2d_97[qi]:.4f} +/- {errs_2d_97[qi]:.4f}\n")

print(f"\nSaved: subharmonic_report.txt")
print(f"All outputs: {RESULTS_DIR}")
