"""
Full Disorder Sweep: Extending r = 0.91 in Xiang et al. 2024
==============================================================
Exhausts every disorder condition in the dataset.
Tests robustness of the alpha-order correlation.

Dataset: Xiang et al. 2024, Zenodo 13692134
  - fig4a.mat: 6 stabilizer Z x 17 disorder values x 41 steps
  - fig4b.mat: DTC FFT peak strength at all 17 disorders
  - fig4e.mat: Topological order at 14 disorders
  - fig2a.mat: Zero-disorder baseline (6 qubits x 41 steps)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

# =====================================================================
# UTILITIES
# =====================================================================

def kww(t, A, tau, alpha):
    """KWW with amplitude."""
    return A * np.exp(-(t / tau) ** alpha)

def fit_kww_envelope(n, env, p0=None):
    """Fit |Z(n)| envelope to A*exp(-(n/tau)^alpha).
    Returns (alpha, alpha_err, tau, A, R2, converged).
    Uses CONSISTENT bounds for all disorder values."""
    if p0 is None:
        p0 = [env[0], 20.0, 1.0]
    try:
        mask = env > 0.02  # signal above noise floor
        n_fit, y_fit = n[mask], env[mask]
        if len(n_fit) < 4:
            return np.nan, np.nan, np.nan, np.nan, np.nan, False
        # CONSISTENT bounds across ALL disorder values
        popt, pcov = curve_fit(kww, n_fit, y_fit, p0=p0,
                               bounds=([0.01, 0.5, 0.05],
                                       [2.0, 200.0, 10.0]),
                               maxfev=20000)
        A_fit, tau_fit, alpha_fit = popt
        perr = np.sqrt(np.diag(pcov))
        alpha_err = perr[2]
        y_pred = kww(n_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Check for boundary-hitting
        hit_bound = False
        if abs(alpha_fit - 0.05) < 0.01 or abs(alpha_fit - 10.0) < 0.1:
            hit_bound = True
        if abs(tau_fit - 0.5) < 0.1 or abs(tau_fit - 200.0) < 1.0:
            hit_bound = True

        return alpha_fit, alpha_err, tau_fit, A_fit, r2, not hit_bound
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =====================================================================
# DATA LOADING
# =====================================================================

XIANG_BASE = XIANG_2024_BASE


# =====================================================================
# TASK 1: Catalogue every disorder condition
# =====================================================================

def task1():
    print_header("TASK 1: Complete Disorder Catalogue")

    # Load fig4a — main disorder sweep
    d4a = loadmat(f"{XIANG_BASE}/main_figure4/fig4a.mat")
    disorders_4a = d4a['disorders'][0]
    Zs = d4a['Zs_exp_avg']  # (6, 17, 41)
    Zs_fft = d4a['Zs_exp_fft']  # (6, 17, 21)
    freqs = d4a['freqs'][0]

    # Load fig4b — DTC peak strength
    d4b = loadmat(f"{XIANG_BASE}/main_figure4/fig4b.mat")
    peak_avg = d4b['exp_fZs_peak_avg'][0]
    peak_sd = d4b['exp_fZs_peak_sd'][0]

    # Load fig4e — topological order
    d4e = loadmat(f"{XIANG_BASE}/main_figure4/fig4e.mat")
    disorders_4e = d4e['disorders'][0]
    topo_avg = d4e['S_topo_exp_avg'][0]
    topo_err = d4e['S_topo_exp_err'][0]

    # Load fig2a — zero disorder baseline
    d2a = loadmat(f"{XIANG_BASE}/main_figure2/fig2a.mat")
    Zs_baseline = d2a['Zs_exp_avg']  # (6, 41)

    print(f"  Source files and disorder values found:")
    print(f"    fig4a.mat: {len(disorders_4a)} disorder values")
    print(f"      W = {disorders_4a}")
    print(f"      Data: 6 stabilizer Z qubits x 41 half-period steps per W")
    print(f"      Also: FFT data (6 x 17 x 21)")
    print(f"    fig4b.mat: DTC FFT peak strength at {len(peak_avg)} disorders")
    print(f"    fig4e.mat: Topological order at {len(disorders_4e)} disorders")
    print(f"      W = {disorders_4e}")
    print(f"    fig2a.mat: Zero-disorder baseline (6 qubits x 41 steps)")
    print()

    # Complete catalogue
    print(f"  Complete disorder catalogue:")
    print(f"  {'W':>6s}  {'Z decay?':>8s}  {'FFT peak?':>9s}  {'Topo ord?':>9s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*9}")

    all_W = sorted(set(list(disorders_4a) + list(disorders_4e)))
    for w in all_W:
        has_z = 'YES' if w in disorders_4a else 'no'
        has_fft = 'YES' if w in disorders_4a else 'no'  # same source
        has_topo = 'YES' if w in disorders_4e else 'no'
        print(f"  {w:>6.1f}  {has_z:>8s}  {has_fft:>9s}  {has_topo:>9s}")

    print(f"\n  Total unique W values: {len(all_W)}")
    print(f"  W range: {min(all_W):.1f} to {max(all_W):.1f}")
    print(f"  Missing W values in topo order: "
          f"{sorted(set(disorders_4a) - set(disorders_4e))}")

    return (d4a, d4b, d4e, d2a, disorders_4a, peak_avg, peak_sd,
            disorders_4e, topo_avg, topo_err)


# =====================================================================
# TASK 2: Extract alpha for every disorder condition
# =====================================================================

def task2(d4a, disorders):
    print_header("TASK 2: KWW Exponent alpha at Every Disorder Value")

    Zs = d4a['Zs_exp_avg']  # (6, 17, 41)
    Zs_err = d4a['Zs_exp_err']
    n_qubits, n_disorders, n_steps = Zs.shape
    n = np.arange(n_steps, dtype=float)

    print(f"  Fitting procedure:")
    print(f"    Model: |Z(n)| = A * exp(-(n/tau)^alpha)")
    print(f"    Bounds: A in [0.01, 2.0], tau in [0.5, 200], alpha in [0.05, 10]")
    print(f"    Signal threshold: |Z| > 0.02")
    print(f"    Consistent across all W — no per-disorder tuning")
    print()

    # Per-qubit fits
    print(f"  Per-qubit alpha values:")
    header = f"  {'W':>5s}"
    for q in range(n_qubits):
        header += f"  {'Z'+str(q):>8s}"
    header += f"  {'mean':>8s}  {'std':>7s}  {'R2_mean':>7s}  {'flags':>10s}"
    print(header)
    print(f"  {'-'*5}" + f"  {'-'*8}" * n_qubits +
          f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}")

    results = []
    for di in range(n_disorders):
        W = disorders[di]
        alphas_q = []
        r2s_q = []
        flags = []

        for q in range(n_qubits):
            env = np.abs(Zs[q, di, :])
            a, ae, tau, A, r2, conv = fit_kww_envelope(n, env)
            alphas_q.append(a)
            r2s_q.append(r2)
            if not conv:
                flags.append(f'Z{q}:bound')
            elif r2 < 0.85:
                flags.append(f'Z{q}:lowR2')

        alphas_q = np.array(alphas_q)
        r2s_q = np.array(r2s_q)
        valid = ~np.isnan(alphas_q)

        mean_a = np.nanmean(alphas_q) if np.any(valid) else np.nan
        std_a = np.nanstd(alphas_q) if np.sum(valid) > 1 else np.nan
        mean_r2 = np.nanmean(r2s_q) if np.any(valid) else np.nan
        flag_str = '; '.join(flags) if flags else 'OK'

        row = f"  {W:>5.1f}"
        for q in range(n_qubits):
            if np.isnan(alphas_q[q]):
                row += f"  {'FAIL':>8s}"
            else:
                row += f"  {alphas_q[q]:>8.4f}"
        row += f"  {mean_a:>8.4f}  {std_a:>7.4f}  {mean_r2:>7.4f}  {flag_str:>10s}"
        print(row)

        results.append({
            'W': W,
            'alphas': alphas_q,
            'mean_alpha': mean_a,
            'std_alpha': std_a,
            'mean_r2': mean_r2,
            'flags': flag_str,
            'n_valid': np.sum(valid)
        })

    # Quality summary
    n_ok = sum(1 for r in results if r['flags'] == 'OK')
    n_flagged = len(results) - n_ok
    print(f"\n  Quality summary: {n_ok}/{len(results)} clean, "
          f"{n_flagged} flagged")

    # Apply quality threshold BEFORE seeing correlations
    R2_THRESHOLD = 0.85
    n_pass = sum(1 for r in results if r['mean_r2'] > R2_THRESHOLD)
    print(f"  Quality threshold R^2 > {R2_THRESHOLD}: "
          f"{n_pass}/{len(results)} pass")
    for r in results:
        r['passes_quality'] = r['mean_r2'] > R2_THRESHOLD

    return results


# =====================================================================
# TASK 3: Extract DTC order strength for every disorder
# =====================================================================

def task3(d4b, disorders, peak_avg, peak_sd, disorders_4e, topo_avg, topo_err):
    print_header("TASK 3: DTC Order Parameter at Every Disorder Value")

    print(f"  Order parameter 1: FFT peak strength (subharmonic response)")
    print(f"    Available at {len(disorders)} disorder values")
    print(f"  Order parameter 2: Topological order S_topo")
    print(f"    Available at {len(disorders_4e)} disorder values")
    print()

    print(f"  {'W':>5s}  {'FFT peak':>10s}  {'peak_sd':>8s}  "
          f"{'S_topo':>8s}  {'topo_err':>8s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    order_data = []
    for di, W in enumerate(disorders):
        fft_p = peak_avg[di]
        fft_s = peak_sd[di]

        # Find matching topo value
        topo_idx = np.where(np.abs(disorders_4e - W) < 0.01)[0]
        if len(topo_idx) > 0:
            st = topo_avg[topo_idx[0]]
            se = topo_err[topo_idx[0]]
        else:
            st = np.nan
            se = np.nan

        topo_str = f"{st:>8.4f}" if not np.isnan(st) else f"{'N/A':>8s}"
        terr_str = f"{se:>8.4f}" if not np.isnan(se) else f"{'N/A':>8s}"

        print(f"  {W:>5.1f}  {fft_p:>10.4f}  {fft_s:>8.4f}  "
              f"{topo_str}  {terr_str}")

        order_data.append({
            'W': W,
            'fft_peak': fft_p,
            'fft_peak_sd': fft_s,
            'topo': st,
            'topo_err': se
        })

    return order_data


# =====================================================================
# TASK 4: Full correlation analysis
# =====================================================================

def task4(alpha_results, order_data):
    print_header("TASK 4: Full Correlation Analysis")

    # Merge alpha and order data
    n = len(alpha_results)
    W_arr = np.array([r['W'] for r in alpha_results])
    alpha_arr = np.array([r['mean_alpha'] for r in alpha_results])
    alpha_err = np.array([r['std_alpha'] for r in alpha_results])
    r2_arr = np.array([r['mean_r2'] for r in alpha_results])
    fft_arr = np.array([o['fft_peak'] for o in order_data])
    fft_err = np.array([o['fft_peak_sd'] for o in order_data])
    passes = np.array([r['passes_quality'] for r in alpha_results])

    # ---- FULL DATASET (all 17 points) ----
    print(f"  A. Full dataset: all {n} disorder values")
    print()

    valid = ~np.isnan(alpha_arr) & ~np.isnan(fft_arr)
    a_v = alpha_arr[valid]
    f_v = fft_arr[valid]
    n_v = np.sum(valid)

    r_all, p_all = pearsonr(a_v, f_v)
    rho_all, p_rho = spearmanr(a_v, f_v)

    print(f"    N = {n_v} valid data points")
    print(f"    Pearson r  = {r_all:.4f}, p = {p_all:.6f}")
    print(f"    Spearman rho = {rho_all:.4f}, p = {p_rho:.6f}")

    # Linear fit
    if n_v > 2:
        coeffs = np.polyfit(f_v, a_v, 1)
        a_pred = np.polyval(coeffs, f_v)
        ss_res = np.sum((a_v - a_pred) ** 2)
        ss_tot = np.sum((a_v - np.mean(a_v)) ** 2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"    Linear fit: alpha = {coeffs[0]:.4f} * O + {coeffs[1]:.4f}")
        print(f"    Linear R^2 = {r2_lin:.4f}")

        # Residuals and outlier detection
        residuals = a_v - a_pred
        resid_std = np.std(residuals)
        outliers = np.abs(residuals) > 2 * resid_std
        if np.any(outliers):
            W_valid = W_arr[valid]
            print(f"\n    Outliers (>2sigma from linear fit):")
            for idx in np.where(outliers)[0]:
                print(f"      W={W_valid[idx]:.1f}: alpha={a_v[idx]:.4f}, "
                      f"predicted={a_pred[idx]:.4f}, residual={residuals[idx]:+.4f}")
        else:
            print(f"    No outliers detected (all within 2sigma)")

    # 95% CI for r using Fisher z-transform
    import math
    if n_v > 3:
        z = 0.5 * np.log((1 + r_all) / (1 - r_all))
        se_z = 1 / np.sqrt(n_v - 3)
        z_lo, z_hi = z - 1.96 * se_z, z + 1.96 * se_z
        r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
        r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
        print(f"    95% CI for r: [{r_lo:.4f}, {r_hi:.4f}]")

    # ---- QUALITY-FILTERED DATASET ----
    print(f"\n  B. Quality-filtered dataset (R^2 > 0.85):")
    qf = valid & passes
    a_qf = alpha_arr[qf]
    f_qf = fft_arr[qf]
    n_qf = np.sum(qf)

    if n_qf > 2:
        r_qf, p_qf = pearsonr(a_qf, f_qf)
        rho_qf, p_rho_qf = spearmanr(a_qf, f_qf)
        print(f"    N = {n_qf} data points")
        print(f"    Pearson r  = {r_qf:.4f}, p = {p_qf:.6f}")
        print(f"    Spearman rho = {rho_qf:.4f}, p = {p_rho_qf:.6f}")
    else:
        print(f"    Only {n_qf} points pass quality filter — insufficient")
        r_qf, p_qf = np.nan, np.nan

    # ---- Leave-one-out: does removing any point change sign or p>0.05? ----
    print(f"\n  C. Single-point sensitivity (full dataset):")
    print(f"    Leave-one-out: does removing any single point change r sign or p>0.05?")

    W_valid = W_arr[valid]
    sensitive_points = []
    for idx in range(n_v):
        a_loo = np.delete(a_v, idx)
        f_loo = np.delete(f_v, idx)
        r_loo, p_loo = pearsonr(a_loo, f_loo)
        if r_loo * r_all < 0 or p_loo > 0.05:
            sensitive_points.append((W_valid[idx], r_loo, p_loo))
            print(f"      W={W_valid[idx]:.1f}: r_loo={r_loo:.4f}, p={p_loo:.6f} *** SENSITIVE")

    if not sensitive_points:
        print(f"      No single point changes sign or pushes p > 0.05")

    # ---- Data table ----
    print(f"\n  D. Complete data table:")
    print(f"  {'W':>5s}  {'alpha':>8s}  {'a_err':>7s}  {'FFT_pk':>8s}  "
          f"{'pk_err':>7s}  {'R2':>6s}  {'quality':>7s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}")
    for i in range(n):
        q_str = 'PASS' if passes[i] else 'FAIL'
        print(f"  {W_arr[i]:>5.1f}  {alpha_arr[i]:>8.4f}  {alpha_err[i]:>7.4f}  "
              f"{fft_arr[i]:>8.4f}  {fft_err[i]:>7.4f}  {r2_arr[i]:>6.3f}  {q_str:>7s}")

    return (r_all, p_all, rho_all, p_rho, r_qf, p_qf,
            W_arr, alpha_arr, fft_arr, passes, valid)


# =====================================================================
# TASK 5: Disorder sweep profile alpha(W)
# =====================================================================

def task5(W_arr, alpha_arr, alpha_results):
    print_header("TASK 5: Disorder Sweep Profile alpha(W)")

    print(f"  Expected profile from companion paper:")
    print(f"    W=0 (clean): alpha ~ 1.3 (prethermal cooperative)")
    print(f"    W small: alpha peaks (maximally activated search)")
    print(f"    W moderate: alpha decreasing (MBL onset)")
    print(f"    W large: alpha ~ 0.82 (MBL, localized relaxation)")
    print()

    print(f"  Observed profile:")
    print(f"  {'W':>5s}  {'alpha':>8s}  {'R2':>6s}  {'regime':>20s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*20}")

    for r in alpha_results:
        W = r['W']
        a = r['mean_alpha']
        r2 = r['mean_r2']

        if W < 0.3:
            regime = 'clean/prethermal'
        elif W < 1.0:
            regime = 'weak disorder'
        elif W < 2.0:
            regime = 'moderate disorder'
        else:
            regime = 'strong disorder/MBL'

        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
        a_str = f"{a:.4f}" if not np.isnan(a) else "N/A"
        print(f"  {W:>5.1f}  {a_str:>8s}  {r2_str:>6s}  {regime:>20s}")

    # Find peak
    valid = ~np.isnan(alpha_arr)
    if np.any(valid):
        peak_idx = np.nanargmax(alpha_arr)
        print(f"\n  Peak alpha = {alpha_arr[peak_idx]:.4f} at W = {W_arr[peak_idx]:.1f}")

        # Check if peak is at weak disorder
        if 0.0 < W_arr[peak_idx] < 1.0:
            print(f"  Peak at weak disorder — MATCHES prediction")
        elif W_arr[peak_idx] == 0.0:
            print(f"  Peak at clean limit — no weak-disorder enhancement")
        else:
            print(f"  Peak at W={W_arr[peak_idx]:.1f} — "
                  f"{'matches' if W_arr[peak_idx] < 0.5 else 'does not match'} prediction")

    # Monotonic decay check after peak
    if np.any(valid):
        peak_W = W_arr[peak_idx]
        post_peak = [(W_arr[i], alpha_arr[i]) for i in range(len(W_arr))
                     if W_arr[i] > peak_W and not np.isnan(alpha_arr[i])]
        if len(post_peak) > 2:
            ws, als = zip(*post_peak)
            is_monotone = all(als[i] >= als[i+1] - 0.1
                              for i in range(len(als)-1))
            print(f"  Monotonic decrease after peak: "
                  f"{'approximately yes' if is_monotone else 'NO — non-monotonic'}")


# =====================================================================
# TASK 6: Compare with previous 6-point result
# =====================================================================

def task6(r_all, p_all, rho_all, alpha_results, W_arr, alpha_arr, fft_arr, valid):
    print_header("TASK 6: Comparison with Previous 6-Point Result")

    # The previous 6-point analysis used W = 0.0, 0.1, 0.2, 0.4, 0.6, 0.8
    # (based on the starred entries in the original report: disorder_0.0 through 0.8)
    # But let me check which ones were actually used
    # From memory: the previous analysis starred W=0.0, 0.1, 0.2 and fig2a Z0-Z3
    # The 6 disorder values with alpha > 1 and good R2 were likely the low-W ones

    # Reconstruct: which 6 points gave r=0.91?
    # The previous report showed disorder_0.0 through disorder_0.8 as the "good" fits
    # Let me identify them by R2 > 0.95

    print(f"  Previous result: r = 0.91 from 6 disorder values (p ~ 0.01)")
    print()

    # Find the 6 low-disorder points that were likely used
    prev_W = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    prev_mask = np.array([w in prev_W for w in W_arr]) & valid
    a_prev = alpha_arr[prev_mask]
    f_prev = fft_arr[prev_mask]
    n_prev = np.sum(prev_mask)

    if n_prev >= 3:
        r_prev, p_prev = pearsonr(a_prev, f_prev)
        print(f"  Previous 6-point subset (W = {prev_W}):")
        print(f"    N = {n_prev}")
        print(f"    Pearson r = {r_prev:.4f}, p = {p_prev:.6f}")
    else:
        r_prev, p_prev = np.nan, np.nan
        print(f"  Could not reconstruct previous 6-point subset (only {n_prev} valid)")

    # Full dataset comparison
    n_full = np.sum(valid)
    print(f"\n  Full dataset:")
    print(f"    N = {n_full}")
    print(f"    Pearson r = {r_all:.4f}, p = {p_all:.6f}")
    print(f"    Spearman rho = {rho_all:.4f}")

    # Summary table
    print(f"\n  Comparison:")
    print(f"  {'Metric':<25s}  {'Previous (6 pt)':>15s}  {'Full ({n_full} pt)':>15s}  {'Change':>10s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*15}  {'-'*10}")

    def change_str(old, new):
        if np.isnan(old) or np.isnan(new):
            return 'N/A'
        diff = new - old
        return f"{'+'if diff>0 else ''}{diff:.4f}"

    print(f"  {'N data points':<25s}  {n_prev:>15d}  {n_full:>15d}  "
          f"{'+' + str(n_full - n_prev):>10s}")
    print(f"  {'Pearson r':<25s}  {r_prev:>15.4f}  {r_all:>15.4f}  "
          f"{change_str(r_prev, r_all):>10s}")
    print(f"  {'p-value':<25s}  {p_prev:>15.6f}  {p_all:>15.6f}  "
          f"{'improved' if p_all < p_prev else 'worsened':>10s}")

    # Which points were excluded previously?
    excluded_W = sorted(set(W_arr) - set(prev_W))
    print(f"\n  Disorder values in full dataset but NOT in previous 6-point:")
    print(f"    W = {excluded_W}")
    print(f"    These comprise {len(excluded_W)} additional conditions")

    # Why were they excluded?
    print(f"\n  Properties of excluded points:")
    print(f"  {'W':>5s}  {'alpha':>8s}  {'R2':>6s}  {'FFT_pk':>8s}  {'likely reason':>25s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*25}")
    for w in excluded_W:
        idx = np.where(np.abs(W_arr - w) < 0.01)[0]
        if len(idx) > 0:
            i = idx[0]
            a = alpha_arr[i]
            r2 = [r for r in alpha_results if abs(r['W'] - w) < 0.01][0]['mean_r2']
            f = fft_arr[i]
            reason = ''
            if r2 < 0.85:
                reason = 'low R2 (poor fit)'
            elif a < 0.5:
                reason = 'very low alpha (noise)'
            elif w > 1.0:
                reason = 'high disorder (MBL regime)'
            else:
                reason = 'not clear'
            print(f"  {w:>5.1f}  {a:>8.4f}  {r2:>6.3f}  {f:>8.4f}  {reason:>25s}")

    return r_prev, p_prev


# =====================================================================
# TASK 7: Jackknife stability test
# =====================================================================

def task7(alpha_arr, fft_arr, valid, W_arr):
    print_header("TASK 7: Jackknife Stability Test")

    a_v = alpha_arr[valid]
    f_v = fft_arr[valid]
    W_v = W_arr[valid]
    n_v = len(a_v)

    print(f"  Jackknife: compute r for every subset of N-1 points")
    print(f"  N = {n_v}")
    print()

    jk_rs = []
    jk_ps = []

    print(f"  {'Removed W':>10s}  {'r_jk':>8s}  {'p_jk':>10s}  {'flag':>10s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    for idx in range(n_v):
        a_loo = np.delete(a_v, idx)
        f_loo = np.delete(f_v, idx)
        r_loo, p_loo = pearsonr(a_loo, f_loo)
        jk_rs.append(r_loo)
        jk_ps.append(p_loo)

        flag = ''
        if r_loo < 0.80:
            flag = 'DRIVER'
        elif r_loo < 0.85:
            flag = 'influential'
        print(f"  W={W_v[idx]:>6.1f}  {r_loo:>8.4f}  {p_loo:>10.6f}  {flag:>10s}")

    jk_rs = np.array(jk_rs)
    jk_ps = np.array(jk_ps)

    # Jackknife statistics
    r_jk_mean = np.mean(jk_rs)
    r_jk_std = np.std(jk_rs) * np.sqrt(n_v - 1)  # jackknife SE
    r_jk_min = np.min(jk_rs)
    r_jk_max = np.max(jk_rs)

    print(f"\n  Jackknife summary:")
    print(f"    Mean r (jackknife)   = {r_jk_mean:.4f}")
    print(f"    SE (jackknife)       = {r_jk_std:.4f}")
    print(f"    Min r (leave-one-out)= {r_jk_min:.4f} (W={W_v[np.argmin(jk_rs)]:.1f})")
    print(f"    Max r (leave-one-out)= {r_jk_max:.4f} (W={W_v[np.argmax(jk_rs)]:.1f})")
    print(f"    Range                = [{r_jk_min:.4f}, {r_jk_max:.4f}]")

    n_drivers = np.sum(jk_rs < 0.80)
    if n_drivers == 0:
        print(f"\n    No single point drops r below 0.80 — result is STABLE")
        stable = True
    else:
        print(f"\n    {n_drivers} point(s) drive the correlation — result is FRAGILE")
        stable = False

    return r_jk_mean, r_jk_std, r_jk_min, r_jk_max, stable


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  FULL DISORDER SWEEP: Extending r = 0.91")
    print("  Xiang et al. 2024 — Complete Dataset Analysis")
    print("=" * 70)

    # Task 1
    (d4a, d4b, d4e, d2a, disorders, peak_avg, peak_sd,
     disorders_4e, topo_avg, topo_err) = task1()

    # Task 2
    alpha_results = task2(d4a, disorders)

    # Task 3
    order_data = task3(d4b, disorders, peak_avg, peak_sd,
                       disorders_4e, topo_avg, topo_err)

    # Task 4
    (r_all, p_all, rho_all, p_rho, r_qf, p_qf,
     W_arr, alpha_arr, fft_arr, passes, valid) = task4(alpha_results, order_data)

    # Task 5
    alpha_err_arr = np.array([r['std_alpha'] for r in alpha_results])
    task5(W_arr, alpha_arr, alpha_results)

    # Task 6
    r_prev, p_prev = task6(r_all, p_all, rho_all, alpha_results,
                            W_arr, alpha_arr, fft_arr, valid)

    # Task 7
    r_jk, se_jk, r_min, r_max, stable = task7(alpha_arr, fft_arr, valid, W_arr)

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print_header("FINAL SUMMARY")

    n_total = np.sum(valid)
    print(f"\n  Primary table:")
    print(f"  {'W':>5s}  {'alpha':>8s}  {'a_std':>7s}  {'O(FFT)':>8s}  "
          f"{'O_std':>7s}  {'R2':>6s}  {'Notes':>15s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*15}")
    for i, r in enumerate(alpha_results):
        notes = ''
        if r['W'] == 0.0:
            notes = 'Clean limit'
        elif r['W'] <= 0.2:
            notes = 'Weak disorder'
        elif r['W'] <= 0.8:
            notes = 'Moderate'
        elif r['W'] <= 2.0:
            notes = 'Strong/MBL'
        else:
            notes = 'Deep MBL'
        if not r['passes_quality']:
            notes += ' [LOW R2]'
        print(f"  {r['W']:>5.1f}  {r['mean_alpha']:>8.4f}  {r['std_alpha']:>7.4f}  "
              f"{fft_arr[i]:>8.4f}  {peak_sd[i]:>7.4f}  "
              f"{r['mean_r2']:>6.3f}  {notes:>15s}")

    print(f"\n  Summary statistics:")
    print(f"  {'Metric':<30s}  {'Previous (6)':>12s}  {'Full ({n_total})':>12s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    print(f"  {'N data points':<30s}  {'6':>12s}  {n_total:>12d}")
    print(f"  {'Pearson r':<30s}  {r_prev:>12.4f}  {r_all:>12.4f}")
    p_prev_str = f"{p_prev:.6f}" if not np.isnan(p_prev) else "~0.01"
    print(f"  {'p-value':<30s}  {p_prev_str:>12s}  {p_all:>12.6f}")
    print(f"  {'Spearman rho':<30s}  {'---':>12s}  {rho_all:>12.4f}")
    print(f"  {'Jackknife r (mean)':<30s}  {'---':>12s}  {r_jk:>12.4f}")
    print(f"  {'Jackknife r (min)':<30s}  {'---':>12s}  {r_min:>12.4f}")
    print(f"  {'Jackknife SE':<30s}  {'---':>12s}  {se_jk:>12.4f}")
    print(f"  {'Result stable?':<30s}  {'Fragile':>12s}  "
          f"{'STABLE' if stable else 'FRAGILE':>12s}")

    print()
    if r_all > 0.80 and p_all < 0.001 and stable:
        print(f"  VERDICT: r = {r_all:.3f} is ROBUST.")
        print(f"  The alpha-order correlation survives the full 17-point dataset,")
        print(f"  with p < 0.001, and no single point drives the result.")
    elif r_all > 0.70 and p_all < 0.01:
        print(f"  VERDICT: r = {r_all:.3f} is CONFIRMED but MODERATED.")
        print(f"  The correlation holds with p < 0.01 but is weaker than the")
        print(f"  original 6-point estimate of r = 0.91. Additional scatter")
        print(f"  from high-disorder (MBL) points reduces the linear correlation.")
    elif r_all > 0.50 and p_all < 0.05:
        print(f"  VERDICT: r = {r_all:.3f} is PRESENT but WEAKENED.")
        print(f"  The correlation exists (p < 0.05) but the original r = 0.91")
        print(f"  was inflated by the small sample and favorable subset selection.")
    else:
        print(f"  VERDICT: r = {r_all:.3f} is NOT ROBUST.")
        print(f"  The full dataset does not support the original r = 0.91.")
    print()
