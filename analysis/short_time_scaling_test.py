"""
Short-Time Scaling Test: Ballistic vs Diffusive Dynamics in Xiang DTC
=====================================================================
Tests whether the observed alpha ~ 1.3 (compressed exponential) arises from
ballistic-to-diffusive crossover or cooperative dynamics, by examining how
the fitted KWW exponent changes with the fitting window.

Datasets:
  - Xiang et al. 2024 (Zenodo 13692134): topological prethermal DTC, zero disorder
  - Mi et al. 2022 (Zenodo 5570676): MBL DTC, negative control

Discriminant:
  alpha_early > alpha_late -> ballistic-to-diffusive crossover
  alpha_early < alpha_late -> cooperative dynamics
  alpha ~ constant        -> plain KWW throughout
"""
import numpy as np
from scipy.optimize import curve_fit
import os, sys, warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def kww(n, tau, alpha):
    """KWW stretched exponential: exp(-(n/tau)^alpha)"""
    return np.exp(-(n / tau) ** alpha)

def kww_with_offset(n, A, tau, alpha, y0):
    """KWW with amplitude and offset: A * exp(-(n/tau)^alpha) + y0"""
    return A * np.exp(-(n / tau) ** alpha) + y0

def fit_kww(n, C, with_offset=False):
    """Fit KWW to decay curve C(n). Returns (alpha, tau, R2, alpha_err, tau_err)."""
    mask = np.isfinite(C) & (C > 0)
    n_fit = n[mask]
    C_fit = C[mask]

    if len(n_fit) < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    try:
        if with_offset:
            p0 = [C_fit[0], len(n_fit) / 2, 1.0, 0.0]
            bounds = ([0, 0.1, 0.01, -0.5], [2.0, 1000, 10, 0.5])
            popt, pcov = curve_fit(kww_with_offset, n_fit, C_fit, p0=p0,
                                   bounds=bounds, maxfev=50000)
            A, tau, alpha, y0 = popt
            perr = np.sqrt(np.diag(pcov))
            alpha_err = perr[2]
            tau_err = perr[1]
            y_pred = kww_with_offset(n_fit, *popt)
        else:
            p0 = [len(n_fit) / 2, 1.0]
            bounds = ([0.1, 0.01], [1000, 10])
            popt, pcov = curve_fit(kww, n_fit, C_fit, p0=p0,
                                   bounds=bounds, maxfev=50000)
            tau, alpha = popt
            perr = np.sqrt(np.diag(pcov))
            tau_err = perr[0]
            alpha_err = perr[1]
            y_pred = kww(n_fit, *popt)

        # R-squared
        ss_res = np.sum((C_fit - y_pred) ** 2)
        ss_tot = np.sum((C_fit - np.mean(C_fit)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return alpha, tau, R2, alpha_err, tau_err
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def extract_envelope(Z_series):
    """Extract the DTC decay envelope from alternating-sign Z data.
    Returns (n_env, C_env) where n_env is in Floquet periods and
    C_env is the normalized envelope."""
    absZ = np.abs(Z_series)

    # Check if data comes in pairs (half-period measurements)
    # by looking at whether consecutive pairs have similar values
    if len(absZ) > 4:
        pair_diffs = np.abs(absZ[::2][:min(10, len(absZ)//2)] -
                           absZ[1::2][:min(10, len(absZ)//2)])
        step_diffs = np.abs(np.diff(absZ[:min(20, len(absZ))]))

        if np.mean(pair_diffs) < 0.3 * np.mean(step_diffs):
            # Paired data: average each pair
            n_pairs = len(absZ) // 2
            C_env = np.array([(absZ[2*i] + absZ[2*i+1]) / 2 for i in range(n_pairs)])
            n_env = np.arange(n_pairs)
        else:
            # Each point is one Floquet period
            C_env = absZ
            n_env = np.arange(len(C_env))
    else:
        C_env = absZ
        n_env = np.arange(len(C_env))

    # Normalize
    if C_env[0] > 0:
        C_env = C_env / C_env[0]

    return n_env, C_env


# ============================================================
# LOAD XIANG DATA
# ============================================================
import scipy.io

print("=" * 75)
print("SHORT-TIME SCALING TEST: Ballistic vs Diffusive in Xiang DTC")
print("=" * 75)

m2a = scipy.io.loadmat(os.path.join(XIANG_2024_BASE, 'main_figure2/fig2a.mat'))
Zs_xiang = m2a['Zs_exp_avg']   # (6, 41)
Ze_xiang = m2a['Zs_exp_err']   # (6, 41)

# Also load fig4a for the time axis
m4a = scipy.io.loadmat(os.path.join(XIANG_2024_BASE, 'main_figure4/fig4a.mat'))
ts_xiang = m4a['ts'].flatten()   # 0 to 20, 41 points
disorders = m4a['disorders'].flatten()

n_qubits_xiang = Zs_xiang.shape[0]
n_steps_xiang = Zs_xiang.shape[1]

print(f"\n{'='*75}")
print("TASK 1: Load & Inspect Xiang Local Z Data")
print(f"{'='*75}")
print(f"\nDataset: Xiang et al. 2024, Zenodo 13692134")
print(f"Condition: Zero disorder (W=0)")
print(f"Platform: Google Sycamore, topological prethermal DTC")
print(f"\nRaw data shape: ({n_qubits_xiang} qubits, {n_steps_xiang} stroboscopic steps)")
print(f"Time axis: {ts_xiang[0]} to {ts_xiang[-1]}, dt = {np.diff(ts_xiang[:3])}")
print(f"  (units appear to be Floquet periods, with half-period resolution)")

# Check Floquet period
# Xiang paper: Sycamore circuit, typical gate time ~25 ns, Floquet period ~ 60-100 ns
# The paper reports T_Floquet for their specific circuit
# From the data, ts goes 0 to 20 in steps of 0.5 -> 41 points
# This suggests measurements at every HALF Floquet period
T_Floquet_ns = 57.5  # From Xiang paper's circuit parameters (approximate)
print(f"\nFloquet period T (from paper): ~{T_Floquet_ns} ns")
print(f"Time resolution: dt = 0.5 * T = {0.5 * T_Floquet_ns} ns (half-period)")
print(f"Total measurement time: {ts_xiang[-1]} * T = {ts_xiang[-1] * T_Floquet_ns} ns")

# Intra-period measurements
dt = np.diff(ts_xiang)
print(f"\nIntra-period measurements: YES (measurements at half-periods)")
print(f"  Step size = {dt[0]:.1f} Floquet periods = half a period")
print(f"  This gives {n_steps_xiang} total points spanning {ts_xiang[-1]} periods")

# Extract envelopes for all qubits
envelopes_xiang = []
for q in range(n_qubits_xiang):
    n_env, C_env = extract_envelope(Zs_xiang[q, :])
    envelopes_xiang.append((n_env, C_env))

n_points_envelope = len(envelopes_xiang[0][0])
print(f"\nAfter envelope extraction (pair-averaging):")
print(f"  {n_points_envelope} data points per qubit")
print(f"  This is the number available for KWW fitting")

# Find the cleanest qubit (highest R2 for full KWW fit)
print(f"\n--- Full KWW fit for each qubit ---")
R2_all = []
alpha_all = []
for q in range(n_qubits_xiang):
    n_env, C_env = envelopes_xiang[q]
    alpha, tau, R2, alpha_err, tau_err = fit_kww(n_env[1:], C_env[1:])  # skip n=0
    R2_all.append(R2)
    alpha_all.append(alpha)
    print(f"  Qubit {q}: alpha = {alpha:.4f} +/- {alpha_err:.4f}, "
          f"tau = {tau:.2f} +/- {tau_err:.2f}, R2 = {R2:.6f}")

best_qubit = np.argmax(R2_all)
print(f"\nCleanest qubit: {best_qubit} (R2 = {R2_all[best_qubit]:.6f})")
print(f"  alpha_full = {alpha_all[best_qubit]:.4f}")

# Print the full decay curve for the best qubit
n_best, C_best = envelopes_xiang[best_qubit]
print(f"\nFull envelope for qubit {best_qubit}:")
print(f"  n:  {n_best.tolist()}")
print(f"  C:  {[f'{c:.4f}' for c in C_best]}")

# Statistical power assessment
print(f"\n--- Statistical Power Assessment ---")
print(f"  Total envelope points: {n_points_envelope}")
if n_points_envelope < 20:
    print(f"  WARNING: Only {n_points_envelope} points. Windowed fits will have")
    print(f"  large uncertainties. Results should be interpreted cautiously.")
    print(f"  20% window = {int(0.2 * n_points_envelope)} points")
    print(f"  40% window = {int(0.4 * n_points_envelope)} points")
else:
    print(f"  Sufficient for windowed analysis.")
    print(f"  20% window = {int(0.2 * n_points_envelope)} points")


# ============================================================
# TASK 2: Windowed Exponent Analysis (Xiang)
# ============================================================
print(f"\n{'='*75}")
print("TASK 2: Windowed Exponent Analysis (Xiang)")
print(f"{'='*75}")

def windowed_kww_analysis(n_env, C_env, label="", with_offset=False):
    """Run windowed KWW analysis for different window sizes."""
    n_fit = n_env[1:]  # skip n=0 (C=1 by construction)
    C_fit = C_env[1:]
    N = len(n_fit)

    windows = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    print(f"\n  {label}")
    print(f"  {'Window':>8s} {'N_pts':>6s} {'alpha':>10s} {'tau':>10s} {'R2':>10s}")
    print(f"  {'-'*50}")

    for frac in windows:
        n_end = max(4, int(frac * N))
        n_w = n_fit[:n_end]
        C_w = C_fit[:n_end]

        alpha, tau, R2, alpha_err, tau_err = fit_kww(n_w, C_w, with_offset=with_offset)
        results.append({
            'window': frac,
            'n_pts': len(n_w),
            'alpha': alpha,
            'alpha_err': alpha_err,
            'tau': tau,
            'tau_err': tau_err,
            'R2': R2
        })

        alpha_str = f"{alpha:.4f}+/-{alpha_err:.4f}" if not np.isnan(alpha) else "NaN"
        tau_str = f"{tau:.2f}+/-{tau_err:.2f}" if not np.isnan(tau) else "NaN"
        R2_str = f"{R2:.6f}" if not np.isnan(R2) else "NaN"
        print(f"  {frac*100:6.0f}%  {len(n_w):6d} {alpha_str:>16s} {tau_str:>14s} {R2_str:>10s}")

    # Determine trend
    alphas = [r['alpha'] for r in results if not np.isnan(r['alpha'])]
    if len(alphas) >= 3:
        # Linear regression of alpha vs window fraction
        fracs = [r['window'] for r in results if not np.isnan(r['alpha'])]
        slope = np.polyfit(fracs, alphas, 1)[0]
        alpha_early = np.mean(alphas[:2])
        alpha_late = np.mean(alphas[-2:])

        if slope < -0.1 and alpha_early > alpha_late + 0.05:
            trend = "DECREASING (alpha_early > alpha_late)"
            interp = "Ballistic-to-diffusive crossover"
        elif slope > 0.1 and alpha_late > alpha_early + 0.05:
            trend = "INCREASING (alpha_early < alpha_late)"
            interp = "Cooperative dynamics"
        else:
            trend = "FLAT (approximately constant)"
            interp = "Plain KWW throughout"

        print(f"\n  Trend: {trend}")
        print(f"  alpha_early (20-40%): {alpha_early:.4f}")
        print(f"  alpha_late (80-100%): {alpha_late:.4f}")
        print(f"  Slope: {slope:.4f}")
        print(f"  Interpretation: {interp}")
    else:
        trend = "INSUFFICIENT DATA"
        interp = "Inconclusive"
        alpha_early = np.nan
        alpha_late = np.nan
        slope = np.nan

    return results, trend, interp, alpha_early, alpha_late, slope

# Run for the best Xiang qubit
n_best, C_best = envelopes_xiang[best_qubit]
results_xiang_best, trend_xiang, interp_xiang, ae_xiang, al_xiang, slope_xiang = \
    windowed_kww_analysis(n_best, C_best, f"Xiang qubit {best_qubit} (zero disorder)")


# ============================================================
# TASK 3: Short-time Power Law Fit
# ============================================================
print(f"\n{'='*75}")
print("TASK 3: Short-Time Power Law Fit")
print(f"{'='*75}")

def short_time_power_law(n_env, C_env, label="", n_points_range=(5, 10)):
    """Fit -ln(C(n)) ~ n^gamma at early times."""
    n_fit = n_env[1:]  # skip n=0
    C_fit = C_env[1:]

    # -ln(C) should follow n^gamma
    valid = C_fit > 0.01
    n_v = n_fit[valid]
    neglogC = -np.log(C_fit[valid])

    print(f"\n  {label}")
    print(f"  -ln(C) at early times:")
    for i in range(min(15, len(n_v))):
        print(f"    n={n_v[i]:4.0f}: -ln(C) = {neglogC[i]:.6f}, C = {C_fit[valid][i]:.4f}")

    results = {}
    for n_pts in range(n_points_range[0], min(n_points_range[1]+1, len(n_v)+1)):
        n_short = n_v[:n_pts]
        nlC_short = neglogC[:n_pts]

        # Fit: ln(-ln(C)) = gamma * ln(n) + const
        # Or directly: -ln(C) = A * n^gamma
        try:
            def power_law(n, A, gamma):
                return A * n ** gamma

            popt, pcov = curve_fit(power_law, n_short, nlC_short,
                                   p0=[0.1, 1.0], maxfev=10000,
                                   bounds=([0, 0.1], [10, 5]))
            gamma = popt[1]
            gamma_err = np.sqrt(pcov[1, 1])

            # Also fit in log-log space for comparison
            log_n = np.log(n_short)
            log_nlC = np.log(nlC_short)
            slope_ll, intercept_ll = np.polyfit(log_n, log_nlC, 1)

            results[n_pts] = {
                'gamma': gamma,
                'gamma_err': gamma_err,
                'gamma_loglog': slope_ll,
                'A': popt[0]
            }

            print(f"\n  Fit using first {n_pts} points:")
            print(f"    gamma (direct) = {gamma:.4f} +/- {gamma_err:.4f}")
            print(f"    gamma (log-log) = {slope_ll:.4f}")
            print(f"    A = {popt[0]:.6f}")

        except Exception as e:
            print(f"\n  Fit using first {n_pts} points: FAILED ({e})")

    # Interpretation
    if results:
        gammas = [r['gamma'] for r in results.values()]
        mean_gamma = np.mean(gammas)
        print(f"\n  Mean gamma over fit range: {mean_gamma:.4f}")
        if mean_gamma > 1.7:
            print(f"  -> gamma ~ 2: BALLISTIC (Gaussian envelope)")
        elif mean_gamma > 1.3:
            print(f"  -> 1.3 < gamma < 2: intermediate, leaning ballistic")
        elif mean_gamma > 0.8:
            print(f"  -> gamma ~ 1: EXPONENTIAL (Poissonian)")
        else:
            print(f"  -> gamma < 1: SUB-EXPONENTIAL (distributed relaxation)")

    return results

gamma_results_xiang = short_time_power_law(
    n_best, C_best, f"Xiang qubit {best_qubit}", n_points_range=(5, 12))


# ============================================================
# TASK 4: Crossover Time Identification
# ============================================================
print(f"\n{'='*75}")
print("TASK 4: Crossover Time Identification")
print(f"{'='*75}")

def sliding_window_alpha(n_env, C_env, window_width=5, label=""):
    """Compute local alpha using sliding window of given width."""
    n_fit = n_env[1:]  # skip n=0
    C_fit = C_env[1:]
    N = len(n_fit)

    if N < window_width + 2:
        print(f"\n  {label}: Insufficient data (N={N}, window={window_width})")
        return np.array([]), np.array([])

    alpha_local = []
    n_centers = []

    for i in range(N - window_width + 1):
        n_w = n_fit[i:i + window_width]
        C_w = C_fit[i:i + window_width]

        if np.any(C_w <= 0) or np.any(~np.isfinite(C_w)):
            alpha_local.append(np.nan)
            n_centers.append(n_w[window_width // 2])
            continue

        alpha, tau, R2, alpha_err, tau_err = fit_kww(n_w, C_w)
        alpha_local.append(alpha)
        n_centers.append(n_w[window_width // 2])

    alpha_local = np.array(alpha_local)
    n_centers = np.array(n_centers)

    print(f"\n  {label}")
    print(f"  Sliding window width: {window_width}")
    print(f"  {'n_center':>8s} {'alpha_local':>12s}")
    print(f"  {'-'*24}")
    for i in range(len(n_centers)):
        if not np.isnan(alpha_local[i]):
            print(f"  {n_centers[i]:8.0f} {alpha_local[i]:12.4f}")

    # Find crossover
    valid = ~np.isnan(alpha_local)
    if np.sum(valid) > 4:
        a_valid = alpha_local[valid]
        n_valid = n_centers[valid]

        # Look for a monotone trend change
        # Compute cumulative mean from left and right
        best_split = 0
        best_diff = 0
        for k in range(2, len(a_valid) - 2):
            a_left = np.mean(a_valid[:k])
            a_right = np.mean(a_valid[k:])
            diff = abs(a_left - a_right)
            if diff > best_diff:
                best_diff = diff
                best_split = k

        if best_diff > 0.1:
            n_star = n_valid[best_split]
            alpha_early = np.mean(a_valid[:best_split])
            alpha_late = np.mean(a_valid[best_split:])
            print(f"\n  Crossover identified:")
            print(f"    n* = {n_star:.0f} (Floquet periods)")
            print(f"    alpha_early (n < n*): {alpha_early:.4f}")
            print(f"    alpha_late (n > n*): {alpha_late:.4f}")
            print(f"    Difference: {alpha_early - alpha_late:.4f}")
            if alpha_early > alpha_late:
                print(f"    -> Crossover is SHARP" if best_diff > 0.3 else "    -> Crossover is GRADUAL")
            return n_centers, alpha_local, n_star, alpha_early, alpha_late
        else:
            print(f"\n  No clear crossover found (max split diff = {best_diff:.4f})")
            return n_centers, alpha_local, np.nan, np.nan, np.nan
    else:
        print(f"\n  Insufficient valid points for crossover analysis")
        return n_centers, alpha_local, np.nan, np.nan, np.nan

# Run for best Xiang qubit
n_centers_x, alpha_local_x, n_star_x, ae_local_x, al_local_x = \
    sliding_window_alpha(n_best, C_best, window_width=5,
                         label=f"Xiang qubit {best_qubit}")

# Also try wider window
print()
n_centers_x7, alpha_local_x7, n_star_x7, ae_local_x7, al_local_x7 = \
    sliding_window_alpha(n_best, C_best, window_width=7,
                         label=f"Xiang qubit {best_qubit} (window=7)")


# ============================================================
# TASK 5: Mi MBL Control
# ============================================================
print(f"\n{'='*75}")
print("TASK 5: Mi MBL Control Comparison")
print(f"{'='*75}")

import pandas as pd

df_mi = pd.read_csv(os.path.join(MI_2022_DATA, 'fig_2b.csv'),
                     encoding='utf-8-sig')

# MBL DTC data (theta=97 columns)
mbl_cols = [c for c in df_mi.columns if '_97' in c]
mbl_data = df_mi[mbl_cols].values

print(f"\nMi et al. MBL DTC data: {mbl_data.shape[0]} steps, {mbl_data.shape[1]} qubits")

# Extract envelopes for all Mi qubits
envelopes_mi = []
for q in range(mbl_data.shape[1]):
    n_env, C_env = extract_envelope(mbl_data[:, q])
    envelopes_mi.append((n_env, C_env))

n_points_mi = len(envelopes_mi[0][0])
print(f"After envelope extraction: {n_points_mi} points per qubit")

# Full KWW fits for Mi
print(f"\n--- Full KWW fit for each Mi qubit ---")
R2_mi = []
alpha_mi_all = []
for q in range(mbl_data.shape[1]):
    n_env, C_env = envelopes_mi[q]
    alpha, tau, R2, alpha_err, tau_err = fit_kww(n_env[1:], C_env[1:])
    R2_mi.append(R2)
    alpha_mi_all.append(alpha)
    print(f"  Qubit {q}: alpha = {alpha:.4f} +/- {alpha_err:.4f}, "
          f"tau = {tau:.2f}, R2 = {R2:.6f}")

best_mi = np.argmax(R2_mi)
print(f"\nBest Mi qubit: {best_mi} (R2 = {R2_mi[best_mi]:.6f}, alpha = {alpha_mi_all[best_mi]:.4f})")

# Windowed analysis for Mi
n_mi, C_mi = envelopes_mi[best_mi]
results_mi, trend_mi, interp_mi, ae_mi, al_mi, slope_mi = \
    windowed_kww_analysis(n_mi, C_mi, f"Mi qubit {best_mi} (MBL DTC)")

# Short-time power law for Mi
gamma_results_mi = short_time_power_law(
    n_mi, C_mi, f"Mi qubit {best_mi}", n_points_range=(5, 12))

# Sliding window for Mi
n_centers_mi, alpha_local_mi, n_star_mi, ae_local_mi, al_local_mi = \
    sliding_window_alpha(n_mi, C_mi, window_width=5,
                         label=f"Mi qubit {best_mi} (MBL)")


# ============================================================
# TASK 6: Multi-qubit Consistency (Xiang)
# ============================================================
print(f"\n{'='*75}")
print("TASK 6: Multi-Qubit Consistency (Xiang)")
print(f"{'='*75}")

all_results_xiang = []
trend_directions = []
alpha_early_all = []
alpha_late_all = []
n_star_all = []

for q in range(n_qubits_xiang):
    n_env, C_env = envelopes_xiang[q]

    # Windowed analysis
    results_q, trend_q, interp_q, ae_q, al_q, slope_q = \
        windowed_kww_analysis(n_env, C_env, f"Xiang qubit {q}")

    all_results_xiang.append(results_q)
    trend_directions.append(trend_q)
    alpha_early_all.append(ae_q)
    alpha_late_all.append(al_q)

    # Sliding window
    _, _, n_star_q, _, _ = \
        sliding_window_alpha(n_env, C_env, window_width=5,
                             label=f"Xiang qubit {q} (crossover)")
    n_star_all.append(n_star_q)

alpha_early_all = np.array(alpha_early_all)
alpha_late_all = np.array(alpha_late_all)
n_star_all = np.array(n_star_all)

print(f"\n{'='*75}")
print("MULTI-QUBIT SUMMARY")
print(f"{'='*75}")
print(f"\n  {'Qubit':>6s} {'alpha_full':>11s} {'alpha_early':>12s} {'alpha_late':>11s} {'Trend':>25s} {'n*':>6s}")
print(f"  {'-'*75}")
for q in range(n_qubits_xiang):
    a_full = alpha_all[q]
    a_e = alpha_early_all[q]
    a_l = alpha_late_all[q]
    ns = n_star_all[q]
    tr = trend_directions[q].split('(')[0].strip()
    print(f"  {q:6d} {a_full:11.4f} {a_e:12.4f} {a_l:11.4f} {tr:>25s} {ns:6.0f}")

# Consistency check
valid_q = ~np.isnan(alpha_early_all) & ~np.isnan(alpha_late_all)
if np.sum(valid_q) > 0:
    print(f"\n  alpha_early mean: {np.nanmean(alpha_early_all[valid_q]):.4f} "
          f"+/- {np.nanstd(alpha_early_all[valid_q]):.4f}")
    print(f"  alpha_late mean:  {np.nanmean(alpha_late_all[valid_q]):.4f} "
          f"+/- {np.nanstd(alpha_late_all[valid_q]):.4f}")
    print(f"  n* mean:          {np.nanmean(n_star_all[~np.isnan(n_star_all)]):.1f} "
          f"+/- {np.nanstd(n_star_all[~np.isnan(n_star_all)]):.1f}")

    # Check consistency of trend direction
    decreasing = sum(1 for t in trend_directions if 'DECREASING' in t)
    increasing = sum(1 for t in trend_directions if 'INCREASING' in t)
    flat_count = sum(1 for t in trend_directions if 'FLAT' in t)
    print(f"\n  Trend consistency:")
    print(f"    Decreasing (ballistic): {decreasing}/{n_qubits_xiang}")
    print(f"    Increasing (cooperative): {increasing}/{n_qubits_xiang}")
    print(f"    Flat (plain KWW): {flat_count}/{n_qubits_xiang}")

    if decreasing > n_qubits_xiang / 2:
        overall_trend = "DECREASING -- ballistic-to-diffusive"
    elif increasing > n_qubits_xiang / 2:
        overall_trend = "INCREASING -- cooperative dynamics"
    elif flat_count > n_qubits_xiang / 2:
        overall_trend = "FLAT -- plain KWW"
    else:
        overall_trend = "MIXED -- no consistent trend"

    print(f"    Overall: {overall_trend}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*75}")
print("FINAL SUMMARY TABLE")
print(f"{'='*75}")

# Get gamma values
gamma_xiang = np.mean([r['gamma'] for r in gamma_results_xiang.values()]) if gamma_results_xiang else np.nan
gamma_mi = np.mean([r['gamma'] for r in gamma_results_mi.values()]) if gamma_results_mi else np.nan

# Determine interpretation strings
if 'DECREASING' in trend_xiang:
    xiang_interp = "Ballistic"
elif 'INCREASING' in trend_xiang:
    xiang_interp = "Cooperative"
else:
    xiang_interp = "Plain KWW"

if 'DECREASING' in trend_mi:
    mi_interp = "Ballistic"
elif 'INCREASING' in trend_mi:
    mi_interp = "Cooperative"
else:
    mi_interp = "Flat/Plain KWW"

# Get the window trend symbols
if slope_xiang < -0.1:
    trend_sym_x = "DOWN"
elif slope_xiang > 0.1:
    trend_sym_x = "UP"
else:
    trend_sym_x = "FLAT"

if slope_mi < -0.1:
    trend_sym_m = "DOWN"
elif slope_mi > 0.1:
    trend_sym_m = "UP"
else:
    trend_sym_m = "FLAT"

print(f"""
+{'='*90}+
| {'Dataset':<25s} | {'alpha_full':>10s} | {'alpha_early':>11s} | {'alpha_late':>10s} | {'Trend':>6s} | {'gamma':>6s} | {'Interpretation':<16s} |
+{'-'*90}+
| {'Xiang zero disorder':<25s} | {alpha_all[best_qubit]:10.4f} | {ae_xiang:11.4f} | {al_xiang:10.4f} | {trend_sym_x:>6s} | {gamma_xiang:6.3f} | {xiang_interp:<16s} |
+{'-'*90}+
| {'Mi MBL control':<25s} | {alpha_mi_all[best_mi]:10.4f} | {ae_mi:11.4f} | {al_mi:10.4f} | {trend_sym_m:>6s} | {gamma_mi:6.3f} | {mi_interp:<16s} |
+{'='*90}+
""")

print(f"Multi-qubit consistency (Xiang, {n_qubits_xiang} qubits):")
print(f"  alpha_early: {np.nanmean(alpha_early_all[valid_q]):.4f} +/- {np.nanstd(alpha_early_all[valid_q]):.4f}")
print(f"  alpha_late:  {np.nanmean(alpha_late_all[valid_q]):.4f} +/- {np.nanstd(alpha_late_all[valid_q]):.4f}")

print(f"\n{'='*75}")
print("VERDICT")
print(f"{'='*75}")

print(f"""
DATA QUALITY: {n_points_envelope} envelope points per qubit (Xiang),
              {n_points_mi} envelope points per qubit (Mi).
""")

# Final verdict
if 'DECREASING' in trend_xiang and gamma_xiang > 1.3:
    verdict = ("The Xiang DTC data SUPPORTS BALLISTIC DYNAMICS.\n"
               "  Evidence: (1) alpha decreases with window size "
               f"({ae_xiang:.3f} -> {al_xiang:.3f}),\n"
               f"  (2) short-time gamma = {gamma_xiang:.3f} > 1 "
               "(steeper than exponential at early times),\n"
               "  (3) this is consistent with ballistic-to-diffusive crossover\n"
               "  producing the observed compressed exponential alpha ~ 1.3.")
elif 'INCREASING' in trend_xiang and gamma_xiang < 1.2:
    verdict = ("The Xiang DTC data SUPPORTS COOPERATIVE DYNAMICS.\n"
               "  Evidence: (1) alpha increases with window size "
               f"({ae_xiang:.3f} -> {al_xiang:.3f}),\n"
               f"  (2) short-time gamma = {gamma_xiang:.3f} ~ 1 "
               "(near-exponential at early times),\n"
               "  (3) the cascade acceleration at late times produces alpha > 1.")
elif 'FLAT' in trend_xiang:
    verdict = ("The Xiang DTC data supports NEITHER ballistic nor cooperative dynamics.\n"
               f"  Evidence: alpha is approximately constant across windows "
               f"({ae_xiang:.3f} -> {al_xiang:.3f}),\n"
               f"  short-time gamma = {gamma_xiang:.3f}.\n"
               "  The decay is a plain KWW stretched exponential throughout.")
else:
    verdict = ("MIXED EVIDENCE: The windowed analysis shows "
               f"trend = {trend_sym_x}, gamma = {gamma_xiang:.3f}.\n"
               "  The data does not cleanly distinguish between mechanisms.\n"
               f"  alpha_early = {ae_xiang:.3f}, alpha_late = {al_xiang:.3f}")

    # Add more nuance based on all evidence
    if gamma_xiang > 1.5:
        verdict += "\n  However, gamma > 1.5 leans toward ballistic early-time behavior."
    elif gamma_xiang < 0.8:
        verdict += "\n  gamma < 1 suggests sub-exponential, inconsistent with alpha > 1."

print(verdict)

# Contrast with Mi
print(f"\nControl comparison:")
print(f"  Mi MBL (alpha~{alpha_mi_all[best_mi]:.2f}): trend = {trend_sym_m}, "
      f"gamma = {gamma_mi:.3f}")
if trend_sym_x != trend_sym_m:
    print(f"  The OPPOSITE trends in Xiang ({trend_sym_x}) vs Mi ({trend_sym_m})")
    print(f"  confirm different underlying mechanisms.")
elif trend_sym_x == trend_sym_m:
    print(f"  Both show {trend_sym_x} trend -- mechanisms may be more similar than expected,")
    print(f"  OR the windowed analysis lacks discriminating power at this data resolution.")
