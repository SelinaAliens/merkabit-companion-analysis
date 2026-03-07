"""
Stress Test: n* = 12 Crossover Period
======================================
Tests whether the cooperative cascade crossover at n* ~ 12 Floquet periods
in the Xiang DTC data is robust, or a methodological/statistical artefact.

All data loaded from scratch. No reuse of previous intermediates.
"""
import numpy as np
from scipy.optimize import curve_fit, minimize
import scipy.io
import pandas as pd
import os, sys, warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# FOUNDATIONS: load data, define helpers
# ============================================================

def kww(n, tau, alpha):
    """exp(-(n/tau)^alpha)"""
    with np.errstate(over='ignore', invalid='ignore'):
        return np.exp(-(n / tau) ** alpha)

def fit_kww_window(n, C):
    """Fit KWW to a short window. Returns (alpha, alpha_err, R2)."""
    mask = np.isfinite(C) & (C > 0.001) & np.isfinite(n) & (n > 0)
    n_f, C_f = n[mask], C[mask]
    if len(n_f) < 3:
        return np.nan, np.nan, np.nan
    try:
        p0 = [max(n_f) * 1.5, 1.0]
        popt, pcov = curve_fit(kww, n_f, C_f, p0=p0,
                               bounds=([0.1, 0.01], [5000, 10]),
                               maxfev=30000)
        tau, alpha = popt
        alpha_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.nan
        y_pred = kww(n_f, *popt)
        ss_res = np.sum((C_f - y_pred) ** 2)
        ss_tot = np.sum((C_f - np.mean(C_f)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return alpha, alpha_err, R2
    except Exception:
        return np.nan, np.nan, np.nan

def extract_envelope(Z_raw):
    """DTC envelope from alternating-sign Z data, pair-averaged."""
    absZ = np.abs(Z_raw)
    # Check for paired structure
    if len(absZ) >= 4:
        pair_d = np.mean(np.abs(absZ[::2][:10] - absZ[1::2][:10]))
        step_d = np.mean(np.abs(np.diff(absZ[:20])))
        if pair_d < 0.3 * step_d:
            n_p = len(absZ) // 2
            C = np.array([(absZ[2*i] + absZ[2*i+1]) / 2 for i in range(n_p)])
            return np.arange(n_p), C / C[0] if C[0] > 0 else C
    return np.arange(len(absZ)), absZ / absZ[0] if absZ[0] > 0 else absZ

def sliding_alpha(n_env, C_env, width=5):
    """Compute alpha_local in a sliding window of given width.
    Returns (n_centers, alpha_local, alpha_err_local)."""
    n_fit = n_env[1:]  # skip n=0 (normalized to 1)
    C_fit = C_env[1:]
    N = len(n_fit)
    centers, alphas, errs = [], [], []
    for i in range(N - width + 1):
        nw = n_fit[i:i+width]
        Cw = C_fit[i:i+width]
        a, ae, r2 = fit_kww_window(nw, Cw)
        centers.append(nw[width // 2])
        alphas.append(a)
        errs.append(ae)
    return np.array(centers), np.array(alphas), np.array(errs)

def find_nstar_sustained(centers, alphas, threshold=1.0, run_length=3):
    """n* = first centre where alpha > threshold for run_length consecutive windows."""
    valid = ~np.isnan(alphas)
    for i in range(len(alphas) - run_length + 1):
        segment = alphas[i:i+run_length]
        if np.all(valid[i:i+run_length]) and np.all(segment > threshold):
            return centers[i]
    return np.nan

def find_nstar_first_exceed(centers, alphas, threshold=1.05):
    """n*_A = first centre where alpha > threshold."""
    for i, a in enumerate(alphas):
        if not np.isnan(a) and a > threshold:
            return centers[i]
    return np.nan

def find_nstar_max_deriv(centers, alphas):
    """n*_B = centre where d(alpha)/dn is maximum."""
    valid = ~np.isnan(alphas)
    if np.sum(valid) < 3:
        return np.nan
    # Smooth lightly to reduce noise in derivative
    a_smooth = np.copy(alphas)
    for i in range(1, len(a_smooth) - 1):
        if valid[i-1] and valid[i] and valid[i+1]:
            a_smooth[i] = (alphas[i-1] + alphas[i] + alphas[i+1]) / 3
    deriv = np.diff(a_smooth)
    dc = (centers[:-1] + centers[1:]) / 2
    valid_d = ~np.isnan(deriv)
    if not np.any(valid_d):
        return np.nan
    idx = np.nanargmax(deriv)
    return dc[idx]


# ---- Load Xiang data from scratch ----
m2a = scipy.io.loadmat(os.path.join(XIANG_2024_BASE, 'main_figure2/fig2a.mat'))
Zs_xiang = m2a['Zs_exp_avg']   # (6, 41)
Ze_xiang = m2a['Zs_exp_err']   # (6, 41)

# ---- Load Mi data from scratch ----
df_mi = pd.read_csv(os.path.join(MI_2022_DATA, 'fig_2b.csv'),
                     encoding='utf-8-sig')
mbl_cols = [c for c in df_mi.columns if '_97' in c]
Mi_mbl = df_mi[mbl_cols].values  # (101, 5)

# ---- Extract all Xiang envelopes ----
xiang_envs = []
for q in range(6):
    ne, ce = extract_envelope(Zs_xiang[q, :])
    xiang_envs.append((ne, ce))

N_env = len(xiang_envs[0][0])

print("=" * 75)
print("STRESS TEST: n* = 12 CROSSOVER PERIOD")
print("=" * 75)
print(f"\nXiang data: 6 qubits, {Zs_xiang.shape[1]} raw steps -> {N_env} envelope points")
print(f"Mi data: {Mi_mbl.shape[1]} qubits, {Mi_mbl.shape[0]} steps")


# ============================================================
# TASK 1: Reproduce the crossover independently
# ============================================================
print(f"\n{'='*75}")
print("TASK 1: Reproduce Crossover Independently")
print(f"{'='*75}")

# Use each qubit individually, then report per-qubit
print("\nSliding window width = 5, all 6 Xiang qubits")
print(f"{'Qubit':>5} | {'n*_sust (>1.0, 3 consec)':>24} | {'n*_A (>1.05)':>14} | {'n*_B (max deriv)':>16}")
print("-" * 70)

nstar_sust_all = []
nstar_A_all = []
nstar_B_all = []

for q in range(6):
    ne, ce = xiang_envs[q]
    centers, alphas, errs = sliding_alpha(ne, ce, width=5)

    ns_sust = find_nstar_sustained(centers, alphas, threshold=1.0, run_length=3)
    ns_A = find_nstar_first_exceed(centers, alphas, threshold=1.05)
    ns_B = find_nstar_max_deriv(centers, alphas)

    nstar_sust_all.append(ns_sust)
    nstar_A_all.append(ns_A)
    nstar_B_all.append(ns_B)

    ns_str = f"{ns_sust:.0f}" if not np.isnan(ns_sust) else "none"
    nA_str = f"{ns_A:.0f}" if not np.isnan(ns_A) else "none"
    nB_str = f"{ns_B:.1f}" if not np.isnan(ns_B) else "none"
    print(f"  Q{q}  | {ns_str:>24} | {nA_str:>14} | {nB_str:>16}")

nstar_sust_all = np.array(nstar_sust_all)
nstar_A_all = np.array(nstar_A_all)
nstar_B_all = np.array(nstar_B_all)

# Also print full alpha_local for each qubit
print("\nFull alpha_local(n) for each qubit (window=5):")
for q in range(6):
    ne, ce = xiang_envs[q]
    centers, alphas, errs = sliding_alpha(ne, ce, width=5)
    print(f"\n  Q{q}: alpha_full_fit = ", end="")
    # Full fit for reference
    a_full, _, _ = fit_kww_window(ne[1:], ce[1:])
    print(f"{a_full:.4f}")
    for i in range(len(centers)):
        marker = " *" if not np.isnan(alphas[i]) and alphas[i] > 1.0 else ""
        a_str = f"{alphas[i]:.4f}" if not np.isnan(alphas[i]) else "  NaN"
        print(f"    n={centers[i]:5.0f}: alpha={a_str}{marker}")

# Summary for Task 1
print(f"\nTask 1 Summary:")
print(f"  Sustained crossover definition (>1.0 for 3 consec):")
valid_sust = ~np.isnan(nstar_sust_all)
print(f"    Qubits with crossover: {np.sum(valid_sust)}/6")
if np.sum(valid_sust) > 0:
    print(f"    Values: {nstar_sust_all[valid_sust]}")
    print(f"    Range: [{np.nanmin(nstar_sust_all):.0f}, {np.nanmax(nstar_sust_all):.0f}]")

print(f"  First exceed 1.05:")
valid_A = ~np.isnan(nstar_A_all)
print(f"    Qubits with crossover: {np.sum(valid_A)}/6")
if np.sum(valid_A) > 0:
    print(f"    Values: {nstar_A_all[valid_A]}")

print(f"  Max derivative:")
valid_B = ~np.isnan(nstar_B_all)
if np.sum(valid_B) > 0:
    print(f"    Values: {[f'{x:.1f}' for x in nstar_B_all[valid_B]]}")

all_in_range = True
for arr in [nstar_sust_all[valid_sust], nstar_A_all[valid_A]]:
    if len(arr) > 0 and (np.nanmin(arr) < 10 or np.nanmax(arr) > 16):
        all_in_range = False
print(f"\n  All definitions in [10, 16]? {all_in_range}")


# ============================================================
# TASK 2: Per-qubit crossover periods
# ============================================================
print(f"\n{'='*75}")
print("TASK 2: Per-Qubit Crossover Periods")
print(f"{'='*75}")

# Already computed above. Focus on statistics.
print(f"\nUsing sustained crossover definition (n*_sust):")
print(f"  Q0: {nstar_sust_all[0]}")
print(f"  Q1: {nstar_sust_all[1]}")
print(f"  Q2: {nstar_sust_all[2]}")
print(f"  Q3: {nstar_sust_all[3]}")
print(f"  Q4: {nstar_sust_all[4]}")
print(f"  Q5: {nstar_sust_all[5]}")

valid_all6 = ~np.isnan(nstar_sust_all)
if np.sum(valid_all6) > 0:
    print(f"\n  All 6 qubits: mean = {np.nanmean(nstar_sust_all[valid_all6]):.1f}, "
          f"SD = {np.nanstd(nstar_sust_all[valid_all6]):.1f}, "
          f"min = {np.nanmin(nstar_sust_all[valid_all6]):.0f}, "
          f"max = {np.nanmax(nstar_sust_all[valid_all6]):.0f}")

# Q0-Q3 only (alpha > 1 qubits)
q03 = nstar_sust_all[:4]
valid_q03 = ~np.isnan(q03)
if np.sum(valid_q03) > 0:
    print(f"\n  Q0-Q3 only (alpha > 1 regime):")
    print(f"    mean = {np.nanmean(q03[valid_q03]):.1f}, "
          f"SD = {np.nanstd(q03[valid_q03]):.1f}, "
          f"min = {np.nanmin(q03[valid_q03]):.0f}, "
          f"max = {np.nanmax(q03[valid_q03]):.0f}")

# Q4 and Q5 behaviour
print(f"\n  Q4 (alpha_full=0.97): n* = {nstar_sust_all[4]} -- ", end="")
if np.isnan(nstar_sust_all[4]):
    print("No sustained crossover above 1.0 (as expected, alpha < 1)")
else:
    print(f"crossover at {nstar_sust_all[4]:.0f}")

print(f"  Q5 (alpha_full=0.93): n* = {nstar_sust_all[5]} -- ", end="")
if np.isnan(nstar_sust_all[5]):
    print("No sustained crossover above 1.0 (as expected, alpha < 1)")
else:
    print(f"crossover at {nstar_sust_all[5]:.0f}")

# Clustering assessment
if np.sum(valid_q03) >= 3:
    spread = np.nanmax(q03[valid_q03]) - np.nanmin(q03[valid_q03])
    print(f"\n  Q0-Q3 spread: {spread:.0f} periods")
    if spread <= 5:
        print(f"  -> TIGHT clustering (spread <= 5): collective property")
    elif spread <= 10:
        print(f"  -> MODERATE clustering: suggestive but not tight")
    else:
        print(f"  -> WIDE scatter: mean may be artefactual")


# ============================================================
# TASK 3: Sensitivity to window width
# ============================================================
print(f"\n{'='*75}")
print("TASK 3: Sensitivity to Window Width")
print(f"{'='*75}")

# Test for the qubits with alpha > 1 (Q0-Q3)
widths = [3, 5, 7, 9]
print(f"\nn* (sustained >1.0, 3 consec) for each window width:")
print(f"{'Width':>5} | {'Q0':>5} {'Q1':>5} {'Q2':>5} {'Q3':>5} {'Q4':>5} {'Q5':>5} | {'Mean Q0-3':>10}")
print("-" * 65)

nstar_by_width = {}
for w in widths:
    nstars_w = []
    for q in range(6):
        ne, ce = xiang_envs[q]
        if len(ne) - 1 < w:
            nstars_w.append(np.nan)
            continue
        centers, alphas, _ = sliding_alpha(ne, ce, width=w)
        ns = find_nstar_sustained(centers, alphas, threshold=1.0, run_length=3)
        nstars_w.append(ns)
    nstars_w = np.array(nstars_w)
    nstar_by_width[w] = nstars_w

    q_strs = [f"{x:5.0f}" if not np.isnan(x) else " none" for x in nstars_w]
    q03_w = nstars_w[:4]
    valid_w = ~np.isnan(q03_w)
    mean_str = f"{np.nanmean(q03_w[valid_w]):.1f}" if np.sum(valid_w) > 0 else "N/A"
    print(f"  {w:3d}   | {' '.join(q_strs)} | {mean_str:>10}")

# Range across widths for Q0-Q3
all_nstars_q03 = []
for w in widths:
    for q in range(4):
        v = nstar_by_width[w][q]
        if not np.isnan(v):
            all_nstars_q03.append(v)

if len(all_nstars_q03) > 0:
    all_nstars_q03 = np.array(all_nstars_q03)
    range_across = np.max(all_nstars_q03) - np.min(all_nstars_q03)
    print(f"\n  All Q0-Q3 n* across all widths: [{np.min(all_nstars_q03):.0f}, {np.max(all_nstars_q03):.0f}]")
    print(f"  Range: {range_across:.0f} periods")
    if range_across <= 4:
        print(f"  -> STABLE (range <= 4): robust to window width")
    elif range_across <= 8:
        print(f"  -> MODERATELY STABLE: some dependence on window width")
    else:
        print(f"  -> UNSTABLE: crossover is poorly localised")
else:
    print("\n  No valid crossovers found across any width")


# ============================================================
# TASK 4: Statistical significance of crossover
# ============================================================
print(f"\n{'='*75}")
print("TASK 4: Statistical Significance of Crossover")
print(f"{'='*75}")

# For each qubit Q0-Q3, fit piecewise model to alpha_local(n)
# and compare to flat model

def piecewise_constant_rising(n, n_star, a1, a2, b):
    """Piecewise: constant a1 for n < n_star, then a2 + b*(n - n_star) for n >= n_star."""
    result = np.where(n < n_star, a1, a2 + b * (n - n_star))
    return result

def fit_piecewise(centers, alphas):
    """Fit piecewise model with n* as free parameter. Returns best_nstar, CI, F-stat, p-value."""
    valid = ~np.isnan(alphas)
    c_v = centers[valid]
    a_v = alphas[valid]
    N = len(a_v)

    if N < 6:
        return np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan

    # Grid search for n*
    best_sse = np.inf
    best_nstar = np.nan
    best_params = None

    for n_try in np.arange(c_v[2], c_v[-3], 0.5):
        mask_left = c_v < n_try
        mask_right = c_v >= n_try

        if np.sum(mask_left) < 2 or np.sum(mask_right) < 2:
            continue

        a1 = np.mean(a_v[mask_left])
        if np.sum(mask_right) >= 2:
            coeffs = np.polyfit(c_v[mask_right] - n_try, a_v[mask_right], 1)
            b, a2 = coeffs[0], coeffs[1]
        else:
            a2 = np.mean(a_v[mask_right])
            b = 0

        pred = piecewise_constant_rising(c_v, n_try, a1, a2, b)
        sse = np.sum((a_v - pred) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_nstar = n_try
            best_params = (a1, a2, b)

    # Flat model (no crossover)
    flat_mean = np.mean(a_v)
    sse_flat = np.sum((a_v - flat_mean) ** 2)

    # F-test: piecewise (4 params: n*, a1, a2, b) vs flat (1 param: mean)
    k_piece = 4
    k_flat = 1
    df1 = k_piece - k_flat  # 3
    df2 = N - k_piece

    if df2 > 0 and best_sse > 0:
        F_stat = ((sse_flat - best_sse) / df1) / (best_sse / df2)
        from scipy.stats import f as f_dist
        p_value = 1 - f_dist.cdf(F_stat, df1, df2)
    else:
        F_stat = np.nan
        p_value = np.nan

    # Bootstrap CI for n*
    n_boot = 2000
    nstar_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(N, N, replace=True)
        c_b = c_v[idx]
        a_b = a_v[idx]
        sort_order = np.argsort(c_b)
        c_b = c_b[sort_order]
        a_b = a_b[sort_order]

        best_sse_b = np.inf
        best_ns_b = np.nan
        for n_try in np.arange(c_b[1], c_b[-2], 1.0):
            ml = c_b < n_try
            mr = c_b >= n_try
            if np.sum(ml) < 2 or np.sum(mr) < 2:
                continue
            a1_b = np.mean(a_b[ml])
            if np.sum(mr) >= 2:
                co = np.polyfit(c_b[mr] - n_try, a_b[mr], 1)
                b_b, a2_b = co[0], co[1]
            else:
                a2_b = np.mean(a_b[mr])
                b_b = 0
            pred_b = piecewise_constant_rising(c_b, n_try, a1_b, a2_b, b_b)
            sse_b = np.sum((a_b - pred_b) ** 2)
            if sse_b < best_sse_b:
                best_sse_b = sse_b
                best_ns_b = n_try
        nstar_boots.append(best_ns_b)

    nstar_boots = np.array(nstar_boots)
    nstar_boots = nstar_boots[~np.isnan(nstar_boots)]
    if len(nstar_boots) > 0:
        ci_lo = np.percentile(nstar_boots, 2.5)
        ci_hi = np.percentile(nstar_boots, 97.5)
    else:
        ci_lo, ci_hi = np.nan, np.nan

    return best_nstar, (ci_lo, ci_hi), F_stat, p_value, best_params, sse_flat

print(f"\nPiecewise model: alpha = a1 (n < n*) ; a2 + b*(n-n*) (n >= n*)")
print(f"vs flat model: alpha = const")
print(f"\n{'Qubit':>5} | {'n*_pw':>6} | {'95% CI':>14} | {'F-stat':>8} | {'p-value':>8} | {'Significant?':>12}")
print("-" * 70)

pw_results = {}
for q in range(6):
    ne, ce = xiang_envs[q]
    centers, alphas, _ = sliding_alpha(ne, ce, width=5)

    ns_pw, ci, F, p, params, sse_f = fit_piecewise(centers, alphas)
    pw_results[q] = {'nstar': ns_pw, 'ci': ci, 'F': F, 'p': p}

    ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if not np.isnan(ci[0]) else "N/A"
    F_str = f"{F:.2f}" if not np.isnan(F) else "N/A"
    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
    sig = "YES" if not np.isnan(p) and p < 0.05 else "NO"
    ns_str = f"{ns_pw:.1f}" if not np.isnan(ns_pw) else "N/A"
    print(f"  Q{q}  | {ns_str:>6} | {ci_str:>14} | {F_str:>8} | {p_str:>8} | {sig:>12}")

# Summary for Q0-Q3
q03_nstar_pw = [pw_results[q]['nstar'] for q in range(4)]
q03_p = [pw_results[q]['p'] for q in range(4)]
q03_ci_lo = [pw_results[q]['ci'][0] for q in range(4)]
q03_ci_hi = [pw_results[q]['ci'][1] for q in range(4)]

print(f"\n  Q0-Q3 summary:")
valid_pw = [not np.isnan(x) for x in q03_nstar_pw]
if any(valid_pw):
    print(f"    n* values: {[f'{x:.1f}' for x, v in zip(q03_nstar_pw, valid_pw) if v]}")
    print(f"    p-values: {[f'{x:.4f}' for x, v in zip(q03_p, valid_pw) if v]}")
    sig_count = sum(1 for p in q03_p if not np.isnan(p) and p < 0.05)
    print(f"    Significant at p<0.05: {sig_count}/4")
    # Combined CI
    all_ci_lo = [x for x in q03_ci_lo if not np.isnan(x)]
    all_ci_hi = [x for x in q03_ci_hi if not np.isnan(x)]
    if all_ci_lo and all_ci_hi:
        print(f"    Broadest CI across Q0-Q3: [{min(all_ci_lo):.1f}, {max(all_ci_hi):.1f}]")


# ============================================================
# TASK 5: Null model
# ============================================================
print(f"\n{'='*75}")
print("TASK 5: Null Model -- What Crossover Does Random Data Give?")
print(f"{'='*75}")

# Estimate noise from Xiang residuals
residual_sigmas = []
for q in range(4):  # Q0-Q3
    ne, ce = xiang_envs[q]
    a_f, _, _ = fit_kww_window(ne[1:], ce[1:])
    if np.isnan(a_f):
        continue
    # Fit full KWW
    try:
        popt, _ = curve_fit(kww, ne[1:], ce[1:], p0=[10, a_f],
                           bounds=([0.1, 0.01], [100, 10]), maxfev=30000)
        resid = ce[1:] - kww(ne[1:], *popt)
        residual_sigmas.append(np.std(resid))
    except:
        pass

noise_sigma = np.mean(residual_sigmas) if residual_sigmas else 0.02
print(f"\nEstimated noise level (from Xiang residuals): sigma = {noise_sigma:.4f}")
print(f"Generating 10,000 synthetic KWW decay curves...")
print(f"  N_points = {N_env}, alpha ~ U[0.8, 1.4], tau ~ U[5, 25]")

n_null = 10000
null_nstars = []

n_synth = np.arange(N_env)

for trial in range(n_null):
    alpha_true = np.random.uniform(0.8, 1.4)
    tau_true = np.random.uniform(5, 25)
    C_true = kww(n_synth, tau_true, alpha_true)
    C_noisy = C_true + np.random.normal(0, noise_sigma, len(n_synth))
    C_noisy = np.clip(C_noisy, 0.001, 2.0)
    C_noisy[0] = 1.0  # normalize

    centers_s, alphas_s, _ = sliding_alpha(n_synth, C_noisy, width=5)
    ns_s = find_nstar_sustained(centers_s, alphas_s, threshold=1.0, run_length=3)
    null_nstars.append(ns_s)

null_nstars = np.array(null_nstars)
null_valid = ~np.isnan(null_nstars)
null_found = null_nstars[null_valid]

print(f"\nNull model results:")
print(f"  Trials with ANY sustained crossover: {np.sum(null_valid)}/{n_null} "
      f"({100*np.sum(null_valid)/n_null:.1f}%)")

if len(null_found) > 0:
    print(f"  Mean n* (when found): {np.mean(null_found):.1f} +/- {np.std(null_found):.1f}")
    print(f"  Median n*: {np.median(null_found):.1f}")
    print(f"  Distribution:")
    for edge in range(3, 18):
        count = np.sum((null_found >= edge) & (null_found < edge + 1))
        pct = 100 * count / len(null_found) if len(null_found) > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    n*=[{edge:2d},{edge+1:2d}): {count:5d} ({pct:5.1f}%) {bar}")

    # Where does Xiang n*=12 fall in the null distribution?
    # Use the mean Q0-Q3 n* from the sustained definition
    xiang_nstar_mean = np.nanmean(nstar_sust_all[:4][~np.isnan(nstar_sust_all[:4])])
    if not np.isnan(xiang_nstar_mean):
        percentile = 100 * np.sum(null_found <= xiang_nstar_mean) / len(null_found)
        print(f"\n  Xiang mean n* (Q0-Q3) = {xiang_nstar_mean:.1f}")
        print(f"  Percentile in null: {percentile:.1f}%")
        if percentile < 5:
            print(f"  -> BELOW 5th percentile: crossover at n*={xiang_nstar_mean:.0f} is UNUSUAL in random data")
        elif percentile < 25:
            print(f"  -> Below 25th percentile: somewhat unusual")
        elif percentile < 75:
            print(f"  -> Near median: NOT distinguishable from random")
        else:
            print(f"  -> Above 75th percentile: later than typical")
    else:
        print(f"\n  No valid Xiang n* for Q0-Q3 to compare")
else:
    print(f"  No crossovers found in null model (all alpha < 1 or no sustained run)")
    print(f"  This means ANY sustained crossover is unusual -> Xiang result is meaningful")


# ============================================================
# TASK 6: Mi MBL Control
# ============================================================
print(f"\n{'='*75}")
print("TASK 6: Mi MBL Control")
print(f"{'='*75}")

mi_envs = []
for q in range(Mi_mbl.shape[1]):
    ne, ce = extract_envelope(Mi_mbl[:, q])
    mi_envs.append((ne, ce))

print(f"\nMi MBL: {Mi_mbl.shape[1]} qubits, {len(mi_envs[0][0])} envelope points each")

print(f"\nAlpha_local(n) for each Mi qubit (window=5, first 30 centres):")
mi_ever_exceeds = {}

for q in range(Mi_mbl.shape[1]):
    ne, ce = mi_envs[q]
    centers, alphas, _ = sliding_alpha(ne, ce, width=5)

    a_full, _, _ = fit_kww_window(ne[1:], ce[1:])
    print(f"\n  Mi Q{q} (alpha_full = {a_full:.4f}):")

    # Check if alpha_local ever exceeds 1.0
    valid_a = ~np.isnan(alphas)
    exceeds = alphas[valid_a] > 1.0
    n_exceed = np.sum(exceeds)
    total_valid = np.sum(valid_a)

    # Find longest consecutive run above 1.0
    max_run = 0
    current_run = 0
    first_exceed_n = np.nan
    for i in range(len(alphas)):
        if not np.isnan(alphas[i]) and alphas[i] > 1.0:
            current_run += 1
            if current_run == 1:
                first_exceed_n = centers[i]
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    mi_ever_exceeds[q] = {
        'n_exceed': n_exceed,
        'total': total_valid,
        'max_run': max_run,
        'first_n': first_exceed_n
    }

    print(f"    Windows where alpha > 1.0: {n_exceed}/{total_valid}")
    print(f"    Longest consecutive run > 1.0: {max_run}")
    if max_run >= 3:
        ns_mi = find_nstar_sustained(centers, alphas, threshold=1.0, run_length=3)
        print(f"    Sustained crossover n*: {ns_mi:.0f}")
    else:
        print(f"    No sustained crossover (run < 3)")

    # Print a sample of alpha_local
    step = max(1, len(centers) // 20)
    print(f"    Sample alpha_local (every {step} windows):")
    for i in range(0, len(centers), step):
        a_str = f"{alphas[i]:.4f}" if not np.isnan(alphas[i]) else "NaN"
        marker = " *" if not np.isnan(alphas[i]) and alphas[i] > 1.0 else ""
        print(f"      n={centers[i]:5.0f}: {a_str}{marker}")

# Summary
print(f"\n  Mi MBL Summary:")
any_sustained = False
for q in range(Mi_mbl.shape[1]):
    if mi_ever_exceeds[q]['max_run'] >= 3:
        any_sustained = True
        print(f"    Q{q}: sustained crossover found (max_run={mi_ever_exceeds[q]['max_run']})")

if not any_sustained:
    print(f"    No qubit shows sustained alpha > 1.0 (3 consecutive windows)")
    print(f"    -> CLEAN CONTRAST with Xiang: cooperative cascade absent in MBL")
else:
    print(f"    Some Mi qubits show sustained alpha > 1.0")
    print(f"    -> Weakens the Xiang result if the crossover reflects fitting artefact")


# ============================================================
# TASK 7: What precision is justified?
# ============================================================
print(f"\n{'='*75}")
print("TASK 7: The h(E6) = 12 Claim -- What Precision is Justified?")
print(f"{'='*75}")

# Collect all evidence
print(f"\n--- Evidence Summary ---")

# T1: Definition robustness
q03_sust = nstar_sust_all[:4][~np.isnan(nstar_sust_all[:4])]
q03_A = nstar_A_all[:4][~np.isnan(nstar_A_all[:4])]
q03_B = nstar_B_all[:4][~np.isnan(nstar_B_all[:4])]

if len(q03_sust) > 0:
    t1_result = f"n*_sust={np.mean(q03_sust):.1f}, n*_A={np.mean(q03_A):.1f}" if len(q03_A) > 0 else f"n*_sust={np.mean(q03_sust):.1f}"
    t1_range_sust = (np.min(q03_sust), np.max(q03_sust))
    t1_support = 10 <= np.mean(q03_sust) <= 16
else:
    t1_result = "No crossover found"
    t1_support = False

# T2: Per-qubit consistency
if len(q03_sust) >= 2:
    t2_mean = np.mean(q03_sust)
    t2_sd = np.std(q03_sust)
    t2_support = t2_sd <= 3
else:
    t2_mean = np.nan
    t2_sd = np.nan
    t2_support = False

# T3: Window width stability
if len(all_nstars_q03) > 0:
    t3_range = (np.min(all_nstars_q03), np.max(all_nstars_q03))
    t3_support = (t3_range[1] - t3_range[0]) <= 4
else:
    t3_range = (np.nan, np.nan)
    t3_support = False

# T4: Statistical significance
sig_qubits = sum(1 for q in range(4) if not np.isnan(pw_results[q]['p']) and pw_results[q]['p'] < 0.05)
t4_any_sig = sig_qubits > 0
ci_ranges = []
for q in range(4):
    lo, hi = pw_results[q]['ci']
    if not np.isnan(lo) and not np.isnan(hi):
        ci_ranges.append((lo, hi))
if ci_ranges:
    broadest = (min(c[0] for c in ci_ranges), max(c[1] for c in ci_ranges))
    t4_ci_contains_12 = broadest[0] <= 12 <= broadest[1]
else:
    broadest = (np.nan, np.nan)
    t4_ci_contains_12 = False

# T5: Null model
if len(null_found) > 0 and not np.isnan(xiang_nstar_mean):
    t5_percentile = 100 * np.sum(null_found <= xiang_nstar_mean) / len(null_found)
    t5_support = t5_percentile < 25  # somewhat unusual
else:
    t5_percentile = np.nan
    t5_support = False

# T6: Mi control
t6_clean = not any_sustained

# ---- Print verdict table ----
print(f"\n{'='*75}")
print("VERDICT TABLE")
print(f"{'='*75}")
print(f"\n{'Test':<35} | {'Result':<25} | {'Supports n*=12?':>15} | {'Confidence':>10}")
print("-" * 95)

def yn(b): return "Yes" if b else "No"
def conf(b, strong=False):
    if not b: return "Low"
    return "High" if strong else "Medium"

print(f"{'T1: Definition robustness':<35} | {t1_result:<25} | {yn(t1_support):>15} | {conf(t1_support):>10}")

t2_str = f"mean={t2_mean:.1f}, SD={t2_sd:.1f}" if not np.isnan(t2_mean) else "N/A"
print(f"{'T2: Per-qubit consistency':<35} | {t2_str:<25} | {yn(t2_support):>15} | {conf(t2_support):>10}")

t3_str = f"range [{t3_range[0]:.0f}, {t3_range[1]:.0f}]" if not np.isnan(t3_range[0]) else "N/A"
print(f"{'T3: Window width sensitivity':<35} | {t3_str:<25} | {yn(t3_support):>15} | {conf(t3_support):>10}")

t4_str = f"{sig_qubits}/4 sig, CI {broadest}" if not np.isnan(broadest[0]) else "N/A"
t4_str_short = f"{sig_qubits}/4 sig"
print(f"{'T4: Statistical significance':<35} | {t4_str_short:<25} | {yn(t4_any_sig):>15} | {conf(t4_any_sig):>10}")

t5_str = f"pctile={t5_percentile:.0f}%" if not np.isnan(t5_percentile) else "N/A"
print(f"{'T5: Null model':<35} | {t5_str:<25} | {yn(t5_support):>15} | {conf(t5_support):>10}")

t6_str = "No sustained alpha>1" if t6_clean else "Some alpha>1 runs"
print(f"{'T6: Mi MBL control':<35} | {t6_str:<25} | {yn(t6_clean):>15} | {conf(t6_clean, strong=True):>10}")

# ---- Recommendation ----
print(f"\n{'='*75}")
print("RECOMMENDATION: Which claim is supported?")
print(f"{'='*75}")

scores = [t1_support, t2_support, t3_support, t4_any_sig, t5_support, t6_clean]
n_support = sum(scores)
n_high_conf = sum([t3_support, t6_clean])  # the strongest tests

print(f"\n  Tests supporting n*=12: {n_support}/6")
print(f"  High-confidence supports: {n_high_conf}/2")

if n_support >= 5 and n_high_conf >= 2:
    recommendation = "A"
    claim = ('Option A (strong): "The cooperative cascade engages at n* = 12 +/- 1 '
             'Floquet periods, matching the Coxeter number h(E6) = 12."')
elif n_support >= 4:
    recommendation = "B"
    claim = ('Option B (moderate): "The cooperative cascade engages at n* ~ 10-15 '
             'Floquet periods, consistent with the ouroboros cycle period h(E6) = 12."')
elif n_support >= 2:
    recommendation = "C"
    claim = ('Option C (weak): "The cooperative cascade engages at n* ~ 12 Floquet '
             'periods, though the precision of this estimate is limited by the '
             '20-period dataset."')
else:
    recommendation = "D"
    claim = ('Option D (null): "A crossover is observed but its period is not precisely '
             'determinable from this dataset. The match to h(E6) = 12 cannot be '
             'confirmed or refuted."')

print(f"\n  RECOMMENDATION: Option {recommendation}")
print(f"\n  {claim}")

# Additional context
print(f"\n  Key factors in this assessment:")
if t6_clean:
    print(f"    + Mi MBL control is clean (no cooperative cascade)")
if t4_any_sig:
    print(f"    + Piecewise model is statistically significant for {sig_qubits}/4 qubits")
if not np.isnan(broadest[0]):
    print(f"    {'+'if t4_ci_contains_12 else '-'} 95% CI [{broadest[0]:.1f}, {broadest[1]:.1f}] "
          f"{'contains' if t4_ci_contains_12 else 'does not contain'} n*=12")
if t2_support:
    print(f"    + Per-qubit clustering is tight (SD={t2_sd:.1f})")
else:
    print(f"    - Per-qubit spread is wide (SD={t2_sd:.1f})" if not np.isnan(t2_sd) else "    - Insufficient per-qubit data")
if not np.isnan(t5_percentile):
    print(f"    {'+'if t5_support else '-'} Null model: Xiang n* at {t5_percentile:.0f}th percentile "
          f"{'(unusual)' if t5_support else '(not unusual)'}")
print(f"    - Dataset limited to 20 Floquet periods (statistical power is modest)")
