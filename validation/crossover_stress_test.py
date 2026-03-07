"""
Stress Test: n* ~ 12 Crossover Period
=======================================
Tests whether the cooperative cascade crossover at n* ~ 12 Floquet periods
is robust, artefactual, or imprecise.

h(E6) = 12 is the Coxeter number of the E6 Lie algebra.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.io import loadmat
from scipy.stats import f as f_dist
import csv
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

# =====================================================================
# UTILITIES
# =====================================================================

def kww(t, tau, alpha):
    return np.exp(-(t / tau) ** alpha)

def kww_A(t, A, tau, alpha):
    return A * np.exp(-(t / tau) ** alpha)

def fit_kww_window(t, y):
    """Fit KWW to a windowed segment. Returns (alpha, R2, converged)."""
    if len(t) < 4 or np.all(y < 0.01):
        return np.nan, np.nan, False
    try:
        p0 = [y[0], max(t[-1] / 2, 1.0), 1.0]
        popt, pcov = curve_fit(kww_A, t, y, p0=p0,
                               bounds=([0.001, 0.1, 0.05], [2.0, 500.0, 5.0]),
                               maxfev=10000)
        A, tau, alpha = popt
        y_pred = kww_A(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return alpha, r2, True
    except:
        return np.nan, np.nan, False


def sliding_window_alpha(envelope, window_width_periods):
    """
    Compute alpha_local in sliding windows across the envelope.

    Parameters:
    -----------
    envelope : array, shape (n_steps,)
        |Z(n)| for n = 0, 1, ..., n_steps-1 (half-period indices)
    window_width_periods : int
        Window width in Floquet periods (1 period = 2 half-period steps)

    Returns:
    --------
    centers_periods : array of window center positions in Floquet period units
    alphas : array of local alpha values
    r2s : array of local R^2 values
    """
    n_steps = len(envelope)
    half_width = window_width_periods  # in half-period steps (width_periods * 2 / 2)
    # Window width in half-period steps = window_width_periods * 2
    # Half of that = window_width_periods
    w_half = window_width_periods  # half-periods on each side of center

    centers = []
    alphas = []
    r2s = []

    # Move center by 1 period = 2 half-period steps
    for center_hp in range(w_half, n_steps - w_half, 2):
        start = center_hp - w_half
        end = center_hp + w_half + 1  # inclusive
        if end > n_steps:
            break

        seg = envelope[start:end]
        t_seg = np.arange(len(seg), dtype=float)

        alpha, r2, conv = fit_kww_window(t_seg, seg)

        center_period = center_hp / 2.0  # convert to Floquet period units
        centers.append(center_period)
        alphas.append(alpha)
        r2s.append(r2)

    return np.array(centers), np.array(alphas), np.array(r2s)


def find_crossover(centers, alphas, threshold=1.0, sustained=3):
    """Find first period where alpha exceeds threshold for 'sustained' consecutive windows."""
    above = alphas > threshold
    count = 0
    for i in range(len(above)):
        if above[i]:
            count += 1
            if count >= sustained:
                return centers[i - sustained + 1]
        else:
            count = 0
    return np.nan


def find_max_gradient(centers, alphas):
    """Find period of steepest rise in alpha."""
    if len(alphas) < 3:
        return np.nan
    grad = np.gradient(alphas, centers)
    valid = ~np.isnan(grad)
    if not np.any(valid):
        return np.nan
    best_idx = np.nanargmax(grad)
    return centers[best_idx]


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =====================================================================
# DATA LOADING
# =====================================================================

XIANG_BASE = XIANG_2024_BASE
MI_BASE = MI_2022_DATA


def load_xiang_fresh():
    """Load Xiang zero-disorder data from scratch."""
    d = loadmat(f"{XIANG_BASE}/main_figure2/fig2a.mat")
    Zs = d['Zs_exp_avg']      # (6, 41)
    Zs_err = d['Zs_exp_err']  # (6, 41)
    return Zs, Zs_err


def load_mi_fresh():
    """Load Mi DTC data from scratch."""
    data = []
    with open(f"{MI_BASE}/fig_2b.csv", 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([float(x) if x else np.nan for x in row])
    data = np.array(data)
    # Use gate=97 columns (indices 5-9)
    mi_data = data[:, 5:10]
    valid = ~np.any(np.isnan(mi_data), axis=1)
    return mi_data[valid]


# =====================================================================
# TASK 1: Reproduce crossover independently
# =====================================================================

def task1():
    print_header("TASK 1: Reproduce Crossover Independently")

    Zs, _ = load_xiang_fresh()
    n_qubits, n_steps = Zs.shape
    envelopes = np.abs(Zs)

    print(f"  Data loaded fresh: {n_qubits} qubits x {n_steps} half-period steps")
    print(f"  = {n_steps // 2} Floquet periods")
    print(f"  Window width: 5 Floquet periods")
    print()

    # Mean envelope across qubits 0-3 (the ones with alpha > 1)
    mean_env = np.mean(envelopes[:4], axis=0)

    # Sliding window with width 5 periods
    centers, alphas, r2s = sliding_window_alpha(mean_env, window_width_periods=5)

    print(f"  Sliding window alpha_local(n) [mean of Q0-Q3]:")
    print(f"  {'n (period)':>10s}  {'alpha_local':>11s}  {'R^2':>6s}")
    print(f"  {'-'*10}  {'-'*11}  {'-'*6}")
    for i in range(len(centers)):
        a_str = f"{alphas[i]:.4f}" if not np.isnan(alphas[i]) else "N/A"
        r2_str = f"{r2s[i]:.3f}" if not np.isnan(r2s[i]) else "N/A"
        marker = " <-- " if not np.isnan(alphas[i]) and alphas[i] > 1.0 else ""
        print(f"  {centers[i]:>10.1f}  {a_str:>11s}  {r2_str:>6s}{marker}")

    # Three definitions of n*
    n_star = find_crossover(centers, alphas, threshold=1.0, sustained=3)
    n_star_A = find_crossover(centers, alphas, threshold=1.05, sustained=1)
    n_star_B = find_max_gradient(centers, alphas)

    print(f"\n  Crossover definitions:")
    print(f"    n* (sustained > 1.0 for 3 windows):  {n_star}")
    print(f"    n*_A (first > 1.05):                  {n_star_A}")
    print(f"    n*_B (max gradient):                   {n_star_B}")
    print()

    all_in_range = True
    for name, val in [("n*", n_star), ("n*_A", n_star_A), ("n*_B", n_star_B)]:
        if np.isnan(val):
            print(f"    {name} = NaN -- crossover NOT FOUND")
            all_in_range = False
        elif 10 <= val <= 16:
            print(f"    {name} = {val:.1f} -- in range [10, 16]")
        else:
            print(f"    {name} = {val:.1f} -- OUTSIDE range [10, 16]")
            all_in_range = False

    if all_in_range:
        print(f"\n  All three definitions fall in [10, 16]: crossover is ROBUST")
    else:
        print(f"\n  Definitions do NOT all fall in [10, 16]: crossover definition-dependent")

    return envelopes, centers, alphas, n_star, n_star_A, n_star_B


# =====================================================================
# TASK 2: Per-qubit crossover periods
# =====================================================================

def task2(envelopes):
    print_header("TASK 2: Per-Qubit Crossover Periods")

    n_qubits = envelopes.shape[0]

    print(f"  {'Qubit':>6s}  {'n*':>6s}  {'n*_A':>6s}  {'n*_B':>6s}  "
          f"{'alpha_range':>15s}  {'note':>15s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*15}  {'-'*15}")

    all_nstars = []
    for q in range(n_qubits):
        env = envelopes[q]
        centers, alphas, r2s = sliding_window_alpha(env, window_width_periods=5)

        ns = find_crossover(centers, alphas, 1.0, 3)
        nA = find_crossover(centers, alphas, 1.05, 1)
        nB = find_max_gradient(centers, alphas)

        valid_a = alphas[~np.isnan(alphas)]
        a_range = f"[{np.min(valid_a):.2f}, {np.max(valid_a):.2f}]" if len(valid_a) > 0 else "N/A"

        note = ""
        if q >= 4:
            note = "alpha < 1 qubit"
        if np.isnan(ns):
            note += " no crossover"

        ns_str = f"{ns:.1f}" if not np.isnan(ns) else "NONE"
        nA_str = f"{nA:.1f}" if not np.isnan(nA) else "NONE"
        nB_str = f"{nB:.1f}" if not np.isnan(nB) else "NONE"

        print(f"  Z{q:>5d}  {ns_str:>6s}  {nA_str:>6s}  {nB_str:>6s}  "
              f"{a_range:>15s}  {note:>15s}")

        all_nstars.append(ns)

    all_nstars = np.array(all_nstars)
    valid_ns = all_nstars[~np.isnan(all_nstars)]

    print(f"\n  All qubits (Q0-Q5):")
    if len(valid_ns) > 0:
        print(f"    Mean n* = {np.mean(valid_ns):.1f}")
        print(f"    SD      = {np.std(valid_ns):.1f}")
        print(f"    Min     = {np.min(valid_ns):.1f}")
        print(f"    Max     = {np.max(valid_ns):.1f}")
    else:
        print(f"    No qubit shows a sustained crossover")

    # Q0-Q3 only (alpha > 1 qubits)
    ns_03 = all_nstars[:4]
    valid_03 = ns_03[~np.isnan(ns_03)]
    print(f"\n  Q0-Q3 only (alpha > 1 qubits):")
    if len(valid_03) > 0:
        print(f"    Mean n* = {np.mean(valid_03):.1f}")
        print(f"    SD      = {np.std(valid_03):.1f}")
        print(f"    Min     = {np.min(valid_03):.1f}")
        print(f"    Max     = {np.max(valid_03):.1f}")
    else:
        print(f"    No Q0-Q3 qubit shows a sustained crossover")

    # Q4-Q5 contribution
    print(f"\n  Q4-Q5 (alpha < 1 qubits):")
    for q in [4, 5]:
        ns_q = all_nstars[q]
        if np.isnan(ns_q):
            print(f"    Z{q}: no crossover (never exceeds alpha=1.0)")
        else:
            print(f"    Z{q}: n* = {ns_q:.1f}")

    return all_nstars


# =====================================================================
# TASK 3: Sensitivity to window width
# =====================================================================

def task3(envelopes):
    print_header("TASK 3: Sensitivity to Window Width")

    mean_env = np.mean(envelopes[:4], axis=0)

    print(f"  Using mean envelope of Q0-Q3")
    print(f"  {'Width (periods)':>15s}  {'n*':>6s}  {'n*_A':>6s}  {'n*_B':>6s}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*6}  {'-'*6}")

    nstars_by_width = []
    for w in [3, 5, 7, 9]:
        centers, alphas, _ = sliding_window_alpha(mean_env, window_width_periods=w)
        ns = find_crossover(centers, alphas, 1.0, 3)
        nA = find_crossover(centers, alphas, 1.05, 1)
        nB = find_max_gradient(centers, alphas)

        ns_str = f"{ns:.1f}" if not np.isnan(ns) else "NONE"
        nA_str = f"{nA:.1f}" if not np.isnan(nA) else "NONE"
        nB_str = f"{nB:.1f}" if not np.isnan(nB) else "NONE"

        print(f"  {w:>15d}  {ns_str:>6s}  {nA_str:>6s}  {nB_str:>6s}")
        nstars_by_width.append(ns)

    valid_ns = [x for x in nstars_by_width if not np.isnan(x)]
    if len(valid_ns) > 1:
        spread = max(valid_ns) - min(valid_ns)
        print(f"\n  n* range across widths: [{min(valid_ns):.1f}, {max(valid_ns):.1f}]")
        print(f"  Spread: {spread:.1f} periods")
        if spread <= 4:
            print(f"  STABLE: n* varies by <= 4 periods across window widths")
        else:
            print(f"  UNSTABLE: n* varies by > 4 periods")
    elif len(valid_ns) == 1:
        print(f"\n  Only one width shows crossover — limited assessment")
    else:
        print(f"\n  No crossover detected at any window width")

    return nstars_by_width


# =====================================================================
# TASK 4: Statistical significance
# =====================================================================

def task4(envelopes):
    print_header("TASK 4: Statistical Significance of Crossover")

    mean_env = np.mean(envelopes[:4], axis=0)
    centers, alphas, _ = sliding_window_alpha(mean_env, window_width_periods=5)

    valid = ~np.isnan(alphas)
    c = centers[valid]
    a = alphas[valid]
    n_pts = len(c)

    print(f"  Data: {n_pts} window-center alpha values")
    print()

    # Fit flat model: alpha = constant
    a_flat = np.mean(a)
    ss_flat = np.sum((a - a_flat) ** 2)

    # Fit piecewise model: alpha = a1 if n < n*, else a2 + b*(n - n*)
    best_sse = np.inf
    best_nstar = None
    best_params = None

    for trial_n in np.arange(c[2], c[-3], 0.5):
        mask1 = c < trial_n
        mask2 = c >= trial_n
        if np.sum(mask1) < 2 or np.sum(mask2) < 2:
            continue

        a1 = np.mean(a[mask1])
        # Linear fit for segment 2
        c2 = c[mask2] - trial_n
        a2_vals = a[mask2]
        if len(c2) > 1:
            coeffs = np.polyfit(c2, a2_vals, 1)
            b = coeffs[0]
            a2 = coeffs[1]
        else:
            b = 0
            a2 = a2_vals[0]

        pred = np.where(c < trial_n, a1, a2 + b * (c - trial_n))
        sse = np.sum((a - pred) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_nstar = trial_n
            best_params = (a1, a2, b)

    if best_params is not None:
        a1, a2, b = best_params
        print(f"  Piecewise model: alpha = {a1:.3f} for n < {best_nstar:.1f}, "
              f"then {a2:.3f} + {b:.4f}*(n - {best_nstar:.1f})")
        print(f"  Best-fit n* = {best_nstar:.1f}")

        # F-test: piecewise (3 params: a1, a2+b, n*) vs flat (1 param)
        df_flat = n_pts - 1
        df_pw = n_pts - 4  # 4 parameters: a1, a2, b, n*
        if df_pw > 0 and best_sse > 0:
            f_stat = ((ss_flat - best_sse) / (df_flat - df_pw)) / (best_sse / df_pw)
            p_value = 1 - f_dist.cdf(f_stat, df_flat - df_pw, df_pw)
            print(f"\n  F-test (piecewise vs flat):")
            print(f"    SS_flat = {ss_flat:.4f}, SS_piecewise = {best_sse:.4f}")
            print(f"    F = {f_stat:.3f}, df = ({df_flat - df_pw}, {df_pw})")
            print(f"    p = {p_value:.6f}")
            if p_value < 0.05:
                print(f"    Piecewise model is SIGNIFICANTLY better than flat (p < 0.05)")
            else:
                print(f"    Piecewise model is NOT significantly better than flat")
        else:
            p_value = np.nan
            print(f"    Insufficient df for F-test")

        # Confidence interval for n* by sweeping and finding delta-chi2 boundary
        print(f"\n  95% CI for n* (from SSE profile):")
        sse_profile = []
        n_grid = np.arange(c[1], c[-2], 0.25)
        for trial_n in n_grid:
            mask1 = c < trial_n
            mask2 = c >= trial_n
            if np.sum(mask1) < 2 or np.sum(mask2) < 2:
                sse_profile.append(np.inf)
                continue
            a1_t = np.mean(a[mask1])
            c2 = c[mask2] - trial_n
            if len(c2) > 1:
                coeffs = np.polyfit(c2, a[mask2], 1)
                pred = np.where(c < trial_n, a1_t,
                               coeffs[1] + coeffs[0] * (c - trial_n))
            else:
                pred = np.where(c < trial_n, a1_t, a[mask2][0])
            sse_profile.append(np.sum((a - pred) ** 2))

        sse_profile = np.array(sse_profile)
        # Delta chi-squared for 1 parameter at 95%: ~3.84
        threshold = best_sse * (1 + 3.84 / df_pw) if df_pw > 0 else np.inf
        in_ci = sse_profile < threshold
        if np.any(in_ci):
            ci_lo = n_grid[in_ci][0]
            ci_hi = n_grid[in_ci][-1]
            print(f"    n* = {best_nstar:.1f}, 95% CI = [{ci_lo:.1f}, {ci_hi:.1f}]")
            ci_width = ci_hi - ci_lo
            print(f"    CI width = {ci_width:.1f} periods")
        else:
            ci_lo, ci_hi = np.nan, np.nan
            print(f"    Could not determine CI")
    else:
        print(f"  Piecewise fit failed")
        best_nstar = np.nan
        ci_lo, ci_hi = np.nan, np.nan
        p_value = np.nan

    return best_nstar, ci_lo, ci_hi, p_value


# =====================================================================
# TASK 5: Null model
# =====================================================================

def task5(envelopes):
    print_header("TASK 5: Null Model — What Crossover Would Random Data Give?")

    mean_env = np.mean(envelopes[:4], axis=0)
    n_steps = len(mean_env)

    # Estimate noise from residuals
    n_arr = np.arange(n_steps, dtype=float)
    try:
        popt, _ = curve_fit(kww_A, n_arr, mean_env, p0=[1.0, 20.0, 1.0],
                            bounds=([0.01, 0.5, 0.05], [2.0, 200.0, 5.0]))
        residuals = mean_env - kww_A(n_arr, *popt)
        noise_std = np.std(residuals)
    except:
        noise_std = 0.02
    print(f"  Estimated noise level from Xiang residuals: sigma = {noise_std:.4f}")

    n_sims = 2000
    print(f"  Running {n_sims} null model simulations...")
    print(f"  Null: KWW with alpha ~ Uniform[0.8, 1.4], same n_steps={n_steps}")
    print()

    rng = np.random.default_rng(42)
    null_nstars = []

    for sim in range(n_sims):
        # Random alpha (no cooperative structure)
        alpha_sim = rng.uniform(0.8, 1.4)
        tau_sim = rng.uniform(10, 40)
        env_sim = np.exp(-(n_arr / tau_sim) ** alpha_sim)
        env_sim += rng.normal(0, noise_std, n_steps)
        env_sim = np.clip(env_sim, 0.01, 1.5)

        centers_sim, alphas_sim, _ = sliding_window_alpha(env_sim, window_width_periods=5)
        ns_sim = find_crossover(centers_sim, alphas_sim, 1.0, 3)
        null_nstars.append(ns_sim)

    null_nstars = np.array(null_nstars)
    valid_null = null_nstars[~np.isnan(null_nstars)]

    n_crossover = len(valid_null)
    frac_crossover = n_crossover / n_sims

    print(f"  Results:")
    print(f"    Simulations with a crossover detected: {n_crossover}/{n_sims} "
          f"({frac_crossover*100:.1f}%)")

    if n_crossover > 10:
        print(f"    Null n* mean = {np.mean(valid_null):.1f}")
        print(f"    Null n* std  = {np.std(valid_null):.1f}")
        print(f"    Null n* median = {np.median(valid_null):.1f}")

        # Histogram
        bins = np.arange(0, 22, 2)
        counts, edges = np.histogram(valid_null, bins=bins)
        print(f"\n    Distribution of null n*:")
        for i in range(len(counts)):
            bar = '#' * int(counts[i] / max(counts) * 30) if max(counts) > 0 else ''
            print(f"      [{edges[i]:>4.0f}-{edges[i+1]:>4.0f}): {counts[i]:>5d}  {bar}")

        # Where does Xiang n*=12 fall?
        xiang_nstar = 12
        percentile = np.mean(valid_null <= xiang_nstar) * 100
        print(f"\n    Xiang n* = {xiang_nstar} falls at percentile {percentile:.1f}%")
        if percentile < 5 or percentile > 95:
            print(f"    UNUSUAL: n*=12 is outside the 90% range of null model")
        else:
            print(f"    NOT UNUSUAL: n*=12 is within the bulk of the null distribution")
    else:
        print(f"    Too few crossovers detected to characterize null distribution")
        print(f"    This means the crossover is RARE in random KWW data")
        percentile = np.nan

    # Also test: null with CONSTANT alpha across the series (no crossover by construction)
    print(f"\n  Control null: constant alpha=1.1 with noise (no real crossover):")
    ctrl_nstars = []
    for sim in range(1000):
        alpha_c = 1.1
        tau_c = 20.0
        env_c = np.exp(-(n_arr / tau_c) ** alpha_c)
        env_c += rng.normal(0, noise_std, n_steps)
        env_c = np.clip(env_c, 0.01, 1.5)
        c_c, a_c, _ = sliding_window_alpha(env_c, window_width_periods=5)
        ns_c = find_crossover(c_c, a_c, 1.0, 3)
        ctrl_nstars.append(ns_c)

    ctrl_nstars = np.array(ctrl_nstars)
    ctrl_valid = ctrl_nstars[~np.isnan(ctrl_nstars)]
    print(f"    Crossovers detected: {len(ctrl_valid)}/1000 ({len(ctrl_valid)/10:.1f}%)")
    if len(ctrl_valid) > 10:
        print(f"    Mean n* = {np.mean(ctrl_valid):.1f}, "
              f"std = {np.std(ctrl_valid):.1f}")

    return percentile, frac_crossover


# =====================================================================
# TASK 6: Mi MBL control
# =====================================================================

def task6():
    print_header("TASK 6: Mi MBL Control")

    mi_data = load_mi_fresh()
    n_steps_mi, n_qubits_mi = mi_data.shape
    mi_env = np.abs(mi_data)
    mean_mi_env = np.mean(mi_env, axis=1)  # average across qubits

    print(f"  Mi MBL data: {n_qubits_mi} qubits x {n_steps_mi} Floquet periods")
    print(f"  (Note: Mi data is already at period resolution, not half-period)")
    print()

    # Sliding window analysis — Mi data is at period resolution
    # so "window_width_periods" applied to half-period conversion doesn't apply
    # We need to adapt: treat each step as a half-period (double the array)
    # OR: implement a direct period-based sliding window

    # Direct implementation for period-resolution data
    window = 5
    half_w = window // 2
    centers_mi = []
    alphas_mi = []

    for center in range(half_w, n_steps_mi - half_w):
        start = center - half_w
        end = center + half_w + 1
        seg = mean_mi_env[start:end]
        t_seg = np.arange(len(seg), dtype=float)
        alpha, r2, conv = fit_kww_window(t_seg, seg)
        centers_mi.append(center)
        alphas_mi.append(alpha)

    centers_mi = np.array(centers_mi)
    alphas_mi = np.array(alphas_mi)

    print(f"  Sliding window alpha_local(n) [mean of all qubits]:")
    print(f"  {'n (period)':>10s}  {'alpha_local':>11s}  {'> 1.0?':>6s}")
    print(f"  {'-'*10}  {'-'*11}  {'-'*6}")

    n_above = 0
    for i in range(min(30, len(centers_mi))):
        above = alphas_mi[i] > 1.0 if not np.isnan(alphas_mi[i]) else False
        a_str = f"{alphas_mi[i]:.4f}" if not np.isnan(alphas_mi[i]) else "N/A"
        print(f"  {centers_mi[i]:>10.0f}  {a_str:>11s}  {'YES' if above else 'no':>6s}")
        if above:
            n_above += 1

    if len(centers_mi) > 30:
        remaining_above = np.sum(alphas_mi[30:] > 1.0)
        n_above += remaining_above
        print(f"  ... ({len(centers_mi) - 30} more windows, "
              f"{remaining_above} above 1.0)")

    valid_mi = ~np.isnan(alphas_mi)
    print(f"\n  Summary:")
    print(f"    Total windows: {len(centers_mi)}")
    print(f"    Windows with alpha > 1.0: {n_above}")
    print(f"    Mean alpha_local: {np.nanmean(alphas_mi):.4f}")
    print(f"    Max alpha_local: {np.nanmax(alphas_mi):.4f} "
          f"at period {centers_mi[np.nanargmax(alphas_mi)]:.0f}")
    print(f"    Min alpha_local: {np.nanmin(alphas_mi):.4f}")

    # Check for sustained crossover
    ns_mi = np.nan
    count = 0
    for i in range(len(alphas_mi)):
        if not np.isnan(alphas_mi[i]) and alphas_mi[i] > 1.0:
            count += 1
            if count >= 3:
                ns_mi = centers_mi[i - 2]
                break
        else:
            count = 0

    if np.isnan(ns_mi):
        print(f"\n  No sustained crossover above alpha=1.0 in Mi MBL data")
        print(f"  CLEAN CONTRAST: cooperative cascade absent in MBL regime")
        mi_crossover = False
    else:
        print(f"\n  Sustained crossover at period {ns_mi:.0f} in Mi MBL data")
        print(f"  WEAKENS Xiang result — crossover may be fitting artefact")
        mi_crossover = True

    return mi_crossover, np.nanmean(alphas_mi)


# =====================================================================
# TASK 7: The h(E6) = 12 claim
# =====================================================================

def task7(n_star, n_star_A, n_star_B, all_nstars, nstars_by_width,
          best_nstar_pw, ci_lo, ci_hi, p_value,
          null_percentile, null_frac, mi_crossover):
    print_header("TASK 7: The h(E6) = 12 Claim — Verdict")

    # Collect evidence
    print(f"\n  VERDICT TABLE:")
    print(f"  {'Test':>5s}  {'Description':<35s}  {'Result':<25s}  "
          f"{'Supports n*=12?':>15s}  {'Confidence':>10s}")
    print(f"  {'-'*5}  {'-'*35}  {'-'*25}  {'-'*15}  {'-'*10}")

    verdicts = []

    # T1: Definition robustness
    t1_vals = [n_star, n_star_A, n_star_B]
    t1_valid = [v for v in t1_vals if not np.isnan(v)]
    if len(t1_valid) >= 2:
        t1_range = f"[{min(t1_valid):.0f}, {max(t1_valid):.0f}]"
        t1_support = all(10 <= v <= 16 for v in t1_valid)
        t1_conf = "High" if t1_support else "Low"
    else:
        t1_range = "insufficient"
        t1_support = False
        t1_conf = "Low"
    print(f"  {'T1':>5s}  {'Definition robustness':<35s}  "
          f"{t1_range:<25s}  {'Yes' if t1_support else 'No':>15s}  {t1_conf:>10s}")
    verdicts.append(t1_support)

    # T2: Per-qubit consistency
    valid_ns = all_nstars[~np.isnan(all_nstars)]
    if len(valid_ns) >= 2:
        t2_result = f"mean={np.mean(valid_ns):.1f}, SD={np.std(valid_ns):.1f}"
        t2_support = np.std(valid_ns) < 5 and abs(np.mean(valid_ns) - 12) < 5
        t2_conf = "High" if np.std(valid_ns) < 3 else ("Med" if np.std(valid_ns) < 5 else "Low")
    elif len(valid_ns) == 1:
        t2_result = f"only 1 qubit: n*={valid_ns[0]:.1f}"
        t2_support = 10 <= valid_ns[0] <= 16
        t2_conf = "Low"
    else:
        t2_result = "no crossovers"
        t2_support = False
        t2_conf = "Low"
    print(f"  {'T2':>5s}  {'Per-qubit consistency':<35s}  "
          f"{t2_result:<25s}  {'Yes' if t2_support else 'No':>15s}  {t2_conf:>10s}")
    verdicts.append(t2_support)

    # T3: Window width sensitivity
    valid_ws = [x for x in nstars_by_width if not np.isnan(x)]
    if len(valid_ws) >= 2:
        t3_result = f"range [{min(valid_ws):.0f}, {max(valid_ws):.0f}]"
        spread = max(valid_ws) - min(valid_ws)
        t3_support = spread <= 4
        t3_conf = "High" if spread <= 2 else ("Med" if spread <= 4 else "Low")
    else:
        t3_result = f"{len(valid_ws)} widths show crossover"
        t3_support = False
        t3_conf = "Low"
    print(f"  {'T3':>5s}  {'Window width sensitivity':<35s}  "
          f"{t3_result:<25s}  {'Yes' if t3_support else 'No':>15s}  {t3_conf:>10s}")
    verdicts.append(t3_support)

    # T4: Statistical significance
    if not np.isnan(p_value) and not np.isnan(ci_lo):
        ci_str = f"[{ci_lo:.0f}, {ci_hi:.0f}], p={p_value:.4f}"
        ci_width = ci_hi - ci_lo
        t4_support = p_value < 0.05 and ci_lo <= 12 <= ci_hi
        t4_conf = "High" if (p_value < 0.01 and ci_width < 6) else \
                  ("Med" if p_value < 0.05 else "Low")
    else:
        ci_str = "fit failed"
        t4_support = False
        t4_conf = "Low"
    print(f"  {'T4':>5s}  {'Statistical significance':<35s}  "
          f"{ci_str:<25s}  {'Yes' if t4_support else 'No':>15s}  {t4_conf:>10s}")
    verdicts.append(t4_support)

    # T5: Null model
    if not np.isnan(null_percentile):
        t5_result = f"pctl={null_percentile:.0f}%, frac={null_frac*100:.0f}%"
        t5_support = null_percentile < 5 or null_percentile > 95 or null_frac < 0.10
        t5_conf = "High" if (null_frac < 0.05 or null_percentile < 5) else \
                  ("Med" if null_frac < 0.20 else "Low")
    else:
        t5_result = f"frac crossover={null_frac*100:.0f}%"
        t5_support = null_frac < 0.10
        t5_conf = "Med" if t5_support else "Low"
    print(f"  {'T5':>5s}  {'Null model':<35s}  "
          f"{t5_result:<25s}  {'Yes' if t5_support else 'No':>15s}  {t5_conf:>10s}")
    verdicts.append(t5_support)

    # T6: Mi MBL control
    t6_result = "no crossover in Mi" if not mi_crossover else "crossover in Mi too"
    t6_support = not mi_crossover
    t6_conf = "High" if t6_support else "Low"
    print(f"  {'T6':>5s}  {'Mi MBL control':<35s}  "
          f"{t6_result:<25s}  {'Yes' if t6_support else 'No':>15s}  {t6_conf:>10s}")
    verdicts.append(t6_support)

    # Determine option
    n_support = sum(verdicts)
    n_total = len(verdicts)
    high_conf = sum(1 for v, t in zip(verdicts,
                    [t1_conf, t2_conf, t3_conf, t4_conf, t5_conf, t6_conf])
                    if v and t == "High")

    print(f"\n  Score: {n_support}/{n_total} tests support n*=12")
    print(f"  High-confidence supports: {high_conf}")

    print(f"\n  RECOMMENDATION:")
    if n_support >= 5 and high_conf >= 3:
        option = 'A'
        print(f"  Option A (STRONG): 'The cooperative cascade engages at")
        print(f"  n* = 12 +/- 1 Floquet periods, matching h(E6) = 12.'")
    elif n_support >= 4:
        option = 'B'
        print(f"  Option B (MODERATE): 'The cooperative cascade engages at")
        print(f"  n* ~ 10-15 Floquet periods, consistent with h(E6) = 12.'")
    elif n_support >= 2:
        option = 'C'
        print(f"  Option C (WEAK): 'The cooperative cascade engages at")
        print(f"  n* ~ 12 Floquet periods, though precision is limited")
        print(f"  by the 20-period dataset.'")
    else:
        option = 'D'
        print(f"  Option D (NULL): 'A crossover is observed but its period")
        print(f"  is not precisely determinable from this dataset.'")

    return option


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  STRESS TEST: n* ~ 12 CROSSOVER PERIOD")
    print("  Does the cooperative cascade crossover match h(E6) = 12?")
    print("=" * 70)

    # Task 1
    envelopes, centers, alphas, n_star, n_star_A, n_star_B = task1()

    # Task 2
    all_nstars = task2(envelopes)

    # Task 3
    nstars_by_width = task3(envelopes)

    # Task 4
    best_nstar_pw, ci_lo, ci_hi, p_value = task4(envelopes)

    # Task 5
    null_percentile, null_frac = task5(envelopes)

    # Task 6
    mi_crossover, mi_mean_alpha = task6()

    # Task 7
    option = task7(n_star, n_star_A, n_star_B, all_nstars, nstars_by_width,
                   best_nstar_pw, ci_lo, ci_hi, p_value,
                   null_percentile, null_frac, mi_crossover)

    print()
