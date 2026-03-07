"""
Q3-Q4 Sliding Window Crossover Analysis - Full Rebuild
=======================================================
Paper: "Hunting for the Merkabit in Existing Hardware"

Primary target: Q3-Q4 pairwise shuttling (counter-rotating) from Figure 3d
Control: Q2-Q3 pairwise shuttling (co-rotating) from Figure 3c
Data: van Riggelen-Doelman et al. 2024, Zenodo 11203148

All results compared to Xiang et al. 2024 prethermal DTC benchmark:
  - n* crossover 95% CI: [6, 15], width = 9
  - Null model: 85th percentile (not significant)

Target: log_11(12) = ln(12)/ln(11) ~ 1.0363
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
import os, sys, warnings, glob, re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(42)

LOG11_12 = np.log(12) / np.log(11)  # 1.03627...

# ============================================================
# DATA LOADING
# ============================================================

# NatCom (van Riggelen-Doelman 2024) F3 data — Zenodo 11203148
# Download and extract to data/natcom_f3/ to run this script
BASE_F3 = os.path.join(DATA_DIR, 'natcom_f3')

def load_pairwise_data(base_path, data_dir, spin_down_pattern, spin_up_pattern, n_extractor):
    """Load pairwise shuttling data from QCoDeS folders.
    Returns dict with 'down' and 'up' keys, each containing (N_array, fraction_array).
    """
    full_path = os.path.join(base_path, data_dir)
    folders = os.listdir(full_path)

    data = {'down': {'N': [], 'frac': []}, 'up': {'N': [], 'frac': []}}

    for folder in folders:
        fpath = os.path.join(full_path, folder)
        if not os.path.isdir(fpath):
            continue

        # Read single_set.dat for m1N_fraction
        single_path = os.path.join(fpath, 'single_set.dat')
        if not os.path.exists(single_path):
            continue

        try:
            with open(single_path, 'r') as f:
                lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
            if lines:
                frac = float(lines[0].split('\t')[1])
            else:
                continue
        except Exception:
            continue

        # Determine spin channel and extract N
        N = n_extractor(folder)
        if N is None:
            continue

        if re.search(spin_down_pattern, folder):
            data['down']['N'].append(N)
            data['down']['frac'].append(frac)
        elif re.search(spin_up_pattern, folder):
            data['up']['N'].append(N)
            data['up']['frac'].append(frac)

    # Sort by N
    for key in ['down', 'up']:
        if len(data[key]['N']) > 0:
            order = np.argsort(data[key]['N'])
            data[key]['N'] = np.array(data[key]['N'])[order]
            data[key]['frac'] = np.array(data[key]['frac'])[order]
        else:
            data[key]['N'] = np.array([])
            data[key]['frac'] = np.array([])

    return data


def extract_N_q3q4(folder):
    """Extract N from Q3-Q4 folder name: *_N_<value>_waitQ3_*"""
    m = re.search(r'_N_(\d+)_waitQ3', folder)
    return int(m.group(1)) if m else None

def extract_N_q2q3(folder):
    """Extract N from Q2-Q3 folder name: *_N_<value> (at end)"""
    m = re.search(r'_N_(\d+)$', folder)
    return int(m.group(1)) if m else None


# ============================================================
# COHERENCE AND KWW MACHINERY
# ============================================================

def extract_coherence(N_arr, frac_arr, label=""):
    """Convert spin fraction to coherence envelope C(n).
    C = 2 * |P - P_inf|, normalized so C(0) = 1.
    """
    P_inf = np.mean(frac_arr[-10:])
    C_raw = np.abs(frac_arr - P_inf)
    C0 = C_raw[0]
    if C0 < 0.01:
        print(f"  WARNING ({label}): C(0) ~ 0, spin not polarized at N=0")
        return N_arr, np.full_like(frac_arr, np.nan)
    return N_arr, C_raw / C0

def kww(n, tau, alpha):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.exp(-(n / tau) ** alpha)

def fit_kww_window(n, C):
    mask = np.isfinite(C) & (C > 0.001) & np.isfinite(n) & (n > 0)
    n_f, C_f = n[mask], C[mask]
    if len(n_f) < 3:
        return np.nan, np.nan, np.nan
    try:
        p0 = [max(n_f) * 1.5, 1.0]
        popt, pcov = curve_fit(kww, n_f, C_f, p0=p0,
                               bounds=([0.1, 0.01], [50000, 10]), maxfev=30000)
        tau, alpha = popt
        alpha_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.nan
        y_pred = kww(n_f, *popt)
        ss_res = np.sum((C_f - y_pred) ** 2)
        ss_tot = np.sum((C_f - np.mean(C_f)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return alpha, alpha_err, R2
    except Exception:
        return np.nan, np.nan, np.nan

def fit_kww_full(n, C):
    mask = np.isfinite(C) & (C > 0.001) & np.isfinite(n) & (n > 0)
    n_f, C_f = n[mask], C[mask]
    if len(n_f) < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        p0 = [max(n_f) * 0.3, 1.0]
        popt, pcov = curve_fit(kww, n_f, C_f, p0=p0,
                               bounds=([0.1, 0.01], [50000, 10]), maxfev=30000)
        tau, alpha = popt
        tau_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else np.nan
        alpha_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.nan
        y_pred = kww(n_f, *popt)
        ss_res = np.sum((C_f - y_pred) ** 2)
        ss_tot = np.sum((C_f - np.mean(C_f)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return tau, alpha, tau_err, alpha_err, R2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def sliding_alpha(n_env, C_env, width=5):
    n_fit = n_env[1:]  # skip n=0
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


# ============================================================
# CROSSOVER DEFINITIONS
# ============================================================

def find_nstar_sustained(centers, alphas, threshold=1.0, run_length=3):
    valid = ~np.isnan(alphas)
    for i in range(len(alphas) - run_length + 1):
        segment = alphas[i:i+run_length]
        if np.all(valid[i:i+run_length]) and np.all(segment > threshold):
            return centers[i]
    return np.nan

def find_nstar_max_deriv(centers, alphas):
    valid = ~np.isnan(alphas)
    if np.sum(valid) < 3:
        return np.nan
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

def find_nstar_intersection(centers, alphas):
    """n*_B: intersection of early-time linear fit and late-time linear fit."""
    valid = ~np.isnan(alphas)
    c_v = centers[valid]
    a_v = alphas[valid]
    if len(c_v) < 6:
        return np.nan
    # Split at midpoint, fit lines to each half
    mid = len(c_v) // 2
    c_early, a_early = c_v[:mid], a_v[:mid]
    c_late, a_late = c_v[mid:], a_v[mid:]
    if len(c_early) < 2 or len(c_late) < 2:
        return np.nan
    try:
        p1 = np.polyfit(c_early, a_early, 1)
        p2 = np.polyfit(c_late, a_late, 1)
        # Intersection: p1[0]*x + p1[1] = p2[0]*x + p2[1]
        if abs(p1[0] - p2[0]) < 1e-12:
            return np.nan
        x_int = (p2[1] - p1[1]) / (p1[0] - p2[0])
        if c_v[0] <= x_int <= c_v[-1]:
            return x_int
        return np.nan
    except Exception:
        return np.nan


# ============================================================
# PIECEWISE MODEL
# ============================================================

def piecewise_constant_rising(n, n_star, a1, a2, b):
    return np.where(n < n_star, a1, a2 + b * (n - n_star))

def fit_piecewise(centers, alphas, n_boot=1000):
    valid = ~np.isnan(alphas)
    c_v = centers[valid]
    a_v = alphas[valid]
    N = len(a_v)
    if N < 6:
        return np.nan, (np.nan, np.nan), np.nan, np.nan, None, np.nan

    best_sse = np.inf
    best_nstar = np.nan
    best_params = None

    step = max(1, (c_v[-3] - c_v[2]) / 80)
    for n_try in np.arange(c_v[2], c_v[-3], step):
        ml = c_v < n_try
        mr = c_v >= n_try
        if np.sum(ml) < 2 or np.sum(mr) < 2:
            continue
        a1 = np.mean(a_v[ml])
        if np.sum(mr) >= 2:
            coeffs = np.polyfit(c_v[mr] - n_try, a_v[mr], 1)
            b, a2 = coeffs[0], coeffs[1]
        else:
            a2 = np.mean(a_v[mr])
            b = 0
        pred = piecewise_constant_rising(c_v, n_try, a1, a2, b)
        sse = np.sum((a_v - pred) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_nstar = n_try
            best_params = (a1, a2, b)

    flat_mean = np.mean(a_v)
    sse_flat = np.sum((a_v - flat_mean) ** 2)

    k_piece = 4
    k_flat = 1
    df1 = k_piece - k_flat
    df2 = N - k_piece
    if df2 > 0 and best_sse > 0:
        F_stat = ((sse_flat - best_sse) / df1) / (best_sse / df2)
        p_value = 1 - f_dist.cdf(F_stat, df1, df2)
    else:
        F_stat, p_value = np.nan, np.nan

    nstar_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(N, N, replace=True)
        c_b = c_v[idx]
        a_b = a_v[idx]
        so = np.argsort(c_b)
        c_b, a_b = c_b[so], a_b[so]
        best_sse_b = np.inf
        best_ns_b = np.nan
        step_b = max(1, (c_b[-2] - c_b[1]) / 40)
        for n_try in np.arange(c_b[1], c_b[-2], step_b):
            ml = c_b < n_try
            mr = c_b >= n_try
            if np.sum(ml) < 2 or np.sum(mr) < 2:
                continue
            a1_b = np.mean(a_b[ml])
            if np.sum(mr) >= 2:
                co = np.polyfit(c_b[mr] - n_try, a_b[mr], 1)
                b_b, a2_b = co[0], co[1]
            else:
                a2_b, b_b = np.mean(a_b[mr]), 0
            pred_b = piecewise_constant_rising(c_b, n_try, a1_b, a2_b, b_b)
            sse_b = np.sum((a_b - pred_b) ** 2)
            if sse_b < best_sse_b:
                best_sse_b = sse_b
                best_ns_b = n_try
        nstar_boots.append(best_ns_b)

    nstar_boots = np.array([x for x in nstar_boots if not np.isnan(x)])
    ci = (np.percentile(nstar_boots, 2.5), np.percentile(nstar_boots, 97.5)) if len(nstar_boots) > 0 else (np.nan, np.nan)
    return best_nstar, ci, F_stat, p_value, best_params, sse_flat


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 80)
print("Q3-Q4 SLIDING WINDOW CROSSOVER ANALYSIS - FULL REBUILD")
print("van Riggelen-Doelman et al. 2024 | Pairwise Shuttling")
print("Primary: Q3-Q4 (counter-rotating) | Control: Q2-Q3 (co-rotating)")
print("=" * 80)

# Load data
print("\nLoading data...")

q3q4 = load_pairwise_data(BASE_F3, 'Data_Figure3d',
                           r'SpinDown', r'SpinUp', extract_N_q3q4)
q2q3 = load_pairwise_data(BASE_F3, 'Data_Figure3c',
                           r'Spin_Down', r'Spin_Up', extract_N_q2q3)

print(f"  Q3-Q4 (Figure 3d): SpinDown={len(q3q4['down']['N'])}, SpinUp={len(q3q4['up']['N'])}")
print(f"    N range: {q3q4['down']['N'][0]}-{q3q4['down']['N'][-1]} (step {q3q4['down']['N'][1]-q3q4['down']['N'][0]})")
print(f"    Shuttles per N: 2 (pairwise round-trip)")
print(f"  Q2-Q3 (Figure 3c): SpinDown={len(q2q3['down']['N'])}, SpinUp={len(q2q3['up']['N'])}")
print(f"    N range: {q2q3['down']['N'][0]}-{q2q3['down']['N'][-1]} (step {q2q3['down']['N'][1]-q2q3['down']['N'][0]})")
print(f"    Shuttles per N: 2 (pairwise round-trip)")


# ============================================================
# TASK 1: Extract decoherence envelopes
# ============================================================
print(f"\n{'='*80}")
print("TASK 1: EXTRACT DECOHERENCE ENVELOPES")
print(f"{'='*80}")

geometries = {
    'Q3-Q4': q3q4,
    'Q2-Q3': q2q3
}

results = {}

for geo_name, geo_data in geometries.items():
    results[geo_name] = {}
    for spin in ['down', 'up']:
        N_arr = geo_data[spin]['N']
        frac_arr = geo_data[spin]['frac']
        label = f"{geo_name} Spin-{spin}"

        N_env, C_env = extract_coherence(N_arr, frac_arr, label)
        idx = np.arange(len(N_env), dtype=float)

        tau, alpha, tau_err, alpha_err, R2 = fit_kww_full(idx, C_env)

        results[geo_name][spin] = {
            'N_arr': N_env, 'frac': frac_arr,
            'idx': idx, 'C': C_env,
            'tau': tau, 'alpha': alpha,
            'tau_err': tau_err, 'alpha_err': alpha_err, 'R2': R2
        }

        print(f"\n  {label}:")
        print(f"    Points: {len(N_env)}")
        print(f"    P(N=0)={frac_arr[0]:.4f}, P(N_max)={frac_arr[-1]:.4f}")
        print(f"    C(0)={C_env[0]:.4f}, C(5)={C_env[5]:.4f}, C(10)={C_env[10]:.4f}")
        print(f"    KWW full fit: alpha={alpha:.4f} +/- {alpha_err:.4f}")
        print(f"    tau={tau:.1f} +/- {tau_err:.1f} (index units), R2={R2:.6f}")

print(f"\n  Xiang comparison: alpha_full ~ 1.07 (mean), R2 ~ 0.99")


# ============================================================
# TASK 2: Sliding window alpha_local(n)
# ============================================================
print(f"\n{'='*80}")
print("TASK 2: SLIDING WINDOW LOCAL EXPONENT (width=5)")
print(f"{'='*80}")
print(f"\n  Target: alpha_local(n=11) ~ log_11(12) = {LOG11_12:.4f}")

for geo_name in ['Q3-Q4', 'Q2-Q3']:
    for spin in ['down', 'up']:
        r = results[geo_name][spin]
        centers, alphas, errs = sliding_alpha(r['idx'], r['C'], width=5)
        r['centers_w5'] = centers
        r['alphas_w5'] = alphas
        r['errs_w5'] = errs

        print(f"\n  {geo_name} Spin-{spin}:")
        # Report alpha at n=11
        alpha_at_11 = np.nan
        for i, c in enumerate(centers):
            if abs(c - 11) < 0.5:
                alpha_at_11 = alphas[i]
                break
        r['alpha_at_11'] = alpha_at_11
        hit = abs(alpha_at_11 - LOG11_12) < 0.05 if not np.isnan(alpha_at_11) else False
        print(f"    alpha_local(n=11) = {alpha_at_11:.4f}" if not np.isnan(alpha_at_11) else "    alpha_local(n=11) = N/A")
        print(f"    log_11(12) = {LOG11_12:.4f}")
        if not np.isnan(alpha_at_11):
            print(f"    |deviation| = {abs(alpha_at_11 - LOG11_12):.4f}, within +/-0.05? {'YES' if hit else 'NO'}")

        # Print table of first 40 windows
        print(f"    {'idx':>5} | {'N':>6} | {'alpha':>8} | {'err':>8} | {'>1?':>4}")
        print(f"    {'-'*45}")
        for i in range(min(40, len(centers))):
            c = centers[i]
            N_phys = int(r['N_arr'][int(c)] if int(c) < len(r['N_arr']) else c * (r['N_arr'][1] - r['N_arr'][0]))
            a = alphas[i]
            e = errs[i]
            a_str = f"{a:.4f}" if not np.isnan(a) else "  NaN"
            e_str = f"{e:.4f}" if not np.isnan(e) else "  NaN"
            mark = " *" if not np.isnan(a) and a > 1.0 else ""
            print(f"    {c:5.0f} | {N_phys:6d} | {a_str:>8} | {e_str:>8} |{mark}")
        if len(centers) > 40:
            print(f"    ... ({len(centers)-40} more windows omitted)")

        valid_a = ~np.isnan(alphas)
        print(f"    Windows with alpha > 1.0: {np.sum(alphas[valid_a] > 1.0)}/{np.sum(valid_a)}")
        print(f"    Mean alpha_local: {np.nanmean(alphas):.4f}, Max: {np.nanmax(alphas):.4f}")


# ============================================================
# TASK 3: Three crossover definitions
# ============================================================
print(f"\n{'='*80}")
print("TASK 3: THREE CROSSOVER DEFINITIONS")
print(f"{'='*80}")

print(f"\n  {'Geometry':<10} {'Spin':<6} | {'n*_sust':>8} | {'n*_A':>8} | {'n*_B':>8} | {'Spread':>7} | {'Converge?':>10}")
print(f"  {'-'*70}")

for geo_name in ['Q3-Q4', 'Q2-Q3']:
    for spin in ['down', 'up']:
        r = results[geo_name][spin]
        c = r['centers_w5']
        a = r['alphas_w5']

        ns_sust = find_nstar_sustained(c, a, threshold=1.0, run_length=3)
        ns_A = find_nstar_max_deriv(c, a)
        ns_B = find_nstar_intersection(c, a)

        r['ns_sust'] = ns_sust
        r['ns_A'] = ns_A
        r['ns_B'] = ns_B

        vals = [x for x in [ns_sust, ns_A, ns_B] if not np.isnan(x)]
        spread = max(vals) - min(vals) if len(vals) >= 2 else np.nan
        conv = "YES" if not np.isnan(spread) and spread <= 4.4 else ("NO" if not np.isnan(spread) else "N/A")

        s1 = f"{ns_sust:.1f}" if not np.isnan(ns_sust) else "none"
        s2 = f"{ns_A:.1f}" if not np.isnan(ns_A) else "none"
        s3 = f"{ns_B:.1f}" if not np.isnan(ns_B) else "none"
        sp = f"{spread:.1f}" if not np.isnan(spread) else "N/A"
        print(f"  {geo_name:<10} {spin:<6} | {s1:>8} | {s2:>8} | {s3:>8} | {sp:>7} | {conv:>10}")

print(f"\n  Xiang comparison: spread = 4.4 steps")


# ============================================================
# TASK 4: Window width sensitivity
# ============================================================
print(f"\n{'='*80}")
print("TASK 4: WINDOW WIDTH SENSITIVITY")
print(f"{'='*80}")

widths = [3, 5, 7, 9]
print(f"\n  n*_sust (>1.0, 3 consec) for each window width:")
print(f"  {'Width':>5} | {'Q3-Q4 dn':>10} | {'Q3-Q4 up':>10} | {'Q2-Q3 dn':>10} | {'Q2-Q3 up':>10}")
print(f"  {'-'*55}")

nstar_sensitivity = {}
for geo_name in ['Q3-Q4', 'Q2-Q3']:
    nstar_sensitivity[geo_name] = {}
    for spin in ['down', 'up']:
        nstar_sensitivity[geo_name][spin] = {}

for w in widths:
    row = []
    for geo_name in ['Q3-Q4', 'Q2-Q3']:
        for spin in ['down', 'up']:
            r = results[geo_name][spin]
            c_w, a_w, _ = sliding_alpha(r['idx'], r['C'], width=w)
            ns = find_nstar_sustained(c_w, a_w, threshold=1.0, run_length=3)
            nstar_sensitivity[geo_name][spin][w] = ns
            row.append(f"{ns:.0f}" if not np.isnan(ns) else "none")
    print(f"  {w:5d} | {row[0]:>10} | {row[1]:>10} | {row[2]:>10} | {row[3]:>10}")

# Range analysis
print(f"\n  Range across widths:")
for geo_name in ['Q3-Q4', 'Q2-Q3']:
    for spin in ['down', 'up']:
        vals = [nstar_sensitivity[geo_name][spin][w] for w in widths
                if not np.isnan(nstar_sensitivity[geo_name][spin][w])]
        if len(vals) >= 2:
            rng = max(vals) - min(vals)
            stable = "STABLE" if rng < 9 else "UNSTABLE"
            print(f"    {geo_name} {spin}: [{min(vals):.0f}, {max(vals):.0f}], range = {rng:.0f} -> {stable}")
        else:
            print(f"    {geo_name} {spin}: insufficient n* values")

print(f"\n  Xiang comparison: range = 9 (too wide to be informative)")


# ============================================================
# TASK 5: Piecewise model with 95% CI (bootstrap)
# ============================================================
print(f"\n{'='*80}")
print("TASK 5: PIECEWISE MODEL WITH 95% CI (1000 bootstrap resamples)")
print(f"{'='*80}")

print(f"\n  {'Geometry':<10} {'Spin':<6} | {'n*_pw':>6} | {'95% CI':>16} | {'Width':>6} | {'F':>8} | {'p-value':>10} | {'Sig?':>5}")
print(f"  {'-'*80}")

for geo_name in ['Q3-Q4', 'Q2-Q3']:
    for spin in ['down', 'up']:
        r = results[geo_name][spin]
        ns_pw, ci, F, p, params, sse_f = fit_piecewise(r['centers_w5'], r['alphas_w5'], n_boot=1000)
        r['pw_nstar'] = ns_pw
        r['pw_ci'] = ci
        r['pw_F'] = F
        r['pw_p'] = p
        r['pw_params'] = params

        ci_w = ci[1] - ci[0] if not np.isnan(ci[0]) else np.nan
        ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if not np.isnan(ci[0]) else "N/A"
        w_str = f"{ci_w:.1f}" if not np.isnan(ci_w) else "N/A"
        F_str = f"{F:.2f}" if not np.isnan(F) else "N/A"
        p_str = f"{p:.6f}" if not np.isnan(p) else "N/A"
        sig = "YES" if not np.isnan(p) and p < 0.05 else "NO"
        ns_str = f"{ns_pw:.1f}" if not np.isnan(ns_pw) else "N/A"

        print(f"  {geo_name:<10} {spin:<6} | {ns_str:>6} | {ci_str:>16} | {w_str:>6} | {F_str:>8} | {p_str:>10} | {sig:>5}")

print(f"\n  Xiang comparison: 95% CI = [6.0, 15.0], width = 9.0")


# ============================================================
# TASK 6: Null model (10,000 permutation trials)
# ============================================================
print(f"\n{'='*80}")
print("TASK 6: NULL MODEL (10,000 PERMUTATION TRIALS)")
print(f"{'='*80}")

# Use Q3-Q4 SpinDown as reference for noise estimation
ref = results['Q3-Q4']['down']
N_pts = len(ref['C'])
idx_ref = ref['idx']

try:
    popt, _ = curve_fit(kww, idx_ref[1:], ref['C'][1:],
                        p0=[ref['tau'], ref['alpha']],
                        bounds=([0.1, 0.01], [50000, 10]), maxfev=30000)
    resid = ref['C'][1:] - kww(idx_ref[1:], *popt)
    noise_sigma = np.std(resid)
except Exception:
    noise_sigma = 0.02

print(f"\n  Noise (Q3-Q4 dn residuals): sigma = {noise_sigma:.4f}")
print(f"  N_points = {N_pts}")
print(f"  Generating 10,000 synthetic KWW curves...")
print(f"  alpha ~ U[0.8, 1.4], tau ~ U[5, {N_pts*0.5:.0f}]")

n_null = 10000
null_nstars = []
n_synth = np.arange(N_pts, dtype=float)

for trial in range(n_null):
    alpha_true = np.random.uniform(0.8, 1.4)
    tau_true = np.random.uniform(5, N_pts * 0.5)
    C_true = kww(n_synth, tau_true, alpha_true)
    C_noisy = C_true + np.random.normal(0, noise_sigma, N_pts)
    C_noisy = np.clip(C_noisy, 0.001, 2.0)
    C_noisy[0] = 1.0
    c_s, a_s, _ = sliding_alpha(n_synth, C_noisy, width=5)
    ns_s = find_nstar_sustained(c_s, a_s, threshold=1.0, run_length=3)
    null_nstars.append(ns_s)

null_nstars = np.array(null_nstars)
null_valid = ~np.isnan(null_nstars)
null_found = null_nstars[null_valid]

print(f"\n  Null model results:")
print(f"    Trials with sustained crossover: {np.sum(null_valid)}/{n_null} ({100*np.sum(null_valid)/n_null:.1f}%)")

if len(null_found) > 0:
    print(f"    Mean n* (when found): {np.mean(null_found):.1f} +/- {np.std(null_found):.1f}")
    print(f"    Median n*: {np.median(null_found):.1f}")

    # Where does each dataset fall?
    for geo_name in ['Q3-Q4', 'Q2-Q3']:
        for spin in ['down', 'up']:
            ns_val = results[geo_name][spin]['ns_sust']
            if not np.isnan(ns_val):
                pctile = 100 * np.sum(null_found <= ns_val) / len(null_found)
                results[geo_name][spin]['null_pctile'] = pctile
                unusual = "HIGHLY UNUSUAL" if pctile < 5 else ("UNUSUAL" if pctile < 25 else ("TYPICAL" if pctile < 75 else "LATE"))
                print(f"    {geo_name} {spin}: n*={ns_val:.0f}, null percentile = {pctile:.1f}% -> {unusual}")
            else:
                results[geo_name][spin]['null_pctile'] = np.nan
                print(f"    {geo_name} {spin}: no sustained crossover found")

print(f"\n  Xiang comparison: 85th percentile (not significant)")


# ============================================================
# TASK 7: Spin-resolved crossover comparison
# ============================================================
print(f"\n{'='*80}")
print("TASK 7: SPIN-RESOLVED CROSSOVER COMPARISON")
print(f"{'='*80}")

print(f"\n  Merkabit prediction: spin-down (stable in Ge holes) should show")
print(f"  sharper crossover than spin-up (unstable mode).")

for geo_name in ['Q3-Q4', 'Q2-Q3']:
    print(f"\n  {geo_name}:")
    print(f"    {'Metric':<25} | {'Spin-down':>12} | {'Spin-up':>12} | {'Match?':>8}")
    print(f"    {'-'*65}")

    rd = results[geo_name]['down']
    ru = results[geo_name]['up']

    for label, key, fmt in [('Full alpha', 'alpha', '.4f'),
                             ('Full tau', 'tau', '.1f'),
                             ('Full R2', 'R2', '.6f')]:
        vd = rd[key]
        vu = ru[key]
        match = "YES" if abs(vd - vu) / max(abs(vd), abs(vu), 0.001) < 0.15 else "NO"
        print(f"    {label:<25} | {vd:>12{fmt}} | {vu:>12{fmt}} | {match:>8}")

    for label, key in [('n*_sust', 'ns_sust'), ('n*_A (max deriv)', 'ns_A'),
                       ('n*_B (intersection)', 'ns_B'), ('Piecewise n*', 'pw_nstar')]:
        vd = rd.get(key, np.nan)
        vu = ru.get(key, np.nan)
        vd_s = f"{vd:.1f}" if not np.isnan(vd) else "none"
        vu_s = f"{vu:.1f}" if not np.isnan(vu) else "none"
        if not np.isnan(vd) and not np.isnan(vu):
            diff = abs(vd - vu)
            match = "YES" if diff <= 3 else f"d={diff:.0f}"
        else:
            match = "N/A"
        print(f"    {label:<25} | {vd_s:>12} | {vu_s:>12} | {match:>8}")

    # Sharper = narrower CI
    ci_d = rd.get('pw_ci', (np.nan, np.nan))
    ci_u = ru.get('pw_ci', (np.nan, np.nan))
    cw_d = ci_d[1] - ci_d[0] if not np.isnan(ci_d[0]) else np.nan
    cw_u = ci_u[1] - ci_u[0] if not np.isnan(ci_u[0]) else np.nan
    if not np.isnan(cw_d) and not np.isnan(cw_u):
        sharper = "DOWN" if cw_d < cw_u else "UP"
        print(f"    {'CI width (sharper?)':<25} | {cw_d:>12.1f} | {cw_u:>12.1f} | {sharper:>8}")
    else:
        print(f"    {'CI width (sharper?)':<25} | {'N/A':>12} | {'N/A':>12} | {'N/A':>8}")


# ============================================================
# TASK 8: Direct comparison table + Verdict
# ============================================================
print(f"\n{'='*80}")
print("TASK 8: DIRECT COMPARISON TABLE")
print(f"{'='*80}")

# Use spin-down as primary for each geometry
rq34 = results['Q3-Q4']['down']
rq23 = results['Q2-Q3']['down']

print(f"\n  {'Metric':<30} | {'Q3-Q4':>15} | {'Q2-Q3 (ctrl)':>15} | {'Xiang DTC':>15}")
print(f"  {'-'*80}")

# alpha_local at n=11
a11_34 = rq34.get('alpha_at_11', np.nan)
a11_23 = rq23.get('alpha_at_11', np.nan)
a34_s = f"{a11_34:.4f}" if not np.isnan(a11_34) else "N/A"
a23_s = f"{a11_23:.4f}" if not np.isnan(a11_23) else "N/A"
print(f"  {'alpha_local at n=11':<30} | {a34_s:>15} | {a23_s:>15} | {'N/A':>15}")

# log_11(12) hit?
hit_34 = abs(a11_34 - LOG11_12) < 0.05 if not np.isnan(a11_34) else False
hit_23 = abs(a11_23 - LOG11_12) < 0.05 if not np.isnan(a11_23) else False
hit_34_s = "YES" if hit_34 else "NO"
hit_23_s = "YES" if hit_23 else "NO"
print(f"  {'log_11(12)=1.036: hit?':<30} | {hit_34_s:>15} | {hit_23_s:>15} | {'N/A':>15}")

# n* (definition A = max deriv)
ns_A_34 = rq34.get('ns_A', np.nan)
ns_A_23 = rq23.get('ns_A', np.nan)
nA34_s = f"{ns_A_34:.1f}" if not np.isnan(ns_A_34) else "N/A"
nA23_s = f"{ns_A_23:.1f}" if not np.isnan(ns_A_23) else "N/A"
print(f"  {'n* (max deriv)':<30} | {nA34_s:>15} | {nA23_s:>15} | {'~12':>15}")

# 95% CI width
ci_34 = rq34.get('pw_ci', (np.nan, np.nan))
ci_23 = rq23.get('pw_ci', (np.nan, np.nan))
cw_34 = ci_34[1] - ci_34[0] if not np.isnan(ci_34[0]) else np.nan
cw_23 = ci_23[1] - ci_23[0] if not np.isnan(ci_23[0]) else np.nan
cw34_s = f"{cw_34:.1f}" if not np.isnan(cw_34) else "N/A"
cw23_s = f"{cw_23:.1f}" if not np.isnan(cw_23) else "N/A"
print(f"  {'95% CI width':<30} | {cw34_s:>15} | {cw23_s:>15} | {'9':>15}")

# Null percentile
np_34 = rq34.get('null_pctile', np.nan)
np_23 = rq23.get('null_pctile', np.nan)
np34_s = f"{np_34:.0f}th" if not np.isnan(np_34) else "N/A"
np23_s = f"{np_23:.0f}th" if not np.isnan(np_23) else "N/A"
print(f"  {'Null percentile':<30} | {np34_s:>15} | {np23_s:>15} | {'85th':>15}")

# Full KWW alpha
print(f"  {'Full KWW alpha':<30} | {rq34['alpha']:>15.4f} | {rq23['alpha']:>15.4f} | {'~1.07':>15}")
print(f"  {'Full KWW R2':<30} | {rq34['R2']:>15.6f} | {rq23['R2']:>15.6f} | {'~0.99':>15}")

# Crossover sharper than Xiang?
sharper_34 = "YES" if not np.isnan(cw_34) and cw_34 < 9.0 else "NO"
sharper_23 = "YES" if not np.isnan(cw_23) and cw_23 < 9.0 else "NO"
print(f"  {'Crossover sharper than Xiang?':<30} | {sharper_34:>15} | {sharper_23:>15} | {'baseline':>15}")


# ============================================================
# BINARY VERDICT
# ============================================================
print(f"\n{'='*80}")
print("BINARY VERDICT")
print(f"{'='*80}")

print(f"\n  KEY NUMBERS:")
print(f"    alpha_local(n=11) for Q3-Q4:  {a34_s}")
print(f"    log_11(12) = {LOG11_12:.4f} -- measured within +/-0.05?  {hit_34_s}")
ns_pw_34 = rq34.get('pw_nstar', np.nan)
ns_pw_s = f"{ns_pw_34:.1f}" if not np.isnan(ns_pw_34) else "N/A"
print(f"    n* point estimate for Q3-Q4:  {ns_pw_s}")
ci_s = f"[{ci_34[0]:.1f}, {ci_34[1]:.1f}]" if not np.isnan(ci_34[0]) else "N/A"
print(f"    95% CI for n* (Q3-Q4):        {ci_s}, width = {cw34_s}")
print(f"    Null model percentile:        {np34_s}")
print(f"    Same for Q2-Q3 control:       alpha_11={a23_s}, CI_width={cw23_s}, null={np23_s}")

# Conditions
cond_a = hit_34
cond_b = not np.isnan(cw_34) and cw_34 < 9.0
cond_c = not np.isnan(np_34) and np_34 > 90.0

print(f"\n  CONDITIONS:")
print(f"    (a) alpha_local(n=11) within 0.05 of 1.036?  {'PASS' if cond_a else 'FAIL'}")
print(f"    (b) Crossover CI narrower than Xiang's 9?    {'PASS' if cond_b else 'FAIL'}")
print(f"    (c) Null percentile above 90th?              {'PASS' if cond_c else 'FAIL'}")

verdict = cond_a and cond_b and cond_c
print(f"\n  VERDICT: {'YES' if verdict else 'NO'}")

if verdict:
    print(f"\n  Standing wave sensitivity argument supported.")
    print(f"  Pre-lock systems broad, near-lock systems sharp.")
    print(f"  Ladder of locks consistent with data.")
else:
    failed = []
    if not cond_a:
        failed.append(f"(a) alpha_local(n=11)={a34_s}, not within 0.05 of {LOG11_12:.4f}")
    if not cond_b:
        failed.append(f"(b) CI width={cw34_s}, not narrower than 9")
    if not cond_c:
        failed.append(f"(c) Null percentile={np34_s}, not above 90th")
    print(f"\n  Failed conditions:")
    for f in failed:
        print(f"    - {f}")
    print(f"\n  Interpretation: The data does not meet all three criteria for")
    print(f"  the Merkabit standing wave first-lock threshold at n=11.")

print(f"\n{'='*80}")
print("END OF ANALYSIS")
print(f"{'='*80}")
