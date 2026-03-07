"""
Multi-Dataset DTC Subharmonic Envelope Analysis
================================================
Dataset 1: Xiang et al. 2024 — Topological prethermal DTC (Google Sycamore)
Dataset 2: Randall et al. 2021 — MBL DTC (diamond NV centers)
Baseline:  Mi et al. 2022 — MBL DTC (Google Sycamore), alpha=0.822

Method: E(n) = (-1)^n * A(n), fit to A0*exp(-(n/n*)^alpha)
"""

import numpy as np
import json
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, io
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import XIANG_2024_BASE, RANDALL_2021_DATA, FIGURES_DIR, REPORTS_DIR

XBASE = XIANG_2024_BASE
RBASE = RANDALL_2021_DATA

ALPHA_MI_BASELINE = 0.822  # Mi et al. MBL DTC subharmonic

def stretched_exp(n, A0, n_star, alpha):
    return A0 * np.exp(-(n / n_star) ** alpha)

def fit_stretched(n_vals, env, label=''):
    mask = (env > 1e-4) & np.isfinite(env)
    n_f, e_f = n_vals[mask], env[mask]
    if len(n_f) < 4:
        return None
    try:
        popt, pcov = curve_fit(stretched_exp, n_f, e_f,
            p0=[e_f[0], len(n_f), 1.0],
            bounds=([0, 0.1, 0.01], [2.0, 50000, 10.0]), maxfev=100000)
        res = e_f - stretched_exp(n_f, *popt)
        ss_res, ss_tot = np.sum(res**2), np.sum((e_f - np.mean(e_f))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {'A0': popt[0], 'n_star': popt[1], 'alpha': popt[2], 'R2': R2,
                'A0_err': perr[0], 'n_star_err': perr[1], 'alpha_err': perr[2]}
    except:
        return None

def subharmonic_envelope(signal):
    """Extract subharmonic envelope: E(n) = (-1)^n * A(n)"""
    n = np.arange(len(signal))
    signs = (-1.0)**n
    return signal * signs

def report_fit(label, result, g_or_param=''):
    if result is None:
        print(f"  {label}: FIT FAILED")
        return
    da = result['alpha'] - ALPHA_MI_BASELINE
    tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
    print(f"  {tag} {label}:")
    print(f"      alpha = {result['alpha']:.4f} +/- {result['alpha_err']:.4f}  "
          f"[delta vs Mi = {da:+.4f}]")
    print(f"      n*    = {result['n_star']:.2f}, R^2 = {result['R2']:.6f}")


# ============================================================
# DATASET 1: XIANG ET AL. 2024 — TOPOLOGICAL DTC
# ============================================================

print("=" * 70)
print("DATASET 1: XIANG ET AL. 2024 — TOPOLOGICAL DTC")
print("Nat. Commun. 15, 8963 (2024)")
print("=" * 70)

xiang_results = {}

# --- fig2a: 6 Z operators x 41 half-cycles (20 full Floquet cycles) ---
d2a = sio.loadmat(os.path.join(XBASE, 'main_figure2', 'fig2a.mat'))
Zs = d2a['Zs_exp_avg']  # (6, 41)
Xs = d2a['Xs_exp_avg']  # (3, 41)

print("\n1. MAIN FIGURE 2: Z and X operators (41 half-cycles = 20 Floquet cycles)")
print("-" * 60)

# Half-cycle data: full Floquet cycles are at indices 0, 2, 4, ..., 40
# Extract full-cycle data and compute subharmonic envelope
n_half = np.arange(41)  # half-cycle indices
full_cycle_idx = np.arange(0, 41, 2)  # indices 0,2,4,...,40 = 21 full cycles
n_full = np.arange(len(full_cycle_idx))

print("\n  Z operators (6 total, likely: Z_local + Z_string/logical):")
for i in range(6):
    signal_full = Zs[i, full_cycle_idx]
    env = subharmonic_envelope(signal_full)

    result = fit_stretched(n_full[1:], env[1:])
    xiang_results[f'fig2a_Z{i}'] = result
    report_fit(f'Z operator {i} (21 full cycles)', result)

print("\n  X operators (3 total):")
for i in range(3):
    signal_full = Xs[i, full_cycle_idx]
    env = subharmonic_envelope(signal_full)
    result = fit_stretched(n_full[1:], env[1:])
    xiang_results[f'fig2a_X{i}'] = result
    report_fit(f'X operator {i}', result)

# --- figS14: Echo protocol (same 41 half-cycles) ---
print("\n\n2. ECHO PROTOCOL (figS14) — envelope without subharmonic")
print("-" * 60)

d14 = sio.loadmat(os.path.join(XBASE, 'supp_figure14', 'figS14.mat'))
Zs_echo = d14['Zs_echo_avg']  # (6, 41) — already has oscillation removed
Zs_raw = d14['Zs_exp_avg']    # (6, 41)

# Echo data is at half-cycle resolution but monotonically decreasing
# Group by full cycles: take every other point
for i in range(6):
    echo_full = Zs_echo[i, full_cycle_idx]
    result_echo = fit_stretched(n_full[1:], echo_full[1:])
    xiang_results[f'echo_Z{i}'] = result_echo
    report_fit(f'Echo Z{i} (envelope directly)', result_echo)

    # Also subharmonic of raw data for comparison
    raw_full = Zs_raw[i, full_cycle_idx]
    env_raw = subharmonic_envelope(raw_full)
    result_raw = fit_stretched(n_full[1:], env_raw[1:])
    xiang_results[f'raw_sub_Z{i}'] = result_raw
    report_fit(f'Raw subharm Z{i}', result_raw)

# --- figS3: LONG DATA — 100 half-cycles (50 full cycles) ---
print("\n\n3. SUPPLEMENTARY FIGURE 3: LONG DATA (100 half-cycles = 50 full cycles)")
print("-" * 60)

dS3 = sio.loadmat(os.path.join(XBASE, 'supp_figure3', 'figS3.mat'))
Zl = dS3['Zl_avg']  # (100, 6) — 6 Z logical operators over 100 half-cycles
Xl = dS3['Xl_avg']  # (100, 3) — 3 X logical operators
Z_all = dS3['Z_avg']  # (100, 18) — 18 individual qubits

full_idx_long = np.arange(0, 100, 2)  # 50 full Floquet cycles
n_full_long = np.arange(len(full_idx_long))

print("\n  Z logical operators (6 total, 50 full cycles):")
for i in range(6):
    signal_full = Zl[full_idx_long, i]
    env = subharmonic_envelope(signal_full)
    result = fit_stretched(n_full_long[1:], env[1:])
    xiang_results[f'figS3_Zl{i}'] = result
    report_fit(f'Z_logical {i} (50 cycles)', result)

print("\n  X logical operators (3 total, 50 full cycles):")
for i in range(3):
    signal_full = Xl[full_idx_long, i]
    env = subharmonic_envelope(signal_full)
    result = fit_stretched(n_full_long[1:], env[1:])
    xiang_results[f'figS3_Xl{i}'] = result
    report_fit(f'X_logical {i} (50 cycles)', result)

# Per-qubit analysis (18 qubits)
print("\n  Per-qubit Z (18 qubits, 50 full cycles):")
per_qubit_alphas_x = []
per_qubit_errs_x = []
for qi in range(18):
    signal_full = Z_all[full_idx_long, qi]
    env = subharmonic_envelope(signal_full)
    result = fit_stretched(n_full_long[1:], env[1:])
    if result and result['R2'] > 0.9:
        per_qubit_alphas_x.append(result['alpha'])
        per_qubit_errs_x.append(result['alpha_err'])
    else:
        per_qubit_alphas_x.append(np.nan)
        per_qubit_errs_x.append(np.nan)

per_qubit_alphas_x = np.array(per_qubit_alphas_x)
valid_x = ~np.isnan(per_qubit_alphas_x)

if valid_x.sum() > 0:
    mean_a = np.nanmean(per_qubit_alphas_x)
    std_a = np.nanstd(per_qubit_alphas_x)
    cv = std_a / mean_a * 100 if mean_a > 0 else 0
    in_range = np.sum((per_qubit_alphas_x[valid_x] >= 1.1) & (per_qubit_alphas_x[valid_x] <= 1.5))
    print(f"    Valid fits: {valid_x.sum()}/18")
    print(f"    Mean alpha = {mean_a:.4f} +/- {std_a:.4f}")
    print(f"    CV = {cv:.1f}%")
    print(f"    Range: [{np.nanmin(per_qubit_alphas_x):.4f}, {np.nanmax(per_qubit_alphas_x):.4f}]")
    print(f"    In Merkabit range [1.1, 1.5]: {in_range}/{valid_x.sum()}")
    print(f"    delta_alpha vs Mi baseline: {mean_a - ALPHA_MI_BASELINE:+.4f}")

    for qi in range(18):
        if valid_x[qi]:
            tag = '*' if 1.1 <= per_qubit_alphas_x[qi] <= 1.5 else ' '
            print(f"      {tag} Q{qi+1:2d}: alpha = {per_qubit_alphas_x[qi]:.4f}")
    xiang_results['per_qubit_mean'] = mean_a
    xiang_results['per_qubit_std'] = std_a
    xiang_results['per_qubit_cv'] = cv

# --- figS4: System size scaling ---
print("\n\n4. SYSTEM SIZE SCALING (figS4)")
print("-" * 60)

dS4 = sio.loadmat(os.path.join(XBASE, 'supp_figure4', 'figS4.mat'))
Ns = dS4['Ns'].flatten()  # [6, 9, 12, 15]
Zl1 = dS4['Zl1_avg']  # (4, 100) — 4 system sizes, 100 half-cycle data

for idx, N in enumerate(Ns):
    signal_half = Zl1[idx]
    signal_full = signal_half[full_idx_long]
    env = subharmonic_envelope(signal_full)
    result = fit_stretched(n_full_long[1:], env[1:])
    xiang_results[f'syssize_N{N}'] = result
    report_fit(f'N={N} qubits', result)

# --- fig4a: Disorder sweep (17 disorder values) ---
print("\n\n5. DISORDER SWEEP (fig4a: 17 disorder values, 41 half-cycles)")
print("-" * 60)

d4a = sio.loadmat(os.path.join(XBASE, 'main_figure4', 'fig4a.mat'))
disorders = d4a['disorders'].flatten()  # 17 values: 0 to 3
Zs_dis = d4a['Zs_exp_avg']  # (6, 17, 41)

# Use Z operator 0 (first, likely primary)
disorder_alphas = []
disorder_errs = []
print("\n  Z operator 0 across disorder strengths:")
for di, dis in enumerate(disorders):
    signal_full = Zs_dis[0, di, full_cycle_idx]
    env = subharmonic_envelope(signal_full)
    result = fit_stretched(n_full[1:], env[1:])
    if result and result['R2'] > 0.9:
        disorder_alphas.append(result['alpha'])
        disorder_errs.append(result['alpha_err'])
        tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
        print(f"    {tag} disorder={dis:.1f}: alpha={result['alpha']:.4f}+/-{result['alpha_err']:.4f}, "
              f"n*={result['n_star']:.1f}, R2={result['R2']:.4f}")
    else:
        disorder_alphas.append(np.nan)
        disorder_errs.append(np.nan)
        print(f"      disorder={dis:.1f}: fit failed or R2<0.9")

    xiang_results[f'disorder_{dis:.1f}'] = result

disorder_alphas = np.array(disorder_alphas)
disorder_errs = np.array(disorder_errs)

# Average over all 6 Z operators for each disorder
print("\n  Averaged over all 6 Z operators:")
for di, dis in enumerate(disorders):
    alphas_ops = []
    for oi in range(6):
        signal_full = Zs_dis[oi, di, full_cycle_idx]
        env = subharmonic_envelope(signal_full)
        result = fit_stretched(n_full[1:], env[1:])
        if result and result['R2'] > 0.9:
            alphas_ops.append(result['alpha'])
    if alphas_ops:
        mean_op = np.mean(alphas_ops)
        tag = '*' if 1.1 <= mean_op <= 1.5 else ' '
        print(f"    {tag} disorder={dis:.1f}: mean alpha={mean_op:.4f} ({len(alphas_ops)}/6 ops)")


# ============================================================
# DATASET 2: RANDALL ET AL. 2021 — NV CENTER MBL DTC
# ============================================================

print("\n\n" + "=" * 70)
print("DATASET 2: RANDALL ET AL. 2021 — NV CENTER MBL DTC")
print("Science 374, 1474 (2021)")
print("=" * 70)

randall_results = {}

# --- Averaged autocorrelator (Neel initial state) ---
with open(os.path.join(RBASE, 'correlation_data_Neel.json')) as f:
    cdn = json.load(f)

n_cycles_r = np.array(cdn[0])  # [0, 1, 2, ..., 100]
A_avg = np.array(cdn[1])       # averaged autocorrelator
A_err = np.array(cdn[2])       # errors

print("\n1. AVERAGED AUTOCORRELATOR (Neel initial state, 101 cycles)")
print("-" * 60)

env = subharmonic_envelope(A_avg)
result = fit_stretched(n_cycles_r[1:], env[1:])
randall_results['avg_neel'] = result
report_fit('Averaged Neel autocorrelator', result)

# Even/odd branch check
even_idx = np.arange(0, len(A_avg), 2)
odd_idx = np.arange(1, len(A_avg), 2)

result_even = fit_stretched(even_idx[1:], A_avg[even_idx[1:]])
result_odd = fit_stretched(odd_idx, -A_avg[odd_idx])
randall_results['even_branch'] = result_even
randall_results['odd_branch'] = result_odd
report_fit('Even-cycle branch', result_even)
report_fit('Odd-cycle branch', result_odd)

# --- Per-spin autocorrelator ---
print("\n\n2. PER-SPIN SUBHARMONIC ENVELOPE (9 NV spins, 101 cycles)")
print("-" * 60)

with open(os.path.join(RBASE, 'individual_spins_Neel.json')) as f:
    isn = json.load(f)

per_spin_alphas = []
per_spin_errs = []
n_spins_r = np.arange(101)

for si in range(9):
    signal = np.array(isn[si])
    env = subharmonic_envelope(signal)
    result = fit_stretched(n_spins_r[1:], env[1:])
    if result and result['R2'] > 0.9:
        per_spin_alphas.append(result['alpha'])
        per_spin_errs.append(result['alpha_err'])
        tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
        print(f"  {tag} Spin {si+1}: alpha={result['alpha']:.4f}+/-{result['alpha_err']:.4f}, "
              f"n*={result['n_star']:.1f}, R2={result['R2']:.4f}")
    else:
        per_spin_alphas.append(np.nan)
        per_spin_errs.append(np.nan)
        print(f"    Spin {si+1}: fit failed or R2<0.9")

    randall_results[f'spin{si+1}'] = result

per_spin_alphas = np.array(per_spin_alphas)
valid_r = ~np.isnan(per_spin_alphas)

if valid_r.sum() > 0:
    mean_r = np.nanmean(per_spin_alphas)
    std_r = np.nanstd(per_spin_alphas)
    cv_r = std_r / mean_r * 100 if mean_r > 0 else 0
    in_range_r = np.sum((per_spin_alphas[valid_r] >= 1.1) & (per_spin_alphas[valid_r] <= 1.5))
    print(f"\n  Mean alpha = {mean_r:.4f} +/- {std_r:.4f}")
    print(f"  CV = {cv_r:.1f}%")
    print(f"  In Merkabit range [1.1, 1.5]: {in_range_r}/{valid_r.sum()}")
    print(f"  delta_alpha vs Mi baseline: {mean_r - ALPHA_MI_BASELINE:+.4f}")
    randall_results['per_spin_mean'] = mean_r
    randall_results['per_spin_std'] = std_r

# --- Per-spin at different tau values ---
print("\n\n3. PER-SPIN AT tau=7.5us (different Floquet period)")
print("-" * 60)

with open(os.path.join(RBASE, 'individual_spins_7pt5_Neel.json')) as f:
    is7 = json.load(f)

per_spin_7_alphas = []
for si in range(9):
    signal = np.array(is7[si])
    env = subharmonic_envelope(signal)
    result = fit_stretched(np.arange(1, len(signal)), env[1:])
    if result and result['R2'] > 0.9:
        per_spin_7_alphas.append(result['alpha'])
        tag = '*' if 1.1 <= result['alpha'] <= 1.5 else ' '
        print(f"  {tag} Spin {si+1}: alpha={result['alpha']:.4f}+/-{result['alpha_err']:.4f}, "
              f"R2={result['R2']:.4f}")
        randall_results[f'spin{si+1}_7pt5'] = result
    else:
        per_spin_7_alphas.append(np.nan)

# --- Tilted initial state ---
print("\n\n4. TILTED INITIAL STATE (Z component, 50 cycles)")
print("-" * 60)

with open(os.path.join(RBASE, 'tilted_state_data.json')) as f:
    tsd = json.load(f)

n_tilt = np.array(tsd['Z'][0])  # time axis
A_tilt = np.array(tsd['Z'][1])  # autocorrelator
env_tilt = subharmonic_envelope(A_tilt)
result_tilt = fit_stretched(n_tilt[1:].astype(float), env_tilt[1:])
randall_results['tilted_Z'] = result_tilt
report_fit('Tilted Z (50 cycles)', result_tilt)

# XY component
A_tilt_xy = np.array(tsd['XY'][1])
result_xy = fit_stretched(n_tilt[1:].astype(float), A_tilt_xy[1:])
randall_results['tilted_XY'] = result_xy
report_fit('Tilted XY (non-oscillating)', result_xy)

# --- Decay histogram ---
print("\n\n5. DECAY TIME HISTOGRAM (10000 bootstrap samples)")
print("-" * 60)

with open(os.path.join(RBASE, 'decay_hist.json')) as f:
    decay_times = np.array(json.load(f))

print(f"  N = {len(decay_times)}")
print(f"  Mean decay time = {np.mean(decay_times):.2f}")
print(f"  Std decay time = {np.std(decay_times):.2f}")
print(f"  Median = {np.median(decay_times):.2f}")
print(f"  (These are n* values from 10000 bootstrap fits)")
randall_results['decay_hist_mean'] = np.mean(decay_times)
randall_results['decay_hist_std'] = np.std(decay_times)

# --- DTC_fit (envelope fit from the paper) ---
with open(os.path.join(RBASE, 'DTC_fit.json')) as f:
    dtc_fit = json.load(f)
print(f"\n  Paper's DTC fit curve: {len(dtc_fit[0])} points")
print(f"  x range: {dtc_fit[0][0]:.1f} to {dtc_fit[0][-1]:.1f}")
print(f"  y range: {dtc_fit[1][0]:.4f} to {dtc_fit[1][-1]:.4f}")
# This is the paper's own envelope fit — we can fit alpha to it
x_fit = np.array(dtc_fit[0])
y_fit = np.array(dtc_fit[1])
result_paper = fit_stretched(x_fit[x_fit > 0], y_fit[x_fit > 0])
randall_results['paper_fit_curve'] = result_paper
report_fit('Paper envelope fit curve', result_paper)

# --- X-basis measurements ---
print("\n\n6. X-BASIS MEASUREMENTS (non-DTC observable)")
print("-" * 60)

with open(os.path.join(RBASE, 'xbasis_data.json')) as f:
    xb = json.load(f)
for key in xb:
    n_xb = np.array(xb[key][0])
    A_xb = np.array(xb[key][1])
    env_xb = subharmonic_envelope(A_xb)
    result_xb = fit_stretched(n_xb[1:].astype(float), env_xb[1:])
    randall_results[f'xbasis_{key}'] = result_xb
    report_fit(f'X-basis config {key}', result_xb)


# ============================================================
# CROSS-DATASET COMPARISON
# ============================================================

print("\n\n" + "=" * 70)
print("CROSS-DATASET COMPARISON: SUBHARMONIC ENVELOPE ALPHA")
print("=" * 70)

print(f"\n{'Dataset':<45} {'alpha':>8} {'+-err':>8} {'delta':>8} {'R2':>8}")
print("-" * 85)

# Mi et al. baseline
print(f"{'Mi et al. 2022 MBL DTC (g=0.97)':<45} {0.822:>8.4f} {'0.006':>8} {'---':>8} {'0.999':>8}")

# Xiang best results
for key, r in sorted(xiang_results.items()):
    if isinstance(r, dict) and r is not None and 'alpha' in r and r['R2'] > 0.99:
        da = r['alpha'] - ALPHA_MI_BASELINE
        tag = '*' if 1.1 <= r['alpha'] <= 1.5 else ' '
        print(f"{tag}Xiang: {key:<41} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {da:>+8.4f} {r['R2']:>8.4f}")

# Randall best results
for key, r in sorted(randall_results.items()):
    if isinstance(r, dict) and r is not None and 'alpha' in r and r['R2'] > 0.99:
        da = r['alpha'] - ALPHA_MI_BASELINE
        tag = '*' if 1.1 <= r['alpha'] <= 1.5 else ' '
        print(f"{tag}Randall: {key:<39} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {da:>+8.4f} {r['R2']:>8.4f}")

# Lower R2 threshold for completeness
print(f"\n  --- Results with R2 > 0.95 (lower threshold) ---")
for key, r in sorted(xiang_results.items()):
    if isinstance(r, dict) and r is not None and 'alpha' in r:
        if 0.95 < r['R2'] <= 0.99:
            da = r['alpha'] - ALPHA_MI_BASELINE
            tag = '*' if 1.1 <= r['alpha'] <= 1.5 else ' '
            print(f"  {tag}Xiang: {key:<39} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {da:>+8.4f} {r['R2']:>8.4f}")

for key, r in sorted(randall_results.items()):
    if isinstance(r, dict) and r is not None and 'alpha' in r:
        if 0.95 < r['R2'] <= 0.99:
            da = r['alpha'] - ALPHA_MI_BASELINE
            tag = '*' if 1.1 <= r['alpha'] <= 1.5 else ' '
            print(f"  {tag}Randall: {key:<37} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {da:>+8.4f} {r['R2']:>8.4f}")


# ============================================================
# THE TREND QUESTION
# ============================================================

print("\n\n" + "=" * 70)
print("THE TREND QUESTION: Does alpha increase away from MBL?")
print("=" * 70)

# Collect best alpha from each dataset
best_mi = 0.822
best_xiang = None
best_randall = None

# Xiang: best from figS3 long data (logical operators)
for i in range(6):
    k = f'figS3_Zl{i}'
    if k in xiang_results and xiang_results[k] is not None and xiang_results[k]['R2'] > 0.95:
        if best_xiang is None or xiang_results[k]['R2'] > best_xiang['R2']:
            best_xiang = xiang_results[k]
            best_xiang['key'] = k

# Randall: best from averaged
if randall_results.get('avg_neel') and randall_results['avg_neel']['R2'] > 0.95:
    best_randall = randall_results['avg_neel']
    best_randall['key'] = 'avg_neel'

print(f"\n  Mi et al. 2022 (MBL DTC, superconducting):     alpha = {best_mi:.4f}")
if best_xiang:
    print(f"  Xiang et al. 2024 (topological DTC, SC):        alpha = {best_xiang['alpha']:.4f} "
          f"[{best_xiang['key']}, R2={best_xiang['R2']:.4f}]")
if best_randall:
    print(f"  Randall et al. 2021 (MBL DTC, NV centers):      alpha = {best_randall['alpha']:.4f} "
          f"[{best_randall['key']}, R2={best_randall['R2']:.4f}]")
print(f"  Merkabit threshold:                              alpha ~ 1.3")


# ============================================================
# PUBLICATION FIGURE
# ============================================================

fig = plt.figure(figsize=(22, 18))

# Panel A: Xiang raw + subharmonic + fit
ax = fig.add_subplot(3, 3, 1)
signal = Zs[0]  # Z operator 0, half-cycle data
ax.plot(np.arange(41)*0.5, signal, 'o', markersize=3, color='gray', alpha=0.5, label='Raw Z0')
signal_fc = signal[full_cycle_idx]
env_fc = subharmonic_envelope(signal_fc)
ax.plot(n_full, env_fc, 's', markersize=4, color='#2ca02c', label='Subharm env')
r = xiang_results.get('fig2a_Z0')
if r and r['R2'] > 0.5:
    n_d = np.linspace(1, 20, 100)
    ax.plot(n_d, stretched_exp(n_d, r['A0'], r['n_star'], r['alpha']),
           '-', color='#2ca02c', linewidth=2,
           label=f'alpha={r["alpha"]:.3f}')
ax.set_xlabel('Floquet cycles')
ax.set_ylabel('Signal')
ax.set_title('A. Xiang: Z0 (20 cycles)', fontweight='bold')
ax.legend(fontsize=8)

# Panel B: Xiang long data (figS3)
ax = fig.add_subplot(3, 3, 2)
for i in range(min(3, 6)):
    signal_fc_l = Zl[full_idx_long, i]
    env_l = subharmonic_envelope(signal_fc_l)
    ax.plot(n_full_long, env_l, 'o-', markersize=2, alpha=0.6, label=f'Zl{i}')
    r = xiang_results.get(f'figS3_Zl{i}')
    if r and r['R2'] > 0.9:
        n_d = np.linspace(1, 49, 200)
        ax.plot(n_d, stretched_exp(n_d, r['A0'], r['n_star'], r['alpha']),
               '--', linewidth=2, label=f'  a={r["alpha"]:.2f}')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
ax.set_xlabel('Floquet cycles')
ax.set_ylabel('Subharm envelope')
ax.set_title('B. Xiang: Z_logical (50 cycles)', fontweight='bold')
ax.legend(fontsize=6, ncol=2)

# Panel C: Xiang echo
ax = fig.add_subplot(3, 3, 3)
for i in range(min(3, 6)):
    echo_fc = Zs_echo[i, full_cycle_idx]
    ax.plot(n_full, echo_fc, 'o-', markersize=3, alpha=0.6)
    r = xiang_results.get(f'echo_Z{i}')
    if r and r['R2'] > 0.9:
        n_d = np.linspace(1, 20, 100)
        ax.plot(n_d, stretched_exp(n_d, r['A0'], r['n_star'], r['alpha']),
               '--', linewidth=2, label=f'Z{i}: a={r["alpha"]:.2f}')
ax.set_xlabel('Floquet cycles')
ax.set_ylabel('Echo envelope')
ax.set_title('C. Xiang: Echo Protocol', fontweight='bold')
ax.legend(fontsize=8)

# Panel D: Randall raw + subharmonic
ax = fig.add_subplot(3, 3, 4)
ax.plot(n_cycles_r, A_avg, 'o', markersize=2, color='gray', alpha=0.4, label='Raw')
env_r = subharmonic_envelope(A_avg)
ax.plot(n_cycles_r, env_r, 's', markersize=2, color='#9467bd', alpha=0.7, label='Subharm env')
r = randall_results.get('avg_neel')
if r and r['R2'] > 0.5:
    n_d = np.linspace(1, 100, 200)
    ax.plot(n_d, stretched_exp(n_d, r['A0'], r['n_star'], r['alpha']),
           '-', color='#9467bd', linewidth=2.5,
           label=f'alpha={r["alpha"]:.3f}')
ax.set_xlabel('Floquet cycles')
ax.set_ylabel('Signal')
ax.set_title('D. Randall: NV Center DTC (101 cycles)', fontweight='bold')
ax.legend(fontsize=8)

# Panel E: Randall per-spin
ax = fig.add_subplot(3, 3, 5)
for si in range(9):
    signal_s = np.array(isn[si])
    env_s = subharmonic_envelope(signal_s)
    ax.plot(np.arange(101), env_s, '-', alpha=0.5, linewidth=0.8)
ax.plot(n_cycles_r, env_r, 'k-', linewidth=2, label='Average')
ax.set_xlabel('Floquet cycles')
ax.set_ylabel('Subharm envelope')
ax.set_title('E. Randall: Per-Spin Envelopes', fontweight='bold')
ax.legend(fontsize=8)

# Panel F: Per-spin alpha bar chart
ax = fig.add_subplot(3, 3, 6)
valid_spins = [i for i in range(9) if not np.isnan(per_spin_alphas[i]) if i < len(per_spin_alphas)]
# Use the Randall per-spin results
rpa = np.array(per_spin_alphas)[:9] if len(per_spin_alphas) >= 9 else per_spin_alphas
valid_r_idx = ~np.isnan(rpa)
if valid_r_idx.sum() > 0:
    ax.bar(np.arange(1, 10)[valid_r_idx], rpa[valid_r_idx],
           color='#9467bd', edgecolor='black', alpha=0.7)
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='alpha=1.3')
    ax.axhline(y=ALPHA_MI_BASELINE, color='orange', linestyle=':', linewidth=2,
               label=f'Mi baseline={ALPHA_MI_BASELINE}')
    ax.set_xlabel('Spin index')
    ax.set_ylabel('alpha')
    ax.set_title('F. Randall: Per-Spin Alpha', fontweight='bold')
    ax.legend(fontsize=8)

# Panel G: Xiang per-qubit
ax = fig.add_subplot(3, 3, 7)
valid_xq = ~np.isnan(per_qubit_alphas_x)
if valid_xq.sum() > 0:
    ax.bar(np.arange(1, 19)[valid_xq], per_qubit_alphas_x[valid_xq],
           color='#2ca02c', edgecolor='black', alpha=0.7)
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='alpha=1.3')
    ax.axhline(y=ALPHA_MI_BASELINE, color='orange', linestyle=':', linewidth=2,
               label=f'Mi baseline={ALPHA_MI_BASELINE}')
    ax.set_xlabel('Qubit index')
    ax.set_ylabel('alpha')
    ax.set_title('G. Xiang: Per-Qubit Alpha', fontweight='bold')
    ax.legend(fontsize=8)

# Panel H: Xiang disorder sweep
ax = fig.add_subplot(3, 3, 8)
valid_dis = ~np.isnan(disorder_alphas)
if valid_dis.sum() > 0:
    ax.errorbar(disorders[valid_dis], disorder_alphas[valid_dis],
               yerr=disorder_errs[valid_dis], fmt='o-', markersize=6, capsize=4,
               color='#2ca02c')
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='alpha=1.3')
    ax.axhline(y=ALPHA_MI_BASELINE, color='orange', linestyle=':', linewidth=2,
               label=f'Mi baseline')
    ax.set_xlabel('Disorder strength')
    ax.set_ylabel('alpha')
    ax.set_title('H. Xiang: Alpha vs Disorder', fontweight='bold')
    ax.legend(fontsize=8)

# Panel I: Summary
ax = fig.add_subplot(3, 3, 9)
ax.axis('off')
txt = "CROSS-DATASET SUMMARY\n" + "=" * 35 + "\n\n"
txt += f"Mi 2022 (MBL, SC):   alpha = {best_mi:.3f}\n"
if best_xiang:
    txt += f"Xiang 2024 (topo):   alpha = {best_xiang['alpha']:.3f}\n"
    txt += f"  delta = {best_xiang['alpha']-best_mi:+.3f}\n"
if best_randall:
    txt += f"Randall 2021 (NV):   alpha = {best_randall['alpha']:.3f}\n"
    txt += f"  delta = {best_randall['alpha']-best_mi:+.3f}\n"
txt += f"\nMerkabit threshold:  alpha ~ 1.3\n"

txt += "\n" + "=" * 35 + "\n"
txt += "IBM Heron 2025: DATA NOT AVAILABLE\n"
txt += "(GitHub repo is private)\n"
txt += "\nKyprianidis 2021: NO PUBLIC DATA\n"

ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Multi-Dataset DTC Subharmonic Envelope Analysis\n'
            'Merkabit Framework: Does alpha increase away from MBL?',
            fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(FIGURES_DIR, 'multi_dtc_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: multi_dtc_analysis.png")


# ============================================================
# SAVE DETAILED REPORTS
# ============================================================

def save_report(filepath, title, results_dict, extra_text=''):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n{'='*70}\n\n")
        f.write(f"Baseline: Mi et al. 2022 MBL DTC alpha = {ALPHA_MI_BASELINE}\n\n")
        f.write(f"{'Key':<45} {'alpha':>8} {'err':>8} {'delta':>8} {'R2':>8}\n")
        f.write("-" * 85 + "\n")
        for k, r in sorted(results_dict.items()):
            if isinstance(r, dict) and r is not None and 'alpha' in r:
                da = r['alpha'] - ALPHA_MI_BASELINE
                tag = '*' if 1.1 <= r['alpha'] <= 1.5 else ' '
                f.write(f"{tag}{k:<44} {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} "
                       f"{da:>+8.4f} {r['R2']:>8.4f}\n")
        f.write(f"\n\n{extra_text}\n")

save_report(os.path.join(REPORTS_DIR, 'xiang_2024_report.txt'),
           'XIANG ET AL. 2024 — TOPOLOGICAL DTC', xiang_results,
           'Stabilisation mechanism: Topology + Prethermalization (NOT MBL)\n'
           'Platform: Google Sycamore superconducting qubits\n'
           'Observable: Local Z + String Z operators on square lattice')

save_report(os.path.join(REPORTS_DIR, 'randall_2021_report.txt'),
           'RANDALL ET AL. 2021 — NV CENTER MBL DTC', randall_results,
           'Stabilisation mechanism: MBL (dipolar disorder)\n'
           'Platform: Diamond NV center spins (9 qubits)\n'
           'Observable: Local Z magnetization\n'
           'Noise type: 1/f-like (laser intensity, motional heating)')

print(f"\nSaved: xiang_2024_report.txt, randall_2021_report.txt")
print(f"Outputs: figures={FIGURES_DIR}, reports={REPORTS_DIR}")
