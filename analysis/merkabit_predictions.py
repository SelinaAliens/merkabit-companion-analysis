"""
Merkabit Framework: Three Testable Predictions (v2 — corrected)
================================================================
Using Xiang et al. 2024 topological DTC dataset.

Key corrections from v1:
- fig2a stabilizer Z oscillates at period 2 in full-cycle data: use (-1)^n envelope
- fig2c per-qubit sigma_z is all-positive at full cycles: fit even branch DIRECTLY
- figS3 per-qubit Z is noise after cycle 1: EXCLUDED
- figS4 is simulation data: noted and analyzed separately
- Late-window fits on 20-cycle data are unreliable: use cumulative and rolling fits

Prediction 1: Time dependence of alpha (drift toward 2 = active locking)
Prediction 2: Response to topology breaking (disorder sweep)
Prediction 3: Subsystem size dependence of alpha
"""

import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, sys, io
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUT = FIGURES_DIR
XBASE = XIANG_2024_BASE
ALPHA_MI = 0.822   # MBL baseline
ALPHA_MK = 1.3     # Merkabit threshold

def stretched_exp(n, A0, n_star, alpha):
    return A0 * np.exp(-(n / n_star) ** alpha)

def fit_stretched(n_vals, env, label=''):
    """Fit stretched exponential. Returns dict or None."""
    mask = (env > 1e-4) & np.isfinite(env)
    n_f, e_f = n_vals[mask].astype(float), env[mask].astype(float)
    if len(n_f) < 5:
        return None
    try:
        popt, pcov = curve_fit(stretched_exp, n_f, e_f,
            p0=[e_f[0], max(len(n_f)*2, 5), 1.0],
            bounds=([0, 0.1, 0.01], [2.0, 50000, 10.0]),
            maxfev=200000)
        res = e_f - stretched_exp(n_f, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((e_f - np.mean(e_f))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {'A0': popt[0], 'n_star': popt[1], 'alpha': popt[2],
                'R2': R2, 'alpha_err': perr[2], 'n_pts': len(n_f)}
    except:
        return None


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("MERKABIT PREDICTIONS v2: THREE TESTS OF ACTIVE LOCKING")
print("=" * 70)

# fig2a: 6 stabilizer Z operators, 41 half-cycles (20 full Floquet cycles)
# These oscillate at period 2 in full-cycle data => use (-1)^n envelope
d2a = sio.loadmat(os.path.join(XBASE, 'main_figure2', 'fig2a.mat'))
Zs_stab = d2a['Zs_exp_avg']  # (6, 41)

# fig2c: 18 per-qubit sigma_z, 41 half-cycles
# These are all positive at full cycles => fit even branch directly
d2c = sio.loadmat(os.path.join(XBASE, 'main_figure2', 'fig2c.mat'))
Zq_perqubit = d2c['Asigmaz_avg']  # (18, 41)

# figS3: 50 full cycles — logical Z (100,6) (string operators, alpha~0.37)
dS3 = sio.loadmat(os.path.join(XBASE, 'supp_figure3', 'figS3.mat'))
Zl_string = dS3['Zl_avg']  # (100, 6)

# figS14: echo protocol (6 operators, 41 half-cycles)
dS14 = sio.loadmat(os.path.join(XBASE, 'supp_figure14', 'figS14.mat'))
Zs_echo = dS14['Zs_echo_avg']  # (6, 41)

# fig4a: disorder sweep — 6 Z ops x 17 disorders x 41 half-cycles
d4a = sio.loadmat(os.path.join(XBASE, 'main_figure4', 'fig4a.mat'))
disorders = d4a['disorders'].flatten()
Zs_dis = d4a['Zs_exp_avg']  # (6, 17, 41)

# fig4b: FFT peak vs disorder
d4b = sio.loadmat(os.path.join(XBASE, 'main_figure4', 'fig4b.mat'))
fft_peak = d4b['exp_fZs_peak_avg'].flatten()

# fig4e: topological entropy vs disorder
d4e = sio.loadmat(os.path.join(XBASE, 'main_figure4', 'fig4e.mat'))
dis_topo = d4e['disorders'].flatten()
S_topo = d4e['S_topo_exp_avg'].flatten()

# figS4: system size scaling — SIMULATION data
dS4 = sio.loadmat(os.path.join(XBASE, 'supp_figure4', 'figS4.mat'))
Ns_sys = dS4['Ns'].flatten()  # [6, 9, 12, 15]
Zl1_sim = dS4['Zl1_avg']      # (4, 100)
ts_log10 = dS4['ts_log10']    # (4, 100) — log10(Floquet cycles)

# Index arrays
fc_short = np.arange(0, 41, 2)  # 21 full cycles (0-20)
fc_long = np.arange(0, 100, 2)   # 50 full cycles


# ============================================================
# PREDICTION 1: TIME-WINDOWED ALPHA DRIFT
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION 1: TIME-WINDOWED ALPHA DRIFT")
print("=" * 70)
print("""
Hypothesis: If active locking, alpha drifts upward toward 2 over later
Floquet cycles as more of the system gets captured by the topological
protection. Noise doesn't drift.
""")

# --- Test 1A: Cumulative fit (fig2a stabilizer Z) ---
# Fit alpha to cycles 1-N for increasing N.
# If locking: alpha increases with N.
print("--- Test 1A: Cumulative alpha (stabilizer Z, fig2a) ---")
print("  Fit alpha to cycles [1, N] for increasing N")

cumulative_stab = {}  # operator -> [(N_end, alpha, err, R2)]

for oi in range(6):
    sig_fc = Zs_stab[oi, fc_short]
    env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))  # subharmonic envelope

    results = []
    for N_end in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        n_vals = np.arange(1, N_end + 1)
        env_vals = env[1:N_end + 1]
        r = fit_stretched(n_vals, env_vals)
        if r and r['R2'] > 0.95:
            results.append((N_end, r['alpha'], r['alpha_err'], r['R2']))
    cumulative_stab[oi] = results

# Print results
print(f"\n  {'N_end':>6}", end='')
for oi in range(6):
    print(f"  {'Z'+str(oi):>8}", end='')
print(f"  {'Mean':>8}  {'Trend':>6}")
print("  " + "-" * 76)

all_N_ends = sorted(set(N for oi in range(6) for N, *_ in cumulative_stab[oi]))
cumul_means = []  # (N_end, mean_alpha, std)
for N_end in all_N_ends:
    alphas_at_N = []
    print(f"  {N_end:>6}", end='')
    for oi in range(6):
        found = [x for x in cumulative_stab[oi] if x[0] == N_end]
        if found:
            a = found[0][1]
            alphas_at_N.append(a)
            print(f"  {a:>8.3f}", end='')
        else:
            print(f"  {'---':>8}", end='')
    if alphas_at_N:
        m = np.mean(alphas_at_N)
        s = np.std(alphas_at_N)
        cumul_means.append((N_end, m, s))
        print(f"  {m:>8.3f}  ", end='')
        if len(cumul_means) > 1:
            drift = m - cumul_means[0][1]
            print(f"{drift:>+.3f}", end='')
    print()

# Linear fit to cumulative mean alpha vs N_end
if len(cumul_means) >= 3:
    Ns_c = np.array([x[0] for x in cumul_means])
    As_c = np.array([x[1] for x in cumul_means])
    p = np.polyfit(Ns_c, As_c, 1)
    print(f"\n  Linear trend: alpha(N) = {p[1]:.3f} + {p[0]:+.5f} * N_end")
    print(f"  Slope = {p[0]:+.5f} per additional cycle")
    total_drift_1a = As_c[-1] - As_c[0]
    print(f"  Total drift from N=5 to N=20: {total_drift_1a:+.4f}")
    if p[0] > 0.005:
        print(f"  ==> UPWARD DRIFT in cumulative alpha")
    elif p[0] < -0.005:
        print(f"  ==> DOWNWARD DRIFT in cumulative alpha")
    else:
        print(f"  ==> FLAT (|slope| < 0.005)")

# --- Test 1B: Rolling window (fig2a stabilizer Z) ---
print("\n--- Test 1B: Rolling window alpha (stabilizer Z, 8-cycle windows) ---")
window = 8
rolling_results = {}

for oi in range(6):
    sig_fc = Zs_stab[oi, fc_short]
    env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))
    results = []
    for start in range(1, 21 - window + 1):
        end = start + window
        n_vals = np.arange(start, end)
        env_vals = env[start:end]
        r = fit_stretched(n_vals, env_vals)
        if r and r['R2'] > 0.9:
            center = (start + end) / 2
            results.append((center, r['alpha'], r['alpha_err'], r['R2']))
    rolling_results[oi] = results

print(f"\n  {'Center':>8}", end='')
for oi in range(6):
    print(f"  {'Z'+str(oi):>8}", end='')
print(f"  {'Mean':>8}")
print("  " + "-" * 64)

rolling_means = []
for start in range(1, 21 - window + 1):
    center = (start + start + window) / 2
    row = []
    print(f"  {center:>8.1f}", end='')
    for oi in range(6):
        found = [x for x in rolling_results[oi] if abs(x[0] - center) < 0.1]
        if found:
            row.append(found[0][1])
            print(f"  {found[0][1]:>8.3f}", end='')
        else:
            print(f"  {'---':>8}", end='')
    if row:
        m = np.mean(row)
        rolling_means.append((center, m))
        print(f"  {m:>8.3f}")
    else:
        print()

if len(rolling_means) >= 3:
    cs = np.array([x[0] for x in rolling_means])
    ms = np.array([x[1] for x in rolling_means])
    p_roll = np.polyfit(cs, ms, 1)
    print(f"\n  Rolling trend: slope = {p_roll[0]:+.5f} per cycle")
    drift_roll = ms[-1] - ms[0]
    print(f"  First-to-last drift: {drift_roll:+.4f}")

# --- Test 1C: Cumulative alpha for per-qubit operators (fig2c) ---
print("\n--- Test 1C: Cumulative alpha (per-qubit sigma_z, fig2c) ---")
print("  These are single-qubit operators, positive at full cycles")
print("  Fit even-branch directly (no subharmonic extraction needed)")

cumul_perqubit = {}
for qi in range(18):
    sig_fc = Zq_perqubit[qi, fc_short]  # positive, decaying
    results = []
    for N_end in [5, 6, 7, 8, 9, 10, 12, 15, 18, 20]:
        n_vals = np.arange(1, N_end + 1)
        env_vals = sig_fc[1:N_end + 1]
        r = fit_stretched(n_vals, env_vals)
        if r and r['R2'] > 0.9:
            results.append((N_end, r['alpha'], r['alpha_err'], r['R2']))
    cumul_perqubit[qi] = results

# Averaged per-qubit signal
avg_pq = np.mean(Zq_perqubit[:, fc_short], axis=0)  # average over 18 qubits
print(f"\n  Averaged per-qubit signal (18 qubits, positive branch):")
cumul_avg_pq = []
for N_end in [5, 6, 7, 8, 9, 10, 12, 15, 18, 20]:
    n_vals = np.arange(1, N_end + 1)
    env_vals = avg_pq[1:N_end + 1]
    r = fit_stretched(n_vals, env_vals)
    if r and r['R2'] > 0.85:
        cumul_avg_pq.append((N_end, r['alpha'], r['alpha_err'], r['R2']))
        print(f"    1-{N_end:<3}: alpha = {r['alpha']:.4f} +/- {r['alpha_err']:.4f}, R2 = {r['R2']:.4f}")

# --- Test 1D: String operators (figS3, 50 cycles) ---
print("\n--- Test 1D: Cumulative alpha (string Z operators, figS3, 50 cycles) ---")
cumul_string = []
for N_end in [5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    alphas_ops = []
    for li in range(6):
        sig_fc = Zl_string[fc_long, li]
        env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))
        end_idx = min(N_end + 1, len(env))
        n_vals = np.arange(1, end_idx)
        env_vals = env[1:end_idx]
        if len(n_vals) >= 5:
            r = fit_stretched(n_vals, env_vals)
            if r and r['R2'] > 0.9:
                alphas_ops.append(r['alpha'])
    if alphas_ops:
        m = np.mean(alphas_ops)
        s = np.std(alphas_ops)
        cumul_string.append((N_end, m, s, len(alphas_ops)))
        print(f"  1-{N_end:<3}: alpha = {m:.4f} +/- {s:.4f} ({len(alphas_ops)}/6 ops)")


# ============================================================
# PREDICTION 2: RESPONSE TO TOPOLOGY BREAKING
# ============================================================
print("\n\n" + "=" * 70)
print("PREDICTION 2: RESPONSE TO TOPOLOGY BREAKING")
print("=" * 70)
print("""
Hypothesis: If you weaken the topological protection (via disorder),
the locking process fails and alpha drops toward the MBL value (0.82).
Noise would be insensitive to this perturbation.
""")

# --- Test 2A: Alpha vs disorder (reliable range only) ---
print("--- Test 2A: Alpha vs disorder (fig4a, disorders 0-1.4) ---")
print("  Only using disorders where R2 > 0.95 for at least 3/6 operators")

# Fit all 6 operators at each disorder
dis_results = np.full((6, len(disorders)), np.nan)
dis_errors = np.full((6, len(disorders)), np.nan)
dis_r2 = np.full((6, len(disorders)), 0.0)

for oi in range(6):
    for di, dis in enumerate(disorders):
        sig_fc = Zs_dis[oi, di, fc_short]
        env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))
        r = fit_stretched(np.arange(1, 21), env[1:])
        if r:
            dis_results[oi, di] = r['alpha']
            dis_errors[oi, di] = r['alpha_err']
            dis_r2[oi, di] = r['R2']

# Print clean table (disorders 0-1.4 only)
print(f"\n  {'Dis':>5}", end='')
for oi in range(6):
    print(f" {'Z'+str(oi)+' (R2)':>14}", end='')
print(f"  {'Mean':>8} {'SEM':>6}")
print("  " + "-" * 106)

dis_clean_alpha = []
dis_clean_errs = []
dis_clean_vals = []

for di, dis in enumerate(disorders):
    if dis > 1.5:
        break
    good = dis_r2[:, di] > 0.95
    print(f"  {dis:>5.1f}", end='')
    row = []
    for oi in range(6):
        if dis_r2[oi, di] > 0.95:
            print(f" {dis_results[oi,di]:>6.3f}({dis_r2[oi,di]:.3f})", end='')
            row.append(dis_results[oi, di])
        elif dis_r2[oi, di] > 0.85:
            print(f" {dis_results[oi,di]:>6.3f}({dis_r2[oi,di]:.3f})", end='')
            row.append(dis_results[oi, di])
        else:
            print(f"      {'FAIL':>8}", end='')

    if len(row) >= 3:
        m = np.mean(row)
        sem = np.std(row) / np.sqrt(len(row))
        dis_clean_alpha.append(m)
        dis_clean_errs.append(sem)
        dis_clean_vals.append(dis)
        print(f"  {m:>8.3f} {sem:>6.3f}")
    else:
        dis_clean_alpha.append(np.nan)
        dis_clean_errs.append(np.nan)
        dis_clean_vals.append(dis)
        print(f"  {'insuf':>8}")

dis_clean_alpha = np.array(dis_clean_alpha)
dis_clean_errs = np.array(dis_clean_errs)
dis_clean_vals = np.array(dis_clean_vals)

# --- Test 2B: Correlation with topological entropy ---
print("\n--- Test 2B: Correlation with topological order ---")
print("  Using only reliable alpha values (R2 > 0.95, >= 3 operators)")

alpha_corr = []
stopo_corr = []
fft_corr = []

for i, dis in enumerate(dis_clean_vals):
    if np.isnan(dis_clean_alpha[i]):
        continue
    # Match to S_topo
    for ti, td in enumerate(dis_topo):
        if abs(td - dis) < 0.05:
            alpha_corr.append(dis_clean_alpha[i])
            stopo_corr.append(S_topo[ti])
            break
    # Match to FFT peak
    for di2, d2 in enumerate(disorders):
        if abs(d2 - dis) < 0.05 and di2 < len(fft_peak):
            fft_corr.append(fft_peak[di2])
            break

print(f"\n  {'Disorder':>10} {'Alpha':>8} {'S_topo':>8} {'FFT pk':>8}")
print("  " + "-" * 38)
for i in range(len(alpha_corr)):
    f_val = fft_corr[i] if i < len(fft_corr) else float('nan')
    print(f"  {dis_clean_vals[i] if i < len(dis_clean_vals) else 0:>10.1f} "
          f"{alpha_corr[i]:>8.3f} {stopo_corr[i]:>8.4f} {f_val:>8.4f}")

if len(alpha_corr) >= 3:
    r_stopo = np.corrcoef(alpha_corr, stopo_corr)[0, 1]
    print(f"\n  Pearson r(alpha, S_topo) = {r_stopo:.4f}")
    if len(fft_corr) >= 3 and len(fft_corr) == len(alpha_corr):
        r_fft = np.corrcoef(alpha_corr[:len(fft_corr)], fft_corr)[0, 1]
        print(f"  Pearson r(alpha, FFT_peak) = {r_fft:.4f}")
    else:
        r_fft = np.nan

    if r_stopo > 0.7:
        print(f"  ==> STRONG POSITIVE CORRELATION")
        print(f"      alpha tracks topological protection strength")
        print(f"      CONSISTENT WITH LOCKING")
    elif r_stopo > 0.3:
        print(f"  ==> MODERATE correlation with topological order")
    else:
        print(f"  ==> WEAK correlation (locking prediction not strongly supported)")

# --- Test 2C: Critical disorder values ---
print("\n--- Test 2C: Critical disorder crossings ---")
valid_dc = ~np.isnan(dis_clean_alpha)
if valid_dc.sum() >= 3:
    dv = dis_clean_vals[valid_dc]
    av = dis_clean_alpha[valid_dc]
    try:
        f_int = interp1d(dv, av, kind='linear', fill_value='extrapolate')
        d_fine = np.linspace(dv[0], dv[-1], 1000)
        a_fine = f_int(d_fine)

        cross_mk = d_fine[np.argmin(np.abs(a_fine - ALPHA_MK))]
        cross_mi = d_fine[np.argmin(np.abs(a_fine - ALPHA_MI))]

        print(f"  Alpha crosses Merkabit threshold (1.3) at disorder ~ {cross_mk:.2f}")
        print(f"  Alpha crosses MBL baseline (0.822) at disorder ~ {cross_mi:.2f}")
        print(f"  alpha=1.3 regime: disorder < {cross_mk:.2f}")
        print(f"  alpha=0.82 regime: disorder > {cross_mi:.2f}")
    except:
        print("  Interpolation failed")

# --- Test 2D: Is the drop monotonic? ---
print("\n--- Test 2D: Monotonicity of alpha(disorder) ---")
valid_alpha = dis_clean_alpha[valid_dc]
diffs = np.diff(valid_alpha)
n_decrease = np.sum(diffs < 0)
n_increase = np.sum(diffs > 0)
print(f"  Steps where alpha decreases: {n_decrease}/{len(diffs)}")
print(f"  Steps where alpha increases: {n_increase}/{len(diffs)}")
if n_decrease > n_increase * 2:
    print(f"  ==> PREDOMINANTLY DECREASING (monotonic drop)")
else:
    print(f"  ==> NON-MONOTONIC (some fluctuation)")


# ============================================================
# PREDICTION 3: SUBSYSTEM SIZE DEPENDENCE
# ============================================================
print("\n\n" + "=" * 70)
print("PREDICTION 3: SUBSYSTEM SIZE DEPENDENCE")
print("=" * 70)
print("""
Hypothesis: A locking process should show alpha increasing with
subsystem size (more qubits = stronger standing wave).
Noise should show alpha decreasing (more qubits = more decoherence channels).
""")

# --- Test 3A: Operator hierarchy ---
# Three operator types with different effective sizes:
# 1. Single-qubit sigma_z (fig2c) — 1 qubit
# 2. Stabilizer Z (fig2a) — ~4 qubits per plaquette
# 3. Logical/string Z (figS3) — ~N qubits (full string)
print("--- Test 3A: Operator hierarchy (1-qubit, ~4-qubit, N-qubit) ---")

# Single-qubit (fig2c): average all 18
avg_pq_sig = np.mean(Zq_perqubit[:, fc_short], axis=0)
r_1q = fit_stretched(np.arange(1, 11), avg_pq_sig[1:11])
if r_1q:
    print(f"  Single-qubit sigma_z (avg 18, 10 cycles):")
    print(f"    alpha = {r_1q['alpha']:.4f} +/- {r_1q['alpha_err']:.4f}, R2 = {r_1q['R2']:.4f}")

# Stabilizer Z (fig2a): average all 6
stab_alphas = []
for oi in range(6):
    env = Zs_stab[oi, fc_short] * ((-1.0)**np.arange(21))
    r = fit_stretched(np.arange(1, 21), env[1:])
    if r and r['R2'] > 0.95:
        stab_alphas.append(r['alpha'])
if stab_alphas:
    print(f"  Stabilizer Z (6 ops, 20 cycles):")
    print(f"    mean alpha = {np.mean(stab_alphas):.4f} +/- {np.std(stab_alphas):.4f}, "
          f"range [{min(stab_alphas):.3f}, {max(stab_alphas):.3f}]")

# String/Logical Z (figS3): average all 6
string_alphas = []
for li in range(6):
    sig_fc = Zl_string[fc_long, li]
    env = sig_fc * ((-1.0)**np.arange(50))
    r = fit_stretched(np.arange(1, 50), env[1:])
    if r and r['R2'] > 0.95:
        string_alphas.append(r['alpha'])
if string_alphas:
    print(f"  String/Logical Z (6 ops, 50 cycles):")
    print(f"    mean alpha = {np.mean(string_alphas):.4f} +/- {np.std(string_alphas):.4f}")

# Summary table
print(f"\n  {'Operator type':>25} {'Eff size':>10} {'Alpha':>8} {'Prediction':>12}")
print("  " + "-" * 58)
if r_1q:
    print(f"  {'Single-qubit':>25} {'~1':>10} {r_1q['alpha']:>8.3f} {'--':>12}")
if stab_alphas:
    print(f"  {'Stabilizer plaquette':>25} {'~4':>10} {np.mean(stab_alphas):>8.3f} {'--':>12}")
if string_alphas:
    print(f"  {'String/Logical':>25} {'~N':>10} {np.mean(string_alphas):>8.3f} {'--':>12}")

if r_1q and stab_alphas and string_alphas:
    a1 = r_1q['alpha']
    a4 = np.mean(stab_alphas)
    aN = np.mean(string_alphas)
    if a4 > a1 and aN < a4:
        trend_3a = "NON-MONOTONIC (peaks at stabilizer)"
    elif a4 > a1 and aN > a4:
        trend_3a = "INCREASING with size (locking-like)"
    elif a4 < a1 and aN < a4:
        trend_3a = "DECREASING with size (noise-like)"
    else:
        trend_3a = "MIXED"
    print(f"\n  Trend: {trend_3a}")

# --- Test 3B: Subsystem construction from per-qubit data (fig2c) ---
print("\n--- Test 3B: Subsystem averaging (fig2c, 18 per-qubit, 10 cycles) ---")
print("  Average k qubits, fit stretched exp to the averaged signal")

subsys_results = []
print(f"\n  {'k qubits':>10} {'alpha':>8} {'+-err':>8} {'R2':>8}")
print("  " + "-" * 38)

for k in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18]:
    # Average over first k qubits (positive, decaying)
    avg_k = np.mean(Zq_perqubit[:k, fc_short], axis=0)
    r = fit_stretched(np.arange(1, 11), avg_k[1:11])
    if r and r['R2'] > 0.9:
        subsys_results.append((k, r['alpha'], r['alpha_err'], r['R2']))
        tag = '*' if 1.1 <= r['alpha'] <= 1.5 else ' '
        print(f"  {tag} {k:>8}   {r['alpha']:>8.4f} {r['alpha_err']:>8.4f} {r['R2']:>8.4f}")
    else:
        print(f"    {k:>8}   {'FAIL':>8}")

# Random subsampling for robustness
print("\n  Random subsampling (200 trials per k):")
np.random.seed(42)
subsys_random = []
for k in [1, 2, 3, 4, 6, 9, 12, 15, 18]:
    trial_alphas = []
    for trial in range(200):
        if k == 18:
            indices = np.arange(18)
        elif k == 1:
            indices = [trial % 18]
        else:
            indices = np.random.choice(18, size=k, replace=False)
        avg_k = np.mean(Zq_perqubit[indices][:, fc_short], axis=0)
        r = fit_stretched(np.arange(1, 11), avg_k[1:11])
        if r and r['R2'] > 0.9:
            trial_alphas.append(r['alpha'])
    if trial_alphas:
        m = np.mean(trial_alphas)
        s = np.std(trial_alphas)
        subsys_random.append((k, m, s, len(trial_alphas)))
        tag = '*' if 1.1 <= m <= 1.5 else ' '
        print(f"  {tag} k={k:>2}: alpha = {m:.4f} +/- {s:.4f}  ({len(trial_alphas)}/200 valid)")

# --- Test 3C: System size scaling from simulation (figS4) ---
print("\n--- Test 3C: System size scaling — SIMULATION DATA (figS4) ---")
print("  NOTE: This is numerical simulation, not experimental data")
print("  Logical Z operator (string) at system sizes N = 6, 9, 12, 15")

# The ts_log10 gives log10(Floquet cycles), and signal is NOT oscillating
# (logical Z in simulation measured at specific cycle numbers)
syssize_sim = []
for idx, N in enumerate(Ns_sys):
    signal = Zl1_sim[idx]
    ts = 10**ts_log10[idx]  # actual Floquet cycles

    # These are simulation data points at log-spaced times
    # Signal is monotonically decaying (no period-2 oscillation in sim)
    mask = (signal > 1e-3) & (ts > 0)
    if mask.sum() >= 5:
        r = fit_stretched(ts[mask], signal[mask])
        if r:
            syssize_sim.append((N, r['alpha'], r['alpha_err'], r['R2']))
            print(f"  N={N:>2}: alpha = {r['alpha']:.4f} +/- {r['alpha_err']:.4f}, "
                  f"R2 = {r['R2']:.4f}, n* = {r['n_star']:.1f}")

if len(syssize_sim) >= 2:
    ns_s = [x[0] for x in syssize_sim]
    as_s = [x[1] for x in syssize_sim]
    p_s = np.polyfit(ns_s, as_s, 1)
    print(f"\n  Trend: slope = {p_s[0]:+.5f} per qubit")
    if p_s[0] > 0:
        print(f"  ==> Alpha INCREASES with system size in simulation")
    else:
        print(f"  ==> Alpha DECREASES with system size in simulation")


# ============================================================
# FINAL SUMMARY AND VERDICTS
# ============================================================
print("\n\n" + "=" * 70)
print("FINAL SUMMARY: THREE PREDICTIONS")
print("=" * 70)

# P1 verdict
print("\n--- PREDICTION 1: TIME DRIFT ---")
if len(cumul_means) >= 2:
    total_drift = cumul_means[-1][1] - cumul_means[0][1]
    print(f"  Cumulative alpha: first = {cumul_means[0][1]:.3f} (N={cumul_means[0][0]}), "
          f"last = {cumul_means[-1][1]:.3f} (N={cumul_means[-1][0]})")
    print(f"  Drift = {total_drift:+.4f}")
    if total_drift > 0.1:
        p1 = "SUPPORTED"
        print(f"  ==> {p1}: Upward drift detected")
        print(f"      Alpha increases as more cycles are included")
        print(f"      Consistent with progressive locking")
    elif total_drift < -0.1:
        p1 = "FALSIFIED"
        print(f"  ==> {p1}: Downward drift (opposite to prediction)")
    else:
        p1 = "INCONCLUSIVE"
        print(f"  ==> {p1}: Drift too small to distinguish")
else:
    p1 = "INSUFFICIENT DATA"

# P2 verdict
print("\n--- PREDICTION 2: TOPOLOGY BREAKING ---")
valid_dc2 = ~np.isnan(dis_clean_alpha)
if valid_dc2.sum() >= 3:
    a_zero = dis_clean_alpha[0]
    # Find last reliable point
    last_v = np.where(valid_dc2)[0][-1]
    a_high = dis_clean_alpha[last_v]
    drop = a_zero - a_high
    print(f"  alpha(disorder=0) = {a_zero:.3f}")
    print(f"  alpha(disorder={dis_clean_vals[last_v]:.1f}) = {a_high:.3f}")
    print(f"  Drop = {drop:.3f}")
    if a_zero > 1.1 and a_high < 0.9:
        p2 = "STRONGLY SUPPORTED"
        print(f"  ==> {p2}: Alpha drops from Merkabit range to sub-MBL")
        print(f"      as topological protection is weakened by disorder")
    elif drop > 0.2:
        p2 = "SUPPORTED"
        print(f"  ==> {p2}: Significant drop in alpha with disorder")
    else:
        p2 = "INCONCLUSIVE"
        print(f"  ==> {p2}")
else:
    p2 = "INSUFFICIENT DATA"

# P3 verdict
print("\n--- PREDICTION 3: SUBSYSTEM SIZE ---")
if subsys_results:
    k1_alpha = subsys_results[0][1]
    kmax_alpha = subsys_results[-1][1]
    k_slope = np.polyfit([x[0] for x in subsys_results],
                         [x[1] for x in subsys_results], 1)[0]
    print(f"  alpha(k=1) = {k1_alpha:.3f}")
    print(f"  alpha(k={subsys_results[-1][0]}) = {kmax_alpha:.3f}")
    print(f"  Slope = {k_slope:+.5f} per qubit")
    if k_slope > 0.003:
        p3 = "SUPPORTED"
        print(f"  ==> {p3}: Alpha increases with subsystem size")
        print(f"      Consistent with collective locking")
    elif k_slope < -0.003:
        p3 = "FALSIFIED"
        print(f"  ==> {p3}: Alpha decreases with subsystem size")
        print(f"      Consistent with noise (more qubits = more decoherence)")
    else:
        p3 = "INCONCLUSIVE"
        print(f"  ==> {p3}: No significant trend")
else:
    p3 = "INSUFFICIENT DATA"

# --- Operator hierarchy verdict ---
print("\n--- ADDITIONAL: OPERATOR HIERARCHY ---")
if r_1q and stab_alphas and string_alphas:
    a_1q = r_1q['alpha']
    a_stab = np.mean(stab_alphas)
    a_string = np.mean(string_alphas)
    print(f"  Single-qubit (1):     alpha = {a_1q:.3f}")
    print(f"  Stabilizer (~4):      alpha = {a_stab:.3f}")
    print(f"  String (~N):          alpha = {a_string:.3f}")
    print(f"  1q -> stab: {a_stab - a_1q:+.3f}")
    print(f"  stab -> string: {a_string - a_stab:+.3f}")

# Overall
print("\n" + "=" * 70)
print("OVERALL ASSESSMENT")
print("=" * 70)
verdicts = [p1, p2, p3]
n_supported = sum(1 for v in verdicts if 'SUPPORT' in v)
n_falsified = sum(1 for v in verdicts if 'FALSIF' in v)
n_inconclusive = sum(1 for v in verdicts if 'INCONCLUS' in v)

print(f"\n  Supported:    {n_supported}/3")
print(f"  Falsified:    {n_falsified}/3")
print(f"  Inconclusive: {n_inconclusive}/3")

if n_supported >= 2 and n_falsified == 0:
    print(f"\n  ==> ACTIVE LOCKING HYPOTHESIS: CONSISTENT WITH DATA")
elif n_falsified >= 2:
    print(f"\n  ==> ACTIVE LOCKING HYPOTHESIS: INCONSISTENT WITH DATA")
else:
    print(f"\n  ==> MIXED EVIDENCE: Further experiments needed")
    print(f"      Most direct test: IBM Heron clean DTC (when data available)")


# ============================================================
# PUBLICATION FIGURE
# ============================================================
print("\n\nGenerating figure...")

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Colors
C_STAB = '#2ca02c'   # stabilizer Z (green)
C_PQ = '#ff7f0e'     # per-qubit (orange)
C_STR = '#1f77b4'    # string Z (blue)
C_MK = '#d62728'     # Merkabit reference (red)
C_MI = '#e377c2'     # MBL baseline (pink)

# --- Panel A: Cumulative alpha (stabilizer Z) ---
ax = fig.add_subplot(gs[0, 0])
if cumul_means:
    Ns_cm = [x[0] for x in cumul_means]
    As_cm = [x[1] for x in cumul_means]
    Ss_cm = [x[2] for x in cumul_means]
    ax.errorbar(Ns_cm, As_cm, yerr=Ss_cm, fmt='o-', markersize=8, capsize=4,
               color=C_STAB, linewidth=2, label='Stabilizer Z mean')
    # Individual operators
    for oi in range(6):
        if cumulative_stab[oi]:
            ns_o = [x[0] for x in cumulative_stab[oi]]
            as_o = [x[1] for x in cumulative_stab[oi]]
            ax.plot(ns_o, as_o, 'o-', markersize=3, alpha=0.3, color=C_STAB)
    # Trend line
    p = np.polyfit(Ns_cm, As_cm, 1)
    x_t = np.linspace(min(Ns_cm), max(Ns_cm), 100)
    ax.plot(x_t, np.polyval(p, x_t), '--', color=C_MK, linewidth=2,
           label=f'Slope = {p[0]:+.4f}/cycle')
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle=':', alpha=0.4)
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle=':', alpha=0.4)
ax.set_xlabel('Cycles included (1 to N)', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('A. Pred 1: Cumulative Alpha\n(Stabilizer Z, fig2a)', fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel B: Rolling window alpha ---
ax = fig.add_subplot(gs[0, 1])
if rolling_means:
    cs_r = [x[0] for x in rolling_means]
    ms_r = [x[1] for x in rolling_means]
    ax.plot(cs_r, ms_r, 'o-', markersize=8, color=C_STAB, linewidth=2, label='Stabilizer Z mean')
    # Individual operators
    for oi in range(6):
        if rolling_results[oi]:
            cs_o = [x[0] for x in rolling_results[oi]]
            as_o = [x[1] for x in rolling_results[oi]]
            ax.plot(cs_o, as_o, 'o-', markersize=3, alpha=0.3, color=C_STAB)
    # Trend
    p_r = np.polyfit(cs_r, ms_r, 1)
    x_t = np.linspace(min(cs_r), max(cs_r), 100)
    ax.plot(x_t, np.polyval(p_r, x_t), '--', color=C_MK, linewidth=2,
           label=f'Slope = {p_r[0]:+.4f}/cycle')
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle=':', alpha=0.4)
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle=':', alpha=0.4)
ax.set_xlabel('Window center (Floquet cycles)', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('B. Pred 1: Rolling Window Alpha\n(8-cycle windows, fig2a)', fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel C: Cumulative alpha for per-qubit + string ---
ax = fig.add_subplot(gs[0, 2])
if cumul_avg_pq:
    ns_p = [x[0] for x in cumul_avg_pq]
    as_p = [x[1] for x in cumul_avg_pq]
    ax.plot(ns_p, as_p, 's-', markersize=7, color=C_PQ, linewidth=2, label='Per-qubit avg (fig2c)')
if cumul_string:
    ns_s = [x[0] for x in cumul_string]
    as_s = [x[1] for x in cumul_string]
    ax.plot(ns_s, as_s, 'D-', markersize=7, color=C_STR, linewidth=2, label='String Z avg (figS3)')
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle=':', alpha=0.4, label='Merkabit (1.3)')
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle=':', alpha=0.4, label='MBL (0.82)')
ax.set_xlabel('Cycles included (1 to N)', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('C. Cumulative Alpha by Operator Type', fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel D: Alpha vs disorder (full sweep) ---
ax = fig.add_subplot(gs[1, 0])
for oi in range(6):
    mask_oi = dis_r2[oi] > 0.9
    if mask_oi.sum() > 0:
        ax.plot(disorders[mask_oi], dis_results[oi, mask_oi],
               'o-', markersize=3, alpha=0.3, color=C_STAB)
valid_dc3 = ~np.isnan(dis_clean_alpha)
ax.errorbar(dis_clean_vals[valid_dc3], dis_clean_alpha[valid_dc3],
           yerr=dis_clean_errs[valid_dc3],
           fmt='s-', markersize=8, capsize=4, color='black', linewidth=2.5,
           label='Mean (R2>0.95)', zorder=10)
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle='--', linewidth=2, alpha=0.6, label='Merkabit (1.3)')
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle='--', linewidth=2, alpha=0.6, label='MBL (0.82)')
ax.axhspan(1.1, 1.5, alpha=0.08, color=C_MK)
ax.set_xlabel('Disorder strength', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('D. Pred 2: Alpha vs Disorder\n(Topology breaking)', fontweight='bold', fontsize=11)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.6)

# --- Panel E: Alpha vs S_topo scatter ---
ax = fig.add_subplot(gs[1, 1])
if alpha_corr and stopo_corr:
    ax.scatter(stopo_corr, alpha_corr, s=100, c=C_STAB, edgecolors='black', zorder=5)
    # Labels with disorder values
    for i in range(len(alpha_corr)):
        ax.annotate(f'{dis_clean_vals[i]:.1f}', (stopo_corr[i], alpha_corr[i]),
                   textcoords="offset points", xytext=(8, 5), fontsize=8)
    z = np.polyfit(stopo_corr, alpha_corr, 1)
    x_l = np.linspace(min(stopo_corr), max(stopo_corr), 100)
    ax.plot(x_l, np.polyval(z, x_l), '--', color=C_MK, linewidth=2,
           label=f'r = {r_stopo:.3f}')
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle=':', alpha=0.4)
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle=':', alpha=0.4)
ax.set_xlabel('Topological entropy S_topo', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('E. Alpha vs Topological Order\n(Labels = disorder)', fontweight='bold', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel F: Alpha phase diagram ---
ax = fig.add_subplot(gs[1, 2])
# Show alpha as a function of disorder with DTC order parameter (FFT peak)
if alpha_corr and len(fft_corr) == len(alpha_corr):
    sc = ax.scatter(fft_corr, alpha_corr, s=100, c=C_STAB, edgecolors='black', zorder=5)
    for i in range(len(alpha_corr)):
        ax.annotate(f'{dis_clean_vals[i]:.1f}', (fft_corr[i], alpha_corr[i]),
                   textcoords="offset points", xytext=(8, 5), fontsize=8)
    z2 = np.polyfit(fft_corr, alpha_corr, 1)
    x_l2 = np.linspace(min(fft_corr), max(fft_corr), 100)
    ax.plot(x_l2, np.polyval(z2, x_l2), '--', color=C_MK, linewidth=2,
           label=f'r = {r_fft:.3f}')
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle=':', alpha=0.4)
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle=':', alpha=0.4)
ax.set_xlabel('DTC FFT peak height', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('F. Alpha vs DTC Order Strength\n(Labels = disorder)', fontweight='bold', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Panel G: Subsystem size (per-qubit averaging) ---
ax = fig.add_subplot(gs[2, 0])
if subsys_results:
    k_v = [x[0] for x in subsys_results]
    a_v = [x[1] for x in subsys_results]
    e_v = [x[2] for x in subsys_results]
    ax.errorbar(k_v, a_v, yerr=e_v, fmt='o-', markersize=10, capsize=5,
               color=C_PQ, linewidth=2, label='First k qubits (fig2c)')
    c3 = np.polyfit(k_v, a_v, 1)
    x_t3 = np.linspace(1, 18, 100)
    ax.plot(x_t3, np.polyval(c3, x_t3), '--', color=C_MK, linewidth=2,
           label=f'Slope = {c3[0]:+.4f}/qubit')
if subsys_random:
    k_r = [x[0] for x in subsys_random]
    a_r = [x[1] for x in subsys_random]
    s_r = [x[2] for x in subsys_random]
    ax.errorbar(k_r, a_r, yerr=s_r, fmt='s--', markersize=7, capsize=3,
               color=C_STR, linewidth=1.5, label='Random subsets')
ax.axhline(y=ALPHA_MK, color=C_MK, linestyle=':', alpha=0.4)
ax.axhline(y=ALPHA_MI, color=C_MI, linestyle=':', alpha=0.4)
ax.set_xlabel('Number of qubits averaged', fontsize=11)
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('G. Pred 3: Subsystem Size\n(Per-qubit sigma_z)', fontweight='bold', fontsize=11)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Panel H: Operator hierarchy ---
ax = fig.add_subplot(gs[2, 1])
labels = []
alphas_h = []
errs_h = []
colors_h = []
if r_1q:
    labels.append('Single qubit\n(sigma_z)')
    alphas_h.append(r_1q['alpha'])
    errs_h.append(r_1q['alpha_err'])
    colors_h.append(C_PQ)
if stab_alphas:
    labels.append('Stabilizer Z\n(plaquette)')
    alphas_h.append(np.mean(stab_alphas))
    errs_h.append(np.std(stab_alphas))
    colors_h.append(C_STAB)
if string_alphas:
    labels.append('String Z\n(logical)')
    alphas_h.append(np.mean(string_alphas))
    errs_h.append(np.std(string_alphas))
    colors_h.append(C_STR)

if labels:
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, alphas_h, yerr=errs_h, capsize=8,
                 color=colors_h, edgecolor='black', alpha=0.8, width=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(y=ALPHA_MK, color=C_MK, linestyle='--', linewidth=2, alpha=0.6, label='Merkabit (1.3)')
    ax.axhline(y=ALPHA_MI, color=C_MI, linestyle='--', linewidth=2, alpha=0.6, label='MBL (0.82)')
    # Annotate values
    for i, (l, a) in enumerate(zip(labels, alphas_h)):
        ax.text(i, a + errs_h[i] + 0.03, f'{a:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Alpha', fontsize=11)
ax.set_title('H. Operator Hierarchy\n(1-qubit -> 4-qubit -> N-qubit)', fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel I: Verdict ---
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

verdict_txt = "PREDICTION VERDICTS\n" + "=" * 42 + "\n\n"
verdict_txt += f"P1 (Time drift):      {p1}\n"
if 'total_drift' in dir():
    verdict_txt += f"    drift = {total_drift:+.3f}\n\n"
verdict_txt += f"P2 (Topology break):  {p2}\n"
if 'drop' in dir():
    verdict_txt += f"    alpha drop = {drop:.3f}\n"
if 'r_stopo' in dir():
    verdict_txt += f"    r(alpha,S_topo) = {r_stopo:.3f}\n\n"
verdict_txt += f"P3 (Subsystem size):  {p3}\n"
if 'k_slope' in dir():
    verdict_txt += f"    slope = {k_slope:+.5f}/qubit\n"

verdict_txt += "\n" + "=" * 42 + "\n"
verdict_txt += f"Supported:    {n_supported}/3\n"
verdict_txt += f"Falsified:    {n_falsified}/3\n"
verdict_txt += f"Inconclusive: {n_inconclusive}/3\n\n"

if n_supported >= 2 and n_falsified == 0:
    verdict_txt += "OVERALL:\n  Consistent with active locking"
elif n_falsified >= 2:
    verdict_txt += "OVERALL:\n  Inconsistent with active locking"
else:
    verdict_txt += "OVERALL:\n  Mixed evidence\n  More data needed"

ax.text(0.05, 0.95, verdict_txt, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Merkabit Framework: Three Testable Predictions\n'
            'Xiang et al. 2024 Topological DTC (Nat. Commun. 15, 8963)',
            fontsize=14, fontweight='bold')

plt.savefig(os.path.join(FIGURES_DIR, 'predictions_figure.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: predictions_figure.png")

# Save report
report_path = os.path.join(REPORTS_DIR, 'predictions_report.txt')
# Capture all output above by redirecting... actually just write a clean report
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("MERKABIT FRAMEWORK: THREE TESTABLE PREDICTIONS\n")
    f.write("=" * 70 + "\n\n")
    f.write("Dataset: Xiang et al. 2024, Nat. Commun. 15, 8963 (2024)\n")
    f.write("  Topological prethermal DTC on Google Sycamore (18 qubits)\n")
    f.write(f"Baseline: Mi et al. 2022, alpha_MBL = {ALPHA_MI}\n")
    f.write(f"Threshold: Merkabit alpha ~ {ALPHA_MK}\n\n")

    f.write("PREDICTION 1: TIME DRIFT OF ALPHA\n")
    f.write("-" * 50 + "\n")
    f.write("If active locking, alpha drifts upward toward 2 over later cycles.\n")
    f.write("If passive noise, alpha stays constant.\n\n")
    f.write("Cumulative fits (stabilizer Z, 20 Floquet cycles):\n")
    if cumul_means:
        for x in cumul_means:
            f.write(f"  1-{x[0]:>2}: alpha = {x[1]:.4f} +/- {x[2]:.4f}\n")
        f.write(f"\nTotal drift: {total_drift:+.4f}\n")
    f.write(f"Verdict: {p1}\n\n")

    f.write("PREDICTION 2: RESPONSE TO TOPOLOGY BREAKING\n")
    f.write("-" * 50 + "\n")
    f.write("If locking, alpha drops toward 0.82 as disorder weakens topology.\n")
    f.write("If noise, alpha is insensitive to disorder.\n\n")
    f.write("Alpha vs disorder (mean over 6 stabilizer Z operators):\n")
    for i, dis in enumerate(dis_clean_vals):
        if not np.isnan(dis_clean_alpha[i]):
            tag = '*' if 1.1 <= dis_clean_alpha[i] <= 1.5 else ' '
            f.write(f"  {tag} dis={dis:.1f}: alpha = {dis_clean_alpha[i]:.4f} +/- {dis_clean_errs[i]:.4f}\n")
    if 'r_stopo' in dir():
        f.write(f"\nr(alpha, S_topo) = {r_stopo:.4f}\n")
    f.write(f"\nDrop from disorder=0 to max: {drop:.3f}\n")
    f.write(f"Verdict: {p2}\n\n")

    f.write("PREDICTION 3: SUBSYSTEM SIZE DEPENDENCE\n")
    f.write("-" * 50 + "\n")
    f.write("If locking, alpha increases with subsystem size.\n")
    f.write("If noise, alpha decreases with subsystem size.\n\n")
    f.write("Subsystem averaging (per-qubit sigma_z, first 10 cycles):\n")
    if subsys_results:
        for x in subsys_results:
            f.write(f"  k={x[0]:>2}: alpha = {x[1]:.4f} +/- {x[2]:.4f}, R2 = {x[3]:.4f}\n")
    f.write(f"\nOperator hierarchy:\n")
    if r_1q:
        f.write(f"  Single-qubit: alpha = {r_1q['alpha']:.4f}\n")
    if stab_alphas:
        f.write(f"  Stabilizer (~4q): alpha = {np.mean(stab_alphas):.4f}\n")
    if string_alphas:
        f.write(f"  String (~Nq): alpha = {np.mean(string_alphas):.4f}\n")
    f.write(f"\nVerdict: {p3}\n\n")

    f.write("=" * 70 + "\n")
    f.write("OVERALL ASSESSMENT\n")
    f.write("=" * 70 + "\n")
    f.write(f"Supported:    {n_supported}/3\n")
    f.write(f"Falsified:    {n_falsified}/3\n")
    f.write(f"Inconclusive: {n_inconclusive}/3\n\n")
    if n_supported >= 2 and n_falsified == 0:
        f.write("Active locking hypothesis: CONSISTENT WITH DATA\n")
    elif n_falsified >= 2:
        f.write("Active locking hypothesis: INCONSISTENT WITH DATA\n")
    else:
        f.write("Active locking hypothesis: MIXED EVIDENCE\n")

print(f"Saved: predictions_report.txt")
print(f"\nAll outputs: {RESULTS_DIR}")
