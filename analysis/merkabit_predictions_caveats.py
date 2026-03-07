"""
Merkabit Predictions: Critical Controls & Artifact Tests
=========================================================
The v2 analysis showed 3/3 predictions supported. Before claiming this,
we need to test for artifacts:

1. Is the cumulative alpha drift real, or a fitting artifact?
   (More data points -> better-constrained fit -> different alpha)
   CONTROL: Fit a KNOWN stretched exp with fixed alpha to synthetic data,
   then measure cumulative alpha. If it also "drifts", the drift is an artifact.

2. Is the subsystem size trend real, or driven by signal-to-noise?
   (Averaging k qubits reduces noise -> changes apparent alpha)
   CONTROL: Add noise to a synthetic signal and test whether averaging
   shifts alpha upward.

3. Does the disorder correlation survive when averaging over all 6 operators?
   (Using only Z0 might cherry-pick the strongest result)
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io as sio
import os, sys, io
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

def stretched_exp(n, A0, n_star, alpha):
    return A0 * np.exp(-(n / n_star) ** alpha)

def fit_stretched(n_vals, env):
    mask = (env > 1e-4) & np.isfinite(env)
    n_f, e_f = n_vals[mask].astype(float), env[mask].astype(float)
    if len(n_f) < 5:
        return None
    try:
        popt, pcov = curve_fit(stretched_exp, n_f, e_f,
            p0=[e_f[0], max(len(n_f)*2, 5), 1.0],
            bounds=([0, 0.1, 0.01], [2.0, 50000, 10.0]), maxfev=200000)
        res = e_f - stretched_exp(n_f, *popt)
        ss_res, ss_tot = np.sum(res**2), np.sum((e_f - np.mean(e_f))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {'alpha': popt[2], 'alpha_err': perr[2], 'R2': R2,
                'A0': popt[0], 'n_star': popt[1]}
    except:
        return None


print("=" * 70)
print("CRITICAL CONTROLS: ARTIFACT TESTING")
print("=" * 70)

# ============================================================
# CONTROL 1: Is cumulative alpha drift a fitting artifact?
# ============================================================
print("\n" + "=" * 70)
print("CONTROL 1: SYNTHETIC DATA -- CUMULATIVE ALPHA DRIFT")
print("=" * 70)
print("""
  If we generate synthetic data with TRUE alpha = 1.3 and fit
  cumulatively (1-5, 1-10, 1-15, 1-20), do we see an artifactual drift?
  If yes, the real-data drift is a fitting artifact.
  If no, the real-data drift is genuine.
""")

np.random.seed(42)
n_cycles = 20
n_full = np.arange(0, n_cycles + 1)

# Test with several true alpha values
for true_alpha in [0.5, 0.8, 1.0, 1.3, 1.5, 2.0]:
    true_A0 = 1.0
    true_nstar = 15.0
    signal_clean = true_A0 * np.exp(-(n_full / true_nstar) ** true_alpha)

    # Add realistic noise (similar to Xiang data)
    noise_levels = [0.0, 0.01, 0.02, 0.05]
    for noise in noise_levels:
        if noise == 0:
            signal = signal_clean
            label = "no noise"
        else:
            signal = signal_clean + np.random.randn(len(signal_clean)) * noise
            signal = np.maximum(signal, 1e-5)  # keep positive
            label = f"noise={noise}"

        cumul_alphas = []
        for N_end in [5, 8, 10, 12, 15, 18, 20]:
            r = fit_stretched(np.arange(1, N_end + 1), signal[1:N_end + 1])
            if r and r['R2'] > 0.9:
                cumul_alphas.append((N_end, r['alpha']))

        if cumul_alphas and noise == 0.02:  # show one noise level
            drift = cumul_alphas[-1][1] - cumul_alphas[0][1]
            vals = ' '.join([f'{x[1]:.3f}' for x in cumul_alphas])
            print(f"  true_alpha={true_alpha:.1f}, {label}: {vals}  "
                  f"drift={drift:+.3f}")

# Now run 1000 Monte Carlo trials for true_alpha=1.3
print("\n  MONTE CARLO: 1000 trials, true_alpha=1.3, noise=0.02")
mc_drifts = []
mc_early = []
mc_late = []
for trial in range(1000):
    signal = true_A0 * np.exp(-(n_full / 15.0) ** 1.3)
    signal += np.random.randn(len(signal)) * 0.02
    signal = np.maximum(signal, 1e-5)

    r5 = fit_stretched(np.arange(1, 6), signal[1:6])
    r20 = fit_stretched(np.arange(1, 21), signal[1:21])
    if r5 and r20 and r5['R2'] > 0.9 and r20['R2'] > 0.9:
        mc_drifts.append(r20['alpha'] - r5['alpha'])
        mc_early.append(r5['alpha'])
        mc_late.append(r20['alpha'])

if mc_drifts:
    mc_drifts = np.array(mc_drifts)
    mc_early = np.array(mc_early)
    mc_late = np.array(mc_late)
    print(f"    Early (1-5):  mean = {np.mean(mc_early):.4f} +/- {np.std(mc_early):.4f}")
    print(f"    Full (1-20):  mean = {np.mean(mc_late):.4f} +/- {np.std(mc_late):.4f}")
    print(f"    Drift:        mean = {np.mean(mc_drifts):+.4f} +/- {np.std(mc_drifts):.4f}")
    print(f"    Drift > +0.1: {np.sum(mc_drifts > 0.1) / len(mc_drifts) * 100:.1f}%")
    print(f"    Drift > +0.3: {np.sum(mc_drifts > 0.3) / len(mc_drifts) * 100:.1f}%")

# Compare to real data drift of +0.315
print(f"\n    REAL DATA drift (Xiang stabilizer Z, 6 operators): +0.315")
if mc_drifts is not None and len(mc_drifts) > 0:
    pval = np.mean(mc_drifts >= 0.315)
    print(f"    Fraction of MC trials with drift >= +0.315: {pval*100:.1f}%")
    if pval < 0.05:
        print(f"    ==> DRIFT IS SIGNIFICANT (p < 0.05): not a fitting artifact")
    else:
        print(f"    ==> DRIFT IS NOT SIGNIFICANT: could be fitting artifact")

# Also test with true_alpha=0.9 (since early window gives ~0.87)
print("\n  MONTE CARLO: 1000 trials, true_alpha=0.9, noise=0.02")
mc_drifts_09 = []
for trial in range(1000):
    signal = 1.0 * np.exp(-(n_full / 15.0) ** 0.9)
    signal += np.random.randn(len(signal)) * 0.02
    signal = np.maximum(signal, 1e-5)
    r5 = fit_stretched(np.arange(1, 6), signal[1:6])
    r20 = fit_stretched(np.arange(1, 21), signal[1:21])
    if r5 and r20 and r5['R2'] > 0.9 and r20['R2'] > 0.9:
        mc_drifts_09.append(r20['alpha'] - r5['alpha'])

if mc_drifts_09:
    mc_drifts_09 = np.array(mc_drifts_09)
    print(f"    Drift: mean = {np.mean(mc_drifts_09):+.4f} +/- {np.std(mc_drifts_09):.4f}")
    print(f"    Drift > +0.3: {np.sum(mc_drifts_09 > 0.3) / len(mc_drifts_09) * 100:.1f}%")

# And with true_alpha=1.2 (intermediate)
print("\n  MONTE CARLO: 1000 trials, true_alpha=1.2, noise=0.02")
mc_drifts_12 = []
for trial in range(1000):
    signal = 1.0 * np.exp(-(n_full / 15.0) ** 1.2)
    signal += np.random.randn(len(signal)) * 0.02
    signal = np.maximum(signal, 1e-5)
    r5 = fit_stretched(np.arange(1, 6), signal[1:6])
    r20 = fit_stretched(np.arange(1, 21), signal[1:21])
    if r5 and r20 and r5['R2'] > 0.9 and r20['R2'] > 0.9:
        mc_drifts_12.append(r20['alpha'] - r5['alpha'])

if mc_drifts_12:
    mc_drifts_12 = np.array(mc_drifts_12)
    print(f"    Drift: mean = {np.mean(mc_drifts_12):+.4f} +/- {np.std(mc_drifts_12):.4f}")
    print(f"    Drift > +0.3: {np.sum(mc_drifts_12 > 0.3) / len(mc_drifts_12) * 100:.1f}%")


# ============================================================
# CONTROL 2: Does averaging change apparent alpha?
# ============================================================
print("\n\n" + "=" * 70)
print("CONTROL 2: SYNTHETIC DATA -- SUBSYSTEM SIZE ARTIFACT")
print("=" * 70)
print("""
  Generate k independent qubits with true alpha, add noise.
  Average k of them, fit alpha. Does averaging change alpha?
""")

for true_alpha in [0.4, 0.8, 1.0, 1.3]:
    print(f"\n  true_alpha = {true_alpha}:")
    n_pts = 10
    ns = np.arange(n_pts + 1)

    for k in [1, 2, 4, 8, 18]:
        trial_alphas = []
        for trial in range(500):
            # Generate k independent qubit signals
            signals = np.zeros((k, n_pts + 1))
            for qi in range(k):
                # Each qubit has same alpha but different n_star and noise
                nstar_i = 3.0 + np.random.randn() * 0.5  # ~3 +/- 0.5
                nstar_i = max(nstar_i, 1.0)
                sig = np.exp(-(ns / nstar_i) ** true_alpha)
                sig += np.random.randn(n_pts + 1) * 0.05
                signals[qi] = np.maximum(sig, 1e-5)

            avg = np.mean(signals, axis=0)
            r = fit_stretched(np.arange(1, n_pts + 1), avg[1:])
            if r and r['R2'] > 0.85:
                trial_alphas.append(r['alpha'])

        if trial_alphas:
            m = np.mean(trial_alphas)
            s = np.std(trial_alphas)
            print(f"    k={k:>2}: fitted alpha = {m:.4f} +/- {s:.4f}  "
                  f"(bias = {m - true_alpha:+.4f}, {len(trial_alphas)}/500 valid)")


# ============================================================
# CONTROL 3: Disorder sweep with all operators averaged
# ============================================================
print("\n\n" + "=" * 70)
print("CONTROL 3: DISORDER SWEEP -- ALL 6 OPERATORS")
print("=" * 70)

d4a = sio.loadmat(os.path.join(XIANG_2024_BASE, 'main_figure4', 'fig4a.mat'))
disorders = d4a['disorders'].flatten()
Zs_dis = d4a['Zs_exp_avg']
fc = np.arange(0, 41, 2)

# For each disorder: average all 6 operators' subharmonic envelopes, then fit
print("\n  Method: Average the 6 stabilizer Z subharmonic envelopes, then fit")
print(f"\n  {'Disorder':>10} {'Alpha(avg)':>12} {'err':>8} {'R2':>8} {'Alpha(Z0)':>12} {'Alpha(indiv)':>14}")
print("  " + "-" * 66)

for di, dis in enumerate(disorders):
    if dis > 1.5:
        break

    # Average envelope approach
    envs = []
    indiv_alphas = []
    for oi in range(6):
        sig = Zs_dis[oi, di, fc]
        env = sig * ((-1.0)**np.arange(len(sig)))
        envs.append(env)
        r_i = fit_stretched(np.arange(1, 21), env[1:])
        if r_i and r_i['R2'] > 0.9:
            indiv_alphas.append(r_i['alpha'])

    avg_env = np.mean(envs, axis=0)
    r_avg = fit_stretched(np.arange(1, 21), avg_env[1:])

    # Z0 only
    r_z0 = fit_stretched(np.arange(1, 21), envs[0][1:])

    avg_str = f"{r_avg['alpha']:.4f}" if r_avg and r_avg['R2'] > 0.9 else "FAIL"
    avg_err = f"{r_avg['alpha_err']:.4f}" if r_avg and r_avg['R2'] > 0.9 else ""
    avg_r2 = f"{r_avg['R2']:.4f}" if r_avg and r_avg['R2'] > 0.9 else ""
    z0_str = f"{r_z0['alpha']:.4f}" if r_z0 and r_z0['R2'] > 0.9 else "FAIL"
    indiv_str = f"{np.mean(indiv_alphas):.4f}" if indiv_alphas else "FAIL"

    print(f"  {dis:>10.1f} {avg_str:>12} {avg_err:>8} {avg_r2:>8} {z0_str:>12} {indiv_str:>14}")


# ============================================================
# KEY TEST: Non-overlapping window comparison
# ============================================================
print("\n\n" + "=" * 70)
print("KEY TEST: NON-OVERLAPPING WINDOWS (independent segments)")
print("=" * 70)
print("""
  The cumulative test is susceptible to fitting artifact because
  all windows share the early data. Use NON-OVERLAPPING windows
  to test if alpha GENUINELY increases in later cycles.
""")

d2a = sio.loadmat(os.path.join(XIANG_2024_BASE, 'main_figure2', 'fig2a.mat'))
Zs = d2a['Zs_exp_avg']

# Split 20 cycles into four 5-cycle non-overlapping windows
windows = [(1, 6), (6, 11), (11, 16), (16, 21)]
print(f"  {'Window':>12}", end='')
for oi in range(6):
    print(f"  {'Z'+str(oi):>8}", end='')
print(f"  {'Mean':>8}")
print("  " + "-" * 66)

window_means = []
for ws, we in windows:
    row = []
    print(f"  [{ws:>2}-{we:>2}]    ", end='')
    for oi in range(6):
        sig_fc = Zs[oi, fc]
        env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))
        n_vals = np.arange(ws, we)
        env_vals = env[ws:we]
        r = fit_stretched(n_vals, env_vals)
        if r and r['R2'] > 0.8:
            row.append(r['alpha'])
            print(f"  {r['alpha']:>8.3f}", end='')
        else:
            print(f"  {'---':>8}", end='')
    if row:
        m = np.mean(row)
        window_means.append(m)
        print(f"  {m:>8.3f}")
    else:
        print(f"  {'N/A':>8}")

if len(window_means) >= 2:
    print(f"\n  Window 1 mean: {window_means[0]:.3f}")
    if len(window_means) >= 3:
        print(f"  Window 3 mean: {window_means[2]:.3f}")
        print(f"  Delta: {window_means[2] - window_means[0]:+.3f}")

# Also try 3 non-overlapping windows of 6-7 cycles each
windows2 = [(1, 8), (8, 14), (14, 21)]
print(f"\n  Three 7-cycle windows:")
print(f"  {'Window':>12}", end='')
for oi in range(6):
    print(f"  {'Z'+str(oi):>8}", end='')
print(f"  {'Mean':>8}")
print("  " + "-" * 66)

window_means2 = []
for ws, we in windows2:
    row = []
    print(f"  [{ws:>2}-{we:>2}]    ", end='')
    for oi in range(6):
        sig_fc = Zs[oi, fc]
        env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))
        n_vals = np.arange(ws, we)
        env_vals = env[ws:we]
        r = fit_stretched(n_vals, env_vals)
        if r and r['R2'] > 0.8:
            row.append(r['alpha'])
            print(f"  {r['alpha']:>8.3f}", end='')
        else:
            print(f"  {'---':>8}", end='')
    if row:
        m = np.mean(row)
        window_means2.append(m)
        print(f"  {m:>8.3f}")
    else:
        print(f"  {'N/A':>8}")

if len(window_means2) >= 2:
    print(f"\n  Early (1-7):  {window_means2[0]:.3f}")
    print(f"  Late (14-20): {window_means2[-1]:.3f}")
    print(f"  Delta: {window_means2[-1] - window_means2[0]:+.3f}")

# ============================================================
# Does alpha(n) = A0*exp(-(n/n*)^alpha) even fit the data?
# Compare residuals of alpha=1.3 fixed vs free alpha
# ============================================================
print("\n\n" + "=" * 70)
print("FIT QUALITY: FIXED alpha=1.3 vs FREE alpha")
print("=" * 70)

def stretched_exp_fixed(n, A0, n_star):
    return A0 * np.exp(-(n / n_star) ** 1.3)

for oi in range(6):
    sig_fc = Zs[oi, fc]
    env = sig_fc * ((-1.0)**np.arange(len(sig_fc)))
    n_fit = np.arange(1, 21)
    e_fit = env[1:21]

    # Free fit
    r_free = fit_stretched(n_fit, e_fit)

    # Fixed alpha=1.3
    try:
        popt_f, _ = curve_fit(stretched_exp_fixed, n_fit, e_fit,
                              p0=[1.0, 15.0], bounds=([0, 0.1], [2.0, 100]),
                              maxfev=100000)
        res_f = e_fit - stretched_exp_fixed(n_fit, *popt_f)
        ss_res_f = np.sum(res_f**2)
        ss_tot = np.sum((e_fit - np.mean(e_fit))**2)
        R2_f = 1 - ss_res_f/ss_tot
        rmse_f = np.sqrt(np.mean(res_f**2))
    except:
        R2_f = 0
        rmse_f = 999

    if r_free:
        res_free = e_fit - stretched_exp(n_fit, r_free['A0'], r_free['n_star'], r_free['alpha'])
        rmse_free = np.sqrt(np.mean(res_free**2))
        print(f"  Z{oi}: free alpha={r_free['alpha']:.3f} (R2={r_free['R2']:.5f}, RMSE={rmse_free:.5f})"
              f"  |  fixed alpha=1.3 (R2={R2_f:.5f}, RMSE={rmse_f:.5f})"
              f"  |  delta_R2 = {r_free['R2'] - R2_f:.6f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("CONTROL TEST SUMMARY")
print("=" * 70)
print("""
This will determine whether the three "supported" predictions
survive scrutiny or are explained by artifacts.
""")
