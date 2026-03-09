"""
P_MED_9 — KWW on Alpha/Gamma Spectral Ratio Time Series
========================================================
The Phase 1 instrument applied to Phase 2 raw EEG.

Phase 1 found HYT alpha/gamma = 1.361 (|dev from 4/3| = 0.028) using
pre-computed spectral summaries. Phase 2 broadband ACF gave null because
it measured the wrong signal — temporal decay of raw EEG, not cross-
frequency coupling.

This script computes the SAME measurement that found 4/3:
  1. Per-channel Welch PSD in sliding windows
  2. Alpha (8-13 Hz) and gamma (30-80 Hz) power per channel per window
  3. Mean alpha/gamma ratio across channels -> ratio time series R(t)
  4. ACF of R(t) -> KWW fit -> alpha_KWW

Prediction: HYT alpha_KWW in window |alpha - 4/3| < 0.15
            with cleaner statistics than Phase 1 (64 subjects vs summary).

Data: Zenodo 57911 raw EEG (EEG_datasets/)
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sp
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, kruskal, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, MEDITATION_RAW, FIGURES_DIR, REPORTS_DIR

# --- CONFIGURATION ---
DATA_DIR = os.path.join(MEDITATION_RAW, 'EEG_datasets')
OUTPUT_DIR = FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = {
    'CTR': {'hours': 0,     'color': '#AAAAAA', 'label': 'CTR (0h)'},
    'ISY': {'hours': 2625,  'color': '#5599DD', 'label': 'ISY (2625h)'},
    'VIP': {'hours': 9201,  'color': '#DD8833', 'label': 'VIP (9201h)'},
    'HYT': {'hours': 15475, 'color': '#AA3333', 'label': 'HYT (15475h)'},
}
GROUP_ORDER = ['CTR', 'ISY', 'VIP', 'HYT']

PREFIX_MAP = {'CTR': 'CTR', 'ISY': 'SNY', 'VIP': 'VIP', 'HYT': 'HT'}
CONDITION_MAP = {
    'CTR': '_breath2.set',
    'ISY': '_medit.set',
    'VIP': '_medit.set',
    'HYT': '_medit.set',
}
NON_EEG_PREFIXES = ('EXG', 'M1', 'M2', 'GSR', 'Erg', 'Status', 'STI')

# Spectral parameters (same as Phase 1)
FREQS_ALPHA = (8, 13)
FREQS_GAMMA = (30, 80)

# Ratio time series: window for PSD computation
PSD_WINDOW_SEC = 4.0    # Welch segment within each ratio window
RATIO_WINDOW_SEC = 4.0  # Window for each ratio sample
RATIO_STEP_SEC = 2.0    # Step between ratio samples (2s -> ~0.5 Hz ratio series)

# KWW fitting
ACF_MAXLAG = 50         # max lag in ratio-series samples (not EEG samples)
ALPHA_43 = 4 / 3
ALPHA_43_WINDOW = 0.15
ZERO_POINT = 1 / 3


# --- FILE DISCOVERY ---
def find_eeg_files(data_dir):
    files_by_group = {g: [] for g in GROUP_ORDER}
    all_sets = sorted(glob.glob(os.path.join(data_dir, '*.set')))
    for f in all_sets:
        fname = os.path.basename(f)
        for g in GROUP_ORDER:
            if fname.startswith(PREFIX_MAP[g]) and fname.endswith(CONDITION_MAP[g]):
                files_by_group[g].append(f)
                break
    for g in GROUP_ORDER:
        print(f"  {g}: {len(files_by_group[g])} files")
    return files_by_group


# --- DATA LOADING ---
def load_eeg_set(filepath):
    try:
        import mne
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        eeg_picks = [i for i, ch in enumerate(raw.ch_names)
                     if not any(ch.startswith(p) for p in NON_EEG_PREFIXES)]
        data = raw.get_data(picks=eeg_picks)
        srate = raw.info['sfreq']
        ch_names = [raw.ch_names[i] for i in eeg_picks]
        return data, srate, ch_names
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, None, None


# --- RATIO TIME SERIES ---
def compute_ratio_timeseries(data, srate):
    """
    Compute alpha/gamma spectral power ratio time series.
    For each window: Welch PSD per channel -> alpha/gamma -> mean across channels.
    Returns: ratio_series (1D), t_centres (1D, in seconds)
    """
    n_ch, n_times = data.shape
    win_n = int(RATIO_WINDOW_SEC * srate)
    step_n = int(RATIO_STEP_SEC * srate)
    nperseg = min(int(PSD_WINDOW_SEC * srate), win_n)

    ratio_series = []
    t_centres = []

    for start in range(0, n_times - win_n, step_n):
        window = data[:, start:start + win_n]

        # Welch PSD per channel
        f, psd = sp.welch(window, fs=srate, nperseg=nperseg, axis=1)
        alpha_mask = (f >= FREQS_ALPHA[0]) & (f <= FREQS_ALPHA[1])
        gamma_mask = (f >= FREQS_GAMMA[0]) & (f <= FREQS_GAMMA[1])

        alpha_pwr = psd[:, alpha_mask].mean(axis=1)  # (n_ch,)
        gamma_pwr = psd[:, gamma_mask].mean(axis=1)
        gamma_pwr = np.where(gamma_pwr < 1e-30, 1e-30, gamma_pwr)

        ratio = alpha_pwr / gamma_pwr  # per-channel ratio
        mean_ratio = ratio.mean()      # spatial mean

        ratio_series.append(mean_ratio)
        t_centres.append((start + win_n / 2) / srate)

    return np.array(ratio_series), np.array(t_centres)


# --- KWW FITTING ---
def kww(t, A, tau, alpha):
    return A * np.exp(-(t / tau) ** alpha)


def compute_acf(signal, maxlag):
    n = len(signal)
    signal = signal - signal.mean()
    var = np.var(signal)
    if var == 0:
        return np.zeros(maxlag)
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[n-1:n-1+maxlag] / (var * n)
    return acf


def fit_kww_to_acf(acf, dt):
    """Fit KWW to ACF. Returns (A, tau, alpha, R2) or None."""
    if len(acf) < 10 or acf[0] <= 0:
        return None

    acf_norm = acf / acf[0]
    t = np.arange(len(acf_norm)) * dt

    # Truncate at first zero crossing or 0.01
    zero_idx = np.where(acf_norm <= 0.01)[0]
    if len(zero_idx) > 0 and zero_idx[0] > 5:
        acf_norm = acf_norm[:zero_idx[0]]
        t = t[:len(acf_norm)]

    if len(acf_norm) < 8:
        return None

    try:
        popt, _ = curve_fit(
            kww, t, acf_norm,
            p0=[1.0, t[len(t)//3], 1.0],
            bounds=([0.5, dt, 0.1], [1.5, t[-1]*5, 3.0]),
            maxfev=5000
        )
        A, tau, alpha = popt
        y_pred = kww(t, *popt)
        ss_res = np.sum((acf_norm - y_pred)**2)
        ss_tot = np.sum((acf_norm - acf_norm.mean())**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return A, tau, alpha, R2
    except Exception:
        return None


# --- PER-SUBJECT ANALYSIS ---
def process_subject(filepath, group, subject_id):
    """Compute alpha/gamma ratio time series, then KWW on its ACF."""
    data, srate, ch_names = load_eeg_set(filepath)
    if data is None:
        return None

    n_ch, n_times = data.shape
    duration = n_times / srate

    # Compute ratio time series
    ratio_ts, t_centres = compute_ratio_timeseries(data, srate)

    if len(ratio_ts) < 20:
        print(f"    Too few ratio samples ({len(ratio_ts)}), skipping")
        return None

    # Time step in the ratio series
    dt_ratio = RATIO_STEP_SEC

    # --- Full-session KWW fit ---
    maxlag = min(ACF_MAXLAG, len(ratio_ts) // 2)
    acf_full = compute_acf(ratio_ts, maxlag)
    kww_full = fit_kww_to_acf(acf_full, dt_ratio)

    if kww_full is not None:
        A_f, tau_f, alpha_f, R2_f = kww_full
    else:
        A_f, tau_f, alpha_f, R2_f = np.nan, np.nan, np.nan, np.nan

    # --- Windowed KWW (sliding windows on ratio series) ---
    # Use 60-sample windows (120 seconds at 2s step) with 50% overlap
    kww_win_n = min(60, len(ratio_ts) // 2)
    kww_step = max(1, kww_win_n // 2)
    kww_maxlag_w = min(30, kww_win_n // 2)

    windowed_alphas = []
    windowed_taus = []
    windowed_r2s = []
    windowed_t = []

    for start in range(0, len(ratio_ts) - kww_win_n, kww_step):
        segment = ratio_ts[start:start + kww_win_n]
        acf_seg = compute_acf(segment, kww_maxlag_w)
        fit = fit_kww_to_acf(acf_seg, dt_ratio)
        if fit is not None:
            A_w, tau_w, alpha_w, R2_w = fit
            if R2_w > 0.3:
                windowed_alphas.append(alpha_w)
                windowed_taus.append(tau_w)
                windowed_r2s.append(R2_w)
                windowed_t.append(t_centres[start + kww_win_n // 2]
                                  if start + kww_win_n // 2 < len(t_centres)
                                  else t_centres[-1])

    windowed_alphas = np.array(windowed_alphas)
    windowed_taus = np.array(windowed_taus)
    windowed_r2s = np.array(windowed_r2s)

    # Summary
    result = {
        'subject': subject_id,
        'group': group,
        'hours': GROUPS[group]['hours'],
        'n_channels': n_ch,
        'duration_s': duration,
        'n_ratio_samples': len(ratio_ts),
        # Full-session KWW
        'full_alpha': float(alpha_f),
        'full_tau': float(tau_f),
        'full_R2': float(R2_f),
        'full_in_window': bool(abs(alpha_f - ALPHA_43) < ALPHA_43_WINDOW) if not np.isnan(alpha_f) else False,
        # Windowed KWW
        'n_kww_fits': len(windowed_alphas),
        'mean_alpha': float(np.mean(windowed_alphas)) if len(windowed_alphas) > 0 else np.nan,
        'std_alpha': float(np.std(windowed_alphas)) if len(windowed_alphas) > 0 else np.nan,
        'median_alpha': float(np.median(windowed_alphas)) if len(windowed_alphas) > 0 else np.nan,
        'frac_near_43': float(np.mean(np.abs(windowed_alphas - ALPHA_43) < ALPHA_43_WINDOW))
                        if len(windowed_alphas) > 0 else 0.0,
        'mean_R2': float(np.mean(windowed_r2s)) if len(windowed_r2s) > 0 else np.nan,
        # Ratio statistics
        'ratio_mean': float(np.mean(ratio_ts)),
        'ratio_std': float(np.std(ratio_ts)),
        'ratio_median': float(np.median(ratio_ts)),
        # Time series for plots
        'ratio_ts': ratio_ts.tolist(),
        'windowed_alphas': windowed_alphas.tolist(),
        'windowed_t': windowed_t,
        'acf_full': acf_full.tolist(),
    }

    in_w = "IN WINDOW" if result['full_in_window'] else ""
    print(f"    {n_ch}ch, {duration:.0f}s, ratio_mean={result['ratio_mean']:.2f}, "
          f"full_alpha={alpha_f:.3f} R2={R2_f:.3f} {in_w}, "
          f"windowed: {len(windowed_alphas)} fits, "
          f"mean={result['mean_alpha']:.3f}" if not np.isnan(result['mean_alpha']) else
          f"    {n_ch}ch, {duration:.0f}s, ratio_mean={result['ratio_mean']:.2f}, "
          f"full_alpha={alpha_f:.3f} R2={R2_f:.3f}")

    return result


# --- MAIN ---
def main():
    print("=" * 70)
    print("P_MED_9: KWW on Alpha/Gamma Spectral Ratio Time Series")
    print("The Phase 1 instrument applied to Phase 2 raw EEG")
    print("=" * 70)

    if not os.path.isdir(DATA_DIR):
        print(f"\nERROR: {DATA_DIR} not found")
        return

    print("\nFinding files...")
    files_by_group = find_eeg_files(DATA_DIR)

    all_results = []
    group_results = {g: [] for g in GROUP_ORDER}

    for g in GROUP_ORDER:
        files = files_by_group[g]
        if not files:
            continue
        print(f"\n--- {GROUPS[g]['label']} ({len(files)} subjects) ---")
        for i, fp in enumerate(files):
            sid = f"{g}_{i+1:02d}"
            print(f"  {sid}: {os.path.basename(fp)}")
            r = process_subject(fp, g, sid)
            if r is not None:
                all_results.append(r)
                group_results[g].append(r)

    if not all_results:
        print("\nNo results.")
        return

    # === GROUP-LEVEL RESULTS ===
    print("\n" + "=" * 70)
    print("GROUP-LEVEL RESULTS")
    print("=" * 70)

    print(f"\n{'Group':<8} {'N':<4} {'Full alpha':<14} {'Full R2':<10} "
          f"{'In Window':<12} {'Win mean':<12} {'Win std':<10} {'Ratio mean':<12}")
    print("-" * 90)

    for g in GROUP_ORDER:
        rs = group_results[g]
        if not rs:
            continue
        full_as = [r['full_alpha'] for r in rs if not np.isnan(r['full_alpha'])]
        full_r2 = [r['full_R2'] for r in rs if not np.isnan(r['full_R2'])]
        in_w = sum(1 for r in rs if r['full_in_window'])
        win_as = [r['mean_alpha'] for r in rs if not np.isnan(r['mean_alpha'])]
        win_stds = [r['std_alpha'] for r in rs if not np.isnan(r['std_alpha'])]
        ratios = [r['ratio_mean'] for r in rs]

        fa_str = f"{np.mean(full_as):.4f}+/-{np.std(full_as):.3f}" if full_as else "N/A"
        fr_str = f"{np.mean(full_r2):.3f}" if full_r2 else "N/A"
        wm_str = f"{np.mean(win_as):.4f}" if win_as else "N/A"
        ws_str = f"{np.mean(win_stds):.4f}" if win_stds else "N/A"

        print(f"{g:<8} {len(rs):<4} {fa_str:<14} {fr_str:<10} "
              f"{in_w}/{len(rs):<10} {wm_str:<12} {ws_str:<10} {np.mean(ratios):<12.2f}")

    # === STATISTICAL TESTS ===
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # Collect per-group full-session alphas
    ga = {}
    for g in GROUP_ORDER:
        vals = [r['full_alpha'] for r in group_results[g]
                if not np.isnan(r['full_alpha']) and r['full_R2'] > 0.3]
        if vals:
            ga[g] = vals

    # P_MED_9a: KWW alpha near 4/3 for HYT
    print("\nP_MED_9a: Full-session alpha_KWW near 4/3")
    for g in GROUP_ORDER:
        if g in ga:
            med = np.median(ga[g])
            mean = np.mean(ga[g])
            dev_med = abs(med - ALPHA_43)
            dev_mean = abs(mean - ALPHA_43)
            status = "IN WINDOW" if dev_med < ALPHA_43_WINDOW else ""
            print(f"  {g}: median={med:.4f} |dev|={dev_med:.4f}, "
                  f"mean={mean:.4f} |dev|={dev_mean:.4f} {status}")

    # Kruskal-Wallis
    valid = [g for g in GROUP_ORDER if g in ga]
    if len(valid) >= 2:
        kw_stat, kw_p = kruskal(*[ga[g] for g in valid])
        print(f"\n  Kruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.4f}")

    # Experience gradient
    if len(valid) >= 3:
        exp_h = [GROUPS[g]['hours'] for g in valid]
        meds = [np.median(ga[g]) for g in valid]
        rho, p_rho = spearmanr(exp_h, meds)
        print(f"  Spearman (experience vs median alpha): rho={rho:.4f}, p={p_rho:.4f}")

    # P_MED_9b: Variance ordering
    gs = {}
    for g in GROUP_ORDER:
        vals = [r['std_alpha'] for r in group_results[g]
                if not np.isnan(r['std_alpha'])]
        if vals:
            gs[g] = vals

    print("\nP_MED_9b: Variance ordering (driven vs self-sustaining)")
    if all(g in gs for g in GROUP_ORDER):
        means = {g: np.mean(gs[g]) for g in GROUP_ORDER}
        ranked = sorted(GROUP_ORDER, key=lambda g: means[g], reverse=True)
        print(f"  Actual:    {' > '.join([f'{g}({means[g]:.4f})' for g in ranked])}")
        print(f"  Predicted: VIP > CTR > ISY ~ HYT")

    if 'VIP' in gs and 'ISY' in gs:
        u, p = mannwhitneyu(gs['VIP'], gs['ISY'], alternative='greater')
        print(f"  VIP std > ISY std: U={u:.1f}, p={p:.4f}")
    if 'VIP' in gs and 'HYT' in gs:
        u, p = mannwhitneyu(gs['VIP'], gs['HYT'], alternative='greater')
        print(f"  VIP std > HYT std: U={u:.1f}, p={p:.4f}")
    if 'HYT' in gs and 'CTR' in gs:
        u, p = mannwhitneyu(gs['HYT'], gs['CTR'], alternative='less')
        print(f"  HYT std < CTR std: U={u:.1f}, p={p:.4f}")

    # P_MED_9c: Alpha/gamma ratio convergence
    print("\nP_MED_9c: Mean alpha/gamma ratio by group")
    for g in GROUP_ORDER:
        rs = group_results[g]
        if rs:
            ratios = [r['ratio_mean'] for r in rs]
            print(f"  {g}: ratio = {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}")

    # P_MED_9d: Fraction near 4/3 (windowed)
    print("\nP_MED_9d: Fraction of windowed fits near 4/3")
    for g in GROUP_ORDER:
        rs = group_results[g]
        if rs:
            fracs = [r['frac_near_43'] for r in rs]
            print(f"  {g}: mean frac = {np.mean(fracs):.4f}")

    # Per-subject detail for HYT
    print("\nHYT per-subject detail:")
    for r in group_results['HYT']:
        in_w = "***" if r['full_in_window'] else ""
        print(f"  {r['subject']}: full_alpha={r['full_alpha']:.4f} R2={r['full_R2']:.3f} "
              f"ratio_mean={r['ratio_mean']:.2f} win_mean={r['mean_alpha']:.4f} {in_w}")

    # === PLOTS ===
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Full-session alpha by group
    ax = axes[0, 0]
    for i, g in enumerate(GROUP_ORDER):
        if g in ga:
            vals = ga[g]
            jitter = np.random.default_rng(42).normal(0, 0.05, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                      color=GROUPS[g]['color'], alpha=0.6, s=50,
                      edgecolors='black', linewidths=0.3)
            ax.hlines(np.median(vals), i-0.2, i+0.2,
                     colors=GROUPS[g]['color'], linewidths=3, zorder=5)
    ax.axhline(y=ALPHA_43, color='red', ls='--', lw=2, label='4/3')
    ax.axhspan(ALPHA_43 - ALPHA_43_WINDOW, ALPHA_43 + ALPHA_43_WINDOW,
               alpha=0.1, color='red')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{g}\n({GROUPS[g]['hours']}h)" for g in GROUP_ORDER])
    ax.set_ylabel('alpha_KWW (full session)')
    ax.set_title('P_MED_9a: KWW alpha on ratio ACF')
    ax.legend()

    # Plot 2: Alpha vs experience (individual subjects)
    ax = axes[0, 1]
    for g in GROUP_ORDER:
        if g in ga:
            hours = GROUPS[g]['hours']
            vals = ga[g]
            ax.scatter([hours]*len(vals), vals,
                      c=GROUPS[g]['color'], alpha=0.5, s=40)
            ax.errorbar(hours, np.mean(vals), yerr=np.std(vals),
                       fmt='D', color=GROUPS[g]['color'], ms=10, capsize=5, zorder=5)
    ax.axhline(y=ALPHA_43, color='red', ls='--', lw=2)
    ax.set_xlabel('Experience (hours)')
    ax.set_ylabel('alpha_KWW')
    ax.set_title('Alpha vs Experience')

    # Plot 3: Representative ACFs by group
    ax = axes[0, 2]
    for g in GROUP_ORDER:
        rs = group_results[g]
        if rs:
            # Pick subject with highest R2
            best = max(rs, key=lambda r: r['full_R2'] if not np.isnan(r['full_R2']) else 0)
            acf = np.array(best['acf_full'])
            t_acf = np.arange(len(acf)) * RATIO_STEP_SEC
            ax.plot(t_acf, acf / acf[0] if acf[0] > 0 else acf,
                   color=GROUPS[g]['color'], lw=2,
                   label=f"{g} (a={best['full_alpha']:.2f})")
    ax.set_xlabel('Lag (seconds)')
    ax.set_ylabel('ACF (normalized)')
    ax.set_title('Best-fit ACF per group')
    ax.legend(fontsize=8)

    # Plot 4: Windowed alpha histograms
    ax = axes[1, 0]
    for g in GROUP_ORDER:
        rs = group_results[g]
        all_wa = []
        for r in rs:
            all_wa.extend(r['windowed_alphas'])
        if all_wa:
            ax.hist(all_wa, bins=25, alpha=0.4, color=GROUPS[g]['color'],
                   label=GROUPS[g]['label'], density=True)
    ax.axvline(x=ALPHA_43, color='red', ls='--', lw=2, label='4/3')
    ax.set_xlabel('alpha_KWW (windowed)')
    ax.set_ylabel('Density')
    ax.set_title('Windowed KWW alpha distribution')
    ax.legend(fontsize=7)

    # Plot 5: Ratio time series (best HYT subject)
    ax = axes[1, 1]
    hyt_rs = group_results['HYT']
    if hyt_rs:
        best_hyt = max(hyt_rs, key=lambda r: r['full_R2'] if not np.isnan(r['full_R2']) else 0)
        ratio = np.array(best_hyt['ratio_ts'])
        t_r = np.arange(len(ratio)) * RATIO_STEP_SEC / 60  # minutes
        ax.plot(t_r, ratio, color='#AA3333', lw=1)
        ax.axhline(y=ALPHA_43, color='red', ls='--', lw=1, alpha=0.5, label='4/3')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Alpha/Gamma ratio')
        ax.set_title(f'Ratio time series - best HYT ({best_hyt["subject"]})')
        ax.legend(fontsize=8)

    # Plot 6: Std(alpha) by group (variance signature)
    ax = axes[1, 2]
    for i, g in enumerate(GROUP_ORDER):
        if g in gs:
            vals = gs[g]
            jitter = np.random.default_rng(42).normal(0, 0.05, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                      color=GROUPS[g]['color'], alpha=0.6, s=40,
                      edgecolors='black', linewidths=0.3)
            ax.bar(i, np.mean(vals), color=GROUPS[g]['color'], alpha=0.3, width=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{g}\n({GROUPS[g]['hours']}h)" for g in GROUP_ORDER])
    ax.set_ylabel('std(alpha_KWW)')
    ax.set_title('Variance signature\n(VIP > CTR > ISY ~ HYT?)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pmed9_results.png'), dpi=150)
    print("Saved: pmed9_results.png")

    # === SAVE JSON ===
    def ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    with open(os.path.join(REPORTS_DIR, 'pmed9_results.json'), 'w') as f:
        json.dump([{k: ser(v) for k, v in r.items()} for r in all_results],
                  f, indent=2, default=ser)
    print("Saved: pmed9_results.json")

    # === FINAL REGISTRY BLOCK ===
    reg = []
    reg.append("\n=== P_MED_9 RESULTS ===")
    reg.append("KWW on alpha/gamma spectral ratio ACF (Phase 1 instrument on Phase 2 data)")
    reg.append("")
    reg.append(f"{'Group':<8} {'N':>3} {'Full alpha':>12} {'|dev 4/3|':>10} {'In Window':>10} "
               f"{'Win mean':>10} {'Win std':>10} {'Ratio':>8}")
    reg.append("-" * 75)

    for g in GROUP_ORDER:
        rs = group_results[g]
        if not rs:
            continue
        full_as = [r['full_alpha'] for r in rs if not np.isnan(r['full_alpha']) and r['full_R2'] > 0.3]
        in_w = sum(1 for r in rs if r['full_in_window'])
        win_as = [r['mean_alpha'] for r in rs if not np.isnan(r['mean_alpha'])]
        win_stds = [r['std_alpha'] for r in rs if not np.isnan(r['std_alpha'])]
        ratios = [r['ratio_mean'] for r in rs]

        fa = np.mean(full_as) if full_as else np.nan
        dev = abs(fa - ALPHA_43) if not np.isnan(fa) else np.nan
        wm = np.mean(win_as) if win_as else np.nan
        ws = np.mean(win_stds) if win_stds else np.nan

        reg.append(f"{g:<8} {len(rs):>3} {fa:>12.4f} {dev:>10.4f} {in_w:>5}/{len(rs):<4} "
                   f"{wm:>10.4f} {ws:>10.4f} {np.mean(ratios):>8.2f}")

    # Verdict
    reg.append("")
    if 'HYT' in ga:
        hyt_med = np.median(ga['HYT'])
        dev = abs(hyt_med - ALPHA_43)
        if dev < ALPHA_43_WINDOW:
            reg.append(f"P_MED_9: CONFIRMED — HYT median alpha = {hyt_med:.4f}, "
                       f"|dev| = {dev:.4f} IN WINDOW")
        else:
            reg.append(f"P_MED_9: NOT CONFIRMED — HYT median alpha = {hyt_med:.4f}, "
                       f"|dev| = {dev:.4f}")

    reg.append("=== END ===")
    registry_text = "\n".join(reg)

    print(registry_text)

    with open(os.path.join(REPORTS_DIR, 'pmed9_summary.txt'), 'w') as f:
        f.write(registry_text)
    print("\nSaved: pmed9_summary.txt")
    print("\nP_MED_9 complete.")


if __name__ == '__main__':
    main()
