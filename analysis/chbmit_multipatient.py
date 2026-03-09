#!/usr/bin/env python3
"""
CHB-MIT EEG Seizure Analysis — Multi-Patient Extension
Merkabit Geometric Signature — Tenth Dataset
March 2026

Extends chb01 analysis to chb02-chb05 (25 total seizures across 5 patients).
Focus: peak alpha in pre-ictal temporal evolution.
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.stats import ttest_1samp, mannwhitneyu, binomtest
import mne
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import CHBMIT_DATA, FIGURES_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
DATA_BASE = CHBMIT_DATA
OUT_DIR   = FIGURES_DIR
SFREQ     = 256
FILT_LO   = 1.0
FILT_HI   = 40.0
PRE_ICTAL_DUR  = 300  # 5 min
EPOCH_DUR = 1.0

# All patients and their seizures
PATIENTS = {
    'chb01': {
        'seizures': [
            ('chb01_03.edf', 2996, 3036),
            ('chb01_04.edf', 1467, 1494),
            ('chb01_15.edf', 1732, 1772),
            ('chb01_16.edf', 1015, 1066),
            ('chb01_18.edf', 1720, 1810),
            ('chb01_21.edf',  327,  420),
            ('chb01_26.edf', 1862, 1963),
        ],
        'interictal': ['chb01_01.edf', 'chb01_09.edf'],
    },
    'chb02': {
        'seizures': [
            ('chb02_16.edf',  130,  212),
            ('chb02_16+.edf', 2972, 3053),
            ('chb02_19.edf', 3369, 3378),
        ],
        'interictal': ['chb02_01.edf'],
    },
    'chb03': {
        'seizures': [
            ('chb03_01.edf',  362,  414),
            ('chb03_02.edf',  731,  796),
            ('chb03_03.edf',  432,  501),
            ('chb03_04.edf', 2162, 2214),
            ('chb03_34.edf', 1982, 2029),
            ('chb03_35.edf', 2592, 2656),
            ('chb03_36.edf', 1725, 1778),
        ],
        'interictal': ['chb03_18.edf'],
    },
    'chb04': {
        'seizures': [
            ('chb04_05.edf', 7804, 7853),
            ('chb04_08.edf', 6446, 6557),
            ('chb04_28.edf', 1679, 1781),
        ],
        'interictal': ['chb04_01.edf'],
    },
    'chb05': {
        'seizures': [
            ('chb05_06.edf',  417,  532),
            ('chb05_13.edf', 1086, 1196),
            ('chb05_16.edf', 2317, 2413),
            ('chb05_17.edf', 2451, 2571),
            ('chb05_22.edf', 2348, 2465),
        ],
        'interictal': ['chb05_01.edf'],
    },
}


# ─────────────────────────────────────────────────────────────────
# Core functions (from chbmit_analysis.py)
# ─────────────────────────────────────────────────────────────────
def kww_func(t, A, tau, alpha, offset):
    return A * np.exp(-(t / tau) ** alpha) + offset

def fit_kww(lags, acf):
    valid = np.isfinite(acf) & np.isfinite(lags)
    lags = lags[valid]
    acf = acf[valid]
    if len(lags) < 5:
        return {'alpha': np.nan, 'alpha_err': np.nan, 'tau': np.nan, 'R2_kww': np.nan}
    acf_norm = acf / (acf[0] if acf[0] != 0 else 1.0)
    try:
        p0 = [1.0, lags[len(lags)//4], 1.0, 0.0]
        bounds = ([0, 0.001, 0.1, -0.5], [2.0, lags[-1]*5, 5.0, 0.5])
        popt, pcov = curve_fit(kww_func, lags, acf_norm, p0=p0, bounds=bounds, maxfev=10000)
        yfit = kww_func(lags, *popt)
        ss_res = np.sum((acf_norm - yfit)**2)
        ss_tot = np.sum((acf_norm - np.mean(acf_norm))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {'alpha': popt[2], 'alpha_err': perr[2], 'tau': popt[1],
                'R2_kww': r2, 'lags': lags, 'acf': acf_norm, 'fit': yfit}
    except:
        return {'alpha': np.nan, 'alpha_err': np.nan, 'tau': np.nan, 'R2_kww': np.nan}


def load_and_filter(filepath, tmin=None, tmax=None):
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"    Load failed: {e}")
        return None, None, None
    # Drop non-EEG
    ch_names = raw.ch_names
    eeg_chs = [ch for ch in ch_names if not ch.startswith(('-', 'ECG', 'VNS', 'STI', 'EKG'))]
    if len(eeg_chs) < len(ch_names) and len(eeg_chs) > 0:
        raw.pick_channels(eeg_chs)
    raw.filter(FILT_LO, FILT_HI, fir_design='firwin', verbose=False)
    if tmin is not None or tmax is not None:
        tmin = tmin if tmin is not None else 0
        tmax = tmax if tmax is not None else raw.times[-1]
        tmax = min(tmax, raw.times[-1])
        tmin = max(tmin, 0)
        if tmin >= tmax:
            return None, None, None
        raw.crop(tmin=tmin, tmax=tmax)
    return raw.get_data(), raw.ch_names, raw.info['sfreq']


def compute_envelope_acf(data, sfreq, max_lag_sec=15):
    envelopes = np.abs(hilbert(data, axis=1))
    mean_env = np.mean(envelopes, axis=0)
    mean_env = mean_env - np.mean(mean_env)
    max_lag_samp = min(int(max_lag_sec * sfreq), len(mean_env) // 2)
    acf = np.correlate(mean_env, mean_env, mode='full')
    acf = acf[len(acf)//2:][:max_lag_samp]
    if acf[0] > 0:
        acf = acf / acf[0]
    lags = np.arange(len(acf)) / sfreq
    return lags, acf


def epochize(data, sfreq, epoch_dur=1.0):
    n_ch, n_samp = data.shape
    samp_per_epoch = int(epoch_dur * sfreq)
    n_epochs = n_samp // samp_per_epoch
    if n_epochs == 0:
        return None
    data_trimmed = data[:, :n_epochs * samp_per_epoch]
    epochs = data_trimmed.reshape(n_ch, n_epochs, samp_per_epoch).transpose(1, 0, 2)
    return epochs


def compute_riemannian_distances(epochs):
    try:
        cov_est = Covariances(estimator='lwf')
        covs = cov_est.fit_transform(epochs)
    except:
        return None, None
    n = covs.shape[0]
    distances = np.zeros(n - 1)
    for i in range(n - 1):
        try:
            distances[i] = distance_riemann(covs[i], covs[i+1])
        except:
            distances[i] = np.nan
    return distances, covs


def rayleigh_test(angles):
    n = len(angles)
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    Z = n * R**2
    p = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n))
    return Z, max(0, min(1, p))


def test_6fold(covs):
    """Test for 6-fold angular structure in tangent space."""
    try:
        from sklearn.decomposition import PCA
        ts = TangentSpace(metric='riemann')
        S = ts.fit_transform(covs)
        if S.shape[0] < 10:
            return None
        pca = PCA(n_components=2)
        c2d = pca.fit_transform(S)
        centroid = np.mean(c2d, axis=0)
        deltas = c2d - centroid
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        angles_6 = (angles * 6) % (2 * np.pi)
        _, p = rayleigh_test(angles_6)
        return p
    except:
        return None


# ─────────────────────────────────────────────────────────────────
# Per-seizure analysis
# ─────────────────────────────────────────────────────────────────
def analyze_one_seizure(patient, fname, sz_start, sz_end, interictal_files):
    """Analyze one seizure. Returns result dict."""
    filepath = os.path.join(DATA_BASE, patient, fname)
    if not os.path.exists(filepath):
        print(f"    FILE NOT FOUND: {filepath}")
        return None

    res = {'patient': patient, 'file': fname, 'sz_start': sz_start, 'sz_end': sz_end}

    # Pre-ictal (5 min)
    pre_start = max(0, sz_start - PRE_ICTAL_DUR)
    pre_end = sz_start
    actual_pre = pre_end - pre_start

    data_pre, _, sfreq = load_and_filter(filepath, pre_start, pre_end)
    if data_pre is None:
        return None

    # Near-onset (60s)
    near_start = max(0, sz_start - 60)
    data_near, _, _ = load_and_filter(filepath, near_start, sz_start)

    # Ictal
    data_ict, _, _ = load_and_filter(filepath, sz_start, sz_end)

    # Interictal (from seizure-free file)
    inter_path = os.path.join(DATA_BASE, patient, interictal_files[0])
    data_int, _, _ = load_and_filter(inter_path, 600, 900)

    # ── KWW on each window ──
    for label, data in [('pre', data_pre), ('near', data_near),
                        ('ict', data_ict), ('int', data_int)]:
        if data is None or data.shape[1] < SFREQ * 5:
            res[f'alpha_{label}'] = np.nan
            res[f'r2_{label}'] = np.nan
            res[f'tau_{label}'] = np.nan
            continue
        lags, acf = compute_envelope_acf(data, sfreq)
        kww = fit_kww(lags[1:], acf[1:])
        res[f'alpha_{label}'] = kww['alpha']
        res[f'r2_{label}'] = kww.get('R2_kww', np.nan)
        res[f'tau_{label}'] = kww.get('tau', np.nan)

    # ── Temporal evolution (1-min sub-windows) ──
    if actual_pre >= 120:
        n_sub = min(5, int(actual_pre // 60))
        sub_dur = actual_pre / n_sub
        sub_alphas = []
        sub_times = []
        for i in range(n_sub):
            t0 = pre_start + i * sub_dur
            t1 = pre_start + (i+1) * sub_dur
            d, _, _ = load_and_filter(filepath, t0, t1)
            if d is None or d.shape[1] < SFREQ * 10:
                sub_alphas.append(np.nan)
            else:
                lg, ac = compute_envelope_acf(d, sfreq, max_lag_sec=10)
                kw = fit_kww(lg[1:], ac[1:])
                sub_alphas.append(kw['alpha'])
            sub_times.append(-(actual_pre - (i+0.5)*sub_dur))

        res['sub_alphas'] = sub_alphas
        res['sub_times'] = sub_times
        valid_alphas = [a for a in sub_alphas if not np.isnan(a)]
        res['peak_alpha'] = max(valid_alphas) if valid_alphas else np.nan
    else:
        res['peak_alpha'] = res.get('alpha_near', np.nan)
        res['sub_alphas'] = []
        res['sub_times'] = []

    # ── Riemannian (pre-ictal only) ──
    if data_pre is not None:
        epochs = epochize(data_pre, sfreq)
        if epochs is not None and len(epochs) >= 10:
            dists, covs = compute_riemannian_distances(epochs)
            if dists is not None:
                res['riem_dist_mean'] = np.nanmean(dists)
                res['riem_dist_std'] = np.nanstd(dists)
                p6 = test_6fold(covs)
                res['p_6fold'] = p6

    return res


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("=" * 70)
    print("CHB-MIT Multi-Patient Analysis — Merkabit Geometric Signature")
    print("=" * 70)

    all_results = []
    patient_results = {}

    for patient, info in PATIENTS.items():
        print(f"\n{'='*60}")
        print(f"Patient: {patient} ({len(info['seizures'])} seizures)")
        print(f"{'='*60}")

        pat_results = []
        for idx, (fname, s, e) in enumerate(info['seizures']):
            print(f"\n  Seizure {idx+1}: {fname}, {s}-{e}s")
            res = analyze_one_seizure(patient, fname, s, e, info['interictal'])
            if res:
                pat_results.append(res)
                all_results.append(res)
                a_pre = res.get('alpha_pre', np.nan)
                a_near = res.get('alpha_near', np.nan)
                a_peak = res.get('peak_alpha', np.nan)
                a_int = res.get('alpha_int', np.nan)
                a_ict = res.get('alpha_ict', np.nan)
                print(f"    pre={a_pre:.3f}, near={a_near:.3f}, peak={a_peak:.3f}, "
                      f"int={a_int:.3f}, ict={a_ict:.3f}" if not np.isnan(a_pre)
                      else f"    FAILED")

        patient_results[patient] = pat_results

    # ─────────────────────────────────────────────────────────────
    # Aggregate statistics
    # ─────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL PATIENTS")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append("CHB-MIT Multi-Patient Analysis -- Merkabit Geometric Signature")
    lines.append("Tenth Dataset -- March 2026")
    lines.append("=" * 70)

    # Per-patient summary
    for patient in PATIENTS:
        pres = patient_results.get(patient, [])
        if not pres:
            continue
        peaks = [r['peak_alpha'] for r in pres if not np.isnan(r.get('peak_alpha', np.nan))]
        nears = [r['alpha_near'] for r in pres if not np.isnan(r.get('alpha_near', np.nan))]
        ints = [r['alpha_int'] for r in pres if not np.isnan(r.get('alpha_int', np.nan))]

        lines.append(f"\n--- {patient} ({len(pres)} seizures) ---")
        if peaks:
            in_w = sum(1 for a in peaks if abs(a - 4/3) < 0.15)
            above_lock = sum(1 for a in peaks if a > 1.036)
            lines.append(f"  Peak alpha: {[f'{a:.3f}' for a in peaks]}")
            lines.append(f"  Mean peak: {np.mean(peaks):.3f} +/- {np.std(peaks):.3f}")
            lines.append(f"  |peak - 4/3| < 0.15: {in_w}/{len(peaks)}")
            lines.append(f"  Peak > 1.036: {above_lock}/{len(peaks)}")
        if nears:
            lines.append(f"  Near-onset alpha: {[f'{a:.3f}' for a in nears]}")
            lines.append(f"  Mean near-onset: {np.mean(nears):.3f}")
        if ints:
            lines.append(f"  Interictal alpha: {[f'{a:.3f}' for a in ints]}")
            lines.append(f"  Mean interictal: {np.mean(ints):.3f}")

    # Grand aggregate
    all_peaks = [r['peak_alpha'] for r in all_results if not np.isnan(r.get('peak_alpha', np.nan))]
    all_nears = [r['alpha_near'] for r in all_results if not np.isnan(r.get('alpha_near', np.nan))]
    all_ints = [r['alpha_int'] for r in all_results if not np.isnan(r.get('alpha_int', np.nan))]
    all_pres = [r['alpha_pre'] for r in all_results if not np.isnan(r.get('alpha_pre', np.nan))]
    all_icts = [r['alpha_ict'] for r in all_results if not np.isnan(r.get('alpha_ict', np.nan))]

    lines.append("\n" + "=" * 70)
    lines.append("GRAND AGGREGATE (all patients)")
    lines.append("=" * 70)

    n_total = len(all_results)
    lines.append(f"Total seizures analyzed: {n_total}")
    lines.append(f"Patients: {len(patient_results)}")

    lines.append(f"\n--- Summary Table ---")
    header = f"{'Window':<15} {'mean_alpha':>10} {'std':>8} {'|a-4/3|':>8} {'N':>4}"
    lines.append(header)
    lines.append("-" * len(header))
    for label, vals in [('Pre-ictal 5m', all_pres), ('Near-onset 60s', all_nears),
                        ('Peak temporal', all_peaks), ('Interictal', all_ints), ('Ictal', all_icts)]:
        if vals:
            m = np.mean(vals)
            s = np.std(vals)
            g = abs(m - 4/3)
            lines.append(f"{label:<15} {m:>10.3f} {s:>8.3f} {g:>8.3f} {len(vals):>4}")

    # Statistical tests on peak alpha
    if len(all_peaks) >= 5:
        lines.append(f"\n--- Peak Alpha Statistics (N={len(all_peaks)}) ---")
        lines.append(f"Values: {[f'{a:.3f}' for a in all_peaks]}")
        lines.append(f"Mean: {np.mean(all_peaks):.3f} +/- {np.std(all_peaks):.3f}")
        lines.append(f"|mean - 4/3| = {abs(np.mean(all_peaks) - 4/3):.3f}")

        in_window = sum(1 for a in all_peaks if abs(a - 4/3) < 0.15)
        above_lock = sum(1 for a in all_peaks if a > 1.036)
        lines.append(f"|peak - 4/3| < 0.15: {in_window}/{len(all_peaks)} ({100*in_window/len(all_peaks):.1f}%)")
        lines.append(f"Peak > 1.036 (1st lock): {above_lock}/{len(all_peaks)} ({100*above_lock/len(all_peaks):.1f}%)")

        t, p = ttest_1samp(all_peaks, 4/3)
        lines.append(f"t-test vs 4/3: t={t:.3f}, p={p:.4f}")
        t1, p1 = ttest_1samp(all_peaks, 1.0)
        lines.append(f"t-test vs 1.0: t={t1:.3f}, p={p1/2:.4f} (one-tailed)")

        # Binomial test: proportion in window
        binom_p = binomtest(in_window, len(all_peaks), 0.3/5.0)  # chance of landing in 0.3-wide window in [0,5]
        lines.append(f"Binomial test (in-window fraction vs chance): p={binom_p.pvalue:.6f}")

    # Near-onset vs interictal
    if len(all_nears) >= 5 and len(all_ints) >= 5:
        lines.append(f"\n--- Near-onset vs Interictal ---")
        n_gt = sum(1 for a, b in zip(all_nears, all_ints[:len(all_nears)]) if a > b)
        lines.append(f"Near-onset > Interictal: {n_gt}/{min(len(all_nears), len(all_ints))}")
        U, p_mw = mannwhitneyu(all_nears, all_ints, alternative='greater')
        lines.append(f"Mann-Whitney U (near > inter): U={U:.0f}, p={p_mw:.4f}")

    # 6-fold test aggregate
    p6_pre = [r['p_6fold'] for r in all_results if r.get('p_6fold') is not None]
    if p6_pre:
        sig = sum(1 for p in p6_pre if p < 0.05)
        lines.append(f"\n--- Riemannian 6-fold (pre-ictal) ---")
        lines.append(f"Significant: {sig}/{len(p6_pre)}")

    # Interpretation
    lines.append("\n" + "=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)

    if all_peaks:
        mean_pk = np.mean(all_peaks)
        in_w = sum(1 for a in all_peaks if abs(a - 4/3) < 0.15)
        above = sum(1 for a in all_peaks if a > 1.036)
        frac_in = in_w / len(all_peaks)
        frac_above = above / len(all_peaks)

        if frac_in >= 0.5:
            lines.append(f"CONFIRMED: {in_w}/{len(all_peaks)} seizures ({frac_in*100:.0f}%) peak near 4/3")
            lines.append(f"Mean peak alpha = {mean_pk:.3f}, |mean - 4/3| = {abs(mean_pk - 4/3):.3f}")
            lines.append("Cooperative cascade CONFIRMED across multiple patients.")
            lines.append("First biological system in the cross-platform table.")
        elif frac_above >= 0.5:
            lines.append(f"PARTIAL: {above}/{len(all_peaks)} ({frac_above*100:.0f}%) cross first lock threshold")
            lines.append(f"Mean peak alpha = {mean_pk:.3f}")
            lines.append("Cooperative cascade present, first lock threshold crossed.")
        else:
            lines.append(f"MIXED: {in_w}/{len(all_peaks)} in window, {above}/{len(all_peaks)} above lock")
            lines.append("Result is patient-dependent. Some brains show cooperative cascade.")

    summary = '\n'.join(lines)
    print('\n' + summary)

    with open(os.path.join(REPORTS_DIR, 'chbmit_multipatient_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\nSaved: chbmit_multipatient_summary.txt")

    # ─────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────
    # 1. Per-patient peak alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: peak alpha per seizure colored by patient
    ax = axes[0]
    colors_map = {'chb01': 'red', 'chb02': 'blue', 'chb03': 'green',
                  'chb04': 'orange', 'chb05': 'purple'}
    x_pos = 0
    for patient in PATIENTS:
        pres = patient_results.get(patient, [])
        peaks = [(r['peak_alpha'], r['file']) for r in pres
                 if not np.isnan(r.get('peak_alpha', np.nan))]
        if peaks:
            xs = list(range(x_pos, x_pos + len(peaks)))
            ys = [p[0] for p in peaks]
            ax.bar(xs, ys, color=colors_map.get(patient, 'gray'), alpha=0.7,
                   label=patient, edgecolor='black', linewidth=0.5)
            x_pos += len(peaks) + 1

    ax.axhline(y=4/3, color='green', linestyle='--', linewidth=2, label='4/3')
    ax.axhline(y=1.036, color='purple', linestyle=':', linewidth=1.5, label='1st lock')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Peak pre-ictal alpha')
    ax.set_title('Peak alpha per seizure (all patients)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2)

    # Right: distribution of peak alphas
    ax2 = axes[1]
    if all_peaks:
        ax2.hist(all_peaks, bins=12, range=(0, 2.4), color='steelblue', alpha=0.7,
                 edgecolor='black')
        ax2.axvline(x=4/3, color='green', linestyle='--', linewidth=2, label='4/3')
        ax2.axvline(x=1.036, color='purple', linestyle=':', linewidth=1.5, label='1st lock')
        ax2.axvspan(4/3 - 0.15, 4/3 + 0.15, alpha=0.15, color='green', label='|a-4/3|<0.15')
        ax2.set_xlabel('Peak pre-ictal alpha')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Distribution (N={len(all_peaks)})')
        ax2.legend(fontsize=8)

    plt.suptitle('CHB-MIT Multi-Patient: Peak Pre-ictal Alpha', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_multipatient_peaks.png'), dpi=150)
    plt.close()
    print("Saved: chbmit_multipatient_peaks.png")

    # 2. Temporal evolution for all seizures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    pat_idx = 0
    for patient in PATIENTS:
        if pat_idx >= 5:
            break
        ax = axes_flat[pat_idx]
        pres = patient_results.get(patient, [])
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(pres), 1)))
        for si, r in enumerate(pres):
            if r.get('sub_times') and r.get('sub_alphas'):
                t = r['sub_times']
                a = r['sub_alphas']
                valid = [(t[i], a[i]) for i in range(len(a)) if not np.isnan(a[i])]
                if valid:
                    ax.plot([v[0] for v in valid], [v[1] for v in valid],
                            'o-', color=colors[si], markersize=5,
                            label=f"Sz{si+1}")
        ax.axhline(y=4/3, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=1.036, color='purple', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time to onset (s)')
        ax.set_ylabel('alpha')
        ax.set_title(patient)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        pat_idx += 1

    axes_flat[5].set_visible(False)
    plt.suptitle('Temporal Evolution: alpha(t) Approaching Seizure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_multipatient_temporal.png'), dpi=150)
    plt.close()
    print("Saved: chbmit_multipatient_temporal.png")

    # 3. Near-onset vs interictal scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    for patient in PATIENTS:
        pres = patient_results.get(patient, [])
        for r in pres:
            an = r.get('alpha_near', np.nan)
            ai = r.get('alpha_int', np.nan)
            if not np.isnan(an) and not np.isnan(ai):
                ax.scatter(ai, an, color=colors_map.get(patient, 'gray'), s=60,
                          edgecolor='black', linewidth=0.5, label=patient, alpha=0.7)

    ax.plot([0, 2.5], [0, 2.5], 'k--', alpha=0.3, label='y=x')
    ax.axhline(y=4/3, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=4/3, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Interictal alpha')
    ax.set_ylabel('Near-onset alpha (60s)')
    ax.set_title('Near-onset vs Interictal alpha')
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_multipatient_scatter.png'), dpi=150)
    plt.close()
    print("Saved: chbmit_multipatient_scatter.png")

    print("\nMulti-patient analysis complete.")
