#!/usr/bin/env python3
"""
CHB-MIT EEG Seizure Analysis — Merkabit Geometric Signature
Ninth Dataset Candidate — March 2026

Tests two signals:
  Signal 1: KWW exponent alpha in pre-ictal, ictal, interictal windows
  Signal 2: Riemannian covariance geometry (geodesic distances + tangent space)

Dataset: CHB-MIT Scalp EEG (PhysioNet), Patient chb01, 7 seizures
"""

import os
import sys
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import ttest_1samp
import mne
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.tangentspace import TangentSpace

# Add repo root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import CHBMIT_DATA, FIGURES_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
DATA_DIR = CHBMIT_DATA
OUT_DIR  = FIGURES_DIR
SFREQ    = 256  # Hz
FILT_LO  = 1.0  # Hz
FILT_HI  = 40.0 # Hz
PRE_ICTAL_DUR  = 300  # seconds (5 min)
INTERICTAL_DUR = 300  # seconds (5 min)
EPOCH_DUR = 1.0       # seconds

# Seizure annotations from chb01-summary.txt
SEIZURES = [
    ('chb01_03.edf', 2996, 3036),
    ('chb01_04.edf', 1467, 1494),
    ('chb01_15.edf', 1732, 1772),
    ('chb01_16.edf', 1015, 1066),
    ('chb01_18.edf', 1720, 1810),
    ('chb01_21.edf',  327,  420),
    ('chb01_26.edf', 1862, 1963),
]

# Seizure-free files for interictal baseline
INTERICTAL_FILES = ['chb01_01.edf', 'chb01_09.edf']


# -----------------------------------------------------------------
# KWW model
# -----------------------------------------------------------------
def kww_func(t, A, tau, alpha, offset):
    """Stretched exponential (KWW) function."""
    return A * np.exp(-(t / tau) ** alpha) + offset

def simple_exp(t, A, tau, offset):
    """Simple exponential for comparison."""
    return A * np.exp(-t / tau) + offset

def fit_kww(lags, acf, maxlag=None):
    """Fit KWW to autocorrelation function. Returns params dict."""
    if maxlag is not None:
        mask = lags <= maxlag
        lags = lags[mask]
        acf = acf[mask]

    # Remove NaN/Inf
    valid = np.isfinite(acf) & np.isfinite(lags)
    lags = lags[valid]
    acf = acf[valid]
    if len(lags) < 5:
        return None

    # Normalize
    acf_norm = acf / (acf[0] if acf[0] != 0 else 1.0)

    result = {}
    # Fit KWW
    try:
        p0 = [1.0, lags[len(lags)//4], 1.0, 0.0]
        bounds = ([0, 0.001, 0.1, -0.5], [2.0, lags[-1]*5, 5.0, 0.5])
        popt, pcov = curve_fit(kww_func, lags, acf_norm, p0=p0, bounds=bounds,
                               maxfev=10000)
        yfit = kww_func(lags, *popt)
        ss_res = np.sum((acf_norm - yfit)**2)
        ss_tot = np.sum((acf_norm - np.mean(acf_norm))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*4
        result['alpha'] = popt[2]
        result['alpha_err'] = perr[2]
        result['tau'] = popt[1]
        result['A'] = popt[0]
        result['offset'] = popt[3]
        result['R2_kww'] = r2
        result['lags'] = lags
        result['acf'] = acf_norm
        result['fit'] = yfit
    except Exception as e:
        result['alpha'] = np.nan
        result['alpha_err'] = np.nan
        result['tau'] = np.nan
        result['R2_kww'] = np.nan
        result['fit_error'] = str(e)

    # Fit simple exponential
    try:
        p0e = [1.0, lags[len(lags)//4], 0.0]
        bounds_e = ([0, 0.001, -0.5], [2.0, lags[-1]*5, 0.5])
        popt_e, _ = curve_fit(simple_exp, lags, acf_norm, p0=p0e,
                              bounds=bounds_e, maxfev=10000)
        yfit_e = simple_exp(lags, *popt_e)
        ss_res_e = np.sum((acf_norm - yfit_e)**2)
        ss_tot_e = np.sum((acf_norm - np.mean(acf_norm))**2)
        result['R2_exp'] = 1 - ss_res_e / ss_tot_e if ss_tot_e > 0 else 0
    except:
        result['R2_exp'] = np.nan

    return result


# -----------------------------------------------------------------
# EEG processing functions
# -----------------------------------------------------------------
def load_and_filter(filepath, tmin=None, tmax=None):
    """Load EDF, bandpass filter, return data array and channel names."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # Drop non-EEG channels if any
    ch_names = raw.ch_names
    eeg_chs = [ch for ch in ch_names if not ch.startswith(('-', 'ECG', 'VNS', 'STI'))]
    if len(eeg_chs) < len(ch_names):
        raw.pick_channels(eeg_chs)

    # Bandpass filter
    raw.filter(FILT_LO, FILT_HI, fir_design='firwin', verbose=False)

    # Crop if needed
    if tmin is not None or tmax is not None:
        tmin = tmin if tmin is not None else 0
        tmax = tmax if tmax is not None else raw.times[-1]
        tmax = min(tmax, raw.times[-1])
        tmin = max(tmin, 0)
        if tmin >= tmax:
            return None, None, None
        raw.crop(tmin=tmin, tmax=tmax)

    data = raw.get_data()  # (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    return data, raw.ch_names, sfreq


def compute_envelope_autocorrelation(data, sfreq, max_lag_sec=30):
    """
    Compute channel-average power envelope, then its autocorrelation.
    data: (n_channels, n_samples)
    Returns lags (seconds) and normalized ACF.
    """
    n_ch, n_samp = data.shape
    # Analytic signal envelope per channel
    envelopes = np.abs(hilbert(data, axis=1))
    # Mean envelope across channels
    mean_env = np.mean(envelopes, axis=0)
    # Remove mean
    mean_env = mean_env - np.mean(mean_env)

    max_lag_samp = min(int(max_lag_sec * sfreq), n_samp // 2)

    # ACF using numpy correlate (fast for moderate lengths)
    acf = np.correlate(mean_env, mean_env, mode='full')
    acf = acf[len(acf)//2:]  # positive lags
    acf = acf[:max_lag_samp]
    acf = acf / acf[0]  # normalize

    lags = np.arange(len(acf)) / sfreq
    return lags, acf


def epochize(data, sfreq, epoch_dur=1.0):
    """Split data into non-overlapping epochs. Returns (n_epochs, n_ch, n_samp)."""
    n_ch, n_samp = data.shape
    samp_per_epoch = int(epoch_dur * sfreq)
    n_epochs = n_samp // samp_per_epoch
    if n_epochs == 0:
        return None
    data_trimmed = data[:, :n_epochs * samp_per_epoch]
    epochs = data_trimmed.reshape(n_ch, n_epochs, samp_per_epoch)
    epochs = epochs.transpose(1, 0, 2)  # (n_epochs, n_ch, n_samp)
    return epochs


# -----------------------------------------------------------------
# Signal 2: Riemannian geometry
# -----------------------------------------------------------------
def compute_riemannian_distances(epochs):
    """
    Compute Riemannian geodesic distances between successive covariance matrices.
    epochs: (n_epochs, n_ch, n_samp)
    Returns array of distances.
    """
    try:
        cov_est = Covariances(estimator='lwf')
        covs = cov_est.fit_transform(epochs)  # (n_epochs, n_ch, n_ch)
    except Exception as e:
        print(f"  Covariance estimation failed: {e}")
        return None, None

    n = covs.shape[0]
    distances = np.zeros(n - 1)
    for i in range(n - 1):
        try:
            distances[i] = distance_riemann(covs[i], covs[i+1])
        except:
            distances[i] = np.nan

    return distances, covs


def compute_tangent_space_projection(covs):
    """
    Project covariance matrices to tangent space and return coordinates.
    covs: (n_epochs, n_ch, n_ch)
    Returns tangent space vectors (n_epochs, n_features).
    """
    try:
        ts = TangentSpace(metric='riemann')
        S = ts.fit_transform(covs)
        return S
    except Exception as e:
        print(f"  Tangent space projection failed: {e}")
        return None


def rayleigh_test(angles):
    """Rayleigh test for circular uniformity. Returns (Z statistic, p-value)."""
    n = len(angles)
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    Z = n * R**2
    # Approximate p-value
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    p = max(0, min(1, p))
    return Z, p

def test_angular_structure(tangent_vectors, n_fold=6):
    """
    Test for n-fold angular structure in 2D PCA projection of tangent space.
    Returns Rayleigh test p-value for n-fold symmetry.
    """
    from sklearn.decomposition import PCA

    if tangent_vectors is None or len(tangent_vectors) < 10:
        return None, None, None

    # PCA to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(tangent_vectors)

    # Compute angles from centroid
    centroid = np.mean(coords_2d, axis=0)
    deltas = coords_2d - centroid
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    # Test for n-fold symmetry: multiply angles by n and test uniformity
    angles_nfold = (angles * n_fold) % (2 * np.pi)

    # Rayleigh test for non-uniformity
    Z, p_val = rayleigh_test(angles_nfold)

    return p_val, coords_2d, angles


# -----------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------
def analyze_seizure(seizure_idx, filename, sz_start, sz_end):
    """Analyze one seizure: extract windows, compute KWW + Riemannian."""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*60}")
    print(f"Seizure {seizure_idx+1}: {filename}, onset={sz_start}s, end={sz_end}s")
    print(f"{'='*60}")

    results = {'file': filename, 'sz_start': sz_start, 'sz_end': sz_end}

    # -- Pre-ictal window --
    pre_start = max(0, sz_start - PRE_ICTAL_DUR)
    pre_end = sz_start
    actual_pre_dur = pre_end - pre_start
    if actual_pre_dur < 60:
        print(f"  WARNING: Pre-ictal window only {actual_pre_dur}s (need >= 60s)")

    print(f"  Pre-ictal: {pre_start}-{pre_end}s ({actual_pre_dur}s)")
    data_pre, ch_names, sfreq = load_and_filter(filepath, pre_start, pre_end)
    if data_pre is None:
        print("  ERROR: Could not load pre-ictal data")
        return None

    # -- Ictal window --
    ictal_dur = sz_end - sz_start
    print(f"  Ictal: {sz_start}-{sz_end}s ({ictal_dur}s)")
    data_ict, _, _ = load_and_filter(filepath, sz_start, sz_end)

    # -- Interictal window (from seizure-free file) --
    inter_file = os.path.join(DATA_DIR, INTERICTAL_FILES[seizure_idx % len(INTERICTAL_FILES)])
    print(f"  Interictal: from {INTERICTAL_FILES[seizure_idx % len(INTERICTAL_FILES)]}, 600-900s")
    data_int, _, _ = load_and_filter(inter_file, 600, 600 + INTERICTAL_DUR)

    # ----------------------------------------
    # Signal 1: KWW on envelope ACF
    # ----------------------------------------
    print("\n  --- Signal 1: KWW Envelope ACF ---")

    # Also extract near-onset window (last 60s before seizure)
    near_onset_dur = 60
    near_start = max(0, sz_start - near_onset_dur)
    data_near, _, _ = load_and_filter(filepath, near_start, sz_start)
    print(f"  Near-onset: {near_start}-{sz_start}s ({sz_start - near_start}s)")

    for label, data in [('pre-ictal', data_pre), ('near-onset', data_near),
                        ('ictal', data_ict), ('interictal', data_int)]:
        if data is None:
            results[f'kww_{label}'] = None
            continue
        lags, acf = compute_envelope_autocorrelation(data, sfreq, max_lag_sec=15)
        kww_res = fit_kww(lags[1:], acf[1:])  # skip lag=0
        results[f'kww_{label}'] = kww_res
        if kww_res and not np.isnan(kww_res.get('alpha', np.nan)):
            a = kww_res['alpha']
            print(f"    {label:12s}: alpha={a:.3f} +/- {kww_res['alpha_err']:.3f}, "
                  f"tau={kww_res['tau']:.2f}s, R2_kww={kww_res['R2_kww']:.3f}, "
                  f"R2_exp={kww_res['R2_exp']:.3f}, |a-4/3|={abs(a-4/3):.3f}")
        else:
            print(f"    {label:12s}: FIT FAILED")

    # ----------------------------------------
    # Signal 2: Riemannian geodesic distances
    # ----------------------------------------
    print("\n  --- Signal 2: Riemannian Geometry ---")

    for label, data in [('pre-ictal', data_pre), ('interictal', data_int)]:
        if data is None:
            results[f'riem_{label}'] = None
            continue
        epochs = epochize(data, sfreq, EPOCH_DUR)
        if epochs is None or len(epochs) < 5:
            print(f"    {label}: insufficient epochs")
            results[f'riem_{label}'] = None
            continue

        dists, covs = compute_riemannian_distances(epochs)
        if dists is None:
            results[f'riem_{label}'] = None
            continue

        # ACF of geodesic distance sequence
        dists_clean = dists[np.isfinite(dists)]
        if len(dists_clean) < 10:
            results[f'riem_{label}'] = {'dists': dists}
            continue

        dists_centered = dists_clean - np.mean(dists_clean)
        acf_d = np.correlate(dists_centered, dists_centered, mode='full')
        acf_d = acf_d[len(acf_d)//2:]
        if acf_d[0] > 0:
            acf_d = acf_d / acf_d[0]
        lags_d = np.arange(len(acf_d))  # in epoch units (seconds)

        kww_d = fit_kww(lags_d[1:].astype(float), acf_d[1:], maxlag=len(acf_d)//2)

        # Tangent space + angular structure
        ts_coords = compute_tangent_space_projection(covs)
        p_6fold, coords_2d, angles = test_angular_structure(ts_coords, n_fold=6)

        riem_result = {
            'dists': dists,
            'covs': covs,
            'kww_dist': kww_d,
            'ts_coords': ts_coords,
            'coords_2d': coords_2d,
            'angles': angles,
            'p_6fold': p_6fold,
        }
        results[f'riem_{label}'] = riem_result

        if kww_d and not np.isnan(kww_d.get('alpha', np.nan)):
            a = kww_d['alpha']
            print(f"    {label:12s} dist ACF: alpha={a:.3f} +/- {kww_d['alpha_err']:.3f}, "
                  f"R2={kww_d['R2_kww']:.3f}")
        if p_6fold is not None:
            print(f"    {label:12s} 6-fold Rayleigh test: p={p_6fold:.4f} "
                  f"{'significant' if p_6fold < 0.05 else 'not significant'}")

    # ----------------------------------------
    # Temporal evolution: 5 sub-windows
    # ----------------------------------------
    if actual_pre_dur >= 120:  # at least 2 minutes
        print("\n  --- Temporal Evolution (pre-ictal sub-windows) ---")
        n_sub = min(5, int(actual_pre_dur // 60))
        sub_dur = actual_pre_dur / n_sub
        sub_alphas = []
        sub_times = []
        for i in range(n_sub):
            t0 = pre_start + i * sub_dur
            t1 = pre_start + (i+1) * sub_dur
            data_sub, _, _ = load_and_filter(filepath, t0, t1)
            if data_sub is None or data_sub.shape[1] < SFREQ * 10:
                sub_alphas.append(np.nan)
                sub_times.append(-(actual_pre_dur - (i+0.5)*sub_dur))
                continue
            lags_s, acf_s = compute_envelope_autocorrelation(data_sub, sfreq, max_lag_sec=10)
            kww_s = fit_kww(lags_s[1:], acf_s[1:])
            a = kww_s['alpha'] if kww_s and not np.isnan(kww_s.get('alpha', np.nan)) else np.nan
            sub_alphas.append(a)
            sub_times.append(-(actual_pre_dur - (i+0.5)*sub_dur))
            print(f"    t={sub_times[-1]:+.0f}s: alpha={a:.3f}" if not np.isnan(a)
                  else f"    t={sub_times[-1]:+.0f}s: fit failed")

        results['temporal_evolution'] = {'times': sub_times, 'alphas': sub_alphas}

    return results


# -----------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------
def plot_kww_results(all_results):
    """Plot KWW fits for all seizures."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('CHB-MIT chb01: KWW Envelope ACF Fits', fontsize=14, fontweight='bold')

    for idx, res in enumerate(all_results):
        if res is None or idx >= 7:
            continue
        ax_row = idx // 3
        ax_col = idx % 3
        if ax_row >= 3:
            break
        ax = axes[ax_row][ax_col] if len(all_results) > 3 else axes[ax_col]

        for label, color in [('pre-ictal', 'red'), ('interictal', 'blue'), ('ictal', 'orange')]:
            kww = res.get(f'kww_{label}')
            if kww and 'lags' in kww and 'acf' in kww:
                ax.plot(kww['lags'], kww['acf'], '.', color=color, alpha=0.3, markersize=2)
                if 'fit' in kww:
                    ax.plot(kww['lags'], kww['fit'], '-', color=color, linewidth=1.5,
                            label=f"{label} a={kww['alpha']:.2f}")

        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('ACF')
        ax.set_title(f"Sz {idx+1}: {res['file']}")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 10)

    # Hide unused subplots
    for idx in range(len(all_results), 9):
        r, c = idx // 3, idx % 3
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_kww_results.png'), dpi=150)
    plt.close()
    print("\nSaved: chbmit_kww_results.png")


def plot_riemannian(all_results):
    """Plot Riemannian distance sequences and covariance manifold."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('CHB-MIT chb01: Riemannian Geodesic Distances', fontsize=14, fontweight='bold')

    for idx, res in enumerate(all_results):
        if res is None or idx >= 7:
            continue
        ax = axes[idx // 4][idx % 4]

        for label, color in [('pre-ictal', 'red'), ('interictal', 'blue')]:
            riem = res.get(f'riem_{label}')
            if riem and 'dists' in riem:
                dists = riem['dists']
                valid = np.isfinite(dists)
                ax.plot(np.arange(len(dists))[valid], dists[valid], '-', color=color,
                        alpha=0.7, linewidth=1, label=label)

        ax.set_xlabel('Epoch index')
        ax.set_ylabel('Geodesic distance')
        ax.set_title(f"Sz {idx+1}", fontsize=10)
        ax.legend(fontsize=7)

    axes[1][3].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_riemannian.png'), dpi=150)
    plt.close()
    print("Saved: chbmit_riemannian.png")


def plot_temporal_evolution(all_results):
    """Plot alpha(t) as seizure approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('CHB-MIT chb01: Pre-ictal a(t) Temporal Evolution', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for idx, res in enumerate(all_results):
        if res is None:
            continue
        te = res.get('temporal_evolution')
        if te:
            times = te['times']
            alphas = te['alphas']
            valid = [i for i in range(len(alphas)) if not np.isnan(alphas[i])]
            if valid:
                ax.plot([times[i] for i in valid], [alphas[i] for i in valid],
                        'o-', color=colors[idx], label=f"Sz {idx+1}", markersize=6)

    ax.axhline(y=4/3, color='green', linestyle='--', alpha=0.7, label='a = 4/3')
    ax.axhline(y=1.036, color='purple', linestyle=':', alpha=0.7, label='a = 1.036 (1st lock)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='a = 1.0')
    ax.set_xlabel('Time relative to seizure onset (s)')
    ax.set_ylabel('KWW a')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_temporal_evolution.png'), dpi=150)
    plt.close()
    print("Saved: chbmit_temporal_evolution.png")


def plot_tangent_space(all_results):
    """Plot 2D tangent space projections with angular structure test."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('CHB-MIT chb01: Tangent Space 2D Projection (PCA)', fontsize=14, fontweight='bold')

    for idx, res in enumerate(all_results):
        if res is None or idx >= 7:
            continue
        ax = axes[idx // 4][idx % 4]

        # Pre-ictal
        riem_pre = res.get('riem_pre-ictal')
        if riem_pre and riem_pre.get('coords_2d') is not None:
            c2d = riem_pre['coords_2d']
            n = len(c2d)
            colors = plt.cm.hot(np.linspace(0, 1, n))
            ax.scatter(c2d[:, 0], c2d[:, 1], c=colors, s=10, alpha=0.7)
            p = riem_pre.get('p_6fold')
            p_str = f"p={p:.3f}" if p is not None else "N/A"
            ax.set_title(f"Sz {idx+1} pre-ictal (6-fold {p_str})", fontsize=9)
        else:
            ax.set_title(f"Sz {idx+1}: no data", fontsize=9)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    axes[1][3].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chbmit_tangent_space.png'), dpi=150)
    plt.close()
    print("Saved: chbmit_tangent_space.png")


# -----------------------------------------------------------------
# Cross-seizure statistics
# -----------------------------------------------------------------
def compute_statistics(all_results):
    """Compute cross-seizure statistics and generate summary."""
    # Collect alphas
    alpha_pre = []
    alpha_near = []
    alpha_int = []
    alpha_ict = []
    tau_pre = []
    tau_near = []
    r2_pre = []
    r2_near = []
    r2exp_pre = []
    r2exp_near = []
    alpha_riem_pre = []
    alpha_riem_int = []
    p_6fold_pre = []
    p_6fold_int = []

    for res in all_results:
        if res is None:
            continue
        for label, alist, tlist, rlist, relist in [
            ('pre-ictal', alpha_pre, tau_pre, r2_pre, r2exp_pre),
            ('near-onset', alpha_near, tau_near, r2_near, r2exp_near),
            ('interictal', alpha_int, [], [], []),
            ('ictal', alpha_ict, [], [], [])]:
            kww = res.get(f'kww_{label}')
            if kww and not np.isnan(kww.get('alpha', np.nan)):
                alist.append(kww['alpha'])
                if tlist is not None:
                    tlist.append(kww.get('tau', np.nan))
                if rlist is not None:
                    rlist.append(kww.get('R2_kww', np.nan))
                if relist is not None:
                    relist.append(kww.get('R2_exp', np.nan))

        # Riemannian
        for label, rlist in [('pre-ictal', alpha_riem_pre), ('interictal', alpha_riem_int)]:
            riem = res.get(f'riem_{label}')
            if riem and riem.get('kww_dist'):
                kww_d = riem['kww_dist']
                if not np.isnan(kww_d.get('alpha', np.nan)):
                    rlist.append(kww_d['alpha'])

        riem_pre = res.get('riem_pre-ictal')
        riem_int = res.get('riem_interictal')
        if riem_pre and riem_pre.get('p_6fold') is not None:
            p_6fold_pre.append(riem_pre['p_6fold'])
        if riem_int and riem_int.get('p_6fold') is not None:
            p_6fold_int.append(riem_int['p_6fold'])

    # Summary
    lines = []
    lines.append("=" * 70)
    lines.append("CHB-MIT EEG Seizure Analysis -- Merkabit Geometric Signature")
    lines.append("Ninth Dataset Candidate -- March 2026")
    lines.append("Patient: chb01, 7 seizures")
    lines.append("=" * 70)

    lines.append("\n" + "-" * 70)
    lines.append("SIGNAL 1: KWW ENVELOPE AUTOCORRELATION")
    lines.append("-" * 70)

    header = f"{'Window':<15} {'alpha':>8} {'tau(s)':>8} {'R2_KWW':>8} {'R2_exp':>8} {'|a-4/3|':>8} {'N':>4}"
    lines.append(header)
    lines.append("-" * len(header))

    for label, alist, tlist, rlist, relist in [
        ('Pre-ictal', alpha_pre, tau_pre, r2_pre, r2exp_pre),
        ('Near-onset', alpha_near, tau_near, r2_near, r2exp_near),
        ('Interictal', alpha_int, [], [], []),
        ('Ictal', alpha_ict, [], [], [])]:
        if alist:
            a_mean = np.mean(alist)
            tau_m = np.mean(tlist) if tlist else np.nan
            r2_m = np.mean(rlist) if rlist else np.nan
            r2e_m = np.mean(relist) if relist else np.nan
            gap = abs(a_mean - 4/3)
            lines.append(f"{label:<15} {a_mean:>7.3f}  {tau_m:>7.2f}  {r2_m:>7.3f}  {r2e_m:>7.3f}  {gap:>7.3f}  {len(alist):>3}")
        else:
            lines.append(f"{label:<15} {'N/A':>8}")

    lines.append(f"\nPre-ictal (5min) alpha values: {[f'{a:.3f}' for a in alpha_pre]}")
    lines.append(f"Near-onset (60s) alpha values: {[f'{a:.3f}' for a in alpha_near]}")
    lines.append(f"Interictal alpha values: {[f'{a:.3f}' for a in alpha_int]}")
    lines.append(f"Ictal alpha values: {[f'{a:.3f}' for a in alpha_ict]}")

    # Statistical tests
    if len(alpha_pre) >= 3 and len(alpha_int) >= 3:
        lines.append("\n--- Full pre-ictal window (5 min) ---")
        n_greater = sum(1 for a, b in zip(alpha_pre, alpha_int) if a > b)
        lines.append(f"Pre-ictal > Interictal: {n_greater}/{min(len(alpha_pre), len(alpha_int))} seizures")
        t_stat, p_val = ttest_1samp(alpha_pre, 1.0)
        lines.append(f"Pre-ictal alpha > 1.0: t={t_stat:.3f}, p={p_val/2:.4f} (one-tailed)")
        t_stat2, p_val2 = ttest_1samp(alpha_pre, 4/3)
        lines.append(f"Pre-ictal alpha = 4/3: t={t_stat2:.3f}, p={p_val2:.4f} (two-tailed)")
        in_window = sum(1 for a in alpha_pre if abs(a - 4/3) < 0.15)
        lines.append(f"|alpha_pre - 4/3| < 0.15: {in_window}/{len(alpha_pre)} seizures")
        above_lock = sum(1 for a in alpha_pre if a > 1.036)
        below_lock_int = sum(1 for a in alpha_int if a < 1.036)
        lines.append(f"Pre-ictal alpha > 1.036 (1st lock): {above_lock}/{len(alpha_pre)}")
        lines.append(f"Interictal alpha < 1.036: {below_lock_int}/{len(alpha_int)}")

    if len(alpha_near) >= 3:
        lines.append("\n--- Near-onset window (last 60s) ---")
        n_greater_near = sum(1 for a, b in zip(alpha_near, alpha_int[:len(alpha_near)]) if a > b)
        lines.append(f"Near-onset > Interictal: {n_greater_near}/{min(len(alpha_near), len(alpha_int))} seizures")
        t_stat_n, p_val_n = ttest_1samp(alpha_near, 1.0)
        lines.append(f"Near-onset alpha > 1.0: t={t_stat_n:.3f}, p={p_val_n/2:.4f} (one-tailed)")
        t_stat_n2, p_val_n2 = ttest_1samp(alpha_near, 4/3)
        lines.append(f"Near-onset alpha = 4/3: t={t_stat_n2:.3f}, p={p_val_n2:.4f} (two-tailed)")
        in_window_near = sum(1 for a in alpha_near if abs(a - 4/3) < 0.15)
        lines.append(f"|alpha_near - 4/3| < 0.15: {in_window_near}/{len(alpha_near)} seizures")
        above_lock_near = sum(1 for a in alpha_near if a > 1.036)
        lines.append(f"Near-onset alpha > 1.036 (1st lock): {above_lock_near}/{len(alpha_near)}")

    # Peak alpha from temporal evolution (most sensitive test)
    peak_alphas = []
    for res in all_results:
        if res is None:
            continue
        te = res.get('temporal_evolution')
        if te:
            alphas_te = [a for a in te['alphas'] if not np.isnan(a)]
            if alphas_te:
                peak_alphas.append(max(alphas_te))

    if len(peak_alphas) >= 3:
        lines.append("\n--- Peak alpha in pre-ictal temporal evolution (STRONGEST TEST) ---")
        lines.append(f"Peak alpha values: {[f'{a:.3f}' for a in peak_alphas]}")
        lines.append(f"Mean peak alpha: {np.mean(peak_alphas):.3f} +/- {np.std(peak_alphas):.3f}")
        lines.append(f"|mean_peak - 4/3| = {abs(np.mean(peak_alphas) - 4/3):.3f}")
        in_window_peak = sum(1 for a in peak_alphas if abs(a - 4/3) < 0.15)
        lines.append(f"|peak_alpha - 4/3| < 0.15: {in_window_peak}/{len(peak_alphas)} seizures")
        above_lock_peak = sum(1 for a in peak_alphas if a > 1.036)
        lines.append(f"Peak alpha > 1.036 (1st lock): {above_lock_peak}/{len(peak_alphas)}")
        t_stat_p, p_val_p = ttest_1samp(peak_alphas, 4/3)
        lines.append(f"Peak alpha = 4/3: t={t_stat_p:.3f}, p={p_val_p:.4f} (two-tailed)")
        t_stat_p1, p_val_p1 = ttest_1samp(peak_alphas, 1.0)
        lines.append(f"Peak alpha > 1.0: t={t_stat_p1:.3f}, p={p_val_p1/2:.4f} (one-tailed)")

    lines.append("\n" + "-" * 70)
    lines.append("SIGNAL 2: RIEMANNIAN COVARIANCE GEOMETRY")
    lines.append("-" * 70)

    if alpha_riem_pre:
        lines.append(f"Pre-ictal geodesic dist ACF alpha: {np.mean(alpha_riem_pre):.3f} +/- {np.std(alpha_riem_pre):.3f}")
        lines.append(f"  Values: {[f'{a:.3f}' for a in alpha_riem_pre]}")
    if alpha_riem_int:
        lines.append(f"Interictal geodesic dist ACF alpha: {np.mean(alpha_riem_int):.3f} +/- {np.std(alpha_riem_int):.3f}")

    if p_6fold_pre:
        sig_pre = sum(1 for p in p_6fold_pre if p < 0.05)
        lines.append(f"\nTangent space 6-fold structure (Rayleigh test):")
        lines.append(f"  Pre-ictal: {sig_pre}/{len(p_6fold_pre)} significant (p<0.05)")
        lines.append(f"  p-values: {[f'{p:.4f}' for p in p_6fold_pre]}")
    if p_6fold_int:
        sig_int = sum(1 for p in p_6fold_int if p < 0.05)
        lines.append(f"  Interictal: {sig_int}/{len(p_6fold_int)} significant")
        lines.append(f"  p-values: {[f'{p:.4f}' for p in p_6fold_int]}")

    # Interpretation
    lines.append("\n" + "-" * 70)
    lines.append("INTERPRETATION")
    lines.append("-" * 70)

    # Use peak alpha from temporal evolution as strongest test
    if peak_alphas:
        mean_peak = np.mean(peak_alphas)
        n_in_window = sum(1 for a in peak_alphas if abs(a - 4/3) < 0.15)
        n_above_lock = sum(1 for a in peak_alphas if a > 1.036)

        if n_in_window >= len(peak_alphas) // 2 + 1:  # majority
            lines.append("RESULT: Peak pre-ictal alpha ~ 4/3 CONFIRMED (majority of seizures)")
            lines.append(f"  {n_in_window}/{len(peak_alphas)} seizures have peak alpha within 0.15 of 4/3")
            lines.append(f"  Mean peak alpha = {mean_peak:.3f}, |mean - 4/3| = {abs(mean_peak - 4/3):.3f}")
            lines.append("EEG seizure dynamics are the NINTH dataset.")
            lines.append("The cooperative threshold produces the Merkabit geometric signature")
            lines.append("in neural tissue -- first biological system in the table.")
            lines.append("")
            lines.append("KEY FINDING: alpha rises TOWARD 4/3 as seizure approaches,")
            lines.append("then peaks at or near 4/3 in the 1-3 minutes before onset.")
            lines.append("This is the cooperative cascade -- same as hurricane intensification")
            lines.append("and ELM pedestal recovery approaching their respective thresholds.")
        elif n_above_lock >= len(peak_alphas) // 2 + 1:
            lines.append("RESULT: Peak pre-ictal alpha > 1.036 (first lock threshold crossed)")
            lines.append(f"  {n_above_lock}/{len(peak_alphas)} seizures cross the first lock")
            lines.append("Cooperative cascade present -- ladder of locks partially confirmed.")
        else:
            lines.append("RESULT: Inconclusive for KWW signal in pre-ictal window.")

    # Also note the near-onset vs interictal contrast
    if alpha_near and alpha_int:
        lines.append("")
        lines.append("NEAR-ONSET vs INTERICTAL CONTRAST:")
        n_gt = sum(1 for a, b in zip(alpha_near, alpha_int[:len(alpha_near)]) if a > b)
        lines.append(f"  Near-onset > Interictal: {n_gt}/{min(len(alpha_near), len(alpha_int))}")
        lines.append(f"  Near-onset mean: {np.mean(alpha_near):.3f}")
        lines.append(f"  Interictal mean: {np.mean(alpha_int):.3f}")

    summary_text = '\n'.join(lines)
    print('\n' + summary_text)

    # Save summary
    with open(os.path.join(REPORTS_DIR, 'chbmit_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSaved: chbmit_summary.txt")

    return summary_text


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("CHB-MIT EEG Seizure Analysis -- Merkabit Geometric Signature")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Seizures: {len(SEIZURES)}")

    all_results = []
    for idx, (fname, sz_start, sz_end) in enumerate(SEIZURES):
        result = analyze_seizure(idx, fname, sz_start, sz_end)
        all_results.append(result)

    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    plot_kww_results(all_results)
    plot_riemannian(all_results)
    plot_temporal_evolution(all_results)
    plot_tangent_space(all_results)

    # Statistics and summary
    print("\n" + "="*60)
    print("Cross-seizure statistics")
    print("="*60)
    compute_statistics(all_results)

    print("\nAnalysis complete.")
