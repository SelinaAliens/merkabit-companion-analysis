"""
Rishikesh Meditation Phase 2 — Full KWW + Geometric Analysis
=============================================================
Requires: Raw EEG dataset (~6.3 GB) from Zenodo record 2348892
          Braboszcz et al. (2017) continuous EEG recordings

Run this script AFTER downloading the raw EEG data.
Expected data location: data/meditation/raw_eeg/ (see download_data.py)

Phase 1 result to build on:
  HYT alpha/gamma power ratio = 1.361 (within 2.1% of 4/3)

Tests implemented:
  P_MED_1: alpha_KWW distribution peaks near 4/3 for experienced groups
  P_MED_2a: HYT/ISY show lower std(alpha_KWW) than CTR — self-sustaining signature
  P_MED_2b: VIP shows HIGHER std(alpha_KWW) than both ISY and HYT — driven vs self-sustaining
  P_MED_3: Plateau |C| converges toward 1/3 at highest experience
  P_MED_4: Inter-channel coherence shifts toward Z3 (3-fold) symmetry
  P_MED_5: Net Berry phase accumulation -> 0 for advanced HYT
  P_MED_6: Shoonya lower var(alpha) than VIP at comparable hours
  P_MED_7: Pre-ictal alpha rises toward 4/3; meditation holds at 4/3
  P_MED_8: Novice rest alpha_KWW well below 4/3
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, MEDITATION_RAW, FIGURES_DIR, REPORTS_DIR

import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.signal import hilbert, welch
from scipy.stats import circmean, circstd, spearmanr, mannwhitneyu, kruskal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───
RAW_EEG_DIR = MEDITATION_RAW

# Group metadata
GROUPS = {
    'CTR': {'experience_hours': 0, 'color': '#4477AA', 'label': 'Control (0h)'},
    'ISY': {'experience_hours': 2625, 'color': '#66CCEE', 'label': 'Shoonya (2625h)'},
    'VIP': {'experience_hours': 9201, 'color': '#228833', 'label': 'Vipassana (9201h)'},
    'HYT': {'experience_hours': 15475, 'color': '#EE6677', 'label': 'HYT (15475h)'},
}
GROUP_ORDER = ['CTR', 'ISY', 'VIP', 'HYT']

# Analysis parameters
FS = 256  # Sampling rate (Hz) — confirm from data
WINDOW_SEC = 30  # Window length for KWW analysis
WINDOW_SAMPLES = FS * WINDOW_SEC
OVERLAP_FRAC = 0.5  # 50% overlap between windows
ACF_MAXLAG_SEC = 5.0  # Max lag for ACF (seconds)
ACF_MAXLAG = int(ACF_MAXLAG_SEC * FS)
ALPHA_43_WINDOW = 0.15  # Threshold for "near 4/3"


# ─── KWW MODEL ───
def kww(t, A, tau, alpha):
    """Kohlrausch-Williams-Watts stretched exponential."""
    return A * np.exp(-(t / tau) ** alpha)


def fit_kww(acf, dt, maxlag=None):
    """
    Fit KWW to autocorrelation function.
    Returns (A, tau, alpha, R2) or None if fit fails.
    """
    if maxlag is None:
        maxlag = len(acf)
    acf = acf[:maxlag]
    t = np.arange(len(acf)) * dt

    # Normalize
    if acf[0] <= 0:
        return None
    acf_norm = acf / acf[0]

    # Remove negative values (truncate at first zero crossing)
    zero_idx = np.where(acf_norm <= 0.01)[0]
    if len(zero_idx) > 0 and zero_idx[0] > 10:
        acf_norm = acf_norm[:zero_idx[0]]
        t = t[:len(acf_norm)]

    if len(acf_norm) < 20:
        return None

    try:
        popt, _ = curve_fit(
            kww, t, acf_norm,
            p0=[1.0, t[len(t)//4], 1.0],
            bounds=([0.5, dt, 0.1], [1.5, t[-1]*2, 3.0]),
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


def compute_acf(signal, maxlag):
    """Compute normalized autocorrelation function."""
    n = len(signal)
    signal = signal - signal.mean()
    var = np.var(signal)
    if var == 0:
        return np.zeros(maxlag)
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[n-1:n-1+maxlag] / (var * n)
    return acf


# ─── MAIN ANALYSIS FUNCTIONS ───

def load_raw_eeg(filepath):
    """
    Load raw EEG data from file.
    Adapt this function based on actual file format (SET, MAT, EDF, etc.)
    Returns: data (n_channels x n_samples), fs, channel_labels
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.mat':
        mat = sio.loadmat(filepath)
        # Try common variable names
        for key in ['EEG', 'data', 'eeg', 'X']:
            if key in mat:
                data = mat[key]
                if data.ndim == 2:
                    return data, FS, None
        # If structured EEG
        if 'EEG' in mat:
            eeg = mat['EEG']
            if hasattr(eeg, 'dtype') and 'data' in eeg.dtype.names:
                return eeg['data'][0, 0], FS, None

    elif ext == '.set':
        # EEGLAB .set format — try mne
        try:
            import mne
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
            return raw.get_data(), raw.info['sfreq'], raw.ch_names
        except ImportError:
            print("MNE not installed. Install with: pip install mne")
            return None, None, None

    elif ext in ['.edf', '.bdf']:
        try:
            import mne
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            return raw.get_data(), raw.info['sfreq'], raw.ch_names
        except ImportError:
            print("MNE not installed.")
            return None, None, None

    print(f"  WARNING: Could not load {filepath}")
    return None, None, None


def analyze_subject(data, fs, subject_id, group):
    """
    Full KWW + geometric analysis for one subject.
    data: (n_channels x n_samples)
    Returns dict with all results.
    """
    n_channels, n_samples = data.shape
    dt = 1.0 / fs
    step = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))
    n_windows = (n_samples - WINDOW_SAMPLES) // step + 1

    results = {
        'subject': subject_id,
        'group': group,
        'experience': GROUPS[group]['experience_hours'],
        'n_windows': n_windows,
        'alpha_kww': [],        # KWW alpha per window
        'tau_kww': [],          # KWW tau per window
        'R2_kww': [],           # KWW R2 per window
        'C_envelope': [],       # |C(t)| envelope per window
        'phase_cumulative': [], # cumulative phase per window
        'coherence_matrix': None,  # inter-channel coherence
        'z3_ratio': None,      # Z3 symmetry metric
    }

    # ─── Per-window KWW analysis ───
    for w in range(n_windows):
        start = w * step
        end = start + WINDOW_SAMPLES

        # Use broadband signal (mean across channels or first principal component)
        segment = data[:, start:end]
        broadband = segment.mean(axis=0)  # simple channel average

        # Compute ACF
        acf = compute_acf(broadband, ACF_MAXLAG)

        # Fit KWW
        fit = fit_kww(acf, dt, ACF_MAXLAG)
        if fit is not None:
            A, tau, alpha, R2 = fit
            if R2 > 0.3:  # quality threshold
                results['alpha_kww'].append(alpha)
                results['tau_kww'].append(tau)
                results['R2_kww'].append(R2)

        # Compute |C| envelope using Hilbert transform
        analytic = hilbert(broadband)
        envelope = np.abs(analytic)
        # Normalize to [0, 1] range
        if envelope.max() > 0:
            envelope_norm = envelope / envelope.max()
            results['C_envelope'].append(np.mean(envelope_norm))

        # Compute cumulative phase
        inst_phase = np.unwrap(np.angle(analytic))
        phase_accum = inst_phase[-1] - inst_phase[0]
        results['phase_cumulative'].append(phase_accum)

    # Convert to arrays
    results['alpha_kww'] = np.array(results['alpha_kww'])
    results['tau_kww'] = np.array(results['tau_kww'])
    results['R2_kww'] = np.array(results['R2_kww'])
    results['C_envelope'] = np.array(results['C_envelope'])
    results['phase_cumulative'] = np.array(results['phase_cumulative'])

    # ─── Summary statistics ───
    if len(results['alpha_kww']) > 0:
        results['mean_alpha'] = np.mean(results['alpha_kww'])
        results['std_alpha'] = np.std(results['alpha_kww'])
        results['median_alpha'] = np.median(results['alpha_kww'])
        results['frac_near_43'] = np.mean(np.abs(results['alpha_kww'] - 4/3) < ALPHA_43_WINDOW)
    else:
        results['mean_alpha'] = np.nan
        results['std_alpha'] = np.nan
        results['median_alpha'] = np.nan
        results['frac_near_43'] = 0.0

    # ─── Plateau |C| (last 25% of session) ───
    if len(results['C_envelope']) > 4:
        n_last = max(1, len(results['C_envelope']) // 4)
        results['plateau_C'] = np.mean(results['C_envelope'][-n_last:])
    else:
        results['plateau_C'] = np.nan

    # ─── Net Berry phase ───
    if len(results['phase_cumulative']) > 0:
        results['net_phase'] = np.sum(results['phase_cumulative'])
    else:
        results['net_phase'] = np.nan

    # ─── Inter-channel phase coherence (Z3 test) ───
    # Use full session, not windowed
    try:
        n_ch_use = min(n_channels, 32)  # limit for speed
        coh_matrix = np.zeros((n_ch_use, n_ch_use))

        # Take a representative segment (middle 60 seconds)
        mid = n_samples // 2
        seg_len = min(60 * int(fs), n_samples)
        seg_start = max(0, mid - seg_len // 2)
        seg_end = seg_start + seg_len
        seg_data = data[:n_ch_use, seg_start:seg_end]

        # Compute analytic signals
        analytic_channels = np.array([hilbert(seg_data[ch]) for ch in range(n_ch_use)])
        phases = np.angle(analytic_channels)

        # Phase coherence matrix
        for i in range(n_ch_use):
            for j in range(i+1, n_ch_use):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                coh_matrix[i, j] = plv
                coh_matrix[j, i] = plv

        results['coherence_matrix'] = coh_matrix

        # Z3 symmetry test: circular harmonic analysis
        # Compute the distribution of phase coherence values
        upper_tri = coh_matrix[np.triu_indices(n_ch_use, k=1)]

        # Compute circular harmonics of the coherence distribution
        # Map coherence to angle: theta = 2*pi*coherence
        theta = 2 * np.pi * upper_tri
        h2 = np.abs(np.mean(np.exp(2j * theta)))  # 2-fold (bilateral)
        h3 = np.abs(np.mean(np.exp(3j * theta)))  # 3-fold (Z3)
        h4 = np.abs(np.mean(np.exp(4j * theta)))  # 4-fold

        results['z3_ratio'] = h3 / h2 if h2 > 0 else np.nan
        results['h2'] = h2
        results['h3'] = h3
        results['h4'] = h4

    except Exception as e:
        print(f"    Z3 analysis failed for {subject_id}: {e}")
        results['z3_ratio'] = np.nan

    return results


def find_eeg_files(base_dir):
    """
    Find and organize raw EEG files by group and subject.
    Returns: dict {group: [list of file paths]}
    Adapt patterns based on actual file naming convention.
    """
    files_by_group = {g: [] for g in GROUP_ORDER}

    # Try common patterns
    patterns = [
        '**/*.set', '**/*.mat', '**/*.edf', '**/*.bdf',
        '**/*MED*.set', '**/*medit*.set'
    ]

    all_files = []
    for pattern in patterns:
        found = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        all_files.extend(found)
    all_files = list(set(all_files))

    # Sort into groups based on filename
    for f in all_files:
        fname = os.path.basename(f).upper()
        for g in GROUP_ORDER:
            if g in fname:
                files_by_group[g].append(f)
                break

    # If no group labels in filenames, try directory structure
    if all(len(v) == 0 for v in files_by_group.values()):
        for g in GROUP_ORDER:
            group_dir = os.path.join(base_dir, g)
            if os.path.isdir(group_dir):
                for ext in ['*.set', '*.mat', '*.edf']:
                    files_by_group[g].extend(glob.glob(os.path.join(group_dir, ext)))

    for g in GROUP_ORDER:
        print(f"  {g}: {len(files_by_group[g])} files")

    return files_by_group


# ─── MAIN EXECUTION ───
def main():
    print("="*70)
    print("PHASE 2: Rishikesh Meditation KWW + Geometric Analysis")
    print("="*70)

    # Check for raw EEG data
    if not os.path.isdir(RAW_EEG_DIR):
        print(f"\nERROR: Raw EEG directory not found: {RAW_EEG_DIR}")
        print("Please download from Zenodo 2348892 and extract to this location.")
        print("Expected size: ~6.3 GB")
        return

    # Find files
    print("\nSearching for EEG files...")
    files_by_group = find_eeg_files(RAW_EEG_DIR)

    total_files = sum(len(v) for v in files_by_group.values())
    if total_files == 0:
        print("\nNo EEG files found. Check directory structure.")
        print(f"Searched in: {RAW_EEG_DIR}")
        return

    # Analyze all subjects
    all_results = []
    for group in GROUP_ORDER:
        print(f"\n--- Processing {group} ({len(files_by_group[group])} subjects) ---")
        for i, filepath in enumerate(sorted(files_by_group[group])):
            subject_id = f"{group}_{i+1:02d}"
            print(f"  {subject_id}: {os.path.basename(filepath)}")

            data, fs, ch_names = load_raw_eeg(filepath)
            if data is None:
                print(f"    SKIPPED (load failed)")
                continue

            if fs != FS:
                print(f"    NOTE: fs={fs}, expected {FS}")

            print(f"    Shape: {data.shape}, fs={fs}")
            result = analyze_subject(data, fs, subject_id, group)
            all_results.append(result)

            n_fits = len(result['alpha_kww'])
            print(f"    KWW fits: {n_fits}, mean_alpha={result['mean_alpha']:.3f}, "
                  f"std={result['std_alpha']:.3f}, frac_near_43={result['frac_near_43']:.3f}")

    if len(all_results) == 0:
        print("\nNo subjects successfully analyzed.")
        return

    # ─── GROUP-LEVEL RESULTS ───
    print("\n" + "="*70)
    print("GROUP-LEVEL RESULTS")
    print("="*70)

    for group in GROUP_ORDER:
        group_results = [r for r in all_results if r['group'] == group]
        if len(group_results) == 0:
            continue

        alphas = [r['mean_alpha'] for r in group_results if not np.isnan(r['mean_alpha'])]
        stds = [r['std_alpha'] for r in group_results if not np.isnan(r['std_alpha'])]
        fracs = [r['frac_near_43'] for r in group_results]
        plateaus = [r['plateau_C'] for r in group_results if not np.isnan(r['plateau_C'])]
        phases = [r['net_phase'] for r in group_results if not np.isnan(r['net_phase'])]
        z3s = [r['z3_ratio'] for r in group_results if not np.isnan(r.get('z3_ratio', np.nan))]

        print(f"\n{group} (n={len(group_results)}, {GROUPS[group]['experience_hours']}h):")
        if alphas:
            print(f"  mean(alpha_KWW) = {np.mean(alphas):.4f} +/- {np.std(alphas):.4f}")
            print(f"  |mean - 4/3| = {abs(np.mean(alphas) - 4/3):.4f}")
        if stds:
            print(f"  mean(std_alpha) = {np.mean(stds):.4f}")
        if fracs:
            print(f"  frac_near_4/3 = {np.mean(fracs):.4f}")
        if plateaus:
            print(f"  plateau |C| = {np.mean(plateaus):.4f} +/- {np.std(plateaus):.4f}")
            print(f"  |plateau - 1/3| = {abs(np.mean(plateaus) - 1/3):.4f}")
        if phases:
            print(f"  net phase = {np.mean(phases):.2f} +/- {np.std(phases):.2f} rad")
        if z3s:
            print(f"  Z3 ratio (h3/h2) = {np.mean(z3s):.4f} +/- {np.std(z3s):.4f}")

    # ─── STATISTICAL TESTS ───
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    # P_MED_1: alpha_KWW near 4/3 for experienced groups
    group_alphas = {}
    for group in GROUP_ORDER:
        vals = [r['mean_alpha'] for r in all_results
                if r['group'] == group and not np.isnan(r['mean_alpha'])]
        if vals:
            group_alphas[group] = vals

    if len(group_alphas) >= 2:
        # Kruskal-Wallis across groups
        kw_stat, kw_p = kruskal(*[group_alphas[g] for g in GROUP_ORDER if g in group_alphas])
        print(f"\nP_MED_1 Kruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.4f}")

        # Monotonic ordering test
        exp_hours = []
        alpha_medians = []
        for g in GROUP_ORDER:
            if g in group_alphas:
                exp_hours.append(GROUPS[g]['experience_hours'])
                alpha_medians.append(np.median(group_alphas[g]))
        rho, p_rho = spearmanr(exp_hours, alpha_medians)
        print(f"  Spearman (experience vs median alpha): rho={rho:.4f}, p={p_rho:.4f}")

    # P_MED_2: Variance signature
    group_stds = {}
    for group in GROUP_ORDER:
        vals = [r['std_alpha'] for r in all_results
                if r['group'] == group and not np.isnan(r['std_alpha'])]
        if vals:
            group_stds[group] = vals

    if 'HYT' in group_stds and 'CTR' in group_stds:
        u_stat, u_p = mannwhitneyu(group_stds['HYT'], group_stds['CTR'], alternative='less')
        print(f"\nP_MED_2a Mann-Whitney (HYT std < CTR std): U={u_stat:.1f}, p={u_p:.4f}")

    if 'ISY' in group_stds and 'CTR' in group_stds:
        u_stat, u_p = mannwhitneyu(group_stds['ISY'], group_stds['CTR'], alternative='less')
        print(f"  Mann-Whitney (ISY std < CTR std): U={u_stat:.1f}, p={u_p:.4f}")

    # P_MED_2b: VIP shows HIGHER variance than ISY and HYT (driven vs self-sustaining)
    # This is the cleanest test: VIP at 9,201h should be MORE variable than ISY at 2,625h
    # because VIP practitioners actively maintain a state (driven) rather than settling (self-sustaining)
    print(f"\n  P_MED_2b: Driven vs Self-Sustaining Variance Test")
    if 'VIP' in group_stds and 'ISY' in group_stds:
        u_stat, u_p = mannwhitneyu(group_stds['VIP'], group_stds['ISY'], alternative='greater')
        print(f"  Mann-Whitney (VIP std > ISY std): U={u_stat:.1f}, p={u_p:.4f}")
        print(f"    VIP mean std = {np.mean(group_stds['VIP']):.4f} (9201h, focused attention)")
        print(f"    ISY mean std = {np.mean(group_stds['ISY']):.4f} (2625h, open awareness)")

    if 'VIP' in group_stds and 'HYT' in group_stds:
        u_stat, u_p = mannwhitneyu(group_stds['VIP'], group_stds['HYT'], alternative='greater')
        print(f"  Mann-Whitney (VIP std > HYT std): U={u_stat:.1f}, p={u_p:.4f}")
        print(f"    VIP mean std = {np.mean(group_stds['VIP']):.4f} (9201h, focused attention)")
        print(f"    HYT mean std = {np.mean(group_stds['HYT']):.4f} (15475h)")

    # Summary: ordering prediction for driven vs self-sustaining
    if all(g in group_stds for g in GROUP_ORDER):
        means = {g: np.mean(group_stds[g]) for g in GROUP_ORDER}
        print(f"\n  Variance ordering: ", end='')
        ranked = sorted(GROUP_ORDER, key=lambda g: means[g], reverse=True)
        print(' > '.join([f"{g}({means[g]:.4f})" for g in ranked]))
        predicted = "VIP > CTR > ISY ~ HYT"
        print(f"  Predicted ordering (driven vs self-sustaining): {predicted}")

    # P_MED_3: |C| convergence to 1/3
    group_plateaus = {}
    for group in GROUP_ORDER:
        vals = [r['plateau_C'] for r in all_results
                if r['group'] == group and not np.isnan(r['plateau_C'])]
        if vals:
            group_plateaus[group] = vals

    if group_plateaus:
        exp_h = []
        plat_med = []
        for g in GROUP_ORDER:
            if g in group_plateaus:
                exp_h.append(GROUPS[g]['experience_hours'])
                plat_med.append(np.median(group_plateaus[g]))
        if len(exp_h) >= 3:
            rho_c, p_c = spearmanr(exp_h, plat_med)
            print(f"\nP_MED_3 Spearman (experience vs plateau |C|): rho={rho_c:.4f}, p={p_c:.4f}")

    # P_MED_4: Z3 symmetry
    group_z3 = {}
    for group in GROUP_ORDER:
        vals = [r['z3_ratio'] for r in all_results
                if r['group'] == group and not np.isnan(r.get('z3_ratio', np.nan))]
        if vals:
            group_z3[group] = vals

    if 'HYT' in group_z3 and 'CTR' in group_z3:
        u_stat, u_p = mannwhitneyu(group_z3['HYT'], group_z3['CTR'], alternative='greater')
        print(f"\nP_MED_4 Mann-Whitney (HYT z3 > CTR z3): U={u_stat:.1f}, p={u_p:.4f}")

    # P_MED_5: Berry phase
    group_phase = {}
    for group in GROUP_ORDER:
        vals = [np.abs(r['net_phase']) for r in all_results
                if r['group'] == group and not np.isnan(r['net_phase'])]
        if vals:
            group_phase[group] = vals

    if 'HYT' in group_phase and 'CTR' in group_phase:
        u_stat, u_p = mannwhitneyu(group_phase['HYT'], group_phase['CTR'], alternative='less')
        print(f"\nP_MED_5 Mann-Whitney (HYT |phase| < CTR |phase|): U={u_stat:.1f}, p={u_p:.4f}")

    # P_MED_6: Shoonya vs Vipassana variance (complements P_MED_2b)
    # ISY (open awareness, 2625h) should have LOWER variance than VIP (focused, 9201h)
    # even though VIP has 3.5x more hours — practice type matters more than duration
    if 'ISY' in group_stds and 'VIP' in group_stds:
        u_stat, u_p = mannwhitneyu(group_stds['ISY'], group_stds['VIP'], alternative='less')
        print(f"\nP_MED_6 Mann-Whitney (ISY std < VIP std): U={u_stat:.1f}, p={u_p:.4f}")
        print(f"    ISY (2625h, open awareness): mean std = {np.mean(group_stds['ISY']):.4f}")
        print(f"    VIP (9201h, focused attention): mean std = {np.mean(group_stds['VIP']):.4f}")
        if np.mean(group_stds['ISY']) < np.mean(group_stds['VIP']):
            print(f"    Direction CONSISTENT with driven vs self-sustaining prediction")

    # ─── PLOTS ───
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Plot 1: alpha_KWW distribution by group
    ax = axes[0, 0]
    for g in GROUP_ORDER:
        group_res = [r for r in all_results if r['group'] == g]
        all_alphas = np.concatenate([r['alpha_kww'] for r in group_res if len(r['alpha_kww']) > 0])
        if len(all_alphas) > 0:
            ax.hist(all_alphas, bins=30, alpha=0.5, color=GROUPS[g]['color'],
                    label=GROUPS[g]['label'], density=True)
    ax.axvline(x=4/3, color='red', ls='--', lw=2, label='4/3')
    ax.set_xlabel('alpha_KWW')
    ax.set_ylabel('Density')
    ax.set_title('P_MED_1: KWW alpha distribution')
    ax.legend(fontsize=7)

    # Plot 2: mean_alpha vs experience
    ax = axes[0, 1]
    for g in GROUP_ORDER:
        if g in group_alphas:
            vals = group_alphas[g]
            ax.scatter([GROUPS[g]['experience_hours']]*len(vals), vals,
                      c=GROUPS[g]['color'], alpha=0.6, s=40, label=GROUPS[g]['label'])
            ax.errorbar(GROUPS[g]['experience_hours'], np.mean(vals),
                       yerr=np.std(vals), fmt='D', color=GROUPS[g]['color'],
                       markersize=10, capsize=5, zorder=5)
    ax.axhline(y=4/3, color='red', ls='--', lw=2)
    ax.set_xlabel('Experience (hours)')
    ax.set_ylabel('mean alpha_KWW')
    ax.set_title('P_MED_1: Alpha vs Experience')
    ax.legend(fontsize=7)

    # Plot 3: var_alpha vs mean_alpha (scatter, colored by group)
    ax = axes[0, 2]
    for r in all_results:
        if not np.isnan(r['mean_alpha']):
            ax.scatter(r['mean_alpha'], r['std_alpha'],
                      c=GROUPS[r['group']]['color'], s=50, alpha=0.7,
                      edgecolors='black', linewidths=0.5)
    ax.axvline(x=4/3, color='red', ls='--', alpha=0.7)
    ax.set_xlabel('mean alpha_KWW')
    ax.set_ylabel('std alpha_KWW')
    ax.set_title('P_MED_2: Variance Signature')
    # Add group labels
    for g in GROUP_ORDER:
        ax.scatter([], [], c=GROUPS[g]['color'], s=50, label=GROUPS[g]['label'])
    ax.legend(fontsize=7)

    # Plot 4: Plateau |C| vs experience
    ax = axes[1, 0]
    for g in GROUP_ORDER:
        if g in group_plateaus:
            vals = group_plateaus[g]
            ax.scatter([GROUPS[g]['experience_hours']]*len(vals), vals,
                      c=GROUPS[g]['color'], alpha=0.6, s=40)
            ax.errorbar(GROUPS[g]['experience_hours'], np.mean(vals),
                       yerr=np.std(vals), fmt='D', color=GROUPS[g]['color'],
                       markersize=10, capsize=5, zorder=5)
    ax.axhline(y=1/3, color='red', ls='--', lw=2, label='1/3')
    ax.set_xlabel('Experience (hours)')
    ax.set_ylabel('Plateau |C|')
    ax.set_title('P_MED_3: |C| -> 1/3?')
    ax.legend(fontsize=7)

    # Plot 5: Z3 ratio vs experience
    ax = axes[1, 1]
    for g in GROUP_ORDER:
        if g in group_z3:
            vals = group_z3[g]
            ax.scatter([GROUPS[g]['experience_hours']]*len(vals), vals,
                      c=GROUPS[g]['color'], alpha=0.6, s=40)
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='h3=h2')
    ax.set_xlabel('Experience (hours)')
    ax.set_ylabel('Z3 ratio (h3/h2)')
    ax.set_title('P_MED_4: Z3 Symmetry')

    # Plot 6: Net Berry phase vs experience
    ax = axes[1, 2]
    for g in GROUP_ORDER:
        if g in group_phase:
            vals = group_phase[g]
            ax.scatter([GROUPS[g]['experience_hours']]*len(vals), vals,
                      c=GROUPS[g]['color'], alpha=0.6, s=40)
    ax.set_xlabel('Experience (hours)')
    ax.set_ylabel('|Net phase| (rad)')
    ax.set_title('P_MED_5: Berry Phase Cancellation')

    # Plot 7: ISY vs VIP variance comparison
    ax = axes[2, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for g in GROUP_ORDER:
        if g in group_stds:
            box_data.append(group_stds[g])
            box_labels.append(f"{g}\n({GROUPS[g]['experience_hours']}h)")
            box_colors.append(GROUPS[g]['color'])
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(box_labels)
    ax.set_ylabel('std(alpha_KWW)')
    ax.set_title('P_MED_6: Variance by Group')

    # Plot 8: Fraction near 4/3 by group
    ax = axes[2, 1]
    for g in GROUP_ORDER:
        fracs = [r['frac_near_43'] for r in all_results if r['group'] == g]
        if fracs:
            ax.bar(g, np.mean(fracs), color=GROUPS[g]['color'], alpha=0.7,
                   yerr=np.std(fracs), capsize=5)
    ax.set_ylabel('Fraction of windows |alpha - 4/3| < 0.15')
    ax.set_title('Fraction Near 4/3')

    # Plot 9: Seizure comparison (P_MED_7)
    ax = axes[2, 2]
    # Load CHB-MIT pre-ictal data if available
    chbmit_file = os.path.join(REPORTS_DIR, 'chbmit_alpha_timeseries.npz')
    if os.path.exists(chbmit_file):
        chbmit = np.load(chbmit_file, allow_pickle=True)
        # Plot pre-ictal alpha trajectory
        if 'alpha_timeseries' in chbmit:
            for ts in chbmit['alpha_timeseries'][:3]:  # first 3 seizures
                t_norm = np.linspace(0, 1, len(ts))
                ax.plot(t_norm, ts, color='black', alpha=0.3, lw=1)
            ax.plot([], [], color='black', alpha=0.5, label='Pre-ictal')
    else:
        print("  CHB-MIT data not found for comparison plot")

    # Plot meditation alpha trajectories (first window to last)
    for g in ['HYT', 'ISY']:
        group_res = [r for r in all_results if r['group'] == g]
        for r in group_res[:3]:
            if len(r['alpha_kww']) > 5:
                t_norm = np.linspace(0, 1, len(r['alpha_kww']))
                ax.plot(t_norm, r['alpha_kww'], color=GROUPS[g]['color'], alpha=0.3, lw=1)
        ax.plot([], [], color=GROUPS[g]['color'], alpha=0.5, label=g)

    ax.axhline(y=4/3, color='red', ls='--', lw=2, label='4/3')
    ax.set_xlabel('Normalized time')
    ax.set_ylabel('alpha_KWW')
    ax.set_title('P_MED_7: Seizure vs Meditation')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_results.png'), dpi=150)
    print(f"Saved: phase2_results.png")

    # ─── SAVE NUMERICAL RESULTS ───
    with open(os.path.join(REPORTS_DIR, 'phase2_results.txt'), 'w') as f:
        f.write("PHASE 2 RESULTS — Rishikesh Meditation\n")
        f.write("="*50 + "\n\n")

        for group in GROUP_ORDER:
            group_res = [r for r in all_results if r['group'] == group]
            if not group_res:
                continue
            f.write(f"\n{group} ({GROUPS[group]['experience_hours']}h, n={len(group_res)}):\n")

            alphas = [r['mean_alpha'] for r in group_res if not np.isnan(r['mean_alpha'])]
            if alphas:
                f.write(f"  mean(alpha_KWW) = {np.mean(alphas):.4f} +/- {np.std(alphas):.4f}\n")
                f.write(f"  |mean - 4/3| = {abs(np.mean(alphas) - 4/3):.4f}\n")

            stds = [r['std_alpha'] for r in group_res if not np.isnan(r['std_alpha'])]
            if stds:
                f.write(f"  mean(std_alpha) = {np.mean(stds):.4f}\n")

            fracs = [r['frac_near_43'] for r in group_res]
            f.write(f"  frac_near_4/3 = {np.mean(fracs):.4f}\n")

            plateaus = [r['plateau_C'] for r in group_res if not np.isnan(r['plateau_C'])]
            if plateaus:
                f.write(f"  plateau |C| = {np.mean(plateaus):.4f} +/- {np.std(plateaus):.4f}\n")
                f.write(f"  |plateau - 1/3| = {abs(np.mean(plateaus) - 1/3):.4f}\n")

            phases = [np.abs(r['net_phase']) for r in group_res if not np.isnan(r['net_phase'])]
            if phases:
                f.write(f"  |net_phase| = {np.mean(phases):.2f} +/- {np.std(phases):.2f} rad\n")

            z3s = [r['z3_ratio'] for r in group_res if not np.isnan(r.get('z3_ratio', np.nan))]
            if z3s:
                f.write(f"  Z3 ratio = {np.mean(z3s):.4f} +/- {np.std(z3s):.4f}\n")

        # P_MED_2b: Write driven vs self-sustaining variance comparison
        f.write("\n\nDRIVEN vs SELF-SUSTAINING VARIANCE (P_MED_2b):\n")
        f.write("-"*50 + "\n")
        f.write("VIP (Vipassana, 9201h, focused attention) should show HIGHER\n")
        f.write("std(alpha_KWW) than ISY (Shoonya, 2625h, open awareness) and\n")
        f.write("HYT (15475h), despite having 3.5x more hours than ISY.\n")
        f.write("This is the cleanest driven vs self-sustaining test:\n")
        f.write("actively maintaining a state = higher temporal variance.\n\n")
        if all(g in group_stds for g in GROUP_ORDER):
            for g in GROUP_ORDER:
                f.write(f"  {g} ({GROUPS[g]['experience_hours']}h): "
                        f"mean std(alpha) = {np.mean(group_stds[g]):.4f}\n")
            ranked = sorted(GROUP_ORDER,
                          key=lambda g: np.mean(group_stds[g]), reverse=True)
            f.write(f"\n  Actual ordering: {' > '.join(ranked)}\n")
            f.write(f"  Predicted:       VIP > CTR > ISY ~ HYT\n")

        f.write("\n\nPREDICTION OUTCOMES:\n")
        f.write("-"*40 + "\n")
        f.write("P_MED_1:  alpha_KWW near 4/3 for experienced — [CHECK ABOVE]\n")
        f.write("P_MED_2a: HYT/ISY lower std(alpha) than CTR — [CHECK ABOVE]\n")
        f.write("P_MED_2b: VIP higher std(alpha) than ISY AND HYT — [CHECK ABOVE]\n")
        f.write("P_MED_3:  plateau |C| -> 1/3 for HYT — [CHECK ABOVE]\n")
        f.write("P_MED_4:  Z3 symmetry shift — [CHECK ABOVE]\n")
        f.write("P_MED_5:  Berry phase -> 0 for HYT — [CHECK ABOVE]\n")
        f.write("P_MED_6:  ISY std < VIP std — [CHECK ABOVE]\n")
        f.write("P_MED_7:  Pre-ictal rises; meditation holds — [SEE PLOT]\n")
        f.write("P_MED_8:  Novice rest << 4/3 — [CHECK CTR ABOVE]\n")

    print(f"Saved: phase2_results.txt")
    print("\nPhase 2 analysis complete.")


if __name__ == '__main__':
    main()
