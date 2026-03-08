"""
Windowed Z3 Time Series — Braboszcz Meditation Dataset
=======================================================
Sliding-window Z3 analysis on each subject's full EEG session.
Extends Phase 1 static Z3 score to a temporal trajectory.

Prediction: Z3 score (H3/H2) spends significantly more time above 1.0
in HYT than in controls, and VIP ~ CTR (focused attention preserves
bilateral structure, open awareness dissolves it).

Data: EEG_datasets.tar.gz from Zenodo 57911 (EEGLAB .set files)
      Requires raw EEG download: python download_data.py meditation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, MEDITATION_RAW, FIGURES_DIR, REPORTS_DIR

import numpy as np
import scipy.io as sio
import scipy.signal as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, spearmanr
import glob
import json
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# EEG_datasets directory is inside the raw EEG download
EEG_DATA_DIR = os.path.join(MEDITATION_RAW, 'EEG_datasets')
CHANLOCS_PATH = os.path.join(MEDITATION_DATA, 'chanlocs.mat')

GROUPS = {
    'control':   {'label': 'CTR', 'hours': 0,     'color': '#AAAAAA'},
    'shoonya':   {'label': 'ISY', 'hours': 2625,  'color': '#5599DD'},
    'vipassana': {'label': 'VIP', 'hours': 9201,  'color': '#DD8833'},
    'hyt':       {'label': 'HYT', 'hours': 15475, 'color': '#AA3333'},
}
GROUP_ORDER = ['control', 'shoonya', 'vipassana', 'hyt']
LABEL_MAP = {g: GROUPS[g]['label'] for g in GROUP_ORDER}

WINDOW_SEC = 30.0
STEP_SEC = 10.0  # 20-second overlap for smooth time series
FREQS_ALPHA = (8, 13)
FREQS_GAMMA = (30, 80)


# --- CHANNEL ANGLES ---
# Load from the chanlocs.mat we already have (64 EEG channels)
def load_channel_angles():
    """
    Load scalp angles from chanlocs.mat (same as Phase 1 analysis).
    Returns: angle_dict {channel_name: angle_rad}, labels list
    """
    mat_cl = sio.loadmat(CHANLOCS_PATH)
    N_CH = 64
    thetas_deg = np.array([float(mat_cl['chanlocs']['theta'][0, i][0, 0]) for i in range(N_CH)])
    radii = np.array([float(mat_cl['chanlocs']['radius'][0, i][0, 0]) for i in range(N_CH)])
    labels = [str(mat_cl['chanlocs']['labels'][0, i][0]) for i in range(N_CH)]

    # Convert to 2D scalp coordinates
    theta_rad = thetas_deg * np.pi / 180
    plot_x = radii * np.sin(theta_rad)
    plot_y = radii * np.cos(theta_rad)
    # Scalp angle from center
    scalp_angle = np.arctan2(plot_x, plot_y)

    # Return as dictionary for name-based lookup
    angle_dict = {lab: ang for lab, ang in zip(labels, scalp_angle)}
    return angle_dict, labels


# Fallback if chanlocs not available
def get_channel_angles_fallback(n_ch):
    return np.linspace(0, 2 * np.pi, n_ch, endpoint=False)


# --- Z3 COMPUTATION ---
def z3_from_window(data_window, srate, angles, freqs_alpha=FREQS_ALPHA, freqs_gamma=FREQS_GAMMA):
    """
    Compute Z3 score from a single window of multi-channel EEG.
    data_window: (n_ch, n_samples)
    angles: (n_ch,) scalp angles in radians
    Returns: H2, H3, H6, Z3=H3/H2
    """
    n_ch = data_window.shape[0]

    # Per-channel alpha and gamma power via Welch
    nperseg = min(int(srate * 4), data_window.shape[1])
    f, psd = sp.welch(data_window, fs=srate, nperseg=nperseg, axis=1)
    alpha_mask = (f >= freqs_alpha[0]) & (f <= freqs_alpha[1])
    gamma_mask = (f >= freqs_gamma[0]) & (f <= freqs_gamma[1])

    alpha_pwr = psd[:, alpha_mask].mean(axis=1)
    gamma_pwr = psd[:, gamma_mask].mean(axis=1)
    gamma_pwr = np.where(gamma_pwr < 1e-30, 1e-30, gamma_pwr)
    ratio_map = alpha_pwr / gamma_pwr  # shape (n_ch,)

    # Demean for spatial pattern (circular harmonics of the deviation)
    r = ratio_map - ratio_map.mean()

    # Truncate angles to match n_ch if needed
    ang = angles[:n_ch] if len(angles) >= n_ch else np.pad(angles, (0, n_ch - len(angles)))

    H1 = np.abs(np.sum(r * np.exp(1j * 1 * ang))) / n_ch
    H2 = np.abs(np.sum(r * np.exp(1j * 2 * ang))) / n_ch
    H3 = np.abs(np.sum(r * np.exp(1j * 3 * ang))) / n_ch
    H4 = np.abs(np.sum(r * np.exp(1j * 4 * ang))) / n_ch
    H6 = np.abs(np.sum(r * np.exp(1j * 6 * ang))) / n_ch

    Z3 = H3 / H2 if H2 > 1e-10 else np.nan

    return H1, H2, H3, H4, H6, Z3, ratio_map.mean()


# --- DATA LOADING ---
# Non-EEG channel prefixes to drop (external electrodes, mastoid refs, etc.)
NON_EEG_PREFIXES = ('EXG', 'M1', 'M2', 'GSR', 'Erg', 'Status', 'STI')

def match_angles_by_name(ch_names, angle_dict):
    """
    Match channel names to angles from the chanlocs dictionary.
    Returns angles array with proper per-channel alignment.
    Channels not found in the dict get fallback uniform spacing.
    """
    n_ch = len(ch_names)
    angles = np.zeros(n_ch)
    matched = 0
    for i, name in enumerate(ch_names):
        if name in angle_dict:
            angles[i] = angle_dict[name]
            matched += 1
        else:
            # Assign a fallback angle for unmatched channels
            angles[i] = 2 * np.pi * i / n_ch
    return angles, matched


def load_eeg_set(filepath, angle_dict=None):
    """
    Load EEGLAB .set file. Try MNE first, fall back to scipy.
    Drops non-EEG channels (EXG, M1, M2) to keep only scalp EEG.
    Uses angle_dict (channel_name -> angle) for proper spatial alignment.
    Returns: data (n_ch x n_samples), srate, angles, ch_names
    """
    # Try MNE
    try:
        import mne
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)

        # Identify and drop non-EEG channels
        eeg_picks = [i for i, ch in enumerate(raw.ch_names)
                     if not any(ch.startswith(p) for p in NON_EEG_PREFIXES)]
        data = raw.get_data(picks=eeg_picks)
        srate = raw.info['sfreq']
        ch_names = [raw.ch_names[i] for i in eeg_picks]

        # Match angles by channel name (not index!)
        if angle_dict is not None:
            angles, n_matched = match_angles_by_name(ch_names, angle_dict)
        else:
            angles = get_channel_angles_fallback(data.shape[0])

        return data, srate, angles, ch_names
    except Exception as e_mne:
        pass

    # Fall back to scipy
    try:
        mat = sio.loadmat(filepath, simplify_cells=True)
        eeg = mat['EEG']
        data = np.array(eeg['data'], dtype=np.float64)
        srate = float(eeg['srate'])
        # Get channel names
        chanlocs = eeg.get('chanlocs', None)
        ch_names = []
        if chanlocs is not None:
            if isinstance(chanlocs, np.ndarray):
                for ch in chanlocs:
                    if isinstance(ch, dict):
                        ch_names.append(ch.get('labels', ''))
                    elif isinstance(ch, np.void):
                        ch_names.append(str(ch['labels']) if 'labels' in ch.dtype.names else '')
            elif isinstance(chanlocs, dict):
                ch_names = [str(l) for l in chanlocs.get('labels', [])]

        # Drop non-EEG channels if we have names
        if ch_names:
            eeg_mask = [not any(ch.startswith(p) for p in NON_EEG_PREFIXES) for ch in ch_names]
            eeg_idx = [i for i, keep in enumerate(eeg_mask) if keep]
            if eeg_idx and len(eeg_idx) < data.shape[0]:
                data = data[eeg_idx]
                ch_names = [ch_names[i] for i in eeg_idx]

        if angle_dict is not None and ch_names:
            angles, _ = match_angles_by_name(ch_names, angle_dict)
        else:
            angles = get_channel_angles_fallback(data.shape[0])

        return data, srate, angles, ch_names
    except Exception as e_scipy:
        print(f"    FAILED to load {filepath}")
        print(f"    MNE error: {e_mne}")
        print(f"    SciPy error: {e_scipy}")
        return None, None, None, None


# --- PER-SUBJECT PROCESSING ---
def process_subject_z3(filepath, group_label, angle_dict=None):
    """
    Compute sliding-window Z3 time series for one subject.
    """
    data, srate, angles, ch_names = load_eeg_set(filepath, angle_dict)
    if data is None:
        return None

    n_ch, n_times = data.shape
    win_n = int(WINDOW_SEC * srate)
    step_n = int(STEP_SEC * srate)

    # Reduce window if session too short
    if n_times < win_n:
        print(f"    Session too short ({n_times/srate:.1f}s), reducing window to 15s")
        win_n = int(15 * srate)
        if n_times < win_n:
            print(f"    Still too short, skipping")
            return None

    z3_series = []
    h1_series = []
    h2_series = []
    h3_series = []
    h4_series = []
    h6_series = []
    ratio_series = []
    t_centres = []

    for start in range(0, n_times - win_n, step_n):
        window = data[:, start:start + win_n]
        H1, H2, H3, H4, H6, Z3, mean_ratio = z3_from_window(window, srate, angles)
        if not np.isnan(Z3):
            z3_series.append(Z3)
            h1_series.append(H1)
            h2_series.append(H2)
            h3_series.append(H3)
            h4_series.append(H4)
            h6_series.append(H6)
            ratio_series.append(mean_ratio)
            t_centres.append((start + win_n / 2) / srate / 60)  # minutes

    z3_arr = np.array(z3_series)
    h2_arr = np.array(h2_series)
    h3_arr = np.array(h3_series)

    if len(z3_arr) == 0:
        print(f"    No valid windows")
        return None

    # Early vs late session
    mid = len(z3_arr) // 2
    early_z3 = np.mean(z3_arr[:mid]) if mid > 0 else np.nan
    late_z3 = np.mean(z3_arr[mid:]) if mid > 0 else np.nan

    result = {
        'group': group_label,
        'filepath': os.path.basename(filepath),
        'n_channels': n_ch,
        'srate': srate,
        'duration_min': n_times / srate / 60,
        'z3_series': z3_series,
        'h2_series': h2_series,
        'h3_series': h3_series,
        'ratio_series': ratio_series,
        't_centres_min': t_centres,
        'median_z3': float(np.median(z3_arr)),
        'mean_z3': float(np.mean(z3_arr)),
        'frac_above_1': float(np.mean(z3_arr > 1.0)),
        'frac_above_08': float(np.mean(z3_arr > 0.8)),
        'z3_std': float(np.std(z3_arr)),
        'n_windows': len(z3_arr),
        'early_z3': float(early_z3),
        'late_z3': float(late_z3),
        'session_drift': float(late_z3 - early_z3),
        'h2_median': float(np.median(h2_arr)),
        'h3_median': float(np.median(h3_arr)),
    }

    print(f"    {n_ch}ch, {srate:.0f}Hz, {result['duration_min']:.1f}min, "
          f"{result['n_windows']} windows, "
          f"median Z3={result['median_z3']:.3f}, "
          f"frac>1: {result['frac_above_1']:.1%}, "
          f"drift={result['session_drift']:+.3f}")

    return result


# --- FIND EEG FILES BY GROUP ---
# File naming in this dataset:
#   CTR##_date_breath2.set  (control - no medit condition, breath2 = their MED equivalent)
#   HT##_date_medit.set     (HYT group, prefix HT)
#   SNY##_date_medit.set    (Shoonya group, prefix SNY)
#   VIP##_date_medit.set    (Vipassana group, prefix VIP)
# All files are in a flat directory (no subdirectories).

PREFIX_MAP = {
    'control':   'CTR',
    'shoonya':   'SNY',
    'vipassana': 'VIP',
    'hyt':       'HT',
}
# CTR has breath2 (not medit); practitioners have medit
CONDITION_MAP = {
    'control':   '_breath2.set',
    'shoonya':   '_medit.set',
    'vipassana': '_medit.set',
    'hyt':       '_medit.set',
}

def find_eeg_files(data_dir):
    """
    Find meditation-condition EEG .set files by group prefix.
    CTR uses breath2 (their MED-equivalent); others use medit.
    """
    files_by_group = {g: [] for g in GROUP_ORDER}

    all_sets = sorted(glob.glob(os.path.join(data_dir, '*.set')))

    for f in all_sets:
        fname = os.path.basename(f)
        for g in GROUP_ORDER:
            prefix = PREFIX_MAP[g]
            cond_suffix = CONDITION_MAP[g]
            # Match: starts with prefix AND ends with correct condition
            if fname.startswith(prefix) and fname.endswith(cond_suffix):
                files_by_group[g].append(f)
                break

    # Report
    for g in GROUP_ORDER:
        print(f"  {LABEL_MAP[g]} ({g}, prefix={PREFIX_MAP[g]}, cond={CONDITION_MAP[g]}): "
              f"{len(files_by_group[g])} files")

    return files_by_group


# --- MAIN ---
def main():
    print("=" * 70)
    print("WINDOWED Z3 TIME SERIES ANALYSIS")
    print("=" * 70)

    # Check data directory
    if not os.path.isdir(EEG_DATA_DIR):
        print(f"\nERROR: Data directory not found: {EEG_DATA_DIR}")
        print("Download and extract raw EEG first:")
        print("  python download_data.py meditation")
        return

    # Load reference channel angles as name->angle dictionary
    try:
        angle_dict, ch_labels = load_channel_angles()
        print(f"\nLoaded reference angles for {len(angle_dict)} channels")
    except Exception:
        angle_dict = None
        print("\nWARNING: Could not load chanlocs.mat, will use fallback angles")

    # Find files
    print("\nSearching for EEG files...")
    files_by_group = find_eeg_files(EEG_DATA_DIR)

    total = sum(len(v) for v in files_by_group.values())
    if total == 0:
        # List directory contents for debugging
        print(f"\nNo files found. Directory contents of {EEG_DATA_DIR}:")
        for item in os.listdir(EEG_DATA_DIR):
            full = os.path.join(EEG_DATA_DIR, item)
            if os.path.isdir(full):
                n_files = len(os.listdir(full))
                print(f"  [DIR] {item}/ ({n_files} items)")
            else:
                sz = os.path.getsize(full)
                print(f"  {item} ({sz} bytes)")
        return

    # Process all subjects
    all_results = []
    group_results = {g: [] for g in GROUP_ORDER}

    for g in GROUP_ORDER:
        label = LABEL_MAP[g]
        files = files_by_group[g]
        if not files:
            continue
        print(f"\n--- {label} ({g}, {len(files)} subjects) ---")
        for i, filepath in enumerate(files):
            subj_id = f"{label}_{i+1:02d}"
            print(f"  {subj_id}: {os.path.basename(filepath)}")
            result = process_subject_z3(filepath, label, angle_dict)
            if result is not None:
                result['subject_id'] = subj_id
                all_results.append(result)
                group_results[g].append(result)

    if not all_results:
        print("\nNo subjects successfully processed.")
        return

    # --- GROUP STATISTICS ---
    print("\n" + "=" * 70)
    print("Z3 TIME SERIES GROUP RESULTS")
    print("=" * 70)

    print(f"\n{'Group':<8} {'N subj':<8} {'Median Z3':<12} {'Frac >1.0':<12} "
          f"{'Frac >0.8':<12} {'Std Z3':<10} {'H2 med':<10} {'H3 med':<10}")
    print("-" * 85)
    for g in GROUP_ORDER:
        rs = group_results[g]
        if not rs:
            continue
        label = LABEL_MAP[g]
        medians = [r['median_z3'] for r in rs]
        frac1 = [r['frac_above_1'] for r in rs]
        frac08 = [r['frac_above_08'] for r in rs]
        stds = [r['z3_std'] for r in rs]
        h2s = [r['h2_median'] for r in rs]
        h3s = [r['h3_median'] for r in rs]
        print(f"{label:<8} {len(rs):<8} {np.median(medians):<12.4f} "
              f"{np.median(frac1):<12.1%} {np.median(frac08):<12.1%} "
              f"{np.median(stds):<10.4f} {np.median(h2s):<10.5f} {np.median(h3s):<10.5f}")

    # --- STATISTICAL TESTS ---
    print("\n--- Statistical Tests ---")

    def get_fracs(g):
        return [r['frac_above_1'] for r in group_results[g]]

    tests = []
    for g1, g2, alt, desc in [
        ('hyt', 'control', 'greater', 'HYT > CTR'),
        ('shoonya', 'control', 'greater', 'ISY > CTR'),
        ('vipassana', 'control', 'greater', 'VIP > CTR'),
        ('hyt', 'vipassana', 'greater', 'HYT > VIP'),
        ('shoonya', 'vipassana', 'greater', 'ISY > VIP'),
    ]:
        f1, f2 = get_fracs(g1), get_fracs(g2)
        if f1 and f2:
            stat, p = mannwhitneyu(f1, f2, alternative=alt)
            tests.append((desc, stat, p))
            print(f"  {desc} (frac above 1.0): U={stat:.1f}, p={p:.4f}")
        else:
            print(f"  {desc}: insufficient data")

    # Kruskal-Wallis across all groups
    all_fracs = [get_fracs(g) for g in GROUP_ORDER if get_fracs(g)]
    if len(all_fracs) >= 3:
        kw_stat, kw_p = kruskal(*all_fracs)
        print(f"\n  Kruskal-Wallis (all groups): H={kw_stat:.4f}, p={kw_p:.4f}")

    # Spearman: experience vs median Z3
    exp_hours = []
    z3_meds = []
    for g in GROUP_ORDER:
        if group_results[g]:
            exp_hours.append(GROUPS[g]['hours'])
            z3_meds.append(np.median([r['median_z3'] for r in group_results[g]]))
    if len(exp_hours) >= 3:
        rho, p_rho = spearmanr(exp_hours, z3_meds)
        print(f"  Spearman (experience vs median Z3): rho={rho:.4f}, p={p_rho:.4f}")

    # --- SESSION DRIFT ---
    print("\n--- Session Drift (early vs late Z3) ---")
    for g in GROUP_ORDER:
        rs = group_results[g]
        if not rs:
            continue
        drifts = [r['session_drift'] for r in rs if not np.isnan(r['session_drift'])]
        if drifts:
            print(f"  {LABEL_MAP[g]}: mean drift = {np.mean(drifts):+.4f}, "
                  f"median = {np.median(drifts):+.4f}")
            for r in rs:
                print(f"    {r.get('subject_id','?')}: early={r['early_z3']:.3f}, "
                      f"late={r['late_z3']:.3f}, drift={r['session_drift']:+.3f}")

    # --- PLOTS ---
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Plot 1: Fraction above 1.0 by group
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(GROUP_ORDER):
        rs = group_results[g]
        if not rs:
            continue
        fracs = [r['frac_above_1'] for r in rs]
        color = GROUPS[g]['color']
        label = LABEL_MAP[g]
        # Individual dots
        jitter = np.random.default_rng(42).normal(0, 0.05, len(fracs))
        ax.scatter(np.full(len(fracs), i) + jitter, fracs,
                   color=color, alpha=0.6, s=40, zorder=3, edgecolors='black', linewidths=0.3)
        # Median bar
        med = np.median(fracs)
        ax.bar(i, med, color=color, alpha=0.3, width=0.5)
        ax.hlines(med, i - 0.25, i + 0.25, colors=color, linewidth=2, zorder=4)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'{LABEL_MAP[g]}\n({GROUPS[g]["hours"]}h)' for g in GROUP_ORDER])
    ax.set_ylabel('Fraction of session with Z3 > 1.0')
    ax.set_title('Time in Z3-dominant state by group\n(prediction: HYT > others; VIP ~ CTR)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'z3_frac_above1_by_group.png'), dpi=150)
    print("Saved: z3_frac_above1_by_group.png")

    # Plot 2: Representative HYT subject Z3 time series
    hyt_results = group_results['hyt']
    if hyt_results:
        best_hyt = max(hyt_results, key=lambda r: r['median_z3'])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(best_hyt['t_centres_min'], best_hyt['z3_series'],
                color='#AA3333', lw=1.2)
        ax.axhline(1.0, color='black', linestyle='--', lw=1, label='Z3 = Z2 parity')
        ax.axhline(0.694, color='grey', linestyle=':', lw=1, label='Phase 1 HYT static (0.694)')
        z3_arr = np.array(best_hyt['z3_series'])
        t_arr = np.array(best_hyt['t_centres_min'])
        ax.fill_between(t_arr, z3_arr, 1.0,
                         where=z3_arr > 1.0, alpha=0.3, color='#AA3333', label='Z3 > 1.0')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Z3 score (H3/H2)')
        ax.set_title(f'Z3 time series - best HYT subject ({best_hyt.get("subject_id","")})')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'z3_timeseries_hyt_example.png'), dpi=150)
        print("Saved: z3_timeseries_hyt_example.png")

    # Plot 3: H2 and H3 trajectories (normalized session time)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # H2 trajectory per group
    ax = axes[0]
    for g in GROUP_ORDER:
        rs = group_results[g]
        if not rs:
            continue
        color = GROUPS[g]['color']
        label = LABEL_MAP[g]
        # Normalize each subject to [0, 1] session time, then average
        n_bins = 20
        h2_binned = np.full((len(rs), n_bins), np.nan)
        for j, r in enumerate(rs):
            h2 = np.array(r['h2_series'])
            if len(h2) >= n_bins:
                # Resample to n_bins
                indices = np.linspace(0, len(h2) - 1, n_bins).astype(int)
                h2_binned[j] = h2[indices]
            elif len(h2) > 0:
                indices = np.linspace(0, len(h2) - 1, n_bins).astype(int)
                indices = np.clip(indices, 0, len(h2) - 1)
                h2_binned[j] = h2[indices]
        mean_h2 = np.nanmean(h2_binned, axis=0)
        sem_h2 = np.nanstd(h2_binned, axis=0) / np.sqrt(np.sum(~np.isnan(h2_binned), axis=0))
        t_norm = np.linspace(0, 1, n_bins)
        ax.plot(t_norm, mean_h2, color=color, lw=2, label=label)
        ax.fill_between(t_norm, mean_h2 - sem_h2, mean_h2 + sem_h2, color=color, alpha=0.2)
    ax.set_xlabel('Normalized session time')
    ax.set_ylabel('H2 (bilateral amplitude)')
    ax.set_title('H2 trajectory over session\n(does bilateral structure collapse?)')
    ax.legend(fontsize=8)

    # H3 trajectory per group
    ax = axes[1]
    for g in GROUP_ORDER:
        rs = group_results[g]
        if not rs:
            continue
        color = GROUPS[g]['color']
        label = LABEL_MAP[g]
        n_bins = 20
        h3_binned = np.full((len(rs), n_bins), np.nan)
        for j, r in enumerate(rs):
            h3 = np.array(r['h3_series'])
            if len(h3) > 0:
                indices = np.linspace(0, len(h3) - 1, n_bins).astype(int)
                indices = np.clip(indices, 0, len(h3) - 1)
                h3_binned[j] = h3[indices]
        mean_h3 = np.nanmean(h3_binned, axis=0)
        sem_h3 = np.nanstd(h3_binned, axis=0) / np.sqrt(np.sum(~np.isnan(h3_binned), axis=0))
        t_norm = np.linspace(0, 1, n_bins)
        ax.plot(t_norm, mean_h3, color=color, lw=2, label=label)
        ax.fill_between(t_norm, mean_h3 - sem_h3, mean_h3 + sem_h3, color=color, alpha=0.2)
    ax.set_xlabel('Normalized session time')
    ax.set_ylabel('H3 (Z3 amplitude)')
    ax.set_title('H3 trajectory over session\n(does Z3 structure emerge?)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'h2_h3_trajectories.png'), dpi=150)
    print("Saved: h2_h3_trajectories.png")

    # --- SAVE JSON RESULTS ---
    # Convert numpy types for JSON
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_results = []
    for r in all_results:
        jr = {k: make_serializable(v) for k, v in r.items()}
        json_results.append(jr)

    json_path = os.path.join(REPORTS_DIR, 'z3_windowed_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=make_serializable)
    print(f"Saved: {json_path}")

    # --- DETERMINE P_MED_4 DYNAMIC STATUS ---
    hyt_frac = get_fracs('hyt')
    ctr_frac = get_fracs('control')
    vip_frac = get_fracs('vipassana')

    hyt_gt_ctr = False
    vip_not_gt_ctr = False

    if hyt_frac and ctr_frac:
        _, p_hc = mannwhitneyu(hyt_frac, ctr_frac, alternative='greater')
        hyt_gt_ctr = p_hc < 0.05
    if vip_frac and ctr_frac:
        _, p_vc = mannwhitneyu(vip_frac, ctr_frac, alternative='greater')
        vip_not_gt_ctr = p_vc > 0.05

    if hyt_gt_ctr and vip_not_gt_ctr:
        p_med4_dyn = "CONFIRMED"
    elif hyt_gt_ctr:
        p_med4_dyn = "PARTIAL (HYT > CTR, but VIP also elevated)"
    elif np.median(hyt_frac) > np.median(ctr_frac) if hyt_frac and ctr_frac else False:
        p_med4_dyn = "PARTIAL (trend but not significant)"
    else:
        p_med4_dyn = "NOT CONFIRMED"

    # --- FINAL REGISTRY BLOCK ---
    hyt_drifts = [r['session_drift'] for r in group_results['hyt']
                  if not np.isnan(r['session_drift'])]

    registry = f"""
=== WINDOWED Z3 RESULTS (Braboszcz dataset) ===

Group | N | Median Z3 | Frac >1.0 | Frac >0.8 | Std Z3"""

    for g in GROUP_ORDER:
        rs = group_results[g]
        if rs:
            label = LABEL_MAP[g]
            n = len(rs)
            med = np.median([r['median_z3'] for r in rs])
            f1 = np.median([r['frac_above_1'] for r in rs])
            f08 = np.median([r['frac_above_08'] for r in rs])
            std = np.median([r['z3_std'] for r in rs])
            registry += f"\n{label:<6}| {n} | {med:.4f}    | {f1:.1%}     | {f08:.1%}     | {std:.4f}"

    registry += "\n\nMann-Whitney p values (frac above 1.0):"
    for desc, stat, p in tests:
        marker = " <-- key test" if "HYT > VIP" in desc else ""
        registry += f"\n  {desc}: p = {p:.4f}{marker}"

    drift_str = f"{np.mean(hyt_drifts):+.4f}" if hyt_drifts else "N/A"
    registry += f"""

HYT session drift (early vs late Z3): {drift_str}
  (positive = Z3 rises over session = deepening state)

P_MED_4 DYNAMIC: {p_med4_dyn}
  Criteria: HYT frac>1.0 significantly > CTR (p<0.05)
            AND VIP frac>1.0 NOT significantly > CTR (p>0.05)

=== END ==="""

    print(registry)

    # Save registry
    summary_path = os.path.join(REPORTS_DIR, 'z3_windowed_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(registry)
    print(f"\nSaved: {summary_path}")
    print("\nWindowed Z3 analysis complete.")


if __name__ == '__main__':
    main()
