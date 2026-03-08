"""
Delorme Mind-Wandering Phase 2 — Within-Subject KWW Analysis
=============================================================
Uses the Rishikesh dataset MW vs MED conditions to test whether
alpha_KWW differs WITHIN subjects between mind-wandering and meditation.

This is a WITHIN-SUBJECT design: each subject has both MW and MED recordings.
The key test: does alpha_KWW shift toward 4/3 during MED vs MW?

Requires: Raw EEG dataset (~6.3 GB) from Zenodo 2348892
Expected data: Both MW and MED condition files for each subject.

Complementary to rishikesh_meditation_phase2.py which focuses on
BETWEEN-GROUP differences. This script focuses on WITHIN-SUBJECT
state changes.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, MEDITATION_RAW, FIGURES_DIR, REPORTS_DIR

import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.stats import wilcoxon, spearmanr, ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───
RAW_EEG_DIR = MEDITATION_RAW

GROUPS = {
    'CTR': {'experience_hours': 0, 'color': '#4477AA', 'label': 'Control'},
    'ISY': {'experience_hours': 2625, 'color': '#66CCEE', 'label': 'Shoonya'},
    'VIP': {'experience_hours': 9201, 'color': '#228833', 'label': 'Vipassana'},
    'HYT': {'experience_hours': 15475, 'color': '#EE6677', 'label': 'HYT'},
}
GROUP_ORDER = ['CTR', 'ISY', 'VIP', 'HYT']

FS = 256
WINDOW_SEC = 30
WINDOW_SAMPLES = FS * WINDOW_SEC
OVERLAP_FRAC = 0.5
ACF_MAXLAG_SEC = 5.0
ACF_MAXLAG = int(ACF_MAXLAG_SEC * FS)
ALPHA_43_WINDOW = 0.15


def kww(t, A, tau, alpha):
    return A * np.exp(-(t / tau) ** alpha)


def fit_kww(acf, dt, maxlag=None):
    if maxlag is None:
        maxlag = len(acf)
    acf = acf[:maxlag]
    t = np.arange(len(acf)) * dt
    if acf[0] <= 0:
        return None
    acf_norm = acf / acf[0]
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
    n = len(signal)
    signal = signal - signal.mean()
    var = np.var(signal)
    if var == 0:
        return np.zeros(maxlag)
    acf = np.correlate(signal, signal, mode='full')
    acf = acf[n-1:n-1+maxlag] / (var * n)
    return acf


def load_raw_eeg(filepath):
    """Load EEG file. Adapt based on actual format."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.mat':
        mat = sio.loadmat(filepath)
        for key in ['EEG', 'data', 'eeg', 'X']:
            if key in mat:
                data = mat[key]
                if data.ndim == 2:
                    return data, FS
        if 'EEG' in mat and hasattr(mat['EEG'], 'dtype'):
            if 'data' in mat['EEG'].dtype.names:
                return mat['EEG']['data'][0, 0], FS
    elif ext == '.set':
        try:
            import mne
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
            return raw.get_data(), raw.info['sfreq']
        except ImportError:
            print("MNE required: pip install mne")
    elif ext in ['.edf', '.bdf']:
        try:
            import mne
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            return raw.get_data(), raw.info['sfreq']
        except ImportError:
            print("MNE required: pip install mne")
    return None, None


def analyze_condition(data, fs):
    """Extract KWW alpha values from all windows in a recording."""
    dt = 1.0 / fs
    n_channels, n_samples = data.shape
    step = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))
    n_windows = (n_samples - WINDOW_SAMPLES) // step + 1

    alphas = []
    taus = []
    R2s = []
    envelopes = []

    for w in range(n_windows):
        start = w * step
        end = start + WINDOW_SAMPLES
        segment = data[:, start:end]
        broadband = segment.mean(axis=0)

        acf = compute_acf(broadband, ACF_MAXLAG)
        fit = fit_kww(acf, dt, ACF_MAXLAG)
        if fit is not None:
            A, tau, alpha, R2 = fit
            if R2 > 0.3:
                alphas.append(alpha)
                taus.append(tau)
                R2s.append(R2)

        # Envelope
        analytic = hilbert(broadband)
        env = np.abs(analytic)
        if env.max() > 0:
            envelopes.append(np.mean(env / env.max()))

    return {
        'alphas': np.array(alphas),
        'taus': np.array(taus),
        'R2s': np.array(R2s),
        'envelopes': np.array(envelopes),
        'mean_alpha': np.mean(alphas) if alphas else np.nan,
        'std_alpha': np.std(alphas) if alphas else np.nan,
        'frac_near_43': np.mean(np.abs(np.array(alphas) - 4/3) < ALPHA_43_WINDOW) if alphas else 0,
        'plateau_C': np.mean(envelopes[-max(1, len(envelopes)//4):]) if envelopes else np.nan,
    }


def find_paired_files(base_dir):
    """
    Find MW and MED file pairs for each subject.
    Returns: dict {group: [{subject_id, mw_file, med_file}, ...]}
    Adapt patterns based on actual naming convention.
    """
    paired = {g: [] for g in GROUP_ORDER}

    # Common naming patterns
    for group in GROUP_ORDER:
        group_dir = os.path.join(base_dir, group)
        if not os.path.isdir(group_dir):
            # Try flat structure
            mw_files = sorted(glob.glob(os.path.join(base_dir, f'*{group}*MW*.*')))
            med_files = sorted(glob.glob(os.path.join(base_dir, f'*{group}*MED*.*')))
        else:
            mw_files = sorted(glob.glob(os.path.join(group_dir, '*MW*.*')))
            med_files = sorted(glob.glob(os.path.join(group_dir, '*MED*.*')))

        # Try to match by subject number
        for i, (mw, med) in enumerate(zip(mw_files, med_files)):
            paired[group].append({
                'subject_id': f'{group}_{i+1:02d}',
                'mw_file': mw,
                'med_file': med,
            })

    return paired


def main():
    print("="*70)
    print("DELORME WITHIN-SUBJECT: MW vs MED KWW Analysis")
    print("="*70)

    if not os.path.isdir(RAW_EEG_DIR):
        print(f"\nERROR: Raw EEG directory not found: {RAW_EEG_DIR}")
        print("Download from Zenodo 2348892 and extract here.")
        return

    # Find paired files
    print("\nSearching for MW/MED file pairs...")
    paired = find_paired_files(RAW_EEG_DIR)
    for g in GROUP_ORDER:
        print(f"  {g}: {len(paired[g])} paired subjects")

    total = sum(len(v) for v in paired.values())
    if total == 0:
        print("\nNo paired files found. Check directory structure.")
        return

    # Analyze all subjects
    all_results = []
    for group in GROUP_ORDER:
        print(f"\n--- {group} ---")
        for subj in paired[group]:
            sid = subj['subject_id']
            print(f"  {sid}:")

            # Load MW
            mw_data, mw_fs = load_raw_eeg(subj['mw_file'])
            if mw_data is None:
                print(f"    MW load failed, skipping")
                continue
            mw_result = analyze_condition(mw_data, mw_fs)
            print(f"    MW: alpha={mw_result['mean_alpha']:.3f}, "
                  f"std={mw_result['std_alpha']:.3f}, n={len(mw_result['alphas'])}")

            # Load MED
            med_data, med_fs = load_raw_eeg(subj['med_file'])
            if med_data is None:
                print(f"    MED load failed, skipping")
                continue
            med_result = analyze_condition(med_data, med_fs)
            print(f"    MED: alpha={med_result['mean_alpha']:.3f}, "
                  f"std={med_result['std_alpha']:.3f}, n={len(med_result['alphas'])}")

            all_results.append({
                'subject_id': sid,
                'group': group,
                'experience': GROUPS[group]['experience_hours'],
                'mw': mw_result,
                'med': med_result,
                'delta_alpha': med_result['mean_alpha'] - mw_result['mean_alpha'],
                'delta_std': med_result['std_alpha'] - mw_result['std_alpha'],
                'delta_frac': med_result['frac_near_43'] - mw_result['frac_near_43'],
            })

    if not all_results:
        print("\nNo subjects analyzed.")
        return

    # ─── RESULTS ───
    print("\n" + "="*70)
    print("WITHIN-SUBJECT RESULTS (MED - MW)")
    print("="*70)

    for group in GROUP_ORDER:
        gres = [r for r in all_results if r['group'] == group]
        if not gres:
            continue

        deltas = [r['delta_alpha'] for r in gres if not np.isnan(r['delta_alpha'])]
        delta_stds = [r['delta_std'] for r in gres if not np.isnan(r['delta_std'])]
        delta_fracs = [r['delta_frac'] for r in gres]

        print(f"\n{group} (n={len(gres)}):")
        if deltas:
            print(f"  delta(alpha) = {np.mean(deltas):.4f} +/- {np.std(deltas):.4f}")
            if len(deltas) >= 5:
                t, p = ttest_rel(
                    [r['med']['mean_alpha'] for r in gres if not np.isnan(r['med']['mean_alpha'])],
                    [r['mw']['mean_alpha'] for r in gres if not np.isnan(r['mw']['mean_alpha'])]
                )
                print(f"  paired t-test: t={t:.3f}, p={p:.4f}")
        if delta_stds:
            print(f"  delta(std) = {np.mean(delta_stds):.4f}")
        if delta_fracs:
            print(f"  delta(frac_near_43) = {np.mean(delta_fracs):.4f}")

    # Cross-group Wilcoxon on delta_alpha
    all_deltas = [r['delta_alpha'] for r in all_results if not np.isnan(r['delta_alpha'])]
    if len(all_deltas) >= 5:
        w, p = wilcoxon(all_deltas)
        print(f"\nAll subjects Wilcoxon (delta_alpha != 0): W={w:.1f}, p={p:.4f}")

    # Correlation: does delta_alpha scale with experience?
    exp_vals = [r['experience'] for r in all_results if not np.isnan(r['delta_alpha'])]
    delta_vals = [r['delta_alpha'] for r in all_results if not np.isnan(r['delta_alpha'])]
    if len(exp_vals) >= 5:
        rho, p_rho = spearmanr(exp_vals, delta_vals)
        print(f"Spearman (experience vs delta_alpha): rho={rho:.4f}, p={p_rho:.4f}")

    # ─── KEY TEST: Does MED push alpha toward 4/3? ───
    print("\n" + "="*70)
    print("KEY TEST: Direction of MED effect on alpha_KWW")
    print("="*70)

    for group in GROUP_ORDER:
        gres = [r for r in all_results if r['group'] == group]
        if not gres:
            continue

        # For each subject: is MED alpha closer to 4/3 than MW alpha?
        closer_to_43 = 0
        total = 0
        for r in gres:
            mw_a = r['mw']['mean_alpha']
            med_a = r['med']['mean_alpha']
            if np.isnan(mw_a) or np.isnan(med_a):
                continue
            total += 1
            if abs(med_a - 4/3) < abs(mw_a - 4/3):
                closer_to_43 += 1

        if total > 0:
            print(f"  {group}: {closer_to_43}/{total} subjects have MED closer to 4/3 "
                  f"({100*closer_to_43/total:.0f}%)")

    # ─── VARIANCE REDUCTION TEST ───
    print("\n--- Does MED reduce alpha_KWW variance? ---")
    for group in GROUP_ORDER:
        gres = [r for r in all_results if r['group'] == group]
        if not gres:
            continue

        mw_stds = [r['mw']['std_alpha'] for r in gres if not np.isnan(r['mw']['std_alpha'])]
        med_stds = [r['med']['std_alpha'] for r in gres if not np.isnan(r['med']['std_alpha'])]
        if mw_stds and med_stds:
            print(f"  {group}: MW std={np.mean(mw_stds):.4f}, MED std={np.mean(med_stds):.4f}, "
                  f"reduction={np.mean(mw_stds)-np.mean(med_stds):.4f}")

    # ─── PLOTS ───
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: MW vs MED alpha per subject (paired lines)
    ax = axes[0, 0]
    for r in all_results:
        if np.isnan(r['mw']['mean_alpha']) or np.isnan(r['med']['mean_alpha']):
            continue
        color = GROUPS[r['group']]['color']
        ax.plot([0, 1], [r['mw']['mean_alpha'], r['med']['mean_alpha']],
               color=color, alpha=0.4, lw=1)
        ax.scatter([0], [r['mw']['mean_alpha']], c=color, s=20, alpha=0.5)
        ax.scatter([1], [r['med']['mean_alpha']], c=color, s=20, alpha=0.5)
    ax.axhline(y=4/3, color='red', ls='--', lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MW', 'MED'])
    ax.set_ylabel('mean alpha_KWW')
    ax.set_title('Within-Subject MW -> MED')
    for g in GROUP_ORDER:
        ax.plot([], [], color=GROUPS[g]['color'], label=GROUPS[g]['label'])
    ax.legend(fontsize=7)

    # Plot 2: Delta alpha by group
    ax = axes[0, 1]
    for i, g in enumerate(GROUP_ORDER):
        deltas = [r['delta_alpha'] for r in all_results
                  if r['group'] == g and not np.isnan(r['delta_alpha'])]
        if deltas:
            bp = ax.boxplot([deltas], positions=[i], patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor(GROUPS[g]['color'])
            bp['boxes'][0].set_alpha(0.7)
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel('delta alpha (MED - MW)')
    ax.set_title('State Effect on alpha_KWW')

    # Plot 3: Delta alpha vs experience
    ax = axes[0, 2]
    for r in all_results:
        if np.isnan(r['delta_alpha']):
            continue
        ax.scatter(r['experience'], r['delta_alpha'],
                  c=GROUPS[r['group']]['color'], s=40, alpha=0.6)
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xlabel('Experience (hours)')
    ax.set_ylabel('delta alpha (MED - MW)')
    ax.set_title('State Effect vs Experience')

    # Plot 4: MW std vs MED std
    ax = axes[1, 0]
    for r in all_results:
        mw_s = r['mw']['std_alpha']
        med_s = r['med']['std_alpha']
        if np.isnan(mw_s) or np.isnan(med_s):
            continue
        ax.scatter(mw_s, med_s, c=GROUPS[r['group']]['color'], s=40, alpha=0.6)
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('MW std(alpha)')
    ax.set_ylabel('MED std(alpha)')
    ax.set_title('Variance Reduction: MW vs MED')

    # Plot 5: MW frac vs MED frac near 4/3
    ax = axes[1, 1]
    for r in all_results:
        ax.scatter(r['mw']['frac_near_43'], r['med']['frac_near_43'],
                  c=GROUPS[r['group']]['color'], s=40, alpha=0.6)
    lims = [0, max(0.5, ax.get_xlim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('MW frac near 4/3')
    ax.set_ylabel('MED frac near 4/3')
    ax.set_title('Fraction Near 4/3: MW vs MED')

    # Plot 6: Plateau |C| comparison
    ax = axes[1, 2]
    for r in all_results:
        mw_c = r['mw']['plateau_C']
        med_c = r['med']['plateau_C']
        if np.isnan(mw_c) or np.isnan(med_c):
            continue
        ax.scatter(mw_c, med_c, c=GROUPS[r['group']]['color'], s=40, alpha=0.6)
    ax.axhline(y=1/3, color='red', ls='--', alpha=0.5, label='1/3')
    ax.axvline(x=1/3, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('MW plateau |C|')
    ax.set_ylabel('MED plateau |C|')
    ax.set_title('|C| Plateau: MW vs MED')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'delorme_within_subject.png'), dpi=150)
    print(f"\nSaved: delorme_within_subject.png")

    # Save text results
    with open(os.path.join(REPORTS_DIR, 'delorme_within_subject_results.txt'), 'w') as f:
        f.write("DELORME WITHIN-SUBJECT ANALYSIS: MW vs MED\n")
        f.write("="*50 + "\n\n")
        f.write("Key question: Does meditation shift alpha_KWW toward 4/3?\n\n")

        for group in GROUP_ORDER:
            gres = [r for r in all_results if r['group'] == group]
            if not gres:
                continue
            f.write(f"\n{group} (n={len(gres)}):\n")
            deltas = [r['delta_alpha'] for r in gres if not np.isnan(r['delta_alpha'])]
            if deltas:
                f.write(f"  delta(alpha) = {np.mean(deltas):.4f} +/- {np.std(deltas):.4f}\n")

        f.write("\n\nInterpretation:\n")
        f.write("If MED consistently pushes alpha_KWW toward 4/3:\n")
        f.write("  -> Meditation is a STATE that approaches cooperative threshold\n")
        f.write("If only experienced groups show the shift:\n")
        f.write("  -> Cooperative threshold requires both STATE + TRAIT\n")
        f.write("If delta_alpha correlates with experience:\n")
        f.write("  -> Practice enables deeper approach to threshold\n")

    print("Saved: delorme_within_subject_results.txt")
    print("\nDone.")


if __name__ == '__main__':
    main()
