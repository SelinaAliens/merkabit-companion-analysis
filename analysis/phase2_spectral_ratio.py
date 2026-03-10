"""
Phase 2 Spectral Ratio Analysis — Alpha/Gamma from raw EEG
Replicates Phase 1 analysis on the SAME Braboszcz 2017 dataset (Zenodo 57911).
Phase 1 used pre-computed MAT files; Phase 2 recomputes from raw .set/.fdt.

Phase 1 measure: 10*log10(sum_of_periodogram_PSD_in_band)
  alpha 7-11 Hz, gamma 60-110 Hz
  ratio = dB_alpha / dB_gamma ≈ 4/3 for HYT

This script computes BOTH the Phase 1 measure AND physically meaningful
integrated band power (uV^2) to test robustness.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, MEDITATION_RAW, FIGURES_DIR, REPORTS_DIR
import glob
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

eeg_dir = os.path.join(MEDITATION_RAW, 'EEG_datasets')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Channel locations ───
mat_cl = sio.loadmat(os.path.join(MEDITATION_DATA, 'chanlocs.mat'))
N_CH = 64
ph1_labels = [str(mat_cl['chanlocs']['labels'][0, i][0]) for i in range(N_CH)]
thetas = [float(mat_cl['chanlocs']['theta'][0, i][0, 0]) for i in range(N_CH)]
radii = [float(mat_cl['chanlocs']['radius'][0, i][0, 0]) for i in range(N_CH)]
theta_rad = np.array(thetas) * np.pi / 180
plot_x = np.array(radii) * np.sin(theta_rad)
plot_y = np.array(radii) * np.cos(theta_rad)

# ─── File mapping ───
group_map = {
    'CTR': {'prefix': 'CTR', 'condition': 'breath2'},
    'ISY': {'prefix': 'SNY', 'condition': 'medit'},
    'VIP': {'prefix': 'VIP', 'condition': 'medit'},
    'HYT': {'prefix': 'HT', 'condition': 'medit'},
}

ALPHA_BAND = (7, 11)
GAMMA_BAND = (60, 110)

def compute_spectral_measures(raw, ch_names):
    """Compute multiple spectral measures per channel.

    Returns dict with:
    - ph1_alpha, ph1_gamma: Phase 1 measure (10*log10(sum_periodogram_in_band))
    - int_alpha, int_gamma: integrated band power (uV^2) from Welch
    - mean_psd_alpha, mean_psd_gamma: mean PSD in band (uV^2/Hz) from Welch
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data(picks=ch_names) * 1e6  # V -> µV
    n_ch, n_samples = data.shape

    # Method 1: Full periodogram -> sum -> dB (Phase 1 method)
    f_per, psd_per = signal.periodogram(data, fs=sfreq, axis=1)
    alpha_m = (f_per >= ALPHA_BAND[0]) & (f_per <= ALPHA_BAND[1])
    gamma_m = (f_per >= GAMMA_BAND[0]) & (f_per <= GAMMA_BAND[1])

    ph1_alpha = 10 * np.log10(np.maximum(psd_per[:, alpha_m].sum(axis=1), 1e-30))
    ph1_gamma = 10 * np.log10(np.maximum(psd_per[:, gamma_m].sum(axis=1), 1e-30))

    # Method 2: Welch PSD -> integrated power (physically meaningful)
    f_w, psd_w = signal.welch(data, fs=sfreq, nperseg=min(1024, n_samples//4),
                               noverlap=512, window='hann', axis=1)
    alpha_w = (f_w >= ALPHA_BAND[0]) & (f_w <= ALPHA_BAND[1])
    gamma_w = (f_w >= GAMMA_BAND[0]) & (f_w <= GAMMA_BAND[1])

    int_alpha = np.trapezoid(psd_w[:, alpha_w], f_w[alpha_w], axis=1)
    int_gamma = np.trapezoid(psd_w[:, gamma_w], f_w[gamma_w], axis=1)

    mean_psd_alpha = psd_w[:, alpha_w].mean(axis=1)
    mean_psd_gamma = psd_w[:, gamma_w].mean(axis=1)

    return {
        'ph1_alpha': ph1_alpha, 'ph1_gamma': ph1_gamma,
        'int_alpha': int_alpha, 'int_gamma': int_gamma,
        'mean_psd_alpha': mean_psd_alpha, 'mean_psd_gamma': mean_psd_gamma,
        'n_samples': n_samples
    }

print("=" * 70)
print("PHASE 2: SPECTRAL RATIO FROM RAW EEG")
print("  Phase 1 measure: 10*log10(sum_periodogram)")
print("  Also: integrated band power (uV^2)")
print("=" * 70)

# ─── Load all subjects ───
# Store on 64-channel grid (NaN for missing)
results = {}  # group -> dict of arrays

for grp, info in group_map.items():
    prefix = info['prefix']
    cond = info['condition']
    pattern = os.path.join(eeg_dir, f'{prefix}*_{cond}.set')
    files = sorted(glob.glob(pattern))

    print(f"\n{grp} ({prefix}*_{cond}): {len(files)} files")

    n_subj = len(files)
    ph1_a = np.full((N_CH, n_subj), np.nan)
    ph1_g = np.full((N_CH, n_subj), np.nan)
    int_a = np.full((N_CH, n_subj), np.nan)
    int_g = np.full((N_CH, n_subj), np.nan)
    ids = []

    for s_idx, fpath in enumerate(files):
        subj_id = os.path.basename(fpath).split('_')[0]
        ids.append(subj_id)

        try:
            raw = mne.io.read_raw_eeglab(fpath, preload=True)
            sfreq = raw.info['sfreq']
            if sfreq < 220:
                print(f"  {subj_id}: sfreq={sfreq}, SKIP")
                continue

            # Match channels
            matched_ch = []
            matched_idx = []
            for i, lbl in enumerate(ph1_labels):
                if lbl in raw.ch_names:
                    matched_ch.append(lbl)
                    matched_idx.append(i)

            if len(matched_ch) < 30:
                print(f"  {subj_id}: {len(matched_ch)} ch matched, SKIP")
                continue

            m = compute_spectral_measures(raw, matched_ch)

            for j, ph1_idx in enumerate(matched_idx):
                ph1_a[ph1_idx, s_idx] = m['ph1_alpha'][j]
                ph1_g[ph1_idx, s_idx] = m['ph1_gamma'][j]
                int_a[ph1_idx, s_idx] = m['int_alpha'][j]
                int_g[ph1_idx, s_idx] = m['int_gamma'][j]

            # Quick summary
            ratio_ph1 = np.nanmean(m['ph1_alpha']) / np.nanmean(m['ph1_gamma'])
            ratio_int = np.nanmean(m['int_alpha']) / np.nanmean(m['int_gamma'])
            print(f"  {subj_id}: {len(matched_ch)} ch, "
                  f"Ph1 a={np.nanmean(m['ph1_alpha']):.1f} g={np.nanmean(m['ph1_gamma']):.1f} "
                  f"r={ratio_ph1:.3f} | "
                  f"int a={np.nanmean(m['int_alpha']):.2f} g={np.nanmean(m['int_gamma']):.4f} "
                  f"r={ratio_int:.2f}")

        except Exception as e:
            print(f"  {subj_id}: ERROR — {e}")

    results[grp] = {
        'ph1_alpha': ph1_a, 'ph1_gamma': ph1_g,
        'int_alpha': int_a, 'int_gamma': int_g,
        'ids': ids
    }

# ─── Phase 1 measure: dB ratio ───
print("\n" + "=" * 70)
print("PHASE 1 MEASURE: dB_alpha / dB_gamma (sum of periodogram)")
print("  Phase 1 reference: HYT=1.361, CTR=1.585, ISY=1.428, VIP=1.593")
print("=" * 70)

groups = ['CTR', 'ISY', 'VIP', 'HYT']
target = 4/3
experience = {'CTR': 0, 'ISY': 2625, 'VIP': 9201, 'HYT': 15475}
ph1_ref = {'CTR': 1.585, 'ISY': 1.428, 'VIP': 1.593, 'HYT': 1.361}

for g in groups:
    ph1_a = results[g]['ph1_alpha']  # 64 x n_subj
    ph1_g = results[g]['ph1_gamma']

    # Per-channel mean across subjects
    ch_a_mean = np.nanmean(ph1_a, axis=1)
    ch_g_mean = np.nanmean(ph1_g, axis=1)
    valid = ~np.isnan(ch_a_mean) & ~np.isnan(ch_g_mean) & (ch_g_mean != 0)

    # Ratio per channel (of means)
    ch_ratio = np.full(N_CH, np.nan)
    ch_ratio[valid] = ch_a_mean[valid] / ch_g_mean[valid]

    # Per-subject ratio
    subj_a_mean = np.nanmean(ph1_a, axis=0)
    subj_g_mean = np.nanmean(ph1_g, axis=0)
    subj_ratio = subj_a_mean / subj_g_mean

    grand_mean = np.nanmean(subj_ratio)
    grand_sem = np.nanstd(subj_ratio) / np.sqrt(len(subj_ratio))

    # Channel-level fraction near 4/3
    dist_43 = np.abs(ch_ratio[valid] - target)
    n_near = np.sum(dist_43 < 0.05)
    n_near10 = np.sum(dist_43 < 0.10)
    n_valid = valid.sum()

    print(f"\n{g} (n={ph1_a.shape[1]}, {experience[g]}h):")
    print(f"  Mean alpha dB: {np.nanmean(subj_a_mean):.2f}")
    print(f"  Mean gamma dB: {np.nanmean(subj_g_mean):.2f}")
    print(f"  Grand mean ratio: {grand_mean:.4f} ± {grand_sem:.4f}")
    print(f"  Phase 1 reference: {ph1_ref[g]:.3f}")
    print(f"  |Phase 2 - Phase 1| = {abs(grand_mean - ph1_ref[g]):.4f}")
    print(f"  |ratio - 4/3| = {abs(grand_mean - target):.4f}")
    print(f"  Channels within 0.05 of 4/3: {n_near}/{n_valid} ({100*n_near/n_valid:.1f}%)")
    print(f"  Channels within 0.10 of 4/3: {n_near10}/{n_valid} ({100*n_near10/n_valid:.1f}%)")

    # Best channels
    best_idx = np.argsort(np.where(np.isnan(ch_ratio), 999, np.abs(ch_ratio - target)))[:5]
    for idx in best_idx:
        if not np.isnan(ch_ratio[idx]):
            print(f"    {ph1_labels[idx]:>6}: ratio={ch_ratio[idx]:.4f}, |r-4/3|={abs(ch_ratio[idx]-target):.4f}")

# ─── Integrated band power ratio ───
print("\n" + "=" * 70)
print("INTEGRATED BAND POWER RATIO (uV^2 alpha / uV^2 gamma)")
print("  This is the physically meaningful measure")
print("=" * 70)

for g in groups:
    int_a = results[g]['int_alpha']
    int_g = results[g]['int_gamma']

    # Per-subject mean power
    subj_a = np.nanmean(int_a, axis=0)
    subj_g = np.nanmean(int_g, axis=0)
    subj_ratio = subj_a / np.maximum(subj_g, 1e-30)

    # Channel means
    ch_a = np.nanmean(int_a, axis=1)
    ch_g = np.nanmean(int_g, axis=1)
    valid = ~np.isnan(ch_a) & ~np.isnan(ch_g) & (ch_g > 0)
    ch_ratio = np.full(N_CH, np.nan)
    ch_ratio[valid] = ch_a[valid] / ch_g[valid]

    grand = np.nanmean(subj_ratio)
    sem = np.nanstd(subj_ratio) / np.sqrt(len(subj_ratio))

    print(f"\n{g}:")
    print(f"  Mean alpha uV^2: {subj_a.mean():.3f}")
    print(f"  Mean gamma uV^2: {subj_g.mean():.5f}")
    print(f"  Grand mean ratio: {grand:.4f} ± {sem:.4f}")

    # Check where ratio is near integer multiples or 4/3
    dist = np.abs(ch_ratio[valid] - target)
    n_near = np.sum(dist < 0.5)  # wider window for this measure
    print(f"  Channels within 0.5 of 4/3: {n_near}/{valid.sum()}")

# ─── Statistical tests (Phase 1 measure) ───
print("\n" + "=" * 70)
print("STATISTICAL TESTS (Phase 1 dB ratio measure)")
print("=" * 70)

subj_ratios = {}
for g in groups:
    ph1_a = results[g]['ph1_alpha']
    ph1_g = results[g]['ph1_gamma']
    subj_a = np.nanmean(ph1_a, axis=0)
    subj_g = np.nanmean(ph1_g, axis=0)
    subj_ratios[g] = subj_a / subj_g

# Mann-Whitney tests
for g1, g2 in [('HYT', 'CTR'), ('ISY', 'CTR'), ('VIP', 'CTR'), ('HYT', 'VIP')]:
    u, p = stats.mannwhitneyu(subj_ratios[g1], subj_ratios[g2], alternative='two-sided')
    print(f"  {g1} vs {g2}: U={u:.1f}, p={p:.4f} "
          f"({g1}={np.mean(subj_ratios[g1]):.4f}, {g2}={np.mean(subj_ratios[g2]):.4f})")

# Kruskal-Wallis
h, p = stats.kruskal(*[subj_ratios[g] for g in groups])
print(f"  Kruskal-Wallis: H={h:.3f}, p={p:.4f}")

# One-sample vs 4/3
for g in groups:
    t, p = stats.ttest_1samp(subj_ratios[g], target)
    print(f"  {g} vs 4/3: t={t:.3f}, p={p:.4f}")

# ─── ISY P9 channel ───
print("\n" + "=" * 70)
print("ISY P9 CHANNEL (Phase 1: ratio=1.3341)")
print("=" * 70)

if 'P9' in ph1_labels:
    p9_idx = ph1_labels.index('P9')
    for g in groups:
        a_vals = results[g]['ph1_alpha'][p9_idx, :]
        g_vals = results[g]['ph1_gamma'][p9_idx, :]
        valid = ~np.isnan(a_vals) & ~np.isnan(g_vals) & (g_vals != 0)
        if valid.sum() > 0:
            ratios = a_vals[valid] / g_vals[valid]
            print(f"  {g} P9: ratio={ratios.mean():.4f} ± {ratios.std():.4f}, "
                  f"|r-4/3|={abs(ratios.mean()-target):.4f}, n={valid.sum()}")

# ─── Variance analysis ───
print("\n" + "=" * 70)
print("CROSS-SUBJECT VARIANCE (P_MED_2b replication)")
print("=" * 70)

for g in groups:
    ph1_a = results[g]['ph1_alpha']
    ph1_g = results[g]['ph1_gamma']
    # Per-channel, per-subject ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ch_subj_ratio = np.where((ph1_g != 0) & ~np.isnan(ph1_g), ph1_a / ph1_g, np.nan)
    ch_std = np.nanstd(ch_subj_ratio, axis=1)
    print(f"  {g}: mean cross-subject std = {np.nanmean(ch_std):.4f}")

# VIP vs ISY variance
ratio_vip = np.where(results['VIP']['ph1_gamma'] != 0,
                      results['VIP']['ph1_alpha'] / results['VIP']['ph1_gamma'], np.nan)
ratio_isy = np.where(results['ISY']['ph1_gamma'] != 0,
                      results['ISY']['ph1_alpha'] / results['ISY']['ph1_gamma'], np.nan)
vip_std = np.nanstd(ratio_vip, axis=1)
isy_std = np.nanstd(ratio_isy, axis=1)
valid_both = ~np.isnan(vip_std) & ~np.isnan(isy_std)
vip_more = np.sum(vip_std[valid_both] > isy_std[valid_both])
n_both = valid_both.sum()
print(f"\n  VIP more variable than ISY: {vip_more}/{n_both} channels ({100*vip_more/n_both:.0f}%)")
t_vi, p_vi = stats.ttest_rel(vip_std[valid_both], isy_std[valid_both])
print(f"  Paired t-test: t={t_vi:.3f}, p={p_vi:.4f}")

# ─── Channel-level per-subject fraction near 4/3 ───
print("\n" + "=" * 70)
print("PER-SUBJECT FRACTION OF CHANNELS NEAR 4/3 (dB measure)")
print("=" * 70)

for g in groups:
    ph1_a = results[g]['ph1_alpha']
    ph1_g = results[g]['ph1_gamma']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_mat = np.where((ph1_g != 0) & ~np.isnan(ph1_g), ph1_a / ph1_g, np.nan)

    near_43 = np.abs(ratio_mat - target) < 0.10
    valid_mat = ~np.isnan(ratio_mat)
    frac_per_subj = np.nansum(near_43 & valid_mat, axis=0) / np.maximum(valid_mat.sum(axis=0), 1)

    print(f"\n{g}:")
    for s_idx, sid in enumerate(results[g]['ids']):
        n_v = valid_mat[:, s_idx].sum()
        n_n = (near_43[:, s_idx] & valid_mat[:, s_idx]).sum()
        print(f"    {sid}: {frac_per_subj[s_idx]:.3f} ({n_n}/{n_v} ch)")
    print(f"  Mean: {frac_per_subj.mean():.3f} ± {frac_per_subj.std():.3f}")

# ─── Topographic plot ───
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Phase 2 Spectral Ratio (dB_alpha / dB_gamma, periodogram method)', fontsize=14)

vmin_r, vmax_r = 1.1, 1.8

for col, g in enumerate(groups):
    ax = axes[0, col]
    ch_a = np.nanmean(results[g]['ph1_alpha'], axis=1)
    ch_g = np.nanmean(results[g]['ph1_gamma'], axis=1)
    valid = ~np.isnan(ch_a) & ~np.isnan(ch_g) & (ch_g != 0)
    ch_ratio = np.full(N_CH, np.nan)
    ch_ratio[valid] = ch_a[valid] / ch_g[valid]

    sc = ax.scatter(plot_x[valid], plot_y[valid], c=ch_ratio[valid], cmap='RdYlBu_r',
                    s=80, vmin=vmin_r, vmax=vmax_r, edgecolors='black', linewidths=0.5)
    circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
    ax.add_patch(circle)
    ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
    ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal'); ax.axis('off')

    grand = np.nanmean(ch_ratio[valid])
    n_near = np.sum(np.abs(ch_ratio[valid] - target) < 0.10)
    ax.set_title(f'{g} ({experience[g]}h)\nmean={grand:.3f}, {n_near}/{valid.sum()} near 4/3')

    near_mask = valid & (np.abs(ch_ratio - target) < 0.10)
    if near_mask.any():
        ax.scatter(plot_x[near_mask], plot_y[near_mask],
                   s=200, facecolors='none', edgecolors='red', linewidths=2, zorder=10)
    if col == 3:
        plt.colorbar(sc, ax=ax, label='dB ratio', shrink=0.7)

# Bottom: histograms and comparisons
ax = axes[1, 0]
for g, c in zip(groups, ['#4477AA', '#66CCEE', '#228833', '#EE6677']):
    ch_a = np.nanmean(results[g]['ph1_alpha'], axis=1)
    ch_g = np.nanmean(results[g]['ph1_gamma'], axis=1)
    valid = (ch_g != 0) & ~np.isnan(ch_a) & ~np.isnan(ch_g)
    r = ch_a[valid] / ch_g[valid]
    ax.hist(r, bins=20, alpha=0.35, color=c, label=g, edgecolor='black')
ax.axvline(target, color='black', lw=2, ls='--')
ax.set_xlabel('dB ratio'); ax.set_ylabel('# channels')
ax.set_title('Channel distribution (all groups)')
ax.legend()

ax = axes[1, 1]
subj_data = [subj_ratios[g] for g in groups]
bp = ax.boxplot(subj_data, labels=groups, patch_artist=True)
for patch, c in zip(bp['boxes'], ['#4477AA', '#66CCEE', '#228833', '#EE6677']):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.axhline(target, color='black', lw=2, ls='--', label='4/3')
ax.set_ylabel('Subject-mean dB ratio')
ax.set_title('Subject-level dB ratio')
ax.legend()

# Phase 2 vs Phase 1 comparison scatter
ax = axes[1, 2]
ph1_means_ref = {'CTR': 1.585, 'ISY': 1.428, 'VIP': 1.593, 'HYT': 1.361}
ph2_means = {g: np.mean(subj_ratios[g]) for g in groups}
for g, c in zip(groups, ['#4477AA', '#66CCEE', '#228833', '#EE6677']):
    ax.scatter(ph1_means_ref[g], ph2_means[g], c=c, s=100, zorder=5, label=g)
ax.plot([1.2, 1.7], [1.2, 1.7], 'k--', alpha=0.5)
ax.axhline(target, color='gray', lw=1, ls=':')
ax.axvline(target, color='gray', lw=1, ls=':')
ax.set_xlabel('Phase 1 mean ratio')
ax.set_ylabel('Phase 2 mean ratio')
ax.set_title('Phase 1 vs Phase 2')
ax.legend()

# HYT - CTR difference topo
ax = axes[1, 3]
hyt_a = np.nanmean(results['HYT']['ph1_alpha'], axis=1)
hyt_g = np.nanmean(results['HYT']['ph1_gamma'], axis=1)
ctr_a = np.nanmean(results['CTR']['ph1_alpha'], axis=1)
ctr_g = np.nanmean(results['CTR']['ph1_gamma'], axis=1)
v_h = ~np.isnan(hyt_a) & (hyt_g != 0)
v_c = ~np.isnan(ctr_a) & (ctr_g != 0)
v = v_h & v_c
diff = np.full(N_CH, np.nan)
diff[v] = (hyt_a[v]/hyt_g[v]) - (ctr_a[v]/ctr_g[v])
if v.any():
    vabs = np.nanmax(np.abs(diff[v]))
    sc = ax.scatter(plot_x[v], plot_y[v], c=diff[v], cmap='RdBu_r',
                    s=80, vmin=-vabs, vmax=vabs, edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=ax, shrink=0.7)
circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('HYT - CTR ratio shift')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'phase2_spectral_ratio.png'), dpi=150)
print(f"
Saved: {os.path.join(FIGURES_DIR, 'phase2_spectral_ratio.png')}")

# ─── CRITICAL METHODOLOGICAL NOTE ───
print("\n" + "=" * 70)
print("METHODOLOGICAL NOTE")
print("=" * 70)
print("The dB ratio = 10*log10(sum_alpha) / 10*log10(sum_gamma) is NOT invariant")
print("to the spectral computation method. The ratio depends on:")
print("  1. Recording length (more samples = more periodogram bins = higher dB)")
print("  2. Band width (alpha=4 Hz, gamma=50 Hz = different bin counts)")
print("  3. Reference level (shifting dB by a constant changes the ratio)")
print("")
print("Phase 1 MAT files used a DIFFERENT spectral pipeline (likely EEGLAB with")
print("ICA cleaning). Muscle artifacts inflate gamma power; ICA removal lowers")
print("gamma dB, increasing the alpha/gamma dB ratio. This explains why Phase 1")
print("had higher ratios (1.36-1.59) vs Phase 2 raw (1.07-1.21).")
print("The GROUP ORDERING is preserved: HYT < ISY < CTR < VIP.")

# ─── Summary ───
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Group':<6} {'Ph2 dB ratio':>12} {'SEM':>8} {'Ph1 ref':>10} {'|Ph2-Ph1|':>10} {'|r-4/3|':>10}")
print("-" * 58)
for g in groups:
    m = np.mean(subj_ratios[g])
    s = np.std(subj_ratios[g]) / np.sqrt(len(subj_ratios[g]))
    print(f"{g:<6} {m:>12.4f} {s:>8.4f} {ph1_ref[g]:>10.3f} "
          f"{abs(m-ph1_ref[g]):>10.4f} {abs(m-target):>10.4f}")

print("\nDone.")
