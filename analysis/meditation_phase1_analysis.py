"""
Meditation Phase 1 Analysis — Zenodo 57911 (Rishikesh study)
Groups: CTR (control), ISY (Isha Shoonya), VIP (Vipassana), HYT (Himalayan Yoga Tradition)
Experience hours: CTR=0, ISY=2625, VIP=9201, HYT=15475
"""
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, FIGURES_DIR, REPORTS_DIR

# ─── Load all data ───
df_alpha = pd.read_csv(os.path.join(MEDITATION_DATA, 'median_alpha_power.csv'))
df_gamma = pd.read_csv(os.path.join(MEDITATION_DATA, 'median_gamma_power.csv'))
df_ba = pd.read_csv(os.path.join(MEDITATION_DATA, 'boxplot_alpha.csv'))
df_bg = pd.read_csv(os.path.join(MEDITATION_DATA, 'boxplot_gamma.csv'))

mat_alpha = sio.loadmat(os.path.join(MEDITATION_DATA, '711Hz_spec_data_medit.mat'))
mat_gamma = sio.loadmat(os.path.join(MEDITATION_DATA, '60110Hz_spec_data_medit.mat'))
mat_chanlocs = sio.loadmat(os.path.join(MEDITATION_DATA, 'chanlocs.mat'))

# Experience hours per group
experience = {'CTR': 0, 'ISY': 2625, 'VIP': 9201, 'HYT': 15475}
group_order = ['CTR', 'ISY', 'VIP', 'HYT']
group_colors = {'CTR': '#4477AA', 'ISY': '#66CCEE', 'VIP': '#228833', 'HYT': '#EE6677'}

# ─── STEP 3: Alpha power analysis ───
print("="*60)
print("STEP 3: GROUP-LEVEL ALPHA POWER (7-11 Hz)")
print("="*60)

# Per-condition alpha
for cond in ['MW', 'MED']:
    print(f"\n--- Condition: {cond} ---")
    for g in group_order:
        vals = df_alpha[(df_alpha['Group']==g) & (df_alpha['Condition']==cond)]['Alpha']
        print(f"  {g} (n={len(vals)}): mean={vals.mean():.3f}, std={vals.std():.3f}, median={vals.median():.3f}")

# Meditation condition only
med_alpha = df_alpha[df_alpha['Condition']=='MED'].copy()
group_medians_alpha = {}
group_vals_alpha = {}
for g in group_order:
    vals = med_alpha[med_alpha['Group']==g]['Alpha'].values
    group_medians_alpha[g] = np.median(vals)
    group_vals_alpha[g] = vals

print("\n--- Meditation condition medians ---")
for g in group_order:
    print(f"  {g}: median={group_medians_alpha[g]:.3f}")

# Monotonic ordering test (Jonckheere-Terpstra via Spearman)
exp_hours = [experience[g] for g in group_order]
med_values = [group_medians_alpha[g] for g in group_order]
rho, p_mono = stats.spearmanr(exp_hours, med_values)
print(f"\nMonotonic ordering test (Spearman): rho={rho:.4f}, p={p_mono:.4f}")

# HYT/CTR ratio
ratio_hyt_ctr_alpha = group_medians_alpha['HYT'] / group_medians_alpha['CTR']
print(f"\nHYT/CTR alpha power ratio: {ratio_hyt_ctr_alpha:.6f}")
print(f"  |ratio - 4/3| = {abs(ratio_hyt_ctr_alpha - 4/3):.6f}")
print(f"  |ratio - 1| = {abs(ratio_hyt_ctr_alpha - 1):.6f}")

# Within-group variance comparison
print("\n--- Within-group variance (MED condition) ---")
for g in group_order:
    vals = group_vals_alpha[g]
    print(f"  {g}: std={np.std(vals):.3f}, CV={np.std(vals)/np.mean(vals):.4f}")

# Levene's test HYT vs CTR
stat_lev, p_lev = stats.levene(group_vals_alpha['HYT'], group_vals_alpha['CTR'])
print(f"\nLevene test HYT vs CTR variance: F={stat_lev:.4f}, p={p_lev:.4f}")

# ─── MAT file spectral data (64 channels x 16 subjects) ───
print("\n--- MAT: 7-11 Hz spectral data (channels x subjects) ---")
for key in ['CTR_medit_711', 'ISY_medit_711', 'VIP_medit_711', 'HYT_medit_711']:
    data = mat_alpha[key]  # 64 channels x 16 subjects
    # Mean across channels for each subject
    subj_means = data.mean(axis=0)
    # Grand mean and std
    print(f"  {key}: grand_mean={subj_means.mean():.3f}, std={subj_means.std():.3f}")

# ─── STEP 4: Gamma power analysis ───
print("\n" + "="*60)
print("STEP 4: GAMMA POWER (60-110 Hz)")
print("="*60)

for cond in ['MW', 'MED', 'BRE']:
    print(f"\n--- Condition: {cond} ---")
    for g in group_order:
        vals = df_gamma[(df_gamma['Group']==g) & (df_gamma['Condition']==cond)]['Gamma_All']
        print(f"  {g} (n={len(vals)}): mean={vals.mean():.3f}, std={vals.std():.3f}")

# Meditation condition gamma
med_gamma = df_gamma[df_gamma['Condition']=='MED'].copy()
group_medians_gamma = {}
group_vals_gamma = {}
for g in group_order:
    vals = med_gamma[med_gamma['Group']==g]['Gamma_All'].values
    group_medians_gamma[g] = np.median(vals)
    group_vals_gamma[g] = vals

print("\n--- Meditation condition gamma medians ---")
for g in group_order:
    print(f"  {g}: median={group_medians_gamma[g]:.3f}")

# Monotonic ordering
med_gamma_vals = [group_medians_gamma[g] for g in group_order]
rho_g, p_g = stats.spearmanr(exp_hours, med_gamma_vals)
print(f"\nGamma monotonic ordering (Spearman): rho={rho_g:.4f}, p={p_g:.4f}")

# HYT/CTR gamma ratio
ratio_hyt_ctr_gamma = group_medians_gamma['HYT'] / group_medians_gamma['CTR']
print(f"\nHYT/CTR gamma ratio: {ratio_hyt_ctr_gamma:.6f}")
print(f"  |ratio - 4/3| = {abs(ratio_hyt_ctr_gamma - 4/3):.6f}")

# Gamma variance (coherence test)
print("\n--- Gamma within-group variance (MED) ---")
for g in group_order:
    vals = group_vals_gamma[g]
    print(f"  {g}: std={np.std(vals):.3f}, CV={np.std(vals)/np.mean(vals):.4f}")

# MAT gamma spectral
print("\n--- MAT: 60-110 Hz spectral data ---")
for key in ['CTR_medit60110', 'ISY_medit60110', 'VIP_medit60110', 'HYT_medit60110']:
    data = mat_gamma[key]
    subj_means = data.mean(axis=0)
    print(f"  {key}: grand_mean={subj_means.mean():.3f}, std={subj_means.std():.3f}")

# ─── STEP 5: Alpha/Gamma ratio ───
print("\n" + "="*60)
print("STEP 5: ALPHA/GAMMA RATIO")
print("="*60)

# Per-subject ratio (MED condition)
# Match subjects between alpha and gamma
for g in group_order:
    a_vals = df_alpha[(df_alpha['Group']==g) & (df_alpha['Condition']=='MED')]['Alpha'].values
    g_vals = df_gamma[(df_gamma['Group']==g) & (df_gamma['Condition']=='MED')]['Gamma_All'].values
    ratios = a_vals / g_vals
    print(f"  {g}: alpha/gamma ratio mean={ratios.mean():.4f}, std={ratios.std():.4f}, median={np.median(ratios):.4f}")
    print(f"       |mean_ratio - 4/3| = {abs(ratios.mean() - 4/3):.4f}")
    print(f"       |mean_ratio - 1| = {abs(ratios.mean() - 1):.4f}")

# ─── ALL RATIO CHECKS ───
print("\n" + "="*60)
print("RATIO CHECKS vs MERKABIT CONSTANTS")
print("="*60)

# Check various ratios against 4/3 and 1/3
checks = []

# HYT/CTR alpha
r = group_medians_alpha['HYT'] / group_medians_alpha['CTR']
checks.append(('HYT/CTR alpha median', r, abs(r-4/3), abs(r-1/3)))

# VIP/CTR alpha
r = group_medians_alpha['VIP'] / group_medians_alpha['CTR']
checks.append(('VIP/CTR alpha median', r, abs(r-4/3), abs(r-1/3)))

# ISY/CTR alpha
r = group_medians_alpha['ISY'] / group_medians_alpha['CTR']
checks.append(('ISY/CTR alpha median', r, abs(r-4/3), abs(r-1/3)))

# HYT/CTR gamma
r = group_medians_gamma['HYT'] / group_medians_gamma['CTR']
checks.append(('HYT/CTR gamma median', r, abs(r-4/3), abs(r-1/3)))

# VIP/CTR gamma
r = group_medians_gamma['VIP'] / group_medians_gamma['CTR']
checks.append(('VIP/CTR gamma median', r, abs(r-4/3), abs(r-1/3)))

# Alpha/gamma ratio per group (MED)
for g in group_order:
    a = group_medians_alpha[g]
    gm = group_medians_gamma[g]
    r = a / gm
    checks.append((f'{g} alpha/gamma median', r, abs(r-4/3), abs(r-1/3)))

# Grand alpha/gamma
grand_alpha = df_alpha[df_alpha['Condition']=='MED']['Alpha'].median()
grand_gamma = df_gamma[df_gamma['Condition']=='MED']['Gamma_All'].median()
r = grand_alpha / grand_gamma
checks.append(('Grand alpha/gamma (MED)', r, abs(r-4/3), abs(r-1/3)))

# VIP/ISY alpha (experience-matched comparison)
r = group_medians_alpha['VIP'] / group_medians_alpha['ISY']
checks.append(('VIP/ISY alpha median', r, abs(r-4/3), abs(r-1/3)))

# HYT/ISY alpha
r = group_medians_alpha['HYT'] / group_medians_alpha['ISY']
checks.append(('HYT/ISY alpha median', r, abs(r-4/3), abs(r-1/3)))

# HYT gamma / CTR gamma (MAT: channel-level)
hyt_mat = mat_gamma['HYT_medit60110'].mean()  # grand mean
ctr_mat = mat_gamma['CTR_medit60110'].mean()
r = hyt_mat / ctr_mat
checks.append(('HYT/CTR gamma MAT', r, abs(r-4/3), abs(r-1/3)))

# Experience ratio checks
r = 15475 / 9201
checks.append(('HYT/VIP experience hours', r, abs(r-4/3), abs(r-1/3)))

print(f"{'Ratio':<35} {'Value':>8} {'|v-4/3|':>8} {'|v-1/3|':>8}")
print("-"*65)
for name, val, d43, d13 in sorted(checks, key=lambda x: min(x[2], x[3])):
    marker = ''
    if d43 < 0.05:
        marker = ' <-- NEAR 4/3'
    elif d13 < 0.05:
        marker = ' <-- NEAR 1/3'
    elif d43 < 0.15:
        marker = ' ~ 4/3'
    elif d13 < 0.15:
        marker = ' ~ 1/3'
    print(f"  {name:<35} {val:>8.4f} {d43:>8.4f} {d13:>8.4f}{marker}")

# ─── PLOTS ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Alpha power by group (MED condition)
ax = axes[0,0]
positions = range(len(group_order))
bp_data = [group_vals_alpha[g] for g in group_order]
bp = ax.boxplot(bp_data, positions=positions, patch_artist=True, widths=0.6)
for patch, g in zip(bp['boxes'], group_order):
    patch.set_facecolor(group_colors[g])
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels([f'{g}\n({experience[g]}h)' for g in group_order])
ax.set_ylabel('Median Alpha Power (7-11 Hz)')
ax.set_title('Alpha Power by Group (MED)')
ax.axhline(y=np.mean([group_medians_alpha[g] for g in group_order]), color='gray', ls='--', alpha=0.5)

# Plot 2: Gamma power by group (MED condition)
ax = axes[0,1]
bp_data_g = [group_vals_gamma[g] for g in group_order]
bp = ax.boxplot(bp_data_g, positions=positions, patch_artist=True, widths=0.6)
for patch, g in zip(bp['boxes'], group_order):
    patch.set_facecolor(group_colors[g])
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels([f'{g}\n({experience[g]}h)' for g in group_order])
ax.set_ylabel('Median Gamma Power (60-110 Hz)')
ax.set_title('Gamma Power by Group (MED)')

# Plot 3: Alpha/Gamma ratio by group
ax = axes[0,2]
ratio_data = []
for g in group_order:
    a_vals = df_alpha[(df_alpha['Group']==g) & (df_alpha['Condition']=='MED')]['Alpha'].values
    g_vals = df_gamma[(df_gamma['Group']==g) & (df_gamma['Condition']=='MED')]['Gamma_All'].values
    ratio_data.append(a_vals / g_vals)
bp = ax.boxplot(ratio_data, positions=positions, patch_artist=True, widths=0.6)
for patch, g in zip(bp['boxes'], group_order):
    patch.set_facecolor(group_colors[g])
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels([f'{g}\n({experience[g]}h)' for g in group_order])
ax.set_ylabel('Alpha/Gamma Ratio')
ax.set_title('Alpha/Gamma Ratio by Group (MED)')
ax.axhline(y=4/3, color='red', ls='--', alpha=0.7, label='4/3')
ax.legend(fontsize=8)

# Plot 4: Alpha power vs experience hours (scatter)
ax = axes[1,0]
for g in group_order:
    vals = group_vals_alpha[g]
    ax.scatter([experience[g]]*len(vals), vals, c=group_colors[g], alpha=0.5, s=30, label=g)
    ax.errorbar(experience[g], np.mean(vals), yerr=np.std(vals), fmt='D',
                color=group_colors[g], markersize=8, capsize=5, zorder=5)
ax.set_xlabel('Experience (hours)')
ax.set_ylabel('Alpha Power (MED)')
ax.set_title(f'Alpha vs Experience (rho={rho:.3f}, p={p_mono:.3f})')
ax.legend(fontsize=8)

# Plot 5: Gamma vs experience
ax = axes[1,1]
for g in group_order:
    vals = group_vals_gamma[g]
    ax.scatter([experience[g]]*len(vals), vals, c=group_colors[g], alpha=0.5, s=30, label=g)
    ax.errorbar(experience[g], np.mean(vals), yerr=np.std(vals), fmt='D',
                color=group_colors[g], markersize=8, capsize=5, zorder=5)
ax.set_xlabel('Experience (hours)')
ax.set_ylabel('Gamma Power (MED)')
ax.set_title(f'Gamma vs Experience (rho={rho_g:.3f}, p={p_g:.3f})')
ax.legend(fontsize=8)

# Plot 6: Channel-averaged spectra from MAT (alpha band)
ax = axes[1,2]
for key, g, color in [('CTR_medit_711','CTR','#4477AA'), ('ISY_medit_711','ISY','#66CCEE'),
                       ('VIP_medit_711','VIP','#228833'), ('HYT_medit_711','HYT','#EE6677')]:
    data = mat_alpha[key]  # 64 channels x 16 subjects
    # Mean across subjects for each channel, then show distribution
    chan_means = data.mean(axis=1)  # 64 channels
    ax.plot(sorted(chan_means), np.linspace(0, 1, len(chan_means)), color=color, label=g, lw=2)
ax.set_xlabel('Channel Alpha Power')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Channel Alpha CDF (MED)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'phase1_group_spectra.png'), dpi=150)
print(f"\nSaved: phase1_group_spectra.png")

# ─── MW vs MED difference ───
print("\n" + "="*60)
print("MW vs MED DIFFERENCE (meditation effect)")
print("="*60)
for g in group_order:
    mw = df_alpha[(df_alpha['Group']==g) & (df_alpha['Condition']=='MW')]['Alpha'].values
    med = df_alpha[(df_alpha['Group']==g) & (df_alpha['Condition']=='MED')]['Alpha'].values
    diff = med - mw  # paired difference
    t, p = stats.ttest_rel(med, mw)
    print(f"  {g}: MED-MW = {diff.mean():.3f} +/- {diff.std():.3f}, t={t:.3f}, p={p:.4f}")
    # Ratio of MED/MW
    r = np.mean(med) / np.mean(mw)
    print(f"       MED/MW ratio = {r:.4f}, |r-1| = {abs(r-1):.4f}")

# ─── Posterior vs Frontal gamma ───
print("\n" + "="*60)
print("POSTERIOR vs FRONTAL GAMMA (MED condition)")
print("="*60)
for g in group_order:
    post = df_gamma[(df_gamma['Group']==g) & (df_gamma['Condition']=='MED')]['Posterior'].values
    front = df_gamma[(df_gamma['Group']==g) & (df_gamma['Condition']=='MED')]['Frontal'].values
    ratio = post / front
    print(f"  {g}: post/front = {ratio.mean():.4f} +/- {ratio.std():.4f}")
    print(f"       |ratio - 1| = {abs(ratio.mean()-1):.4f}")

# ─── ANOVA ───
print("\n" + "="*60)
print("ONE-WAY ANOVA (MED condition)")
print("="*60)
# Alpha
f_alpha, p_alpha = stats.f_oneway(*[group_vals_alpha[g] for g in group_order])
print(f"Alpha: F={f_alpha:.4f}, p={p_alpha:.4f}")

# Gamma
f_gamma, p_gamma = stats.f_oneway(*[group_vals_gamma[g] for g in group_order])
print(f"Gamma: F={f_gamma:.4f}, p={p_gamma:.4f}")

# Post-hoc: VIP vs others (VIP has highest alpha)
for g in ['CTR', 'ISY', 'HYT']:
    t, p = stats.ttest_ind(group_vals_alpha['VIP'], group_vals_alpha[g])
    print(f"  Alpha VIP vs {g}: t={t:.3f}, p={p:.4f}")

print("\nDone.")
