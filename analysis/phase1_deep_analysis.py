"""
Phase 1 Deep Dive — ISY Distribution, Z3 Spatial Symmetry, Spatial Uniformity
==============================================================================
Three analyses from pre-computed spectral data (no raw EEG needed):
  1. ISY individual distribution: bimodal vs unimodal, P9/P10 context
  2. Z3 spatial symmetry score across groups
  3. HYT at-4/3 channel spatial uniformity (Moran's I, sector chi-squared)
"""
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import diptest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, FIGURES_DIR, REPORTS_DIR

# ─── LOAD DATA ───
mat_alpha = sio.loadmat(os.path.join(MEDITATION_DATA, '711Hz_spec_data_medit.mat'))
mat_gamma = sio.loadmat(os.path.join(MEDITATION_DATA, '60110Hz_spec_data_medit.mat'))
mat_cl = sio.loadmat(os.path.join(MEDITATION_DATA, 'chanlocs.mat'))

N_CH = 64
N_SUBJ = 16
labels = [str(mat_cl['chanlocs']['labels'][0, i][0]) for i in range(N_CH)]
thetas_deg = np.array([float(mat_cl['chanlocs']['theta'][0, i][0, 0]) for i in range(N_CH)])
radii = np.array([float(mat_cl['chanlocs']['radius'][0, i][0, 0]) for i in range(N_CH)])
xs = np.array([float(mat_cl['chanlocs']['X'][0, i][0, 0]) for i in range(N_CH)])
ys = np.array([float(mat_cl['chanlocs']['Y'][0, i][0, 0]) for i in range(N_CH)])

# Topoplot coordinates
theta_rad = thetas_deg * np.pi / 180
plot_x = radii * np.sin(theta_rad)
plot_y = radii * np.cos(theta_rad)

# Scalp angle for each channel (for circular harmonic analysis)
# Use the 2D topoplot position: angle from center
scalp_angle = np.arctan2(plot_x, plot_y)  # angle from nose (top)

groups = ['CTR', 'ISY', 'VIP', 'HYT']
experience = {'CTR': 0, 'ISY': 2625, 'VIP': 9201, 'HYT': 15475}
colors = {'CTR': '#4477AA', 'ISY': '#66CCEE', 'VIP': '#228833', 'HYT': '#EE6677'}

# Compute per-subject per-channel ratios for all groups
# ratios[group] = (64 channels x 16 subjects)
ratios = {}
for g in groups:
    a = mat_alpha[f'{g}_medit_711']
    gm = mat_gamma[f'{g}_medit60110']
    ratios[g] = a / gm

# Key channel indices
P9_IDX = labels.index('P9')    # 23
P10_IDX = labels.index('P10')  # 60

# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 1: ISY INDIVIDUAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ANALYSIS 1: ISY INDIVIDUAL DISTRIBUTION")
print("=" * 70)

# Per-subject statistics for ISY and HYT
def subject_stats(ratio_matrix):
    """Compute per-subject summary from (64 x N_subj) ratio matrix."""
    n_subj = ratio_matrix.shape[1]
    results = []
    for s in range(n_subj):
        r = ratio_matrix[:, s]
        results.append({
            'subject': s + 1,
            'median': np.median(r),
            'mean': np.mean(r),
            'std': np.std(r),
            'frac_010': np.mean(np.abs(r - 4/3) < 0.10),
            'frac_005': np.mean(np.abs(r - 4/3) < 0.05),
            'p9_ratio': r[P9_IDX],
            'p10_ratio': r[P10_IDX],
            'dist_43': abs(np.median(r) - 4/3),
        })
    return results

isy_stats = subject_stats(ratios['ISY'])
hyt_stats = subject_stats(ratios['HYT'])
ctr_stats = subject_stats(ratios['CTR'])
vip_stats = subject_stats(ratios['VIP'])

# Print ISY table
print("\nISY Subject Table:")
print(f"{'Subj':>4} | {'Median':>7} | {'|r-4/3|':>7} | {'Ch<0.10':>7} | {'Ch<0.05':>7} | "
      f"{'P9':>7} | {'P10':>7} | {'Std':>6}")
print("-" * 75)
for s in sorted(isy_stats, key=lambda x: x['dist_43']):
    print(f"  {s['subject']:>2} | {s['median']:>7.4f} | {s['dist_43']:>7.4f} | "
          f"{s['frac_010']:>7.3f} | {s['frac_005']:>7.3f} | "
          f"{s['p9_ratio']:>7.4f} | {s['p10_ratio']:>7.4f} | {s['std']:>6.4f}")

# ISY median ratios (per subject)
isy_medians = np.array([s['median'] for s in isy_stats])
hyt_medians = np.array([s['median'] for s in hyt_stats])
ctr_medians = np.array([s['median'] for s in ctr_stats])
vip_medians = np.array([s['median'] for s in vip_stats])

# Bimodality tests on ISY
print("\n--- ISY Bimodality Tests ---")

# Hartigan's dip test
dip_stat, dip_p = diptest.diptest(isy_medians)
print(f"Hartigan's dip test: dip={dip_stat:.4f}, p={dip_p:.4f}")
if dip_p < 0.05:
    print("  -> BIMODAL (p < 0.05)")
else:
    print("  -> Not significantly bimodal")

# Bimodality coefficient
n = len(isy_medians)
skew = stats.skew(isy_medians)
kurt = stats.kurtosis(isy_medians, fisher=True)  # excess kurtosis
# BC = (skew^2 + 1) / (kurt + 3*(n-1)^2/((n-2)*(n-3)))
# Simplified: BC = (g1^2 + 1) / (g2 + 3)  where g2 is excess kurtosis
bc = (skew**2 + 1) / (kurt + 3)
print(f"Bimodality coefficient: BC = {bc:.4f}")
print(f"  skewness = {skew:.4f}, excess kurtosis = {kurt:.4f}")
if bc > 0.555:
    print(f"  -> BC > 0.555: SUGGESTS BIMODALITY")
else:
    print(f"  -> BC <= 0.555: unimodal")

# Shapiro-Wilk normality test
sw_stat, sw_p = stats.shapiro(isy_medians)
print(f"Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}")

# If bimodal, find valley using kernel density
from scipy.signal import argrelmin
kde_x = np.linspace(isy_medians.min() - 0.1, isy_medians.max() + 0.1, 500)
kde = stats.gaussian_kde(isy_medians, bw_method=0.3)
kde_y = kde(kde_x)
valleys = argrelmin(kde_y, order=20)[0]
if len(valleys) > 0:
    valley_ratios = kde_x[valleys]
    print(f"KDE valleys at ratios: {valley_ratios}")
    for v in valley_ratios:
        print(f"  Valley at {v:.4f}, |v - 4/3| = {abs(v - 4/3):.4f}")
else:
    print("No KDE valleys found (unimodal shape)")

# P9/P10 percentiles within ISY
p9_vals = ratios['ISY'][P9_IDX, :]  # 16 subjects' P9 values
p10_vals = ratios['ISY'][P10_IDX, :]
# P9 channel mean across ISY subjects
print(f"\nISY P9 channel: mean={p9_vals.mean():.4f}, std={p9_vals.std():.4f}")
print(f"ISY P10 channel: mean={p10_vals.mean():.4f}, std={p10_vals.std():.4f}")

# P9/P10 in terms of the ISY subject distribution
# "P9 percentile" = where does the ISY group-mean P9 ratio fall in the per-subject median distribution
p9_group_mean = p9_vals.mean()
p10_group_mean = p10_vals.mean()
p9_pctile = stats.percentileofscore(isy_medians, p9_group_mean)
p10_pctile = stats.percentileofscore(isy_medians, p10_group_mean)
print(f"ISY P9 group mean ({p9_group_mean:.4f}) is at {p9_pctile:.1f}th percentile of subject medians")
print(f"ISY P10 group mean ({p10_group_mean:.4f}) is at {p10_pctile:.1f}th percentile of subject medians")

# Fraction of ISY subjects within 0.10 of 4/3
frac_isy_near = np.mean(np.abs(isy_medians - 4/3) < 0.10)
n_isy_near = np.sum(np.abs(isy_medians - 4/3) < 0.10)
print(f"\nISY subjects within 0.10 of 4/3: {n_isy_near}/{N_SUBJ} ({frac_isy_near:.3f})")

frac_hyt_near = np.mean(np.abs(hyt_medians - 4/3) < 0.10)
n_hyt_near = np.sum(np.abs(hyt_medians - 4/3) < 0.10)
print(f"HYT subjects within 0.10 of 4/3: {n_hyt_near}/{N_SUBJ} ({frac_hyt_near:.3f})")

# ─── ISY Distribution Plot ───
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: ISY vs HYT violin/dot comparison
ax = axes[0]
all_groups_medians = [ctr_medians, isy_medians, vip_medians, hyt_medians]
parts = ax.violinplot(all_groups_medians, positions=[0, 1, 2, 3], showmedians=True,
                       showextrema=True, widths=0.7)
for i, (pc, g) in enumerate(zip(parts['bodies'], groups)):
    pc.set_facecolor(colors[g])
    pc.set_alpha(0.5)
# Overlay individual dots
for i, (g, medians) in enumerate(zip(groups, all_groups_medians)):
    jitter = np.random.default_rng(42).normal(0, 0.05, len(medians))
    ax.scatter(np.full_like(medians, i) + jitter, medians,
               c=colors[g], s=30, alpha=0.8, edgecolors='black', linewidths=0.3, zorder=5)
ax.axhline(y=4/3, color='red', ls='--', lw=2, label='4/3', zorder=3)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([f'CTR\n(0h)', f'ISY\n(2625h)', f'VIP\n(9201h)', f'HYT\n(15475h)'])
ax.set_ylabel('Subject median alpha/gamma ratio')
ax.set_title('Per-Subject Distribution')
ax.legend(fontsize=8)

# Plot 2: ISY individual histogram + KDE
ax = axes[1]
ax.hist(isy_medians, bins=8, alpha=0.5, color=colors['ISY'], edgecolor='black',
        density=True, label='ISY subjects')
ax.plot(kde_x, kde_y, color=colors['ISY'], lw=2)
ax.axvline(x=4/3, color='red', ls='--', lw=2, label='4/3')
# Mark valleys
for v in (kde_x[valleys] if len(valleys) > 0 else []):
    ax.axvline(x=v, color='orange', ls=':', lw=1.5, label=f'valley={v:.3f}')
ax.set_xlabel('Subject median alpha/gamma ratio')
ax.set_ylabel('Density')
ax.set_title(f'ISY Distribution (dip p={dip_p:.3f}, BC={bc:.3f})')
ax.legend(fontsize=7)

# Plot 3: Per-subject fraction of channels near 4/3
ax = axes[2]
for i, (g, st) in enumerate(zip(groups, [ctr_stats, isy_stats, vip_stats, hyt_stats])):
    fracs = [s['frac_010'] for s in st]
    jitter = np.random.default_rng(42).normal(0, 0.05, len(fracs))
    ax.scatter(np.full(len(fracs), i) + jitter, fracs,
               c=colors[g], s=40, alpha=0.7, edgecolors='black', linewidths=0.3)
    ax.bar(i, np.mean(fracs), color=colors[g], alpha=0.3, width=0.6)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([f'CTR\n(0h)', f'ISY\n(2625h)', f'VIP\n(9201h)', f'HYT\n(15475h)'])
ax.set_ylabel('Fraction of channels |r - 4/3| < 0.10')
ax.set_title('Per-Subject Channel Fraction Near 4/3')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'isy_individual_distribution.png'), dpi=150)
print(f"\nSaved: isy_individual_distribution.png")

# Save bimodality test results
with open(os.path.join(REPORTS_DIR, 'isy_bimodality_test.txt'), 'w') as f:
    f.write("ISY BIMODALITY TEST RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"N subjects: {N_SUBJ}\n")
    f.write(f"Subject medians: {sorted(isy_medians)}\n\n")
    f.write(f"Hartigan's dip test: dip={dip_stat:.4f}, p={dip_p:.4f}\n")
    f.write(f"  Bimodal? {'YES' if dip_p < 0.05 else 'NO'}\n\n")
    f.write(f"Bimodality coefficient: BC={bc:.4f}\n")
    f.write(f"  skewness={skew:.4f}, excess kurtosis={kurt:.4f}\n")
    f.write(f"  Bimodal (BC>0.555)? {'YES' if bc > 0.555 else 'NO'}\n\n")
    f.write(f"Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}\n\n")
    if len(valleys) > 0:
        f.write(f"KDE valleys: {kde_x[valleys]}\n")
    else:
        f.write("No KDE valleys found\n")
    f.write(f"\nISY P9 channel group mean: {p9_group_mean:.4f}\n")
    f.write(f"ISY P10 channel group mean: {p10_group_mean:.4f}\n")
    f.write(f"P9 percentile in subject medians: {p9_pctile:.1f}\n")
    f.write(f"P10 percentile in subject medians: {p10_pctile:.1f}\n")
    f.write(f"\nISY subjects within 0.10 of 4/3: {n_isy_near}/{N_SUBJ}\n")
    f.write(f"HYT subjects within 0.10 of 4/3: {n_hyt_near}/{N_SUBJ}\n")
print("Saved: isy_bimodality_test.txt")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Z3 SPATIAL SYMMETRY SCORE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 2: Z3 SPATIAL SYMMETRY SCORE")
print("=" * 70)

def compute_circular_harmonics(ratio_vec, angles):
    """
    Compute circular harmonics of a spatial ratio map.
    ratio_vec: (N_CH,) alpha/gamma ratios
    angles: (N_CH,) angular position of each channel on scalp
    Returns: H1, H2, H3, H4, H6
    """
    n = len(ratio_vec)
    # Demean the ratio to focus on spatial pattern
    r = ratio_vec - ratio_vec.mean()
    H1 = np.abs(np.sum(r * np.exp(1j * 1 * angles))) / n
    H2 = np.abs(np.sum(r * np.exp(1j * 2 * angles))) / n
    H3 = np.abs(np.sum(r * np.exp(1j * 3 * angles))) / n
    H4 = np.abs(np.sum(r * np.exp(1j * 4 * angles))) / n
    H6 = np.abs(np.sum(r * np.exp(1j * 6 * angles))) / n
    return H1, H2, H3, H4, H6

# Per-subject Z3 scores
z3_scores = {g: [] for g in groups}
h_values = {g: {'H1': [], 'H2': [], 'H3': [], 'H4': [], 'H6': []} for g in groups}

for g in groups:
    for s in range(N_SUBJ):
        r_vec = ratios[g][:, s]
        H1, H2, H3, H4, H6 = compute_circular_harmonics(r_vec, scalp_angle)
        z3_score = H3 / H2 if H2 > 1e-10 else np.nan
        z3_scores[g].append(z3_score)
        h_values[g]['H1'].append(H1)
        h_values[g]['H2'].append(H2)
        h_values[g]['H3'].append(H3)
        h_values[g]['H4'].append(H4)
        h_values[g]['H6'].append(H6)

# Group statistics
print("\nZ3 Symmetry Score (H3/H2) by Group:")
print(f"{'Group':>5} | {'Z3 (median)':>11} | {'Z3 (mean)':>10} | {'H2':>8} | {'H3':>8} | "
      f"{'H6':>8} | {'Z3>1?':>6} | {'p vs CTR':>8}")
print("-" * 80)

ctr_z3 = np.array(z3_scores['CTR'])
for g in groups:
    z3 = np.array(z3_scores[g])
    h2 = np.array(h_values[g]['H2'])
    h3 = np.array(h_values[g]['H3'])
    h6 = np.array(h_values[g]['H6'])

    med_z3 = np.median(z3)
    mean_z3 = np.mean(z3)
    z3_gt1 = "YES" if med_z3 > 1 else "NO"

    if g != 'CTR':
        u, p = stats.mannwhitneyu(z3, ctr_z3, alternative='greater')
        p_str = f"{p:.4f}"
    else:
        p_str = "---"

    print(f"  {g:>3} | {med_z3:>11.4f} | {mean_z3:>10.4f} | {np.median(h2):>8.5f} | "
          f"{np.median(h3):>8.5f} | {np.median(h6):>8.5f} | {z3_gt1:>6} | {p_str:>8}")

# Monotonic ordering test
z3_medians = [np.median(z3_scores[g]) for g in groups]
exp_hours = [experience[g] for g in groups]
rho_z3, p_z3 = stats.spearmanr(exp_hours, z3_medians)
print(f"\nSpearman (experience vs Z3 score): rho={rho_z3:.4f}, p={p_z3:.4f}")

# All harmonics comparison
print("\n--- Full Harmonic Spectrum (group medians) ---")
print(f"{'Group':>5} | {'H1':>8} | {'H2':>8} | {'H3':>8} | {'H4':>8} | {'H6':>8} | {'H3/H2':>8} | {'H6/H2':>8}")
print("-" * 75)
for g in groups:
    h1m = np.median(h_values[g]['H1'])
    h2m = np.median(h_values[g]['H2'])
    h3m = np.median(h_values[g]['H3'])
    h4m = np.median(h_values[g]['H4'])
    h6m = np.median(h_values[g]['H6'])
    print(f"  {g:>3} | {h1m:>8.5f} | {h2m:>8.5f} | {h3m:>8.5f} | {h4m:>8.5f} | "
          f"{h6m:>8.5f} | {h3m/h2m:>8.4f} | {h6m/h2m:>8.4f}")

# ─── Z3 Plots ───
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Z3 score by group (bar + dots)
ax = axes[0]
for i, g in enumerate(groups):
    z3 = z3_scores[g]
    jitter = np.random.default_rng(42).normal(0, 0.05, len(z3))
    ax.scatter(np.full(len(z3), i) + jitter, z3,
               c=colors[g], s=40, alpha=0.7, edgecolors='black', linewidths=0.3, zorder=5)
    ax.bar(i, np.median(z3), color=colors[g], alpha=0.3, width=0.6)
ax.axhline(y=1.0, color='gray', ls='--', lw=1.5, label='H3 = H2 parity')
ax.set_xticks(range(4))
ax.set_xticklabels([f'{g}\n({experience[g]}h)' for g in groups])
ax.set_ylabel('Z3 Score (H3/H2)')
ax.set_title('Z3 Symmetry by Group')
ax.legend(fontsize=8)

# Plot 2: Full harmonic spectrum per group
ax = axes[1]
harmonics = [1, 2, 3, 4, 6]
for g in groups:
    h_medians = [np.median(h_values[g][f'H{h}']) for h in harmonics]
    ax.plot(harmonics, h_medians, 'o-', color=colors[g], label=g, lw=2, markersize=6)
ax.set_xlabel('Harmonic order')
ax.set_ylabel('Amplitude (median)')
ax.set_title('Spatial Harmonic Spectrum')
ax.set_xticks(harmonics)
ax.set_xticklabels(['H1\n(dipole)', 'H2\n(bilateral)', 'H3\n(Z3)', 'H4\n(quad)', 'H6\n(hex)'])
ax.legend(fontsize=8)

# Plot 3: Z3 topographic sectors for HYT
ax = axes[2]
# HYT group mean ratio
hyt_mean_ratio = ratios['HYT'].mean(axis=1)  # 64 channels

# Define three 120-degree sectors
sector_colors_rgb = ['#E74C3C', '#2ECC71', '#3498DB']
sector_labels = []
sector_means = []
for sec in range(3):
    # Sector boundaries: [-60+120*sec, 60+120*sec] degrees
    center = -60 + 120 * sec  # -60, 60, 180
    center_rad = center * np.pi / 180
    # Angular distance from sector center
    angle_diff = np.abs(np.angle(np.exp(1j * (scalp_angle - center_rad))))
    in_sector = angle_diff < (60 * np.pi / 180)  # within 60 degrees

    sec_ratio = hyt_mean_ratio[in_sector].mean()
    sec_n = in_sector.sum()
    sector_labels.append(f'S{sec+1} ({sec_n}ch)')
    sector_means.append(sec_ratio)

    # Draw sector wedge
    theta_wedge = np.linspace(center_rad - np.pi/3, center_rad + np.pi/3, 50)
    r_wedge = 0.55
    wedge_x = np.concatenate([[0], r_wedge * np.sin(theta_wedge), [0]])
    wedge_y = np.concatenate([[0], r_wedge * np.cos(theta_wedge), [0]])
    ax.fill(wedge_x, wedge_y, color=sector_colors_rgb[sec], alpha=0.1)
    ax.plot(wedge_x, wedge_y, color=sector_colors_rgb[sec], lw=1, alpha=0.5)

# Plot channels
sc = ax.scatter(plot_x, plot_y, c=hyt_mean_ratio, cmap='RdYlBu_r',
                s=60, vmin=1.2, vmax=1.6, edgecolors='black', linewidths=0.5, zorder=5)
# Head outline
circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.75, 0.75)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'HYT Z3 Sectors\nS1={sector_means[0]:.3f}, S2={sector_means[1]:.3f}, S3={sector_means[2]:.3f}')
plt.colorbar(sc, ax=ax, label='Alpha/Gamma', shrink=0.7)

# Z3 uniformity across sectors
sector_range = max(sector_means) - min(sector_means)
sector_cv = np.std(sector_means) / np.mean(sector_means)
print(f"\nHYT Z3 Sector Analysis:")
for i, (lbl, m) in enumerate(zip(sector_labels, sector_means)):
    print(f"  {lbl}: mean ratio = {m:.4f}, |r-4/3| = {abs(m-4/3):.4f}")
print(f"  Sector range: {sector_range:.4f}")
print(f"  Sector CV: {sector_cv:.4f}")
print(f"  Three sectors {'UNIFORM' if sector_range < 0.05 else 'NON-UNIFORM'}")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'z3_symmetry_by_group.png'), dpi=150)
print(f"\nSaved: z3_symmetry_by_group.png")

# Separate topographic sector plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for col, g in enumerate(groups):
    ax = axes[col]
    g_mean = ratios[g].mean(axis=1)

    # Three sectors
    secs = []
    for sec in range(3):
        center_rad = (-60 + 120 * sec) * np.pi / 180
        angle_diff = np.abs(np.angle(np.exp(1j * (scalp_angle - center_rad))))
        in_sector = angle_diff < (60 * np.pi / 180)
        secs.append(g_mean[in_sector].mean())

        theta_wedge = np.linspace(center_rad - np.pi/3, center_rad + np.pi/3, 50)
        wedge_x = np.concatenate([[0], 0.55 * np.sin(theta_wedge), [0]])
        wedge_y = np.concatenate([[0], 0.55 * np.cos(theta_wedge), [0]])
        ax.fill(wedge_x, wedge_y, color=sector_colors_rgb[sec], alpha=0.1)

    sc = ax.scatter(plot_x, plot_y, c=g_mean, cmap='RdYlBu_r',
                    s=50, vmin=1.2, vmax=1.8, edgecolors='black', linewidths=0.3, zorder=5)
    circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
    ax.add_patch(circle)
    ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75)
    ax.set_aspect('equal'); ax.axis('off')
    sec_range = max(secs) - min(secs)
    ax.set_title(f'{g} ({experience[g]}h)\nS1={secs[0]:.3f} S2={secs[1]:.3f} S3={secs[2]:.3f}\nrange={sec_range:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'z3_topographic_sectors.png'), dpi=150)
print("Saved: z3_topographic_sectors.png")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 3: SPATIAL UNIFORMITY OF HYT AT-4/3 CHANNELS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 3: SPATIAL UNIFORMITY OF HYT AT-4/3 CHANNELS")
print("=" * 70)

hyt_mean_ratio = ratios['HYT'].mean(axis=1)
at_43 = np.abs(hyt_mean_ratio - 4/3) < 0.10
n_at_43 = at_43.sum()
print(f"\nHYT channels within 0.10 of 4/3: {n_at_43}/64")

# ─── Moran's I ───
# Spatial weight matrix: inverse distance
coords = np.column_stack([plot_x, plot_y])
dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(dist_matrix, np.inf)
W = 1.0 / dist_matrix  # inverse distance weights
np.fill_diagonal(W, 0)
# Row-standardize
W_row = W / W.sum(axis=1, keepdims=True)

# Moran's I for binary at-4/3 indicator
z = at_43.astype(float) - at_43.mean()
n_ch = len(z)
numerator = n_ch * np.sum(W_row * np.outer(z, z))
denominator = W_row.sum() * np.sum(z**2)
morans_I = numerator / denominator if denominator > 0 else 0

# Permutation test for significance
n_perm = 9999
perm_I = np.zeros(n_perm)
rng = np.random.default_rng(42)
for p in range(n_perm):
    z_perm = rng.permutation(z)
    num_p = n_ch * np.sum(W_row * np.outer(z_perm, z_perm))
    den_p = W_row.sum() * np.sum(z_perm**2)
    perm_I[p] = num_p / den_p if den_p > 0 else 0

moran_p = np.mean(np.abs(perm_I) >= np.abs(morans_I))
expected_I = -1 / (n_ch - 1)

print(f"\nMoran's I = {morans_I:.4f}")
print(f"Expected I (random) = {expected_I:.4f}")
print(f"Permutation p-value (two-sided): {moran_p:.4f}")
if morans_I > 0 and moran_p < 0.05:
    moran_interp = "CLUSTERED"
elif morans_I < expected_I and moran_p < 0.05:
    moran_interp = "DISPERSED"
else:
    moran_interp = "UNIFORM (not significantly different from random)"
print(f"Interpretation: {moran_interp}")

# ─── Sector analysis ───
# Define regions
region_map = {
    'frontal': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'Fpz', 'Fp2',
                'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    'central': ['FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
                'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8'],
    'parietal': ['TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                 'CPz', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'Pz'],
    'occipital': ['PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'PO8', 'PO4', 'O2'],
}

print(f"\n--- Sector breakdown (at-4/3 channels) ---")
sector_obs = []
sector_exp = []
sector_names = []
total_channels_in_sectors = 0
for reg_name in ['frontal', 'central', 'parietal', 'occipital']:
    reg_channels = region_map[reg_name]
    reg_idx = [i for i, l in enumerate(labels) if l in reg_channels]
    n_reg = len(reg_idx)
    n_at_43_reg = at_43[reg_idx].sum()
    frac_at_43_reg = n_at_43_reg / n_reg if n_reg > 0 else 0

    sector_obs.append(n_at_43_reg)
    sector_exp.append(n_reg * n_at_43 / N_CH)  # expected under uniform
    sector_names.append(reg_name)
    total_channels_in_sectors += n_reg

    print(f"  {reg_name:>10}: {n_at_43_reg:>2}/{n_reg:>2} at 4/3 ({frac_at_43_reg:.3f}), "
          f"expected={n_reg * n_at_43 / N_CH:.1f}")

# Chi-squared test
sector_obs = np.array(sector_obs, dtype=float)
sector_exp = np.array(sector_exp, dtype=float)
# Avoid zero expected
sector_exp = np.maximum(sector_exp, 0.5)
chi2, chi2_p = stats.chisquare(sector_obs, f_exp=sector_exp)
print(f"\nChi-squared test (uniform distribution): chi2={chi2:.4f}, p={chi2_p:.4f}")
if chi2_p > 0.05:
    print("  -> Not significantly different from uniform across sectors")
else:
    print("  -> Significantly non-uniform across sectors")

# Save spatial uniformity results
with open(os.path.join(REPORTS_DIR, 'spatial_uniformity_test.txt'), 'w') as f:
    f.write("SPATIAL UNIFORMITY TEST — HYT at-4/3 channels\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"HYT channels within 0.10 of 4/3: {n_at_43}/64\n\n")
    f.write(f"Moran's I = {morans_I:.4f}\n")
    f.write(f"Expected I (random) = {expected_I:.4f}\n")
    f.write(f"Permutation p = {moran_p:.4f}\n")
    f.write(f"Interpretation: {moran_interp}\n\n")
    f.write("Sector breakdown:\n")
    for name, obs, exp in zip(sector_names, sector_obs, sector_exp):
        f.write(f"  {name}: {int(obs)} at 4/3 (expected {exp:.1f})\n")
    f.write(f"\nChi-squared: chi2={chi2:.4f}, p={chi2_p:.4f}\n")
print("Saved: spatial_uniformity_test.txt")


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════

# Determine P_MED_4 status
hyt_z3_median = np.median(z3_scores['HYT'])
ctr_z3_median = np.median(z3_scores['CTR'])
if 'HYT' in z3_scores and 'CTR' in z3_scores:
    u_z3, p_z3_test = stats.mannwhitneyu(z3_scores['HYT'], z3_scores['CTR'], alternative='greater')
else:
    p_z3_test = 1.0

if hyt_z3_median > 1 and p_z3_test < 0.05:
    p_med4_status = "CONFIRMED"
elif hyt_z3_median > ctr_z3_median and p_z3_test < 0.10:
    p_med4_status = "PARTIAL"
else:
    p_med4_status = "NOT CONFIRMED"

# ISY bimodality
bimodal = "YES" if (dip_p < 0.05 or bc > 0.555) else "NO"

summary = f"""
=== PHASE 1 DEEP DIVE RESULTS ===

ISY DISTRIBUTION:
  Bimodal? {bimodal} (dip test p={dip_p:.4f}, BC={bc:.4f})
  {"Valley at ratio = " + ", ".join(f"{v:.4f}" for v in kde_x[valleys]) if len(valleys) > 0 else "No valley detected"}
  P9 percentile within ISY: {p9_pctile:.1f}
  P10 percentile within ISY: {p10_pctile:.1f}
  Fraction ISY subjects within 0.10 of 4/3: {n_isy_near}/{N_SUBJ}
  Fraction HYT subjects within 0.10 of 4/3: {n_hyt_near}/{N_SUBJ}

Z3 SYMMETRY:
  CTR median Z3 score: {np.median(z3_scores['CTR']):.4f}
  ISY median Z3 score: {np.median(z3_scores['ISY']):.4f}
  VIP median Z3 score: {np.median(z3_scores['VIP']):.4f}
  HYT median Z3 score: {np.median(z3_scores['HYT']):.4f}
  HYT Z3 > 1? {"YES" if hyt_z3_median > 1 else "NO"}
  HYT vs CTR Mann-Whitney p = {p_z3_test:.4f}
  Spearman (experience vs Z3): rho={rho_z3:.4f}, p={p_z3:.4f}
  P_MED_4 prediction: {p_med4_status}

SPATIAL UNIFORMITY (HYT at-4/3 channels):
  Moran's I = {morans_I:.4f} (p = {moran_p:.4f})
  Interpretation: {moran_interp}
  Sector breakdown: F={int(sector_obs[0])}/{sum(1 for l in labels if l in region_map['frontal'])} C={int(sector_obs[1])}/{sum(1 for l in labels if l in region_map['central'])} P={int(sector_obs[2])}/{sum(1 for l in labels if l in region_map['parietal'])} O={int(sector_obs[3])}/{sum(1 for l in labels if l in region_map['occipital'])}
  Chi-squared: chi2={chi2:.4f}, p={chi2_p:.4f}

=== END ===
"""

print(summary)

# Save full summary
with open(os.path.join(REPORTS_DIR, 'phase1_deep_summary.txt'), 'w') as f:
    f.write(summary)
print("Saved: phase1_deep_summary.txt")

print("\nAll Phase 1 Deep Dive analyses complete.")
