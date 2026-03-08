"""
Phase 1 Channel-Level Analysis — Topographic alpha/gamma ratio
Uses 64-channel spectral data from MAT files + chanlocs
"""
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, FIGURES_DIR, REPORTS_DIR

# ─── Load data ───
mat_alpha = sio.loadmat(os.path.join(MEDITATION_DATA, '711Hz_spec_data_medit.mat'))
mat_gamma = sio.loadmat(os.path.join(MEDITATION_DATA, '60110Hz_spec_data_medit.mat'))
mat_cl = sio.loadmat(os.path.join(MEDITATION_DATA, 'chanlocs.mat'))

# Extract channel info (first 64 = EEG channels)
N_CH = 64
labels = [str(mat_cl['chanlocs']['labels'][0, i][0]) for i in range(N_CH)]
thetas = [float(mat_cl['chanlocs']['theta'][0, i][0, 0]) for i in range(N_CH)]
radii = [float(mat_cl['chanlocs']['radius'][0, i][0, 0]) for i in range(N_CH)]
xs = [float(mat_cl['chanlocs']['X'][0, i][0, 0]) for i in range(N_CH)]
ys = [float(mat_cl['chanlocs']['Y'][0, i][0, 0]) for i in range(N_CH)]
zs = [float(mat_cl['chanlocs']['Z'][0, i][0, 0]) for i in range(N_CH)]

# Classify channels by region
regions = {}
for i, lbl in enumerate(labels):
    if lbl.startswith(('Fp', 'AF')):
        regions[i] = 'prefrontal'
    elif lbl.startswith(('F',)):
        if lbl.startswith(('FT', 'FC')):
            regions[i] = 'fronto-central'
        else:
            regions[i] = 'frontal'
    elif lbl.startswith(('C',)) and not lbl.startswith(('CP',)):
        regions[i] = 'central'
    elif lbl.startswith(('T',)):
        regions[i] = 'temporal'
    elif lbl.startswith(('TP', 'CP')):
        regions[i] = 'centro-parietal'
    elif lbl.startswith(('P',)) and not lbl.startswith(('PO',)):
        regions[i] = 'parietal'
    elif lbl.startswith(('PO', 'O', 'I')):
        regions[i] = 'parieto-occipital'
    else:
        regions[i] = 'other'

groups = ['CTR', 'ISY', 'VIP', 'HYT']
experience = {'CTR': 0, 'ISY': 2625, 'VIP': 9201, 'HYT': 15475}
colors = {'CTR': '#4477AA', 'ISY': '#66CCEE', 'VIP': '#228833', 'HYT': '#EE6677'}

# ─── Compute per-channel, per-subject alpha/gamma ratio ───
print("="*70)
print("CHANNEL-LEVEL ALPHA/GAMMA RATIO ANALYSIS")
print("="*70)

channel_ratios = {}  # group -> (64 x 16) array of ratios
channel_ratio_means = {}  # group -> (64,) mean across subjects

for g in groups:
    alpha_data = mat_alpha[f'{g}_medit_711']   # 64 x 16
    gamma_data = mat_gamma[f'{g}_medit60110']  # 64 x 16
    ratios = alpha_data / gamma_data            # 64 x 16
    channel_ratios[g] = ratios
    channel_ratio_means[g] = ratios.mean(axis=1)  # 64

    dist_43 = np.abs(channel_ratio_means[g] - 4/3)
    best_idx = np.argsort(dist_43)

    print(f"\n{g} ({experience[g]}h):")
    print(f"  Grand mean ratio: {channel_ratio_means[g].mean():.4f}")
    print(f"  |grand mean - 4/3| = {abs(channel_ratio_means[g].mean() - 4/3):.4f}")
    print(f"  Channels with ratio nearest 4/3:")
    for idx in best_idx[:8]:
        print(f"    {labels[idx]:>6} ({regions[idx]:<18}): "
              f"ratio={channel_ratio_means[g][idx]:.4f}, "
              f"|r-4/3|={dist_43[idx]:.4f}")
    print(f"  Fraction of channels within 0.05 of 4/3: "
          f"{np.mean(dist_43 < 0.05):.3f} ({np.sum(dist_43 < 0.05)}/64)")
    print(f"  Fraction of channels within 0.10 of 4/3: "
          f"{np.mean(dist_43 < 0.10):.3f} ({np.sum(dist_43 < 0.10)}/64)")

# ─── Regional analysis ───
print("\n" + "="*70)
print("REGIONAL ALPHA/GAMMA RATIOS")
print("="*70)

region_names = sorted(set(regions.values()))
for reg in region_names:
    ch_idx = [i for i, r in regions.items() if r == reg]
    print(f"\n{reg} ({len(ch_idx)} channels: {', '.join(labels[i] for i in ch_idx[:6])}{'...' if len(ch_idx)>6 else ''}):")
    for g in groups:
        reg_ratios = channel_ratio_means[g][ch_idx]
        print(f"  {g}: mean={reg_ratios.mean():.4f}, |mean-4/3|={abs(reg_ratios.mean()-4/3):.4f}")

# ─── Which region is closest to 4/3 for HYT? ───
print("\n--- HYT: regions sorted by proximity to 4/3 ---")
hyt_reg_dists = []
for reg in region_names:
    ch_idx = [i for i, r in regions.items() if r == reg]
    mean_ratio = channel_ratio_means['HYT'][ch_idx].mean()
    hyt_reg_dists.append((reg, mean_ratio, abs(mean_ratio - 4/3), len(ch_idx)))

hyt_reg_dists.sort(key=lambda x: x[2])
for reg, ratio, dist, n in hyt_reg_dists:
    marker = " **" if dist < 0.05 else ""
    print(f"  {reg:<20}: ratio={ratio:.4f}, |r-4/3|={dist:.4f}, n_ch={n}{marker}")

# ─── Per-subject channel ratio: fraction of subjects x channels near 4/3 ───
print("\n" + "="*70)
print("PER-SUBJECT x PER-CHANNEL: FRACTION NEAR 4/3")
print("="*70)

for g in groups:
    ratios = channel_ratios[g]  # 64 x 16
    near_43 = np.abs(ratios - 4/3) < 0.10
    frac_per_subject = near_43.mean(axis=0)  # fraction of channels per subject
    frac_per_channel = near_43.mean(axis=1)  # fraction of subjects per channel
    print(f"\n{g}:")
    print(f"  Overall fraction near 4/3 (|r-4/3|<0.10): {near_43.mean():.4f}")
    print(f"  Per-subject: mean frac = {frac_per_subject.mean():.4f}, "
          f"std = {frac_per_subject.std():.4f}")
    print(f"  Best subject: {frac_per_subject.max():.4f} "
          f"({int(frac_per_subject.max()*64)}/64 channels)")

# ─── HYT vs CTR: which channels show LARGEST group difference? ───
print("\n" + "="*70)
print("HYT vs CTR: CHANNEL-LEVEL RATIO SHIFT")
print("="*70)

# For each channel, test HYT ratio vs CTR ratio
shift_results = []
for i in range(N_CH):
    hyt_vals = channel_ratios['HYT'][i, :]  # 16 subjects
    ctr_vals = channel_ratios['CTR'][i, :]   # 16 subjects
    t, p = stats.ttest_ind(hyt_vals, ctr_vals)
    shift = hyt_vals.mean() - ctr_vals.mean()
    shift_results.append((i, labels[i], regions[i], shift, t, p,
                          hyt_vals.mean(), ctr_vals.mean()))

# Sort by shift (most negative = HYT lower than CTR)
shift_results.sort(key=lambda x: x[3])
print("\nChannels where HYT ratio is MOST BELOW CTR (toward 4/3):")
for i, lbl, reg, shift, t, p, hyt_m, ctr_m in shift_results[:10]:
    marker = ""
    if abs(hyt_m - 4/3) < 0.05:
        marker = " <- HYT near 4/3"
    print(f"  {lbl:>6} ({reg:<18}): HYT={hyt_m:.4f}, CTR={ctr_m:.4f}, "
          f"shift={shift:+.4f}, p={p:.4f}{marker}")

print("\nChannels where HYT ratio is MOST ABOVE CTR:")
for i, lbl, reg, shift, t, p, hyt_m, ctr_m in shift_results[-5:]:
    print(f"  {lbl:>6} ({reg:<18}): HYT={hyt_m:.4f}, CTR={ctr_m:.4f}, "
          f"shift={shift:+.4f}, p={p:.4f}")

# Count significant channels
sig_channels = [(lbl, reg, shift) for _, lbl, reg, shift, _, p, _, _ in shift_results if p < 0.05]
print(f"\nSignificant channels (p<0.05): {len(sig_channels)}/64")
for lbl, reg, shift in sig_channels:
    direction = "HYT<CTR (toward 4/3)" if shift < 0 else "HYT>CTR"
    print(f"  {lbl:>6} ({reg}): {direction}")

# ─── ISY vs VIP: spatial pattern of driven vs self-sustaining ───
print("\n" + "="*70)
print("ISY vs VIP: SPATIAL VARIANCE PATTERN")
print("="*70)

# Per-channel variance across subjects
for g in groups:
    ratios = channel_ratios[g]
    ch_var = ratios.std(axis=1)  # std across 16 subjects, per channel
    print(f"\n{g}: mean cross-subject std per channel = {ch_var.mean():.4f}")

# ISY vs VIP per-channel variance comparison
isy_var = channel_ratios['ISY'].std(axis=1)
vip_var = channel_ratios['VIP'].std(axis=1)
hyt_var = channel_ratios['HYT'].std(axis=1)
ctr_var = channel_ratios['CTR'].std(axis=1)

# Count channels where VIP is more variable than ISY
vip_more_variable = np.sum(vip_var > isy_var)
print(f"\nVIP more variable than ISY: {vip_more_variable}/64 channels ({100*vip_more_variable/64:.0f}%)")
vip_more_than_hyt = np.sum(vip_var > hyt_var)
print(f"VIP more variable than HYT: {vip_more_than_hyt}/64 channels ({100*vip_more_than_hyt/64:.0f}%)")

# Paired test across channels
t_isy_vip, p_isy_vip = stats.ttest_rel(vip_var, isy_var)
print(f"Paired t-test (VIP var > ISY var across channels): t={t_isy_vip:.3f}, p={p_isy_vip:.4f}")

t_hyt_vip, p_hyt_vip = stats.ttest_rel(vip_var, hyt_var)
print(f"Paired t-test (VIP var > HYT var across channels): t={t_hyt_vip:.3f}, p={p_hyt_vip:.4f}")

# ─── TOPOGRAPHIC PLOT ───
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Convert theta/radius to x,y for plotting (standard EEG topoplot coords)
# theta: angle in degrees, radius: distance from center
theta_rad = np.array(thetas) * np.pi / 180
plot_x = np.array(radii) * np.sin(theta_rad)
plot_y = np.array(radii) * np.cos(theta_rad)

# Top row: alpha/gamma ratio topoplots for each group
vmin_ratio = 1.1
vmax_ratio = 1.9
for col, g in enumerate(groups):
    ax = axes[0, col]
    sc = ax.scatter(plot_x, plot_y, c=channel_ratio_means[g], cmap='RdYlBu_r',
                    s=80, vmin=vmin_ratio, vmax=vmax_ratio, edgecolors='black', linewidths=0.5)
    # Draw head outline
    circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
    ax.add_patch(circle)
    # Nose
    ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.set_title(f'{g} ({experience[g]}h)\nmean={channel_ratio_means[g].mean():.3f}')
    ax.axis('off')
    if col == 3:
        plt.colorbar(sc, ax=ax, label='Alpha/Gamma ratio', shrink=0.7)

# Mark channels nearest 4/3 for HYT
ax = axes[0, 3]
dist_43_hyt = np.abs(channel_ratio_means['HYT'] - 4/3)
near_mask = dist_43_hyt < 0.05
ax.scatter(plot_x[near_mask], plot_y[near_mask], s=200, facecolors='none',
           edgecolors='red', linewidths=2.5, zorder=10)

# Bottom row: difference maps and variance
# HYT - CTR ratio difference
ax = axes[1, 0]
diff = channel_ratio_means['HYT'] - channel_ratio_means['CTR']
vabs = max(abs(diff.min()), abs(diff.max()))
sc = ax.scatter(plot_x, plot_y, c=diff, cmap='RdBu_r', s=80,
                vmin=-vabs, vmax=vabs, edgecolors='black', linewidths=0.5)
circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('HYT - CTR\n(ratio shift)')
plt.colorbar(sc, ax=ax, shrink=0.7)

# ISY - CTR
ax = axes[1, 1]
diff = channel_ratio_means['ISY'] - channel_ratio_means['CTR']
sc = ax.scatter(plot_x, plot_y, c=diff, cmap='RdBu_r', s=80,
                vmin=-vabs, vmax=vabs, edgecolors='black', linewidths=0.5)
circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('ISY - CTR\n(ratio shift)')

# VIP cross-subject variance (driven signature)
ax = axes[1, 2]
sc = ax.scatter(plot_x, plot_y, c=vip_var, cmap='hot', s=80,
                edgecolors='black', linewidths=0.5)
circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('VIP variance\n(cross-subject std)')
plt.colorbar(sc, ax=ax, shrink=0.7)

# VIP - ISY variance difference (positive = VIP more variable)
ax = axes[1, 3]
var_diff = vip_var - isy_var
vabs_v = max(abs(var_diff.min()), abs(var_diff.max()))
sc = ax.scatter(plot_x, plot_y, c=var_diff, cmap='RdBu_r', s=80,
                vmin=-vabs_v, vmax=vabs_v, edgecolors='black', linewidths=0.5)
circle = plt.Circle((0, 0), 0.55, fill=False, color='black', lw=2)
ax.add_patch(circle)
ax.plot([0, -0.03, 0.03, 0], [0.55, 0.6, 0.6, 0.55], 'k-', lw=2)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('VIP - ISY variance\n(driven > self-sustaining?)')
plt.colorbar(sc, ax=ax, shrink=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'phase1_topographic.png'), dpi=150)
print(f"\nSaved: phase1_topographic.png")

# ─── ANTERIOR-POSTERIOR GRADIENT ───
print("\n" + "="*70)
print("ANTERIOR-POSTERIOR GRADIENT")
print("="*70)

# Y coordinate: positive = anterior, negative = posterior
y_arr = np.array(ys)
anterior_mask = y_arr > 0
posterior_mask = y_arr < 0

for g in groups:
    ant_ratio = channel_ratio_means[g][anterior_mask].mean()
    post_ratio = channel_ratio_means[g][posterior_mask].mean()
    ap_gradient = ant_ratio - post_ratio
    print(f"  {g}: anterior={ant_ratio:.4f}, posterior={post_ratio:.4f}, "
          f"A-P gradient={ap_gradient:+.4f}")

# ─── MIDLINE vs LATERAL ───
print("\n--- Midline vs Lateral ---")
midline_labels = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Iz']
midline_idx = [i for i, l in enumerate(labels) if l in midline_labels]
lateral_idx = [i for i in range(N_CH) if i not in midline_idx]

for g in groups:
    mid_ratio = channel_ratio_means[g][midline_idx].mean()
    lat_ratio = channel_ratio_means[g][lateral_idx].mean()
    print(f"  {g}: midline={mid_ratio:.4f} ({len(midline_idx)} ch), "
          f"lateral={lat_ratio:.4f} ({len(lateral_idx)} ch), "
          f"|midline-4/3|={abs(mid_ratio-4/3):.4f}")

# ─── HEMISPHERE ASYMMETRY ───
print("\n--- Hemisphere Asymmetry ---")
# Negative theta = left hemisphere, positive = right
theta_arr = np.array(thetas)
left_mask = theta_arr < -5
right_mask = theta_arr > 5

for g in groups:
    left_ratio = channel_ratio_means[g][left_mask].mean()
    right_ratio = channel_ratio_means[g][right_mask].mean()
    asym = (right_ratio - left_ratio) / (right_ratio + left_ratio) * 100
    print(f"  {g}: left={left_ratio:.4f}, right={right_ratio:.4f}, "
          f"asymmetry={asym:+.2f}%")

print("\nDone.")
