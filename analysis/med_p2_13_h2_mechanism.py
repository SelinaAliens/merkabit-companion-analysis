"""
MED-P2-13: VIP H2 mechanism test
Prediction: In VIP windows where Z3 is low, H2 (bilateral symmetric component)
is specifically elevated. VIP doesn't suppress Z3 randomly - it maintains
bilateral H2 dominance.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import MEDITATION_DATA, FIGURES_DIR, REPORTS_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

# Load Z3 windowed results
with open(os.path.join(MEDITATION_DATA, 'z3_windowed', 'z3_windowed_results.json')) as f:
    data = json.load(f)

# Organize by group
groups_order = ['CTR', 'ISY', 'VIP', 'HYT']
group_data = {g: [] for g in groups_order}
for entry in data:
    g = entry['group']
    if g in group_data:
        group_data[g].append(entry)

print("=" * 70)
print("MED-P2-13: VIP H2 MECHANISM IN LOW-Z3 WINDOWS")
print("=" * 70)

# ─── 1. Basic group summaries ───
print("\n--- Group summaries ---")
for g in groups_order:
    h2_all = []
    h3_all = []
    z3_all = []
    for s in group_data[g]:
        h2_all.extend(s['h2_series'])
        h3_all.extend(s['h3_series'])
        z3_all.extend(s['z3_series'])
    h2_arr = np.array(h2_all)
    h3_arr = np.array(h3_all)
    z3_arr = np.array(z3_all)
    print(f"  {g}: n_windows={len(z3_arr)}, "
          f"H2={np.median(h2_arr):.4f}, H3={np.median(h3_arr):.4f}, "
          f"Z3={np.median(z3_arr):.4f}")

# ─── 2. Low-Z3 windows: compare H2 across groups ───
print("\n" + "=" * 70)
print("TEST 1: H2 in LOW-Z3 windows (Z3 < 0.5)")
print("=" * 70)

z3_threshold = 0.5

for g in groups_order:
    h2_low = []
    h2_high = []
    n_total = 0
    n_low = 0
    for s in group_data[g]:
        for z3, h2 in zip(s['z3_series'], s['h2_series']):
            n_total += 1
            if z3 < z3_threshold:
                h2_low.append(h2)
                n_low += 1
            else:
                h2_high.append(h2)

    h2_low = np.array(h2_low)
    h2_high = np.array(h2_high) if h2_high else np.array([np.nan])

    frac_low = n_low / n_total if n_total > 0 else 0
    print(f"\n  {g}: {n_low}/{n_total} windows with Z3<{z3_threshold} ({100*frac_low:.1f}%)")
    print(f"    H2 in low-Z3 windows:  median={np.median(h2_low):.4f}, mean={np.mean(h2_low):.4f}")
    if len(h2_high) > 1 and not np.isnan(h2_high[0]):
        print(f"    H2 in high-Z3 windows: median={np.median(h2_high):.4f}, mean={np.mean(h2_high):.4f}")

# ─── 3. VIP vs other groups: H2 in low-Z3 windows ───
print("\n" + "=" * 70)
print("TEST 2: VIP H2 vs other groups (in low-Z3 windows)")
print("=" * 70)

group_h2_low = {}
for g in groups_order:
    h2_vals = []
    for s in group_data[g]:
        for z3, h2 in zip(s['z3_series'], s['h2_series']):
            if z3 < z3_threshold:
                h2_vals.append(h2)
    group_h2_low[g] = np.array(h2_vals)

for g2 in ['CTR', 'ISY', 'HYT']:
    vip_h2 = group_h2_low['VIP']
    other_h2 = group_h2_low[g2]
    if len(vip_h2) > 0 and len(other_h2) > 0:
        u, p = stats.mannwhitneyu(vip_h2, other_h2, alternative='greater')
        print(f"  VIP H2 > {g2} H2 (low-Z3): U={u:.0f}, p={p:.4f} "
              f"(VIP med={np.median(vip_h2):.4f}, {g2} med={np.median(other_h2):.4f})")

# ─── 4. Z3-H2 anti-correlation within groups ───
print("\n" + "=" * 70)
print("TEST 3: Z3 vs H2 correlation (per group)")
print("=" * 70)

for g in groups_order:
    z3_all = []
    h2_all = []
    for s in group_data[g]:
        z3_all.extend(s['z3_series'])
        h2_all.extend(s['h2_series'])
    z3_arr = np.array(z3_all)
    h2_arr = np.array(h2_all)

    # Spearman correlation
    rho, p = stats.spearmanr(z3_arr, h2_arr)
    print(f"  {g}: Spearman rho(Z3, H2) = {rho:.4f}, p={p:.6f}")

# ─── 5. Per-subject analysis: H2/H3 ratio in low-Z3 windows ───
print("\n" + "=" * 70)
print("TEST 4: H2/H3 ratio in low-Z3 windows (mechanism specificity)")
print("=" * 70)
print("  If VIP suppresses Z3 via H2 dominance, H2/H3 should be high in low-Z3 windows")

for g in groups_order:
    h2h3_ratios = []
    for s in group_data[g]:
        for z3, h2, h3 in zip(s['z3_series'], s['h2_series'], s['h3_series']):
            if z3 < z3_threshold and h3 > 1e-10:
                h2h3_ratios.append(h2 / h3)
    h2h3 = np.array(h2h3_ratios)
    print(f"  {g}: H2/H3 in low-Z3 windows: median={np.median(h2h3):.4f}, "
          f"mean={np.mean(h2h3):.4f}, n={len(h2h3)}")

# VIP vs others for H2/H3
group_h2h3_low = {}
for g in groups_order:
    vals = []
    for s in group_data[g]:
        for z3, h2, h3 in zip(s['z3_series'], s['h2_series'], s['h3_series']):
            if z3 < z3_threshold and h3 > 1e-10:
                vals.append(h2 / h3)
    group_h2h3_low[g] = np.array(vals)

print("\n  VIP vs others (H2/H3 in low-Z3):")
for g2 in ['CTR', 'ISY', 'HYT']:
    vip_vals = group_h2h3_low['VIP']
    other_vals = group_h2h3_low[g2]
    if len(vip_vals) > 0 and len(other_vals) > 0:
        u, p = stats.mannwhitneyu(vip_vals, other_vals, alternative='greater')
        print(f"    VIP > {g2}: U={u:.0f}, p={p:.4f}")

# ─── 6. Per-subject H2 in low-Z3 windows ───
print("\n" + "=" * 70)
print("PER-SUBJECT SUMMARY: H2 in low-Z3 windows")
print("=" * 70)

for g in groups_order:
    print(f"\n  {g}:")
    subj_h2_low_medians = []
    for s in group_data[g]:
        h2_low = [h2 for z3, h2 in zip(s['z3_series'], s['h2_series']) if z3 < z3_threshold]
        n_low = len(h2_low)
        n_total = len(s['z3_series'])
        if h2_low:
            med = np.median(h2_low)
            subj_h2_low_medians.append(med)
            print(f"    {s['subject_id']:>10}: {n_low}/{n_total} low-Z3, "
                  f"H2_median={med:.4f}")
        else:
            print(f"    {s['subject_id']:>10}: 0/{n_total} low-Z3 windows")
    if subj_h2_low_medians:
        print(f"  Group median of subject medians: {np.median(subj_h2_low_medians):.4f}")

# ─── 7. Subject-level test (more conservative) ───
print("\n" + "=" * 70)
print("SUBJECT-LEVEL TEST: median H2 in low-Z3 windows per subject")
print("=" * 70)

group_subj_h2_low = {}
for g in groups_order:
    medians = []
    for s in group_data[g]:
        h2_low = [h2 for z3, h2 in zip(s['z3_series'], s['h2_series']) if z3 < z3_threshold]
        if h2_low:
            medians.append(np.median(h2_low))
    group_subj_h2_low[g] = np.array(medians)
    print(f"  {g}: n={len(medians)} subjects with low-Z3 windows, "
          f"median H2={np.median(medians):.4f}" if medians else
          f"  {g}: NO subjects with low-Z3 windows")

print("\n  Subject-level VIP vs others:")
for g2 in ['CTR', 'ISY', 'HYT']:
    v = group_subj_h2_low.get('VIP', np.array([]))
    o = group_subj_h2_low.get(g2, np.array([]))
    if len(v) > 1 and len(o) > 1:
        u, p = stats.mannwhitneyu(v, o, alternative='greater')
        print(f"    VIP > {g2}: U={u:.0f}, p={p:.4f} "
              f"(VIP={np.median(v):.4f}, {g2}={np.median(o):.4f})")

# ─── Plot ───
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = {'CTR': '#4477AA', 'ISY': '#66CCEE', 'VIP': '#228833', 'HYT': '#EE6677'}

# Panel 1: H2 vs Z3 scatter per group
for idx, g in enumerate(groups_order):
    ax = axes[0, idx] if idx < 3 else axes[1, 0]
    z3_all = []
    h2_all = []
    for s in group_data[g]:
        z3_all.extend(s['z3_series'])
        h2_all.extend(s['h2_series'])
    ax.scatter(z3_all, h2_all, alpha=0.2, s=5, c=colors[g])
    ax.axvline(z3_threshold, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Z3')
    ax.set_ylabel('H2')
    ax.set_title(f'{g}: Z3 vs H2')
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(0, 10)
    rho = stats.spearmanr(z3_all, h2_all)[0]
    ax.text(0.95, 0.95, f'rho={rho:.3f}', transform=ax.transAxes, ha='right', va='top')

# Panel 4 (axes[1,0] already used for HYT): H2 boxplot in low-Z3
ax = axes[1, 0]
z3_all = []
h2_all = []
for s in group_data['HYT']:
    z3_all.extend(s['z3_series'])
    h2_all.extend(s['h2_series'])
ax.scatter(z3_all, h2_all, alpha=0.2, s=5, c=colors['HYT'])
ax.axvline(z3_threshold, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Z3')
ax.set_ylabel('H2')
ax.set_title('HYT: Z3 vs H2')
ax.set_xlim(-0.5, 5)
ax.set_ylim(0, 10)
rho = stats.spearmanr(z3_all, h2_all)[0]
ax.text(0.95, 0.95, f'rho={rho:.3f}', transform=ax.transAxes, ha='right', va='top')

# Panel 5: H2 in low-Z3 boxplot
ax = axes[1, 1]
box_data = []
labels = []
for g in groups_order:
    vals = group_h2_low.get(g, np.array([]))
    if len(vals) > 0:
        box_data.append(vals)
        labels.append(f'{g}\nn={len(vals)}')
if box_data:
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    for patch, g in zip(bp['boxes'], groups_order):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.7)
ax.set_ylabel('H2 (in low-Z3 windows)')
ax.set_title('H2 when Z3 < 0.5')

# Panel 6: H2/H3 in low-Z3
ax = axes[1, 2]
box_data2 = []
labels2 = []
for g in groups_order:
    vals = group_h2h3_low.get(g, np.array([]))
    if len(vals) > 0:
        box_data2.append(vals)
        labels2.append(f'{g}\nn={len(vals)}')
if box_data2:
    bp = ax.boxplot(box_data2, labels=labels2, patch_artist=True, showfliers=False)
    for patch, g in zip(bp['boxes'], groups_order):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.7)
ax.axhline(1.0, color='black', ls='--', alpha=0.5, label='H2=H3')
ax.set_ylabel('H2/H3 ratio (in low-Z3 windows)')
ax.set_title('H2/H3 when Z3 < 0.5')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'med_p2_13_h2_mechanism.png'), dpi=150)
print(f"
Saved: {os.path.join(FIGURES_DIR, 'med_p2_13_h2_mechanism.png')}")

# ─── Final verdict ───
print("\n" + "=" * 70)
print("MED-P2-13 VERDICT")
print("=" * 70)
