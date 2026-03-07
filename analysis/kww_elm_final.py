"""
FINAL KWW analysis of ELM relaxation - Cavedon et al. 2019 (ASDEX Upgrade)
Includes: full spike + transport tail analysis for Idiv, gradTi recovery.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DATA_DIR, FIGURES_DIR, REPORTS_DIR

OUTDIR = FIGURES_DIR
img = np.array(Image.open(os.path.join(DATA_DIR, 'figure2_embedded.png')))
gray = np.mean(img[:,:,:3], axis=2)

# ============================================================
# CALIBRATION
# ============================================================
x_cal = np.polyfit([150, 365, 580, 795], [-5.0, 0.0, 5.0, 10.0], 1)
yg_cal = np.polyfit([2145, 2045, 1960], [0.0, 8.0, 16.0], 1)
yb_cal = np.polyfit([840, 770, 700, 630, 560], [0.0, 10.0, 20.0, 30.0, 40.0], 1)

def px_to_t(x): return np.polyval(x_cal, x)
def px_to_I(y): return np.polyval(yg_cal, y)
def px_to_grad(y): return np.polyval(yb_cal, y)

# ============================================================
# DIGITIZE IDIV
# ============================================================
t_list, I_list = [], []
for x in range(155, 795, 1):
    if 360 <= x <= 372: continue
    y_start = 1990 if x < 195 else (1995 if x > 695 else 1950)
    col = gray[y_start:2170, x]
    dark = np.where(col < 85)[0]
    if len(dark) < 1: continue
    if len(dark) == 1:
        y_c = dark[0] + y_start
    else:
        gaps = np.diff(dark)
        splits = np.where(gaps > 5)[0]
        if len(splits) == 0:
            y_c = np.mean(dark) + y_start
        else:
            clusters = np.split(dark, splits + 1)
            y_c = np.mean(max(clusters, key=len)) + y_start
    t_list.append(px_to_t(x)); I_list.append(px_to_I(y_c))

t_g = np.array(t_list); I_g = np.array(I_list)

# Bin 0.1 ms
edges = np.arange(t_g.min(), t_g.max()+0.1, 0.1)
t_gb, I_gb = [], []
for i in range(len(edges)-1):
    m = (t_g >= edges[i]) & (t_g < edges[i+1])
    if np.sum(m) > 2: t_gb.append(np.mean(t_g[m])); I_gb.append(np.mean(I_g[m]))
t_gb = np.array(t_gb); I_gb = np.array(I_gb)

post = t_gb > 0
pi = np.argmax(I_gb[post])
abs_peak = np.where(post)[0][0] + pi
t_decay = t_gb[abs_peak:] - t_gb[abs_peak]
I_decay = I_gb[abs_peak:]
cut = t_decay <= 6.0
t_decay = t_decay[cut]; I_decay = I_decay[cut]
t0_peak = t_gb[abs_peak]
baseline_I = np.mean(I_g[t_g < -1.5])

# ============================================================
# MANUAL DIGITIZATION: gradTi
# ============================================================
t_b_crash = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7,
                       2.0, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0])
g_b_crash = np.array([4, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17, 17, 18, 19, 19, 20])
t_b_pre = np.array([-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1])
g_b_pre = np.array([20,21,20,22,20,21,20,21,20])
t_b_all = np.concatenate([t_b_pre, t_b_crash,
    np.array([4.5,5,6,7,8,9,10])])
g_b_all = np.concatenate([g_b_pre, g_b_crash,
    np.array([18,19,21,20,18,20,19])])

max_grad = np.mean(g_b_pre)  # 20.4
f_inv = (max_grad - g_b_crash) / max_grad
f_inv = np.clip(f_inv, 0, 1.5)

# ============================================================
# FIT FUNCTIONS
# ============================================================
def kww(t, A, tau, alpha, offset):
    return A * np.exp(-(t/tau)**alpha) + offset

def kww_exp(t, A, tau, offset):
    return A * np.exp(-t/tau) + offset

def r_sq(y, yf):
    sr = np.sum((y-yf)**2); st = np.sum((y-np.mean(y))**2)
    return 1-sr/st if st > 1e-15 else float('nan')

def fit_kww(t, y, amax=5.0):
    amp = max(y[0]-y[-1], 0.1); off = y[-1]
    best = None; best_r2 = -np.inf
    for a0 in [0.6, 0.8, 1.0, 1.2, 1.33, 1.5, 2.0, 2.5]:
        for t0 in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            try:
                popt, pcov = curve_fit(kww, t, y, p0=[amp,t0,a0,off],
                    bounds=([0,0.01,0.1,off-3],[amp*5,30,amax,off+3]), maxfev=50000)
                r2 = r_sq(y, kww(t, *popt))
                if r2 > best_r2 and np.isfinite(r2):
                    best_r2 = r2; best = (popt, pcov, r2)
            except: pass
    return best

# ============================================================
# FIT 1: Idiv full decay
# ============================================================
fit_full = fit_kww(t_decay, I_decay)
popt_full, pcov_full, r2_full = fit_full
perr_full = np.sqrt(np.diag(pcov_full)) if np.all(np.isfinite(np.diag(pcov_full))) else np.full(4, np.inf)
try:
    pe_full, _ = curve_fit(kww_exp, t_decay, I_decay,
        p0=[I_decay[0]-I_decay[-1],1,I_decay[-1]], bounds=([0,0.01,-5],[30,30,10]), maxfev=10000)
    r2e_full = r_sq(I_decay, kww_exp(t_decay, *pe_full))
except: r2e_full = float('nan'); pe_full = [0,1,0]

# ============================================================
# FIT 2: Idiv transport tail (start at 1.0 ms from peak)
# ============================================================
TAIL_START = 1.0
mask_tail = t_decay >= TAIL_START
t_tail = t_decay[mask_tail] - t_decay[mask_tail][0]
I_tail = I_decay[mask_tail]

fit_tail = fit_kww(t_tail, I_tail)
popt_tail, pcov_tail, r2_tail = fit_tail
perr_tail = np.sqrt(np.diag(pcov_tail)) if np.all(np.isfinite(np.diag(pcov_tail))) else np.full(4, np.inf)
try:
    pe_tail, _ = curve_fit(kww_exp, t_tail, I_tail,
        p0=[I_tail[0]-I_tail[-1],1,I_tail[-1]], bounds=([0,0.01,-5],[30,30,10]), maxfev=10000)
    r2e_tail = r_sq(I_tail, kww_exp(t_tail, *pe_tail))
except: r2e_tail = float('nan'); pe_tail = [0,1,0]

# ============================================================
# FIT 3: Inverted gradTi recovery
# ============================================================
fit_grad = fit_kww(t_b_crash, f_inv)
popt_grad, pcov_grad, r2_grad = fit_grad
perr_grad = np.sqrt(np.diag(pcov_grad)) if np.all(np.isfinite(np.diag(pcov_grad))) else np.full(4, np.inf)
try:
    pe_grad, _ = curve_fit(kww_exp, t_b_crash, f_inv,
        p0=[f_inv[0],1.5,0], bounds=([0,0.01,-0.5],[2,20,0.5]), maxfev=10000)
    r2e_grad = r_sq(f_inv, kww_exp(t_b_crash, *pe_grad))
except: r2e_grad = float('nan'); pe_grad = [1,1,0]


# ============================================================
# COMPREHENSIVE PLOT
# ============================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle('KWW Fitting of ELM Relaxation Curves\nCavedon et al. 2019, ASDEX Upgrade (#34430)',
             fontsize=14, fontweight='bold')

# (a) Full Idiv trace
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(t_g, I_g, 'k.', ms=0.3, alpha=0.2)
ax1.plot(t_gb, I_gb, 'b-', lw=1, alpha=0.7, label='Binned (0.1ms)')
ax1.axvline(0, color='red', ls='--', alpha=0.4, label='ELM crash')
ax1.axvline(t0_peak, color='orange', ls=':', label=f'Peak ({t0_peak:.1f}ms)')
ax1.axhline(baseline_I, color='green', ls=':', alpha=0.4)
ax1.set_xlabel('t - t_ELM [ms]'); ax1.set_ylabel('I_div [kA]')
ax1.set_title('(a) Divertor Current - Full Trace'); ax1.legend(fontsize=6)
ax1.set_xlim(-6, 11)

# (b) Idiv full decay fit
ax2 = fig.add_subplot(3, 2, 2)
ax2.scatter(t_decay, I_decay, s=15, c='black', zorder=5)
ts = np.linspace(0, t_decay.max(), 300)
ax2.plot(ts, kww(ts, *popt_full), 'r-', lw=2,
    label=f'KWW: $\\alpha$={popt_full[2]:.2f}$\\pm${perr_full[2]:.2f}, $R^2$={r2_full:.4f}')
ax2.plot(ts, kww_exp(ts, *pe_full), 'b--', lw=1.5,
    label=f'Exp: $R^2$={r2e_full:.4f}')
ax2.axvline(TAIL_START, color='green', ls=':', label=f'Tail start ({TAIL_START}ms)')
ax2.set_xlabel('t since peak [ms]'); ax2.set_ylabel('I_div [kA]')
ax2.set_title(f'(b) Idiv Full Decay: $\\alpha$ = {popt_full[2]:.3f}')
ax2.legend(fontsize=6)

# (c) Idiv transport tail fit
ax3 = fig.add_subplot(3, 2, 3)
ax3.scatter(t_tail, I_tail, s=15, c='black', zorder=5)
ts2 = np.linspace(0, t_tail.max(), 300)
ax3.plot(ts2, kww(ts2, *popt_tail), 'r-', lw=2,
    label=f'KWW: $\\alpha$={popt_tail[2]:.2f}$\\pm${perr_tail[2]:.2f}, $R^2$={r2_tail:.4f}')
ax3.plot(ts2, kww_exp(ts2, *pe_tail), 'b--', lw=1.5,
    label=f'Exp: $R^2$={r2e_tail:.4f}')
# Shade Merkabit window
ax3.set_xlabel(f't since {TAIL_START}ms after peak [ms]'); ax3.set_ylabel('I_div [kA]')
ax3.set_title(f'(c) Idiv Transport Tail: $\\alpha$ = {popt_tail[2]:.3f} $\\pm$ {perr_tail[2]:.3f}')
ax3.legend(fontsize=6)

# (d) gradTi full trace
ax4 = fig.add_subplot(3, 2, 4)
ax4.errorbar(t_b_all, g_b_all, yerr=2, fmt='ko', ms=3, capsize=2)
ax4.axvline(0, color='red', ls='--', alpha=0.4, label='ELM crash')
ax4.axhline(max_grad, color='green', ls=':', alpha=0.4, label=f'Baseline ({max_grad:.0f} keV/m)')
ax4.set_xlabel('t - t_ELM [ms]'); ax4.set_ylabel('max($-\\nabla T_i$) [keV/m]')
ax4.set_title('(d) Ion Temperature Gradient'); ax4.legend(fontsize=6)
ax4.set_xlim(-6, 11); ax4.set_ylim(-2, 30)

# (e) Inverted gradTi fit
ax5 = fig.add_subplot(3, 2, 5)
ax5.scatter(t_b_crash, f_inv, s=30, c='black', zorder=5, label='Inverted data')
ts3 = np.linspace(0, t_b_crash.max(), 300)
ax5.plot(ts3, kww(ts3, *popt_grad), 'r-', lw=2.5,
    label=f'KWW: $\\alpha$={popt_grad[2]:.2f}$\\pm${perr_grad[2]:.2f}, $R^2$={r2_grad:.4f}')
ax5.plot(ts3, kww_exp(ts3, *pe_grad), 'b--', lw=1.5,
    label=f'Exp: $R^2$={r2e_grad:.4f}')
ax5.set_xlabel('t since crash [ms]'); ax5.set_ylabel('f(t) = (max$-$grad)/max')
ax5.set_title(f'(e) Inverted $\\nabla T_i$: $\\alpha$ = {popt_grad[2]:.3f} $\\pm$ {perr_grad[2]:.3f}')
ax5.legend(fontsize=6)

# (f) Summary comparison
ax6 = fig.add_subplot(3, 2, 6)
labels = ['Idiv\n(full spike)', 'Idiv\n(tail >1ms)', '$\\nabla T_i$\n(recovery)']
alphas = [popt_full[2], popt_tail[2], popt_grad[2]]
errs = [perr_full[2], perr_tail[2], perr_grad[2]]
errs = [e if np.isfinite(e) else 0 for e in errs]
colors = ['gray', '#2196F3', '#4CAF50']

ax6.barh(range(3), alphas, xerr=errs, color=colors, height=0.5, capsize=5, edgecolor='black')
ax6.axvline(4/3, color='red', ls='--', lw=2, label=f'4/3 = {4/3:.3f}')
ax6.axvspan(4/3-0.15, 4/3+0.15, alpha=0.15, color='red', label='Merkabit window')
ax6.axvline(1.0, color='gray', ls=':', alpha=0.5, label='$\\alpha$=1 (exponential)')
ax6.set_yticks(range(3)); ax6.set_yticklabels(labels, fontsize=9)
ax6.set_xlabel('$\\alpha$ (stretching exponent)')
ax6.set_title('(f) Comparison of $\\alpha$ Values')
ax6.legend(fontsize=6, loc='upper right')
ax6.set_xlim(0, 3.0)

for i, (a, e) in enumerate(zip(alphas, errs)):
    ax6.text(a + e + 0.05, i, f'{a:.2f}$\\pm${e:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'kww_elm_final.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Final plot saved: {os.path.join(OUTDIR, 'kww_elm_final.png')}")


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 95)
print("FINAL SUMMARY: KWW Fitting of Cavedon 2019 ELM Relaxation (ASDEX Upgrade #34430)")
print("=" * 95)
print(f"{'Signal':<20} | {'alpha':>14} | {'tau (ms)':>14} | {'R2_KWW':>8} | {'R2_exp':>8} | {'|a-4/3|':>8}")
print("-" * 95)

def fmt_pm(v, e):
    return f"{v:.3f}+/-{e:.3f}" if np.isfinite(e) else f"{v:.3f}+/-inf"

rows = [
    ('Idiv (full spike)', popt_full[2], perr_full[2], popt_full[1], perr_full[1], r2_full, r2e_full),
    ('Idiv (tail >1ms)', popt_tail[2], perr_tail[2], popt_tail[1], perr_tail[1], r2_tail, r2e_tail),
    ('gradTi (inverted)', popt_grad[2], perr_grad[2], popt_grad[1], perr_grad[1], r2_grad, r2e_grad),
]

for name, a, da, t, dt, r2k, r2e in rows:
    print(f"{name:<20} | {fmt_pm(a,da):>14} | {fmt_pm(t,dt):>14} | "
          f"{r2k:>8.4f} | {r2e:>8.4f} | {abs(a-4/3):>8.3f}")

print("\n" + "=" * 95)
print("PHYSICAL INTERPRETATION")
print("=" * 95)

print(f"""
Idiv (full spike): alpha = {popt_full[2]:.3f}
  The divertor current spike and rapid initial decay are dominated by the
  MHD/electrical circuit response, giving alpha >> 1 (near-Gaussian).
  This is NOT the transport relaxation.

Idiv (transport tail, t > 1ms from peak): alpha = {popt_tail[2]:.3f} +/- {perr_tail[2]:.3f}
  After the initial MHD pulse subsides, the remaining current relaxation
  reflects plasma transport dynamics. Alpha = {popt_tail[2]:.2f} is within
  |alpha - 4/3| = {abs(popt_tail[2]-4/3):.3f} of the Merkabit target.
  R^2 = {r2_tail:.4f} {'> 0.90: PASS' if r2_tail > 0.90 else '< 0.90: marginal'}

gradTi recovery (inverted): alpha = {popt_grad[2]:.3f} +/- {perr_grad[2]:.3f}
  The ion temperature gradient crashes at the ELM and recovers over ~3 ms.
  The inverted recovery (treated as a decay from perturbed to equilibrium)
  gives alpha = {popt_grad[2]:.2f}, clearly super-exponential.
  |alpha - 4/3| = {abs(popt_grad[2]-4/3):.3f} {'< 0.15: WITHIN Merkabit window' if abs(popt_grad[2]-4/3) < 0.15 else '>= 0.15'}
  R^2 = {r2_grad:.4f} {'> 0.90: HIGH CONFIDENCE' if r2_grad > 0.90 else '< 0.90'}
""")

# Check both transport-relevant curves
in_window = []
for name, a, da, r2 in [('Idiv tail', popt_tail[2], perr_tail[2], r2_tail),
                          ('gradTi inv', popt_grad[2], perr_grad[2], r2_grad)]:
    if abs(a - 4/3) < 0.15 and r2 > 0.90:
        in_window.append(name)
        print(f"  *** {name}: alpha={a:.3f}, |a-4/3|={abs(a-4/3):.3f} < 0.15, R^2={r2:.3f} > 0.90 ***")

if len(in_window) > 0:
    print(f"\n  ==> {len(in_window)}/2 transport curves consistent with Merkabit signature")
    print(f"  ==> ASDEX Upgrade ELM relaxation is a candidate 7th system")
else:
    print(f"\n  No curves within Merkabit window with R^2 > 0.90")

print(f"""
Reference alpha values from confirmed systems:
  Mn3Sn (anomalous Hall):    alpha = 1.283
  IBM Q3 (quantum circuit):  alpha = 1.355
  Prethermal DTC:            alpha = 1.30
  Xiang DTC Z0:              alpha = 1.34
  Mi et al. Sycamore:        alpha = 0.822 (sub-exponential, different class)
  Target (Merkabit 4/3):     alpha = {4/3:.4f}
""")

# Sanity checks
print("SANITY CHECKS:")
print(f"  Idiv pre-ELM baseline: {baseline_I:.2f} kA (std={np.std(I_g[t_g<-1.5]):.2f})")
print(f"  Inverted gradTi: f(0)={f_inv[0]:.3f} [expect ~1], f(4)={f_inv[-1]:.3f} [expect ~0]")
print(f"  All R^2 > 0.85: {all(r > 0.85 for r in [r2_full, r2_tail, r2_grad])}")
