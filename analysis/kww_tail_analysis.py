"""
Tail analysis: fit Idiv decay starting from different time offsets
using ACTUAL digitized data to see if transport tail differs from spike.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DATA_DIR, FIGURES_DIR

OUTDIR = FIGURES_DIR
img = np.array(Image.open(os.path.join(DATA_DIR, 'figure2_embedded.png')))
gray = np.mean(img[:,:,:3], axis=2)

# Calibration
x_cal = np.polyfit([150, 365, 580, 795], [-5.0, 0.0, 5.0, 10.0], 1)
yg_cal = np.polyfit([2145, 2045, 1960], [0.0, 8.0, 16.0], 1)
def px_to_t(x): return np.polyval(x_cal, x)
def px_to_I(y): return np.polyval(yg_cal, y)

# Digitize Idiv (same as v4)
t_list, I_list = [], []
for x in range(155, 795, 1):
    if 360 <= x <= 372: continue
    y_start = 1990 if x < 195 else (1995 if x > 695 else 1950)
    col = gray[y_start:2170, x]
    dark = np.where(col < 85)[0]
    if len(dark) < 1: continue
    gaps = np.diff(dark) if len(dark) > 1 else np.array([])
    splits = np.where(gaps > 5)[0] if len(gaps) > 0 else np.array([])
    if len(splits) == 0:
        y_center = np.mean(dark) + y_start
    else:
        clusters = np.split(dark, splits + 1)
        largest = max(clusters, key=len)
        y_center = np.mean(largest) + y_start
    t_list.append(px_to_t(x)); I_list.append(px_to_I(y_center))

t_g = np.array(t_list); I_g = np.array(I_list)

# Bin at 0.1 ms
edges = np.arange(t_g.min(), t_g.max()+0.1, 0.1)
t_b, I_b = [], []
for i in range(len(edges)-1):
    m = (t_g >= edges[i]) & (t_g < edges[i+1])
    if np.sum(m) > 2:
        t_b.append(np.mean(t_g[m])); I_b.append(np.mean(I_g[m]))
t_b = np.array(t_b); I_b = np.array(I_b)

# Find peak
post = t_b > 0
pi = np.argmax(I_b[post])
abs_peak = np.where(post)[0][0] + pi
t_decay = t_b[abs_peak:] - t_b[abs_peak]
I_decay = I_b[abs_peak:]

# Limit to 6 ms
cut = t_decay <= 6.0
t_decay = t_decay[cut]; I_decay = I_decay[cut]

print(f"Peak at t_ELM + {t_b[abs_peak]:.2f} ms, I_peak = {I_decay[0]:.2f} kA")
print(f"Decay data: {len(t_decay)} points over {t_decay[-1]:.1f} ms")

def kww(t, A, tau, alpha, offset):
    return A * np.exp(-(t/tau)**alpha) + offset

def kww_exp(t, A, tau, offset):
    return A * np.exp(-t/tau) + offset

def r_sq(y, yf):
    sr = np.sum((y-yf)**2); st = np.sum((y-np.mean(y))**2)
    return 1-sr/st if st > 1e-15 else float('nan')

def fit_kww(t, y):
    amp = max(y[0]-y[-1], 0.1); off = y[-1]
    best = None; best_r2 = -np.inf
    for a0 in [0.6, 0.8, 1.0, 1.2, 1.33, 1.5, 2.0, 2.5]:
        for t0 in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            try:
                popt, pcov = curve_fit(kww, t, y, p0=[amp,t0,a0,off],
                    bounds=([0,0.01,0.1,off-3],[amp*5,30,5,off+3]), maxfev=50000)
                r2 = r_sq(y, kww(t, *popt))
                if r2 > best_r2 and np.isfinite(r2):
                    best_r2 = r2; best = (popt, pcov, r2)
            except: pass
    return best

# Test different starting points
print(f"\n{'Start':>8} {'Npts':>6} {'alpha':>8} {'+-':>6} {'tau':>8} {'+-':>6} {'R2':>8} {'R2exp':>8} {'|a-4/3|':>8}")
print("-" * 75)

tail_results = []
for t_start in [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
    mask = t_decay >= t_start
    t_sub = t_decay[mask] - t_decay[mask][0]
    I_sub = I_decay[mask]
    if len(t_sub) < 5: continue

    result = fit_kww(t_sub, I_sub)
    if result:
        popt, pcov, r2 = result
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(np.diag(pcov))) else np.full(4, np.inf)
        # Exp fit
        try:
            pe, _ = curve_fit(kww_exp, t_sub, I_sub, p0=[I_sub[0]-I_sub[-1],1,I_sub[-1]],
                bounds=([0,0.01,-5],[30,30,10]), maxfev=10000)
            r2e = r_sq(I_sub, kww_exp(t_sub, *pe))
        except: r2e = float('nan')

        da = f"{perr[2]:.3f}" if np.isfinite(perr[2]) else "inf"
        dt = f"{perr[1]:.3f}" if np.isfinite(perr[1]) else "inf"
        print(f"{t_start:>8.1f} {len(t_sub):>6d} {popt[2]:>8.3f} {da:>6} {popt[1]:>8.3f} {dt:>6} "
              f"{r2:>8.4f} {r2e:>8.4f} {abs(popt[2]-4/3):>8.3f}")
        tail_results.append((t_start, popt[2], perr[2], popt[1], r2, r2e))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Idiv Decay: Sensitivity to Fit Start Time', fontsize=12, fontweight='bold')

# Plot 1: Full decay with different start regions
ax = axes[0]
ax.plot(t_decay, I_decay, 'k.', ms=3, label='Data')
for ts in [0.0, 0.5, 1.0, 1.5]:
    ax.axvline(ts, ls=':', alpha=0.5)
ax.set_xlabel('t since peak [ms]'); ax.set_ylabel('I_div [kA]')
ax.set_title('Decay data with start markers'); ax.legend(fontsize=8)

# Plot 2: Alpha vs start time
ax = axes[1]
tr = np.array(tail_results)
ax.errorbar(tr[:,0], tr[:,1], yerr=tr[:,2], fmt='ko-', capsize=3)
ax.axhline(4/3, color='red', ls='--', label='4/3 = 1.333')
ax.axhspan(4/3-0.15, 4/3+0.15, alpha=0.1, color='red', label='Merkabit window')
ax.set_xlabel('Fit start time [ms]'); ax.set_ylabel('alpha')
ax.set_title('Alpha vs start time'); ax.legend(fontsize=8)
ax.set_ylim(0.5, 3.0)

# Plot 3: R^2 comparison
ax = axes[2]
ax.plot(tr[:,0], tr[:,4], 'ro-', label='KWW R^2')
ax.plot(tr[:,0], tr[:,5], 'b^--', label='Exp R^2')
ax.axhline(0.90, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Fit start time [ms]'); ax.set_ylabel('R^2')
ax.set_title('Fit quality vs start time'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'idiv_tail_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {os.path.join(OUTDIR, 'idiv_tail_analysis.png')}")
