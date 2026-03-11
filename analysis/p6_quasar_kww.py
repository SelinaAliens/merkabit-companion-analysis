#!/usr/bin/env python3
"""
P6 — Quasar Structure Function KWW Analysis (SDSS Stripe 82 + ZTF)
Merkabit Prediction P6: alpha ~ 4/3 for quasars near lambda_Edd ~ 1

Computes first-order structure functions and KWW fits on quasar light curves
selected by p6_super_edd.py.  If light curve data is unavailable, generates
synthetic DRW (damped random walk) demo light curves.

Data sources:
  - SDSS Stripe 82 multi-epoch photometry
  - ZTF DR light curves (optional, via IRSA)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DATA_DIR, FIGURES_DIR, REPORTS_DIR

import warnings
import json
import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

QUASAR_DATA = os.path.join(DATA_DIR, 'quasars')
os.makedirs(QUASAR_DATA, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

ALPHA_TARGET = 4.0 / 3.0
ALPHA_WINDOW = (1.183, 1.483)


# ─────────────────────────────────────────────────────────────────
# KWW functions
# ─────────────────────────────────────────────────────────────────
def kww_sf(tau, SF_inf, tau_char, alpha, offset):
    """KWW structure function: SF_inf * (1 - exp(-(tau/tau_char)^alpha)) + offset"""
    return SF_inf * (1.0 - np.exp(-(tau / tau_char)**alpha)) + offset


def kww_acf(tau, A, tau_char, alpha, offset):
    """KWW ACF: A * exp(-(tau/tau_char)^alpha) + offset"""
    return A * np.exp(-(tau / tau_char)**alpha) + offset


def compute_structure_function(times, mags, mag_errs, tau_grid):
    """Compute first-order structure function from irregularly sampled light curve."""
    n = len(times)
    sf_vals = np.full(len(tau_grid), np.nan)
    sf_counts = np.zeros(len(tau_grid), dtype=int)

    # Bin width: 20% of tau (log-spaced)
    for k, tau in enumerate(tau_grid):
        tau_lo = tau * 0.8
        tau_hi = tau * 1.2
        diffs_sq = []
        for i in range(n):
            for j in range(i + 1, n):
                dt = abs(times[j] - times[i])
                if tau_lo <= dt <= tau_hi:
                    dm2 = (mags[j] - mags[i])**2 - mag_errs[i]**2 - mag_errs[j]**2
                    diffs_sq.append(max(dm2, 0))
        if len(diffs_sq) >= 3:
            sf_vals[k] = np.sqrt(np.mean(diffs_sq))
            sf_counts[k] = len(diffs_sq)

    return sf_vals, sf_counts


def compute_acf_irregular(times, mags, tau_grid):
    """Compute autocorrelation function for irregularly sampled data."""
    mean_mag = np.mean(mags)
    dm = mags - mean_mag
    var = np.var(mags)
    if var == 0:
        return np.full(len(tau_grid), np.nan)

    acf_vals = np.full(len(tau_grid), np.nan)
    n = len(times)

    for k, tau in enumerate(tau_grid):
        tau_lo = tau * 0.8
        tau_hi = tau * 1.2
        prods = []
        for i in range(n):
            for j in range(i + 1, n):
                dt = abs(times[j] - times[i])
                if tau_lo <= dt <= tau_hi:
                    prods.append(dm[i] * dm[j])
        if len(prods) >= 3:
            acf_vals[k] = np.mean(prods) / var

    return acf_vals


def fit_kww_sf(tau_grid, sf_vals):
    """Fit KWW to structure function."""
    valid = np.isfinite(sf_vals) & (tau_grid > 0) & (sf_vals > 0)
    if np.sum(valid) < 4:
        return None

    tau_fit = tau_grid[valid]
    sf_fit = sf_vals[valid]
    sf_max = np.max(sf_fit)
    tau_half = tau_fit[np.argmin(np.abs(sf_fit - 0.5 * sf_max))]

    try:
        popt, pcov = curve_fit(
            kww_sf, tau_fit, sf_fit,
            p0=[sf_max, max(tau_half, 10), 1.3, 0.0],
            bounds=([0, 1, 0.3, -0.1], [sf_max * 3, 1e5, 3.0, sf_max]),
            maxfev=10000
        )
        yfit = kww_sf(tau_fit, *popt)
        ss_res = np.sum((sf_fit - yfit)**2)
        ss_tot = np.sum((sf_fit - np.mean(sf_fit))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {
            'SF_inf': popt[0], 'tau_char': popt[1], 'alpha': popt[2],
            'offset': popt[3], 'R2': r2, 'alpha_err': perr[2],
            'tau_fit': tau_fit, 'sf_fit': sf_fit, 'sf_model': yfit,
        }
    except (RuntimeError, ValueError):
        return None


def fit_kww_acf(tau_grid, acf_vals):
    """Fit KWW to ACF."""
    valid = np.isfinite(acf_vals) & (tau_grid > 0)
    if np.sum(valid) < 4:
        return None

    tau_fit = tau_grid[valid]
    acf_fit = acf_vals[valid]
    tau_half = tau_fit[max(0, np.argmin(np.abs(acf_fit - 0.5 * acf_fit[0])))]

    try:
        popt, pcov = curve_fit(
            kww_acf, tau_fit, acf_fit,
            p0=[1.0, max(tau_half, 10), 1.3, 0.0],
            bounds=([0, 1, 0.3, -0.5], [2.0, 1e5, 3.0, 0.5]),
            maxfev=10000
        )
        yfit = kww_acf(tau_fit, *popt)
        ss_res = np.sum((acf_fit - yfit)**2)
        ss_tot = np.sum((acf_fit - np.mean(acf_fit))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return {
            'A': popt[0], 'tau_char': popt[1], 'alpha': popt[2],
            'offset': popt[3], 'R2': r2, 'alpha_err': perr[2],
            'tau_fit': tau_fit, 'acf_fit': acf_fit, 'acf_model': yfit,
        }
    except (RuntimeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────
def load_sample():
    """Load near-Eddington quasar sample from p6_super_edd.py output."""
    sample_path = os.path.join(QUASAR_DATA, 'super_edd_sample.csv')
    if not os.path.exists(sample_path):
        print("  Sample file not found. Run p6_super_edd.py first.")
        return None

    quasars = []
    with open(sample_path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            vals = line.strip().split(',')
            if len(vals) >= 9:
                quasars.append({
                    'name': vals[0],
                    'ra': float(vals[1]),
                    'dec': float(vals[2]),
                    'z': float(vals[3]),
                    'Mi': float(vals[4]),
                    'logMBH': float(vals[5]),
                    'logLbol': float(vals[6]),
                    'lambda_edd': float(vals[7]),
                    'log_lambda_edd': float(vals[8]),
                })
    return quasars


def fetch_sdss_lightcurve(ra, dec):
    """Try to fetch SDSS Stripe 82 light curve for a quasar."""
    try:
        import urllib.request
        url = (f"https://cas.sdss.org/stripe82/en/tools/search/"
               f"x_results.aspx?searchtool=SQL&TaskName=Skyserver&"
               f"cmd=SELECT+mjd_r,psfmag_r,psfmagerr_r+"
               f"FROM+RunFieldQuality+WHERE+"
               f"ra+BETWEEN+{ra-0.001}+AND+{ra+0.001}+AND+"
               f"dec+BETWEEN+{dec-0.001}+AND+{dec+0.001}"
               f"&format=csv")
        response = urllib.request.urlopen(url, timeout=10)
        text = response.read().decode()
        lines = text.strip().split('\n')
        if len(lines) > 2:
            times, mags, errs = [], [], []
            for row in lines[1:]:
                parts = row.split(',')
                if len(parts) >= 3:
                    try:
                        times.append(float(parts[0]))
                        mags.append(float(parts[1]))
                        errs.append(float(parts[2]))
                    except ValueError:
                        continue
            if len(times) >= 10:
                return np.array(times), np.array(mags), np.array(errs)
    except Exception:
        pass
    return None, None, None


# ─────────────────────────────────────────────────────────────────
# Synthetic light curves (DRW / CAR(1) process)
# ─────────────────────────────────────────────────────────────────
def generate_drw_lightcurve(tau_drw, sf_inf, n_epochs=60, baseline=3000,
                            mag_err=0.02, seed=None):
    """
    Generate a DRW (damped random walk, CAR(1)) light curve.
    Standard model for quasar optical variability (Kelly et al. 2009).

    Parameters:
      tau_drw: characteristic timescale (days)
      sf_inf: long-term structure function amplitude (mag)
      n_epochs: number of observations
      baseline: total observing span (days)
      mag_err: photometric uncertainty
    """
    rng = np.random.default_rng(seed)

    # Irregular time sampling (mimic S82 cadence)
    times = np.sort(rng.uniform(0, baseline, n_epochs))

    # DRW variance: sigma^2 = SF_inf^2 / 2
    sigma_drw = sf_inf / np.sqrt(2)

    # Generate DRW process
    mags = np.zeros(n_epochs)
    mags[0] = rng.normal(0, sigma_drw)
    for i in range(1, n_epochs):
        dt = times[i] - times[i - 1]
        decay = np.exp(-dt / tau_drw)
        noise_var = sigma_drw**2 * (1 - decay**2)
        mags[i] = mags[i - 1] * decay + rng.normal(0, np.sqrt(max(noise_var, 1e-10)))

    # Add photometric noise
    mag_errs = np.full(n_epochs, mag_err)
    mags_obs = mags + rng.normal(0, mag_err, n_epochs)

    return times, mags_obs, mag_errs


def generate_synthetic_sample(quasars):
    """Generate synthetic DRW light curves for the sample."""
    print("\n  Generating synthetic DRW light curves (demo mode)")
    print("  DRW model: tau_drw ~ 200-600 days, SF_inf ~ 0.1-0.3 mag")

    rng = np.random.default_rng(seed=123)
    results = []

    for i, q in enumerate(quasars):
        lam = q['lambda_edd']

        # Tau and SF_inf depend on lambda_Edd (physical scaling)
        # Near-Edd quasars have shorter variability timescales
        tau_drw = 400 * (1 + q['z'])**0.5 / max(lam, 0.1)**0.3
        sf_inf = 0.15 * max(lam, 0.1)**0.2

        times, mags, errs = generate_drw_lightcurve(
            tau_drw, sf_inf, n_epochs=55, baseline=2800,
            mag_err=0.02, seed=rng.integers(0, 100000)
        )

        results.append({
            'quasar': q,
            'times': times,
            'mags': mags,
            'mag_errs': errs,
            'tau_drw_true': tau_drw,
            'sf_inf_true': sf_inf,
            'is_synthetic': True,
        })

    return results


# ─────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────
def analyze_quasar(lc_data, tau_grid):
    """Compute SF and ACF, fit KWW, return results."""
    times = lc_data['times']
    mags = lc_data['mags']
    errs = lc_data['mag_errs']
    q = lc_data['quasar']

    # Structure function
    sf_vals, sf_counts = compute_structure_function(times, mags, errs, tau_grid)
    sf_fit = fit_kww_sf(tau_grid, sf_vals)

    # ACF
    acf_vals = compute_acf_irregular(times, mags, tau_grid)
    acf_fit = fit_kww_acf(tau_grid, acf_vals)

    result = {
        'name': q['name'],
        'z': q['z'],
        'lambda_edd': q['lambda_edd'],
        'log_lambda_edd': q['log_lambda_edd'],
        'logMBH': q['logMBH'],
        'n_epochs': len(times),
        'baseline': float(times[-1] - times[0]),
    }

    if sf_fit:
        result['sf_alpha'] = sf_fit['alpha']
        result['sf_alpha_err'] = sf_fit['alpha_err']
        result['sf_tau'] = sf_fit['tau_char']
        result['sf_R2'] = sf_fit['R2']
    else:
        result['sf_alpha'] = None

    if acf_fit:
        result['acf_alpha'] = acf_fit['alpha']
        result['acf_alpha_err'] = acf_fit['alpha_err']
        result['acf_tau'] = acf_fit['tau_char']
        result['acf_R2'] = acf_fit['R2']
    else:
        result['acf_alpha'] = None

    return result, sf_vals, sf_fit, acf_vals, acf_fit


if __name__ == '__main__':
    print("=" * 70)
    print("P6 -- Quasar Structure Function KWW Analysis")
    print("Prediction: alpha ~ 4/3 near lambda_Edd ~ 1")
    print("=" * 70)

    # Load sample
    quasars = load_sample()
    if not quasars:
        print("  No sample file. Generating fallback sample...")
        from p6_super_edd import FALLBACK_SAMPLE, compute_eddington_ratio, use_fallback
        names, ra, dec, z, mi, logmbh, loglbol = use_fallback()
        lam, log_lam = compute_eddington_ratio(logmbh, loglbol)
        quasars = [{'name': names[i], 'ra': ra[i], 'dec': dec[i],
                     'z': z[i], 'Mi': mi[i], 'logMBH': logmbh[i],
                     'logLbol': loglbol[i], 'lambda_edd': lam[i],
                     'log_lambda_edd': log_lam[i]}
                    for i in range(len(names))]

    print(f"  Sample: {len(quasars)} quasars")

    # Try to fetch real light curves
    lc_data_list = []
    n_real = 0
    print("\n  Attempting SDSS light curve downloads...")
    for q in quasars[:5]:
        t, m, e = fetch_sdss_lightcurve(q['ra'], q['dec'])
        if t is not None:
            lc_data_list.append({
                'quasar': q, 'times': t, 'mags': m, 'mag_errs': e,
                'is_synthetic': False,
            })
            n_real += 1

    if n_real < 3:
        print(f"  Only {n_real} real LCs fetched. Using synthetic DRW for full sample.")
        lc_data_list = generate_synthetic_sample(quasars)
        is_demo = True
    else:
        is_demo = False

    # Time lag grid (log-spaced, days)
    tau_grid = np.logspace(np.log10(10), np.log10(2500), 25)

    # Analyze each quasar
    print(f"\n  Analyzing {len(lc_data_list)} light curves...")
    all_results = []
    best_sf_fit = None
    best_acf_fit = None
    best_sf_vals = None
    best_acf_vals = None
    best_r2 = -1

    for i, lc in enumerate(lc_data_list):
        result, sf_vals, sf_fit, acf_vals, acf_fit = analyze_quasar(lc, tau_grid)
        all_results.append(result)

        a = result.get('sf_alpha')
        r2 = result.get('sf_R2', 0) or 0
        lam = result['lambda_edd']
        status = ''
        if a is not None and ALPHA_WINDOW[0] <= a <= ALPHA_WINDOW[1]:
            status = ' IN WINDOW'
        if a is not None:
            print(f"    {i+1:3d}/{len(lc_data_list)}  lam={lam:.3f}  "
                  f"SF_alpha={a:.3f}  R2={r2:.3f}{status}")

        if r2 > best_r2 and sf_fit is not None:
            best_r2 = r2
            best_sf_fit = sf_fit
            best_acf_fit = acf_fit
            best_sf_vals = sf_vals
            best_acf_vals = acf_vals

    # Bin by lambda_Edd
    bins = [
        ('Sub-Edd (<0.1)', 0, 0.1),
        ('Moderate (0.1-0.5)', 0.1, 0.5),
        ('Near-Edd (0.5-2)', 0.5, 2.0),
        ('Super-Edd (>2)', 2.0, 100),
    ]
    bin_results = {}
    for label, lo, hi in bins:
        alphas = [r['sf_alpha'] for r in all_results
                  if r['sf_alpha'] is not None and lo <= r['lambda_edd'] < hi]
        bin_results[label] = {
            'n': len(alphas),
            'mean': float(np.mean(alphas)) if alphas else None,
            'std': float(np.std(alphas)) if alphas else None,
            'median': float(np.median(alphas)) if alphas else None,
            'in_window': sum(1 for a in alphas if ALPHA_WINDOW[0] <= a <= ALPHA_WINDOW[1]),
        }

    # Summary
    lines = []
    lines.append("=" * 70)
    lines.append("P6 -- Quasar Structure Function KWW Analysis")
    lines.append(f"Data: {'SYNTHETIC DRW (demo)' if is_demo else 'SDSS Stripe 82'}")
    lines.append("=" * 70)
    lines.append(f"\nQuasars analyzed: {len(all_results)}")

    valid_sf = [r for r in all_results if r['sf_alpha'] is not None]
    lines.append(f"Valid SF fits: {len(valid_sf)}")

    lines.append(f"\n--- Results by Eddington ratio bin ---")
    lines.append(f"  {'Bin':<22} {'N':>4} {'mean_a':>8} {'std':>8} {'in_window':>10}")
    lines.append("  " + "-" * 56)
    for label, lo, hi in bins:
        b = bin_results[label]
        if b['n'] > 0:
            lines.append(f"  {label:<22} {b['n']:>4} {b['mean']:>8.3f} "
                         f"{b['std']:>8.3f} {b['in_window']:>10}")
        else:
            lines.append(f"  {label:<22} {b['n']:>4}     N/A      N/A        N/A")

    # Test prediction
    near_edd_alphas = [r['sf_alpha'] for r in all_results
                       if r['sf_alpha'] is not None
                       and 0.5 <= r['lambda_edd'] < 2.0]

    lines.append(f"\n--- Prediction P6 Test ---")
    if near_edd_alphas:
        mean_a = np.mean(near_edd_alphas)
        dev = abs(mean_a - ALPHA_TARGET)
        lines.append(f"  Near-Edd mean alpha: {mean_a:.3f} +/- {np.std(near_edd_alphas):.3f}")
        lines.append(f"  |mean - 4/3| = {dev:.3f}")
        if dev < 0.15:
            lines.append(f"  CONSISTENT with prediction (in cooperative window)")
        else:
            lines.append(f"  Outside cooperative window")
    else:
        lines.append(f"  No near-Eddington quasars with valid fits")

    if is_demo:
        lines.append(f"\n  *** SYNTHETIC DATA -- results are DRW demo only ***")
        lines.append(f"  Download real SDSS S82 / ZTF light curves for production test.")

    summary = '\n'.join(lines)
    print('\n' + summary)

    with open(os.path.join(REPORTS_DIR, 'p6_quasar_kww_summary.txt'), 'w') as f:
        f.write(summary)

    # Save JSON
    json_out = {
        'is_demo': is_demo,
        'n_quasars': len(all_results),
        'bin_results': bin_results,
        'per_quasar': [{k: v for k, v in r.items()} for r in all_results],
    }
    with open(os.path.join(REPORTS_DIR, 'p6_quasar_kww_results.json'), 'w') as f:
        json.dump(json_out, f, indent=2)

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Best SF with KWW fit
    ax = axes[0, 0]
    if best_sf_fit:
        ax.errorbar(best_sf_fit['tau_fit'], best_sf_fit['sf_fit'],
                     fmt='ko', markersize=5)
        tau_plot = np.logspace(np.log10(best_sf_fit['tau_fit'][0]),
                               np.log10(best_sf_fit['tau_fit'][-1]), 100)
        sf_model = kww_sf(tau_plot, best_sf_fit['SF_inf'], best_sf_fit['tau_char'],
                           best_sf_fit['alpha'], best_sf_fit['offset'])
        ax.plot(tau_plot, sf_model, 'r-', lw=2,
                label=f"alpha={best_sf_fit['alpha']:.3f}, R2={best_sf_fit['R2']:.3f}")
        ax.set_xscale('log')
        ax.legend(fontsize=9)
    ax.set_xlabel('Time lag (days)')
    ax.set_ylabel('SF (mag)')
    ax.set_title('Best Structure Function + KWW')

    # Panel 2: Best ACF with KWW fit
    ax = axes[0, 1]
    if best_acf_fit:
        ax.plot(best_acf_fit['tau_fit'], best_acf_fit['acf_fit'], 'ko', markersize=5)
        tau_plot = np.logspace(np.log10(best_acf_fit['tau_fit'][0]),
                               np.log10(best_acf_fit['tau_fit'][-1]), 100)
        acf_model = kww_acf(tau_plot, best_acf_fit['A'], best_acf_fit['tau_char'],
                             best_acf_fit['alpha'], best_acf_fit['offset'])
        ax.plot(tau_plot, acf_model, 'b-', lw=2,
                label=f"alpha={best_acf_fit['alpha']:.3f}, R2={best_acf_fit['R2']:.3f}")
        ax.set_xscale('log')
        ax.legend(fontsize=9)
    ax.set_xlabel('Time lag (days)')
    ax.set_ylabel('ACF')
    ax.set_title('Best ACF + KWW')

    # Panel 3: alpha vs lambda_Edd
    ax = axes[0, 2]
    for r in all_results:
        a = r.get('sf_alpha')
        if a is not None:
            c = 'green' if ALPHA_WINDOW[0] <= a <= ALPHA_WINDOW[1] else 'gray'
            ax.scatter(r['lambda_edd'], a, c=c, s=40, edgecolor='black', linewidth=0.5)
    ax.axhline(ALPHA_TARGET, color='red', ls='--', lw=2, label='4/3')
    ax.axhspan(ALPHA_WINDOW[0], ALPHA_WINDOW[1], alpha=0.1, color='green')
    ax.axvline(1.0, color='orange', ls=':', alpha=0.5, label='lam=1')
    ax.set_xscale('log')
    ax.set_xlabel('lambda_Edd')
    ax.set_ylabel('SF alpha')
    ax.set_title('alpha vs Eddington Ratio')
    ax.legend(fontsize=8)

    # Panel 4: Mean alpha per bin
    ax = axes[1, 0]
    bin_names = [b[0] for b in bins]
    bin_means = [bin_results[b[0]]['mean'] for b in bins]
    bin_stds = [bin_results[b[0]]['std'] for b in bins]
    bin_ns = [bin_results[b[0]]['n'] for b in bins]
    x_pos = np.arange(len(bins))
    valid_bars = [(i, m, s) for i, (m, s) in enumerate(zip(bin_means, bin_stds))
                  if m is not None]
    if valid_bars:
        xs = [v[0] for v in valid_bars]
        ms = [v[1] for v in valid_bars]
        ss = [v[2] for v in valid_bars]
        ax.bar(xs, ms, yerr=ss, color='steelblue', alpha=0.7, edgecolor='black', capsize=5)
    ax.axhline(ALPHA_TARGET, color='red', ls='--', lw=2, label='4/3')
    ax.axhspan(ALPHA_WINDOW[0], ALPHA_WINDOW[1], alpha=0.1, color='green')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n}\n(N={c})" for n, c in zip(bin_names, bin_ns)],
                        fontsize=7, rotation=15)
    ax.set_ylabel('Mean SF alpha')
    ax.set_title('Mean alpha per Eddington bin')

    # Panel 5: Alpha distribution (near-Edd)
    ax = axes[1, 1]
    if near_edd_alphas:
        ax.hist(near_edd_alphas, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(ALPHA_TARGET, color='red', ls='--', lw=2, label='4/3')
        ax.axvspan(ALPHA_WINDOW[0], ALPHA_WINDOW[1], alpha=0.1, color='green')
        ax.legend(fontsize=9)
    ax.set_xlabel('SF alpha')
    ax.set_ylabel('Count')
    ax.set_title('Near-Edd alpha distribution')

    # Panel 6: R2 distribution
    ax = axes[1, 2]
    r2_vals = [r['sf_R2'] for r in all_results if r.get('sf_R2') is not None]
    if r2_vals:
        ax.hist(r2_vals, bins=15, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(0.9, color='red', ls=':', lw=1.5, label='R2=0.9')
    ax.set_xlabel('R2')
    ax.set_ylabel('Count')
    ax.set_title('SF fit quality')
    ax.legend(fontsize=9)

    tag = 'SYNTHETIC DRW' if is_demo else 'SDSS S82'
    plt.suptitle(f'P6 -- Quasar KWW Structure Function Analysis ({tag})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p6_quasar_kww.png'), dpi=150)
    plt.close()
    print(f"\nSaved: p6_quasar_kww.png")
    print("P6 KWW analysis complete.")
