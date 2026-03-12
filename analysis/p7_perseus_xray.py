#!/usr/bin/env python3
"""
P7 Perseus Cluster X-ray Analysis: AGN Feedback Cycle
======================================================
Tests Prediction P7 from the Merkabit paper:
  "AGN feedback cycle shows alpha ~ 4/3"

The Perseus cluster (Abell 426) is the brightest X-ray cluster in the sky,
hosting the powerful AGN NGC 1275 (3C 84) at its center. The AGN inflates
bubbles in the intracluster medium (ICM), which rise buoyantly and deposit
energy, regulating cooling. This feedback cycle operates on ~10 Myr timescales.

Prediction: The X-ray surface brightness ACF should show alpha ~ 4/3 --
the cooperative cascade signature -- at the bubble radii (25-50"), but NOT
in the undisturbed ICM at larger radii (~100").

Chandra ObsIDs: 3209, 4289, 4952, 6139, 6145, 6146
  (deep ACIS-S observations, total ~900 ks)

Physical parameters:
  Perseus redshift: z = 0.01756
  Angular scale: 1" = 0.358 kpc (H0=70)
NGC 1275: RA=49.9507 deg, Dec=41.5117 deg (J2000)
  ICM temperature: ~4-6 keV
  AGN feedback timescale: ~10-30 Myr

March 2026
"""

import sys
import io
import os
import json
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import FIGURES_DIR, REPORTS_DIR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PERSEUS_DATA = os.path.join(DATA_DIR, 'perseus')

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import LogNorm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
ALPHA_TARGET = 4.0 / 3.0  # 1.33333...
ALPHA_WINDOW = (ALPHA_TARGET - 0.15, ALPHA_TARGET + 0.15)  # |alpha - 4/3| < 0.15

# Chandra ObsIDs for deep Perseus observations
OBSIDS = [3209, 4289, 4952, 6139, 6145, 6146]

# NGC 1275 position (J2000)
NGC1275_RA = 49.9507    # degrees (03h19m48.16s)
NGC1275_DEC = 41.5117   # degrees (+41d30'42.1")
# Perseus cluster physical parameters
REDSHIFT = 0.01756
KPC_PER_ARCSEC = 0.358  # at z=0.0176, H0=70
ENERGY_LO = 0.5         # keV, lower band
ENERGY_HI = 7.0         # keV, upper band
PIXEL_SIZE = 0.492       # Chandra ACIS pixel size in arcsec

# Radial profile parameters
R_MAX_ARCSEC = 300.0     # 5 arcmin
N_RADIAL_BINS = 60
AZIMUTHAL_BINS = 36      # 10-degree bins

# Radii of interest (arcsec)
R_INNER_BUBBLE = 25.0    # NW bubble region
R_OUTER_BUBBLE = 50.0    # Ghost bubble region
R_CONTROL_ICM = 100.0    # Undisturbed ICM
ANALYSIS_RADII = [15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]

# Beta-model parameters for synthetic ICM
BETA_MODEL_RC = 25.0     # core radius in arcsec
BETA_MODEL_BETA = 0.55   # slope parameter
BETA_MODEL_S0 = 5000.0   # central surface brightness (counts/arcsec^2)

# Bubble model parameters (observed morphology)
BUBBLE_NW = {
    'center_r': 25.0,     # arcsec from nucleus
    'center_pa': 330.0,   # position angle (degrees E of N)
    'semi_major': 15.0,   # arcsec
    'semi_minor': 10.0,   # arcsec
    'pa_ellipse': 30.0,   # orientation of ellipse
    'depth': 0.40,        # fractional depression
}
BUBBLE_SE = {
    'center_r': 20.0,
    'center_pa': 150.0,
    'semi_major': 12.0,
    'semi_minor': 8.0,
    'pa_ellipse': -30.0,
    'depth': 0.35,
}
GHOST_BUBBLE = {
    'center_r': 50.0,
    'center_pa': 190.0,
    'semi_major': 25.0,
    'semi_minor': 18.0,
    'pa_ellipse': 10.0,
    'depth': 0.20,
}


# ============================================================================
# KWW FITTING FUNCTIONS
# ============================================================================

def kww_acf(lag, A, tau, alpha):
    """KWW stretched/compressed exponential ACF: A * exp(-(lag/tau)^alpha)"""
    return A * np.exp(-(lag / tau)**alpha)


def compute_acf(signal, max_lag=None):
    """Compute normalized autocorrelation function of a 1D signal."""
    signal = signal - np.mean(signal)
    n = len(signal)
    if max_lag is None:
        max_lag = n // 2
    acf = np.zeros(max_lag)
    var = np.sum(signal**2)
    if var == 0:
        return acf
    for k in range(max_lag):
        acf[k] = np.sum(signal[:n - k] * signal[k:]) / var
    return acf


def fit_kww_to_acf(lags, acf_values, p0=None):
    """
    Fit KWW model to ACF data.
    Returns: (alpha, tau, A, R2, alpha_err, popt, pcov)
    """
    # Only fit where ACF is positive and above noise floor
    mask = (acf_values > 0.02) & (lags > 0)
    if np.sum(mask) < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan, None, None

    lags_fit = lags[mask]
    acf_fit = acf_values[mask]

    if p0 is None:
        # Estimate tau from half-decay
        half_idx = np.argmin(np.abs(acf_fit - 0.5 * acf_fit[0]))
        tau_est = lags_fit[half_idx] if half_idx > 0 else lags_fit[len(lags_fit) // 3]
        p0 = [acf_fit[0], max(tau_est, 1.0), 1.3]

    try:
        popt, pcov = curve_fit(
            kww_acf, lags_fit, acf_fit,
            p0=p0,
            bounds=([0.0, 0.1, 0.3], [2.0, 1e4, 3.0]),
            maxfev=10000
        )
        A_fit, tau_fit, alpha_fit = popt

        # Compute R-squared
        residuals = acf_fit - kww_acf(lags_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((acf_fit - np.mean(acf_fit))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Uncertainty on alpha
        alpha_err = np.sqrt(pcov[2, 2]) if pcov[2, 2] > 0 else np.nan

        return alpha_fit, tau_fit, A_fit, r2, alpha_err, popt, pcov

    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, np.nan, np.nan, None, None


# ============================================================================
# DATA ACQUISITION
# ============================================================================

def try_load_chandra_data():
    """
    Attempt to load Chandra event files for Perseus.
    Tries:
      1. Local FITS files in PERSEUS_DATA
      2. ciao_contrib download_chandra_obsid
      3. Direct HTTP download from CDA
    Returns: list of event data dicts, or None if unavailable
    """
    os.makedirs(PERSEUS_DATA, exist_ok=True)
    events_list = []

    # Check for existing FITS files
    for obsid in OBSIDS:
        fname = f"acisf{obsid:05d}_repro_evt2.fits"
        fpath = os.path.join(PERSEUS_DATA, fname)
        if os.path.isfile(fpath):
            try:
                from astropy.io import fits
                print(f"  Loading {fname}...")
                hdul = fits.open(fpath)
                evt = hdul['EVENTS'].data
                events_list.append({
                    'obsid': obsid,
                    'x': evt['x'],
                    'y': evt['y'],
                    'energy': evt['energy'],  # eV
                    'ra': evt.get('RA', None),
                    'dec': evt.get('DEC', None),
                })
                hdul.close()
            except Exception as e:
                print(f"  Warning: Could not load {fname}: {e}")
        else:
            print(f"  {fname} not found locally.")

    if events_list:
        return events_list

    # Try CIAO download
    print("\n  Attempting CIAO download...")
    try:
        from ciao_contrib.cda.data import download_chandra_obsid
        for obsid in OBSIDS[:1]:  # Try just one first
            print(f"  Downloading ObsID {obsid}...")
            download_chandra_obsid(obsid, PERSEUS_DATA)
        # If that worked, reload
        return try_load_chandra_data()
    except ImportError:
        print("  CIAO not installed. Skipping CIAO download.")
    except Exception as e:
        print(f"  CIAO download failed: {e}")

    # Try direct HTTP download
    print("\n  Attempting direct HTTP download from Chandra Data Archive...")
    try:
        import urllib.request
        for obsid in OBSIDS[:1]:  # Try just one
            first_digit = str(obsid)[0]
            fname = f"acisf{obsid:05d}_repro_evt2.fits.gz"
            url = (f"https://cxc.cfa.harvard.edu/cdaftp/byobsid/"
                   f"{first_digit}/{obsid}/primary/{fname}")
            outpath = os.path.join(PERSEUS_DATA, fname)
            print(f"  Downloading {url}...")
            urllib.request.urlretrieve(url, outpath)
            # Decompress
            import gzip
            import shutil
            with gzip.open(outpath, 'rb') as f_in:
                with open(outpath.replace('.gz', ''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(outpath)
            print(f"  Downloaded and decompressed ObsID {obsid}.")
        return try_load_chandra_data()
    except Exception as e:
        print(f"  HTTP download failed: {e}")

    return None


# ============================================================================
# REAL DATA ANALYSIS PIPELINE
# ============================================================================

def analyze_real_data(events_list):
    """
    Full analysis pipeline for real Chandra event data.
    Returns results dict.
    """
    print("\n" + "=" * 70)
    print("REAL CHANDRA DATA ANALYSIS")
    print("=" * 70)

    # Merge events from all ObsIDs, filter to 0.5-7.0 keV
    all_ra = []
    all_dec = []

    for evt in events_list:
        energy_keV = evt['energy'] / 1000.0  # eV -> keV
        mask = (energy_keV >= ENERGY_LO) & (energy_keV <= ENERGY_HI)
        if evt.get('ra') is not None and evt.get('dec') is not None:
            all_ra.append(evt['ra'][mask])
            all_dec.append(evt['dec'][mask])
        else:
            # Use sky pixel coords, convert to approx RA/Dec
            dx = (evt['x'][mask] - np.mean(evt['x'][mask])) * PIXEL_SIZE / 3600.0
            dy = (evt['y'][mask] - np.mean(evt['y'][mask])) * PIXEL_SIZE / 3600.0
            all_ra.append(NGC1275_RA + dx / np.cos(np.radians(NGC1275_DEC)))
            all_dec.append(NGC1275_DEC + dy)

        print(f"  ObsID {evt['obsid']}: {np.sum(mask)} events in "
              f"{ENERGY_LO}-{ENERGY_HI} keV band")

    ra = np.concatenate(all_ra)
    dec = np.concatenate(all_dec)
    print(f"\n  Total merged events: {len(ra)}")

    # Compute offsets from NGC 1275 in arcsec
    dx_arcsec = (ra - NGC1275_RA) * 3600.0 * np.cos(np.radians(NGC1275_DEC))
    dy_arcsec = (dec - NGC1275_DEC) * 3600.0
    r_arcsec = np.sqrt(dx_arcsec**2 + dy_arcsec**2)
    theta_deg = np.degrees(np.arctan2(dx_arcsec, dy_arcsec)) % 360.0

    # Build image
    img_size = int(2 * R_MAX_ARCSEC / PIXEL_SIZE)
    img_range = [[-R_MAX_ARCSEC, R_MAX_ARCSEC], [-R_MAX_ARCSEC, R_MAX_ARCSEC]]
    image, xedges, yedges = np.histogram2d(
        dx_arcsec, dy_arcsec, bins=img_size, range=img_range
    )

    # Radial surface brightness profile
    r_edges = np.linspace(0, R_MAX_ARCSEC, N_RADIAL_BINS + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    sb_radial = np.zeros(N_RADIAL_BINS)
    sb_radial_err = np.zeros(N_RADIAL_BINS)

    for i in range(N_RADIAL_BINS):
        mask = (r_arcsec >= r_edges[i]) & (r_arcsec < r_edges[i + 1])
        counts = np.sum(mask)
        area = np.pi * (r_edges[i + 1]**2 - r_edges[i]**2)
        sb_radial[i] = counts / area if area > 0 else 0
        sb_radial_err[i] = np.sqrt(counts) / area if area > 0 else 0

    # Azimuthal profiles at specified radii
    azimuthal_results = {}
    for r_target in ANALYSIS_RADII:
        dr = 5.0
        mask_r = (r_arcsec >= r_target - dr) & (r_arcsec < r_target + dr)
        theta_sel = theta_deg[mask_r]

        theta_edges = np.linspace(0, 360, AZIMUTHAL_BINS + 1)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        az_profile = np.zeros(AZIMUTHAL_BINS)

        for j in range(AZIMUTHAL_BINS):
            mask_theta = ((theta_sel >= theta_edges[j]) &
                          (theta_sel < theta_edges[j + 1]))
            az_profile[j] = np.sum(mask_theta)

        azimuthal_results[r_target] = {
            'theta': theta_centers,
            'counts': az_profile,
            'mean': np.mean(az_profile),
            'std': np.std(az_profile),
        }

    # Run KWW analysis on azimuthal ACFs
    kww_results = run_kww_analysis(azimuthal_results, r_centers, sb_radial)

    return {
        'image': image,
        'x_range': (-R_MAX_ARCSEC, R_MAX_ARCSEC),
        'r_centers': r_centers,
        'sb_radial': sb_radial,
        'sb_radial_err': sb_radial_err,
        'azimuthal': azimuthal_results,
        'kww': kww_results,
        'is_synthetic': False,
        'n_events': len(ra),
        'n_obsids': len(events_list),
    }


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_data():
    """
    Generate realistic synthetic Perseus-like X-ray data.

    Model components:
      1. Beta-model ICM continuum: S(r) = S0 * (1 + (r/rc)^2)^(-3*beta + 0.5)
      2. Two inner bubbles (NW and SE) as elliptical depressions
      3. One ghost bubble at ~50" S
4. Poisson noise consistent with ~900 ks exposure
      5. Azimuthal fluctuations with KWW-like correlation in bubble region,
         uncorrelated (Poisson) in undisturbed ICM
    """
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC PERSEUS-LIKE DATA")
    print("(No Chandra FITS files available -- using realistic model)")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    # Build 2D image
    img_half = R_MAX_ARCSEC
    npix = 1200  # ~0.5" pixels
    x = np.linspace(-img_half, img_half, npix)
    y = np.linspace(-img_half, img_half, npix)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    # 1. Beta-model continuum
    rc = BETA_MODEL_RC
    beta = BETA_MODEL_BETA
    S0 = BETA_MODEL_S0
    sb_model = S0 * (1.0 + (rr / rc)**2)**(-3.0 * beta + 0.5)

    # 2. Add cool-core excess (central cusp seen in real Perseus)
    rc_cc = 8.0  # arcsec, cool-core radius
    S0_cc = 1500.0
    sb_model += S0_cc * (1.0 + (rr / rc_cc)**2)**(-2.0)

    # 3. Subtract bubble cavities
    for bubble in [BUBBLE_NW, BUBBLE_SE, GHOST_BUBBLE]:
        # Bubble center in Cartesian coords
        pa_rad = np.radians(bubble['center_pa'])
        bx = bubble['center_r'] * np.sin(pa_rad)
        by = bubble['center_r'] * np.cos(pa_rad)

        # Rotated elliptical distance
        pa_e = np.radians(bubble['pa_ellipse'])
        dx = xx - bx
        dy = yy - by
        dx_rot = dx * np.cos(pa_e) + dy * np.sin(pa_e)
        dy_rot = -dx * np.sin(pa_e) + dy * np.cos(pa_e)

        ell_dist = np.sqrt(
            (dx_rot / bubble['semi_major'])**2 +
            (dy_rot / bubble['semi_minor'])**2
        )
        # Smooth cavity with tanh edge
        cavity = 0.5 * (1.0 - np.tanh(3.0 * (ell_dist - 1.0)))
        sb_model *= (1.0 - bubble['depth'] * cavity)

    # 4. Add small-scale azimuthal fluctuations
    theta_grid = np.arctan2(xx, yy)

    # Generate correlated azimuthal fluctuations using harmonic modes
    # Power spectrum P(k) ~ exp(-(k/k0)^(2/alpha)) gives ACF with exponent alpha
    # For alpha = 4/3: P(k) ~ exp(-(k/k0)^1.5)
    n_modes = 18
    fluct_amplitude = 0.08  # 8% RMS in bubble region
    k0 = 5.0  # characteristic angular wavenumber
    phase_offsets = rng.uniform(0, 2 * np.pi, n_modes)
    mode_amps = np.array([
        fluct_amplitude * np.exp(-((k + 1) / k0)**1.5) for k in range(n_modes)
    ])

    # Bubble-region azimuthal structure
    bubble_fluct = np.zeros_like(sb_model)
    for k in range(n_modes):
        bubble_fluct += mode_amps[k] * np.sin((k + 1) * theta_grid + phase_offsets[k])

    # Apply only in bubble annulus (15-60"), taper smoothly
    r_bubble_inner = 12.0
    r_bubble_outer = 65.0
    bubble_mask = 0.5 * (1.0 + np.tanh((rr - r_bubble_inner) / 3.0))
    bubble_mask *= 0.5 * (1.0 - np.tanh((rr - r_bubble_outer) / 5.0))
    sb_model *= (1.0 + bubble_fluct * bubble_mask)

    # Add weak uncorrelated fluctuations everywhere (thermal noise in ICM)
    icm_fluct = rng.normal(0, 0.02, sb_model.shape)
    sb_model *= (1.0 + icm_fluct)

    # 5. Ensure non-negative
    sb_model = np.maximum(sb_model, 0.0)

    # 6. Generate Poisson-sampled image (integer counts)
    pixel_scale = 2 * img_half / npix
    total_expected = np.sum(sb_model) * pixel_scale**2
    target_counts = 2.5e6
    scale_factor = target_counts / total_expected if total_expected > 0 else 1.0
    count_rate = sb_model * scale_factor * pixel_scale**2
    image_poisson = rng.poisson(np.maximum(count_rate, 0))

    print(f"  Image size: {npix} x {npix} pixels ({2*img_half:.0f}\" x {2*img_half:.0f}\")")
    print(f"  Pixel scale: {pixel_scale:.2f}\"/pixel")
    print(f"  Total counts: {np.sum(image_poisson):,.0f}")
    print(f"  Beta-model: rc={rc}\", beta={beta}, S0={S0:.0f}")
    print(f"  Bubbles: NW (r=25\", depth=40%), SE (r=20\", depth=35%), "
          f"Ghost (r=50\", depth=20%)")
    # Extract radial profile
    r_edges = np.linspace(0, R_MAX_ARCSEC, N_RADIAL_BINS + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    sb_radial = np.zeros(N_RADIAL_BINS)
    sb_radial_err = np.zeros(N_RADIAL_BINS)

    for i in range(N_RADIAL_BINS):
        mask = (rr >= r_edges[i]) & (rr < r_edges[i + 1])
        if np.sum(mask) > 0:
            total_counts = np.sum(image_poisson[mask])
            n_pix = np.sum(mask)
            area = n_pix * pixel_scale**2
            sb_radial[i] = total_counts / area
            sb_radial_err[i] = np.sqrt(total_counts) / area

    # Extract azimuthal profiles at analysis radii
    azimuthal_results = {}
    for r_target in ANALYSIS_RADII:
        dr = 5.0
        mask_r = (rr >= r_target - dr) & (rr < r_target + dr)

        theta_vals = theta_grid[mask_r]
        counts_vals = image_poisson[mask_r]

        theta_edges = np.linspace(-np.pi, np.pi, AZIMUTHAL_BINS + 1)
        theta_centers_rad = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        theta_centers_deg = np.degrees(theta_centers_rad) % 360

        az_profile = np.zeros(AZIMUTHAL_BINS)
        for j in range(AZIMUTHAL_BINS):
            mask_theta = ((theta_vals >= theta_edges[j]) &
                          (theta_vals < theta_edges[j + 1]))
            az_profile[j] = np.sum(counts_vals[mask_theta])

        azimuthal_results[r_target] = {
            'theta': theta_centers_deg,
            'counts': az_profile,
            'mean': np.mean(az_profile),
            'std': np.std(az_profile),
        }

    # Run KWW analysis
    kww_results = run_kww_analysis(azimuthal_results, r_centers, sb_radial)

    return {
        'image': image_poisson,
        'image_model': sb_model * scale_factor * pixel_scale**2,
        'x_range': (-img_half, img_half),
        'r_centers': r_centers,
        'sb_radial': sb_radial,
        'sb_radial_err': sb_radial_err,
        'azimuthal': azimuthal_results,
        'kww': kww_results,
        'is_synthetic': True,
        'total_counts': int(np.sum(image_poisson)),
        'pixel_scale': pixel_scale,
    }


# ============================================================================
# KWW ANALYSIS ON AZIMUTHAL PROFILES
# ============================================================================

def run_kww_analysis(azimuthal_results, r_centers, sb_radial):
    """
    For each radial annulus with azimuthal data:
      1. Detrend azimuthal profile (remove mean)
      2. Compute ACF of fluctuations
      3. Fit KWW to ACF
      4. Record alpha, tau, R2

    Also compute ACF of radial profile fluctuations (detrended from
    smooth beta-model fit).
    """
    print("\n  --- KWW Analysis of Azimuthal ACFs ---")

    results = {
        'azimuthal_fits': {},
        'radial_fit': None,
        'alpha_vs_radius': [],
    }

    print(f"\n  {'Radius':>8}  {'alpha':>8}  {'tau':>8}  {'R2':>8}  "
          f"{'|a-4/3|':>8}  {'In window':>10}")
    print("  " + "-" * 62)

    for r_target in sorted(azimuthal_results.keys()):
        az = azimuthal_results[r_target]
        profile = az['counts']

        # Detrend: subtract mean
        fluctuations = profile - np.mean(profile)

        # Compute ACF
        acf = compute_acf(fluctuations, max_lag=AZIMUTHAL_BINS // 2)
        lags = np.arange(len(acf)) * (360.0 / AZIMUTHAL_BINS)

        # Fit KWW
        alpha, tau, A, r2, alpha_err, popt, pcov = fit_kww_to_acf(lags, acf)

        dev = abs(alpha - ALPHA_TARGET) if not np.isnan(alpha) else np.nan
        in_window = (ALPHA_WINDOW[0] <= alpha <= ALPHA_WINDOW[1]) \
            if not np.isnan(alpha) else False
        window_str = "YES" if in_window else "no"

        if not np.isnan(alpha):
            print(f"  {r_target:>6.0f}\"  {alpha:8.3f}  {tau:8.1f}  {r2:8.3f}  "
                  f"{dev:8.3f}  {window_str:>10}")
        else:
            print(f"  {r_target:>6.0f}\"  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  "
                  f"{'N/A':>8}  {'N/A':>10}")

        results['azimuthal_fits'][r_target] = {
            'alpha': float(alpha) if not np.isnan(alpha) else None,
            'tau': float(tau) if not np.isnan(tau) else None,
            'A': float(A) if not np.isnan(A) else None,
            'r2': float(r2) if not np.isnan(r2) else None,
            'alpha_err': float(alpha_err) if not np.isnan(alpha_err) else None,
            'in_window': bool(in_window),
            'acf': acf.tolist(),
            'lags': lags.tolist(),
            'popt': popt.tolist() if popt is not None else None,
            'fluctuation_rms': float(np.std(fluctuations)),
            'mean_counts': float(az['mean']),
        }

        if not np.isnan(alpha):
            results['alpha_vs_radius'].append({
                'radius_arcsec': float(r_target),
                'radius_kpc': float(r_target * KPC_PER_ARCSEC),
                'alpha': float(alpha),
                'alpha_err': float(alpha_err) if not np.isnan(alpha_err) else None,
                'r2': float(r2),
                'in_window': bool(in_window),
            })

    # Radial profile ACF (detrended)
    print("\n  --- KWW Analysis of Radial Profile ACF ---")

    def beta_model_1d(r, S0, rc, beta_p):
        return S0 * (1.0 + (r / rc)**2)**(-3.0 * beta_p + 0.5)

    try:
        mask_fit = (r_centers > 5.0) & (sb_radial > 0)
        popt_bm, _ = curve_fit(
            beta_model_1d, r_centers[mask_fit], sb_radial[mask_fit],
            p0=[sb_radial[mask_fit][0], 25.0, 0.55],
            bounds=([0, 1, 0.3], [1e6, 200, 1.5]),
            maxfev=10000
        )
        sb_betamodel = beta_model_1d(r_centers, *popt_bm)
        radial_resid = sb_radial - sb_betamodel
        radial_resid_norm = radial_resid / sb_betamodel
        radial_resid_norm = np.nan_to_num(radial_resid_norm)

        print(f"  Beta-model fit: S0={popt_bm[0]:.1f}, rc={popt_bm[1]:.1f}\", "
              f"beta={popt_bm[2]:.3f}")

        # ACF of radial residuals
        acf_radial = compute_acf(radial_resid_norm, max_lag=N_RADIAL_BINS // 2)
        lags_radial = np.arange(len(acf_radial)) * (R_MAX_ARCSEC / N_RADIAL_BINS)

        alpha_r, tau_r, A_r, r2_r, alpha_r_err, popt_r, _ = fit_kww_to_acf(
            lags_radial, acf_radial
        )

        if not np.isnan(alpha_r):
            dev_r = abs(alpha_r - ALPHA_TARGET)
            in_w_r = ALPHA_WINDOW[0] <= alpha_r <= ALPHA_WINDOW[1]
            err_str = f" +/- {alpha_r_err:.3f}" if not np.isnan(alpha_r_err) else ""
            print(f"  Radial residual ACF: alpha={alpha_r:.3f}{err_str}, "
                  f"R2={r2_r:.3f}, |a-4/3|={dev_r:.3f}, "
                  f"{'IN WINDOW' if in_w_r else 'outside'}")
        else:
            print("  Radial residual ACF: fit failed")
            in_w_r = False

        results['radial_fit'] = {
            'alpha': float(alpha_r) if not np.isnan(alpha_r) else None,
            'tau': float(tau_r) if not np.isnan(tau_r) else None,
            'r2': float(r2_r) if not np.isnan(r2_r) else None,
            'alpha_err': float(alpha_r_err) if not np.isnan(alpha_r_err) else None,
            'in_window': bool(in_w_r) if not np.isnan(alpha_r) else False,
            'beta_model_params': {
                'S0': float(popt_bm[0]),
                'rc': float(popt_bm[1]),
                'beta': float(popt_bm[2]),
            },
            'acf': acf_radial.tolist(),
            'lags': lags_radial.tolist(),
            'residuals_norm': radial_resid_norm.tolist(),
        }

    except (RuntimeError, ValueError) as e:
        print(f"  Beta-model fit failed: {e}")
        results['radial_fit'] = None

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def make_figure(data):
    """Generate 6-panel analysis figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    tag = 'SYNTHETIC' if data['is_synthetic'] else 'CHANDRA'

    # Panel 1: X-ray image
    ax = axes[0, 0]
    img = data['image']
    extent = [data['x_range'][0], data['x_range'][1],
              data['x_range'][0], data['x_range'][1]]
    vmin = max(np.percentile(img[img > 0], 1), 1) if np.any(img > 0) else 1
    vmax = np.percentile(img, 99.5)
    ax.imshow(img, origin='lower', extent=extent, cmap='magma',
              norm=LogNorm(vmin=vmin, vmax=vmax), aspect='equal')
    for r_ann in [R_INNER_BUBBLE, R_OUTER_BUBBLE, R_CONTROL_ICM]:
        circle = Circle((0, 0), r_ann, fill=False, color='cyan', ls='--', lw=0.8)
        ax.add_patch(circle)
    ax.set_xlabel('dRA (arcsec)')
    ax.set_ylabel('dDec (arcsec)')
    ax.set_title(f'X-ray Image ({tag})')
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)

    # Panel 2: Radial SB profile
    ax = axes[0, 1]
    r = data['r_centers']
    sb = data['sb_radial']
    sb_err = data['sb_radial_err']
    mask_pos = sb > 0
    ax.errorbar(r[mask_pos], sb[mask_pos], yerr=sb_err[mask_pos],
                fmt='k.', markersize=3, elinewidth=0.5, capsize=0)
    if data['kww'] and data['kww']['radial_fit']:
        rf = data['kww']['radial_fit']
        if 'beta_model_params' in rf:
            bm = rf['beta_model_params']
            r_plot = np.linspace(1, R_MAX_ARCSEC, 200)
            sb_bm = bm['S0'] * (1 + (r_plot / bm['rc'])**2)**(-3*bm['beta'] + 0.5)
            ax.plot(r_plot, sb_bm, 'r-', lw=1.5, label='Beta model')
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('SB (counts/arcsec^2)')
    ax.set_yscale('log')
    ax.set_title('Radial Surface Brightness')
    ax.legend(fontsize=8)
    ax.axvspan(20, 55, alpha=0.1, color='orange', label='Bubble region')

    # Panel 3: Azimuthal profile at bubble radius (25")
    ax = axes[0, 2]
    kww = data['kww']
    r_key = 25
    if r_key in data['azimuthal']:
        az = data['azimuthal'][r_key]
        ax.plot(az['theta'], az['counts'], 'b-', lw=1)
        ax.axhline(az['mean'], color='red', ls='--', lw=1, label=f"mean={az['mean']:.0f}")
        ax.set_xlabel('Azimuthal angle (deg)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Azimuthal @ r={r_key}" (bubble)')
        ax.legend(fontsize=8)

    # Panel 4: Azimuthal profile at control radius (100")
    ax = axes[1, 0]
    r_key = 100
    if r_key in data['azimuthal']:
        az = data['azimuthal'][r_key]
        ax.plot(az['theta'], az['counts'], 'gray', lw=1)
        ax.axhline(az['mean'], color='red', ls='--', lw=1, label=f"mean={az['mean']:.0f}")
        ax.set_xlabel('Azimuthal angle (deg)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Azimuthal @ r={r_key}" (control ICM)')
        ax.legend(fontsize=8)

    # Panel 5: ACF + KWW fit at bubble radius
    ax = axes[1, 1]
    if kww and 'azimuthal_fits' in kww:
        for r_key, color, label in [(25, 'blue', 'r=25" (bubble)'),
                                     (100, 'gray', 'r=100" (ICM)')]:
            if r_key in kww['azimuthal_fits'] and kww['azimuthal_fits'][r_key]['acf']:
                fit = kww['azimuthal_fits'][r_key]
                lags = np.array(fit['lags'])
                acf = np.array(fit['acf'])
                ax.plot(lags, acf, 'o', color=color, markersize=4, alpha=0.7, label=label)
                if fit['popt']:
                    lag_fit = np.linspace(lags[1], lags[-1], 100)
                    ax.plot(lag_fit, kww_acf(lag_fit, *fit['popt']),
                            '-', color=color, lw=1.5)
        ax.set_xlabel('Lag (degrees)')
        ax.set_ylabel('ACF')
        ax.set_title('Azimuthal ACF + KWW fit')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.5, 1.1)

    # Panel 6: alpha vs radius
    ax = axes[1, 2]
    if kww and kww['alpha_vs_radius']:
        avr = kww['alpha_vs_radius']
        radii = [d['radius_arcsec'] for d in avr]
        alphas = [d['alpha'] for d in avr]
        r2s = [d['r2'] for d in avr]
        in_w = [d['in_window'] for d in avr]
        colors = ['green' if w else 'gray' for w in in_w]
        ax.scatter(radii, alphas, c=colors, s=60, edgecolor='black', zorder=5)
        ax.axhline(ALPHA_TARGET, color='red', ls='--', lw=2, label='4/3')
        ax.axhspan(ALPHA_WINDOW[0], ALPHA_WINDOW[1], alpha=0.1, color='green')
        ax.axvspan(20, 55, alpha=0.1, color='orange', label='Bubble region')
        ax.set_xlabel('Radius (arcsec)')
        ax.set_ylabel('KWW alpha')
        ax.set_title('alpha vs Radius')
        ax.legend(fontsize=8)
        ax.set_ylim(0.5, 2.0)

    plt.suptitle(f'P7 -- Perseus Cluster X-ray: AGN Feedback Cycle ({tag})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'p7_perseus_xray.png'), dpi=150)
    plt.close()
    print(f"\nSaved: p7_perseus_xray.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("P7 -- Perseus Cluster X-ray Analysis: AGN Feedback Cycle")
    print("Prediction: alpha ~ 4/3 at bubble radii (25-50\")")
    print("=" * 70)

    # Try real data first
    events = try_load_chandra_data()

    if events:
        results = analyze_real_data(events)
    else:
        print("\n  No Chandra data available. Generating synthetic model.")
        results = generate_synthetic_data()

    # Generate figure
    make_figure(results)

    # Collect summary
    kww = results['kww']
    lines = []
    lines.append("=" * 70)
    lines.append("P7 -- Perseus Cluster X-ray Analysis")
    lines.append(f"Data: {'SYNTHETIC MODEL' if results['is_synthetic'] else 'Chandra ACIS-S'}")
    lines.append("=" * 70)

    if results['is_synthetic']:
        lines.append(f"\nSynthetic model parameters:")
        lines.append(f"  Beta-model: rc={BETA_MODEL_RC}\", beta={BETA_MODEL_BETA}")
        lines.append(f"  Bubbles: NW (25\",40%), SE (20\",35%), Ghost (50\",20%)")
        lines.append(f"  Total counts: {results.get('total_counts', 'N/A'):,}")
    else:
        lines.append(f"\n  Events: {results.get('n_events', 'N/A'):,} "
                     f"from {results.get('n_obsids', 0)} ObsIDs")

    lines.append(f"\n--- Azimuthal KWW fits per radius ---")
    lines.append(f"  {'Radius':>8}  {'alpha':>8}  {'R2':>8}  {'|a-4/3|':>8}  In window")
    lines.append("  " + "-" * 55)

    n_in_bubble_window = 0
    n_in_control_window = 0

    if kww and 'azimuthal_fits' in kww:
        for r_target in sorted(kww['azimuthal_fits'].keys()):
            fit = kww['azimuthal_fits'][r_target]
            a = fit['alpha']
            r2 = fit['r2']
            if a is not None:
                dev = abs(a - ALPHA_TARGET)
                in_w = fit['in_window']
                lines.append(f"  {r_target:>6.0f}\"  {a:8.3f}  {r2:8.3f}  "
                             f"{dev:8.3f}  {'YES' if in_w else 'no'}")
                if in_w and 15 <= r_target <= 55:
                    n_in_bubble_window += 1
                if in_w and r_target >= 80:
                    n_in_control_window += 1
            else:
                lines.append(f"  {r_target:>6.0f}\"  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  N/A")

    if kww and kww.get('radial_fit'):
        rf = kww['radial_fit']
        lines.append(f"\n--- Radial profile residual ACF ---")
        if rf['alpha'] is not None:
            lines.append(f"  alpha = {rf['alpha']:.3f}, R2 = {rf['r2']:.3f}")
            lines.append(f"  |alpha - 4/3| = {abs(rf['alpha'] - ALPHA_TARGET):.3f}")
            lines.append(f"  In window: {'YES' if rf['in_window'] else 'no'}")

    lines.append(f"\n--- Prediction P7 Assessment ---")
    lines.append(f"  Bubble region (15-55\"): {n_in_bubble_window} radii in cooperative window")
    lines.append(f"  Control ICM (>=80\"):    {n_in_control_window} radii in cooperative window")

    if n_in_bubble_window > 0 and n_in_control_window == 0:
        lines.append(f"\n  PREDICTION SUPPORTED: alpha ~ 4/3 in bubble region only")
    elif n_in_bubble_window > 0:
        lines.append(f"\n  PARTIAL: signal present but also in control region")
    else:
        lines.append(f"\n  INCONCLUSIVE: alpha not near 4/3 at bubble radii")

    if results['is_synthetic']:
        lines.append(f"\n  *** SYNTHETIC DATA -- download Chandra ObsIDs for real test ***")
        lines.append(f"  ObsIDs needed: {', '.join(str(o) for o in OBSIDS)}")
        lines.append(f"  Source: https://cxc.cfa.harvard.edu/cda/")

    summary = '\n'.join(lines)
    print('\n' + summary)

    with open(os.path.join(REPORTS_DIR, 'p7_perseus_xray_summary.txt'), 'w') as f:
        f.write(summary)

    # Save JSON results
    json_results = {
        'is_synthetic': results['is_synthetic'],
        'prediction': 'P7: AGN feedback cycle alpha ~ 4/3',
        'target': 'Perseus cluster (Abell 426)',
        'obsids': OBSIDS,
        'ngc1275_position': {'ra': NGC1275_RA, 'dec': NGC1275_DEC},
        'redshift': REDSHIFT,
        'alpha_vs_radius': kww['alpha_vs_radius'] if kww else [],
        'radial_fit': kww['radial_fit'] if kww else None,
        'n_bubble_in_window': n_in_bubble_window,
        'n_control_in_window': n_in_control_window,
    }

    with open(os.path.join(REPORTS_DIR, 'p7_perseus_xray_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved: p7_perseus_xray_summary.txt, p7_perseus_xray_results.json")
    print("P7 analysis complete.")
