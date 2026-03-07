# -*- coding: utf-8 -*-
"""
Tsoukalas et al. (2026) -- Independent Alpha Verification
==========================================================
Questions to answer:
  (1) Fitting method used
  (2) Whether +/-0.306 is 1-sigma or 2-sigma
  (3) 95% confidence interval on alpha
  (4) R^2 of the fit
  (5) Number of data points
  Plus: delta-R^2 for alpha fixed at 1.333 (4/3) and 0.82 (MBL)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import *

# Tsoukalas data: download from Zenodo 17144607 into data/tsoukalas/
DATA_DIR_TSOUKALAS = os.path.join(DATA_DIR, 'tsoukalas')

# ==========================================================================
# Load Fig2e1 FID data
# ==========================================================================
df = pd.read_csv(os.path.join(DATA_DIR_TSOUKALAS, 'Fig2e1.csv'))
t_fid = df['xdata'].values      # microseconds
p_fid = df['z_T2'].values       # signal (singlet probability)
N_pts = len(t_fid)

print("=" * 75)
print("TSOUKALAS FID (Fig2e1) -- INDEPENDENT ALPHA VERIFICATION")
print("=" * 75)
print(f"\nData: Fig2e1.csv")
print(f"N = {N_pts} data points")
print(f"t range: [{t_fid[0]:.4f}, {t_fid[-1]:.4f}] us")
print(f"Signal range: [{p_fid.min():.4f}, {p_fid.max():.4f}]")

# ==========================================================================
# Define fitting models
# ==========================================================================
def osc_stretched(t, A, f, T2, alpha, C, phi):
    """Oscillating stretched exponential:
       C + A * exp(-(t/T2)^alpha) * cos(2*pi*f*t + phi)
    """
    return C + A * np.exp(-(t / T2)**alpha) * np.cos(2 * np.pi * f * t + phi)

def osc_stretched_fixed_alpha(alpha_fixed):
    """Returns a 5-parameter version with alpha fixed."""
    def model(t, A, f, T2, C, phi):
        return C + A * np.exp(-(t / T2)**alpha_fixed) * np.cos(2 * np.pi * f * t + phi)
    return model

# ==========================================================================
# Step 1: FFT to estimate oscillation frequency
# ==========================================================================
dt = t_fid[1] - t_fid[0]
fft_y = np.abs(np.fft.fft(p_fid - np.mean(p_fid)))
freqs = np.fft.fftfreq(len(p_fid), d=dt)
pos = freqs > 0.05
f_guess = freqs[pos][np.argmax(fft_y[pos])]
print(f"\nFFT frequency guess: {f_guess:.4f} MHz")

# ==========================================================================
# Step 2: FREE FIT -- all 6 parameters free
# ==========================================================================
print("\n" + "=" * 75)
print("FIT 1: FREE ALPHA (6-parameter fit)")
print("=" * 75)

p0 = [0.5, f_guess, 1.5, 1.0, 0.5, 0.0]
bounds_lo = [0, 0.1, 0.1, 0.3, -1, -np.pi]
bounds_hi = [2, 5, 50, 5.0, 2, np.pi]

popt, pcov = curve_fit(osc_stretched, t_fid, p_fid, p0=p0,
                       bounds=(bounds_lo, bounds_hi), maxfev=50000)
perr = np.sqrt(np.diag(pcov))

A_fit, f_fit, T2_fit, alpha_fit, C_fit, phi_fit = popt
A_err, f_err, T2_err, alpha_err, C_err, phi_err = perr

# Compute R^2
y_pred = osc_stretched(t_fid, *popt)
SS_res = np.sum((p_fid - y_pred)**2)
SS_tot = np.sum((p_fid - np.mean(p_fid))**2)
R2_free = 1 - SS_res / SS_tot

# Degrees of freedom
n_params_free = 6
dof_free = N_pts - n_params_free

print(f"\nModel: A(t) = C + A0 * exp(-(t/T2*)^alpha) * cos(2*pi*f*t + phi)")
print(f"Method: scipy.optimize.curve_fit (Levenberg-Marquardt / Trust Region Reflective)")
print(f"        Nonlinear least squares with bounds")
print(f"\nFitted parameters:")
print(f"  A0    = {A_fit:.6f} +/- {A_err:.6f}")
print(f"  f     = {f_fit:.6f} +/- {f_err:.6f} MHz")
print(f"  T2*   = {T2_fit:.6f} +/- {T2_err:.6f} us")
print(f"  alpha = {alpha_fit:.6f} +/- {alpha_err:.6f}")
print(f"  C     = {C_fit:.6f} +/- {C_err:.6f}")
print(f"  phi   = {phi_fit:.6f} +/- {phi_err:.6f}")
print(f"\n  R^2   = {R2_free:.6f}")
print(f"  N     = {N_pts}")
print(f"  k     = {n_params_free} parameters")
print(f"  DoF   = {dof_free}")

# ==========================================================================
# Step 3: UNCERTAINTY ANALYSIS
# ==========================================================================
print("\n" + "=" * 75)
print("UNCERTAINTY ANALYSIS")
print("=" * 75)

# The +/- from curve_fit is sqrt(diag(covariance)) = 1-sigma SE
print(f"\nalpha = {alpha_fit:.4f} +/- {alpha_err:.4f}")
print(f"\nThe +/- {alpha_err:.4f} from curve_fit is:")
print(f"  sqrt(pcov[3,3]) = sqrt({pcov[3,3]:.6f}) = {np.sqrt(pcov[3,3]):.6f}")
print(f"  This is the 1-SIGMA standard error from the covariance matrix.")

# Paper says +/- 0.306 -- let's check
print(f"\nPaper reports: alpha = 1.442 +/- 0.306")
print(f"Our fit gives: alpha = {alpha_fit:.4f} +/- {alpha_err:.4f}")
print(f"Ratio paper_err / our_err = {0.306 / alpha_err:.4f}")

# 95% CI using t-distribution
t_crit = t_dist.ppf(0.975, dof_free)
ci_95_lo = alpha_fit - t_crit * alpha_err
ci_95_hi = alpha_fit + t_crit * alpha_err
print(f"\n95% Confidence Interval on alpha:")
print(f"  t-critical (dof={dof_free}, two-sided 95%) = {t_crit:.4f}")
print(f"  95% CI = [{ci_95_lo:.4f}, {ci_95_hi:.4f}]")
print(f"  95% CI = {alpha_fit:.4f} +/- {t_crit * alpha_err:.4f}")

# Also check if 0.306 is the 2-sigma value
two_sigma = 2 * alpha_err
print(f"\n  1-sigma = {alpha_err:.4f}")
print(f"  2-sigma = {two_sigma:.4f}")
print(f"  t-crit * sigma (95% CI half-width) = {t_crit * alpha_err:.4f}")

# Check: does 0.306 match any of these?
print(f"\n  Paper's 0.306 vs 1-sigma ({alpha_err:.4f}): ratio = {0.306/alpha_err:.3f}")
print(f"  Paper's 0.306 vs 2-sigma ({two_sigma:.4f}): ratio = {0.306/two_sigma:.3f}")
print(f"  Paper's 0.306 vs 95% CI hw ({t_crit*alpha_err:.4f}): ratio = {0.306/(t_crit*alpha_err):.3f}")

# Profile likelihood CI (more robust for nonlinear models)
print(f"\n  NOTE: For nonlinear models, the covariance-based CI is approximate.")
print(f"  The Wald CI (symmetric) may not be optimal. Profile likelihood")
print(f"  would give asymmetric intervals but is computationally expensive.")

# ==========================================================================
# Step 4: FIT WITH ALPHA FIXED AT 4/3 = 1.333
# ==========================================================================
print("\n" + "=" * 75)
print("FIT 2: ALPHA FIXED AT 4/3 = 1.3333")
print("=" * 75)

model_43 = osc_stretched_fixed_alpha(4.0/3.0)
p0_fixed = [0.5, f_guess, 1.5, 0.5, 0.0]
bounds_lo_f = [0, 0.1, 0.1, -1, -np.pi]
bounds_hi_f = [2, 5, 50, 2, np.pi]

popt_43, pcov_43 = curve_fit(model_43, t_fid, p_fid, p0=p0_fixed,
                              bounds=(bounds_lo_f, bounds_hi_f), maxfev=50000)
perr_43 = np.sqrt(np.diag(pcov_43))

y_pred_43 = model_43(t_fid, *popt_43)
SS_res_43 = np.sum((p_fid - y_pred_43)**2)
R2_43 = 1 - SS_res_43 / SS_tot
n_params_43 = 5
dof_43 = N_pts - n_params_43

dR2_43 = R2_free - R2_43

print(f"  alpha (fixed) = {4.0/3.0:.6f}")
print(f"  A0    = {popt_43[0]:.6f} +/- {perr_43[0]:.6f}")
print(f"  f     = {popt_43[1]:.6f} +/- {perr_43[1]:.6f} MHz")
print(f"  T2*   = {popt_43[2]:.6f} +/- {perr_43[2]:.6f} us")
print(f"  C     = {popt_43[3]:.6f} +/- {perr_43[3]:.6f}")
print(f"  phi   = {popt_43[4]:.6f} +/- {perr_43[4]:.6f}")
print(f"\n  R^2   = {R2_43:.6f}")
print(f"  delta_R^2 (free - fixed@4/3) = {dR2_43:.6f}")
print(f"  DoF   = {dof_43}")

# F-test: is the extra alpha parameter justified?
F_stat_43 = ((SS_res_43 - SS_res) / (n_params_free - n_params_43)) / (SS_res / dof_free)
from scipy.stats import f as f_dist
p_val_43 = 1 - f_dist.cdf(F_stat_43, n_params_free - n_params_43, dof_free)
print(f"\n  F-test (free vs fixed@4/3):")
print(f"    F = {F_stat_43:.4f}, p = {p_val_43:.4f}")
print(f"    {'Significant' if p_val_43 < 0.05 else 'NOT significant'} (p {'<' if p_val_43 < 0.05 else '>'} 0.05)")

# ==========================================================================
# Step 5: FIT WITH ALPHA FIXED AT 0.82 (MBL BASELINE)
# ==========================================================================
print("\n" + "=" * 75)
print("FIT 3: ALPHA FIXED AT 0.82 (MBL BASELINE)")
print("=" * 75)

model_mbl = osc_stretched_fixed_alpha(0.82)
popt_mbl, pcov_mbl = curve_fit(model_mbl, t_fid, p_fid, p0=p0_fixed,
                                bounds=(bounds_lo_f, bounds_hi_f), maxfev=50000)
perr_mbl = np.sqrt(np.diag(pcov_mbl))

y_pred_mbl = model_mbl(t_fid, *popt_mbl)
SS_res_mbl = np.sum((p_fid - y_pred_mbl)**2)
R2_mbl = 1 - SS_res_mbl / SS_tot
n_params_mbl = 5
dof_mbl = N_pts - n_params_mbl

dR2_mbl = R2_free - R2_mbl

print(f"  alpha (fixed) = 0.82")
print(f"  A0    = {popt_mbl[0]:.6f} +/- {perr_mbl[0]:.6f}")
print(f"  f     = {popt_mbl[1]:.6f} +/- {perr_mbl[1]:.6f} MHz")
print(f"  T2*   = {popt_mbl[2]:.6f} +/- {perr_mbl[2]:.6f} us")
print(f"  C     = {popt_mbl[3]:.6f} +/- {perr_mbl[3]:.6f}")
print(f"  phi   = {popt_mbl[4]:.6f} +/- {perr_mbl[4]:.6f}")
print(f"\n  R^2   = {R2_mbl:.6f}")
print(f"  delta_R^2 (free - fixed@0.82) = {dR2_mbl:.6f}")
print(f"  DoF   = {dof_mbl}")

F_stat_mbl = ((SS_res_mbl - SS_res) / (n_params_free - n_params_mbl)) / (SS_res / dof_free)
p_val_mbl = 1 - f_dist.cdf(F_stat_mbl, n_params_free - n_params_mbl, dof_free)
print(f"\n  F-test (free vs fixed@0.82):")
print(f"    F = {F_stat_mbl:.4f}, p = {p_val_mbl:.4f}")
print(f"    {'Significant' if p_val_mbl < 0.05 else 'NOT significant'} (p {'<' if p_val_mbl < 0.05 else '>'} 0.05)")

# ==========================================================================
# Step 6: Additional alpha fixed points for context
# ==========================================================================
print("\n" + "=" * 75)
print("ADDITIONAL FIXED-ALPHA COMPARISONS")
print("=" * 75)

for alpha_test in [0.5, 1.0, 1.3, 1.5, 2.0]:
    model_t = osc_stretched_fixed_alpha(alpha_test)
    try:
        popt_t, _ = curve_fit(model_t, t_fid, p_fid, p0=p0_fixed,
                              bounds=(bounds_lo_f, bounds_hi_f), maxfev=50000)
        y_t = model_t(t_fid, *popt_t)
        R2_t = 1 - np.sum((p_fid - y_t)**2) / SS_tot
        dR2_t = R2_free - R2_t
        print(f"  alpha={alpha_test:.1f}: R^2={R2_t:.6f}, dR^2={dR2_t:.6f}")
    except Exception as e:
        print(f"  alpha={alpha_test:.1f}: FAILED ({e})")

# ==========================================================================
# Step 7: BOOTSTRAP for robust uncertainty estimate
# ==========================================================================
print("\n" + "=" * 75)
print("BOOTSTRAP UNCERTAINTY (1000 resamples)")
print("=" * 75)

np.random.seed(42)
n_boot = 1000
alpha_boot = []
residuals = p_fid - y_pred

for i in range(n_boot):
    # Residual bootstrap
    boot_residuals = np.random.choice(residuals, size=N_pts, replace=True)
    y_boot = y_pred + boot_residuals
    try:
        popt_b, _ = curve_fit(osc_stretched, t_fid, y_boot, p0=popt,
                              bounds=(bounds_lo, bounds_hi), maxfev=30000)
        alpha_boot.append(popt_b[3])
    except:
        pass

alpha_boot = np.array(alpha_boot)
n_success = len(alpha_boot)
print(f"  Successful fits: {n_success}/{n_boot}")
print(f"  Bootstrap mean(alpha)   = {np.mean(alpha_boot):.4f}")
print(f"  Bootstrap median(alpha) = {np.median(alpha_boot):.4f}")
print(f"  Bootstrap std(alpha)    = {np.std(alpha_boot):.4f}")
print(f"  Bootstrap 95% CI        = [{np.percentile(alpha_boot, 2.5):.4f}, {np.percentile(alpha_boot, 97.5):.4f}]")
print(f"  Bootstrap IQR           = [{np.percentile(alpha_boot, 25):.4f}, {np.percentile(alpha_boot, 75):.4f}]")

# Compare uncertainty estimates
print(f"\n  Comparison of uncertainty estimates:")
print(f"    curve_fit 1-sigma:     {alpha_err:.4f}")
print(f"    Bootstrap std:         {np.std(alpha_boot):.4f}")
print(f"    Bootstrap 95% CI width: {np.percentile(alpha_boot, 97.5) - np.percentile(alpha_boot, 2.5):.4f}")
print(f"    Wald 95% CI width:     {2 * t_crit * alpha_err:.4f}")

# ==========================================================================
# SUMMARY TABLE
# ==========================================================================
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"""
(1) FITTING METHOD:
    Model: A(t) = C + A0 * exp(-(t/T2*)^alpha) * cos(2*pi*f*t + phi)
    6 free parameters: A0, f, T2*, alpha, C, phi
    Method: Nonlinear least squares (scipy.optimize.curve_fit)
    with Trust Region Reflective algorithm and parameter bounds
    Data: Fig2e1.csv (free induction decay / Ramsey T2* measurement)

(2) UNCERTAINTY TYPE:
    The +/- {alpha_err:.4f} is 1-SIGMA (standard error from covariance matrix)
    Paper rounds to +/- 0.306 (vs our {alpha_err:.4f})
    This is sqrt(pcov[alpha, alpha]) from the Jacobian-based covariance estimate.
    It is NOT a 2-sigma or 95% CI -- it is 1-sigma.

(3) 95% CONFIDENCE INTERVAL:
    Wald (t-distribution):  [{ci_95_lo:.4f}, {ci_95_hi:.4f}]
    Bootstrap (percentile): [{np.percentile(alpha_boot, 2.5):.4f}, {np.percentile(alpha_boot, 97.5):.4f}]
    t-critical = {t_crit:.4f} (dof = {dof_free})

(4) R-SQUARED:
    R^2 = {R2_free:.6f}

(5) NUMBER OF DATA POINTS:
    N = {N_pts}
    k = 6 parameters
    DoF = {dof_free}

FIXED-ALPHA COMPARISONS:
    Free fit:      alpha = {alpha_fit:.4f}, R^2 = {R2_free:.6f}
    Fixed 4/3:     alpha = 1.3333, R^2 = {R2_43:.6f}, dR^2 = {dR2_43:.6f}, F-test p = {p_val_43:.4f}
    Fixed 0.82:    alpha = 0.82,   R^2 = {R2_mbl:.6f}, dR^2 = {dR2_mbl:.6f}, F-test p = {p_val_mbl:.4f}
""")

# ==========================================================================
# Write report to file
# ==========================================================================
report_path = os.path.join(REPORTS_DIR, 'tsoukalas_alpha_verification.txt')
with open(report_path, 'w') as f:
    f.write("TSOUKALAS et al. (2026) -- ALPHA VERIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Data source: Fig2e1.csv (FID / Ramsey T2*)\n")
    f.write(f"N = {N_pts} data points\n")
    f.write(f"t range: [{t_fid[0]:.4f}, {t_fid[-1]:.4f}] us\n\n")

    f.write("FREE FIT (6 parameters)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Model: C + A0*exp(-(t/T2*)^alpha)*cos(2*pi*f*t + phi)\n")
    f.write(f"  Method: NLS (curve_fit, TRF with bounds)\n")
    f.write(f"  alpha  = {alpha_fit:.6f} +/- {alpha_err:.6f} (1-sigma)\n")
    f.write(f"  T2*    = {T2_fit:.6f} +/- {T2_err:.6f} us\n")
    f.write(f"  f      = {f_fit:.6f} +/- {f_err:.6f} MHz\n")
    f.write(f"  R^2    = {R2_free:.6f}\n")
    f.write(f"  DoF    = {dof_free}\n\n")

    f.write("95% CI on alpha\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Wald:      [{ci_95_lo:.4f}, {ci_95_hi:.4f}]\n")
    f.write(f"  Bootstrap: [{np.percentile(alpha_boot, 2.5):.4f}, {np.percentile(alpha_boot, 97.5):.4f}]\n\n")

    f.write("FIXED-ALPHA COMPARISONS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Free:        R^2 = {R2_free:.6f}\n")
    f.write(f"  alpha=4/3:   R^2 = {R2_43:.6f}, dR^2 = {dR2_43:.6f}, F p = {p_val_43:.4f}\n")
    f.write(f"  alpha=0.82:  R^2 = {R2_mbl:.6f}, dR^2 = {dR2_mbl:.6f}, F p = {p_val_mbl:.4f}\n")

print(f"\nReport written to: {report_path}")
print("DONE.")
